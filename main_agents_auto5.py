import os
import json
import time
import logging
import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from functools import lru_cache
import hashlib
import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Date as SQLDate, select, text

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("crewai-event-planner")

# === CACHING LAYER ===
class VendorCache:
    """In-memory vendor cache with TTL and smart invalidation"""   
    def __init__(self, ttl_seconds: int = 3600):  
        self._cache: Dict[str, Dict] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
        
    def _generate_key(self, **kwargs) -> str:
        """Generate cache key from query parameters"""
        key_parts = []
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:16]
    
    async def get(self, **query_params) -> Optional[List[Dict]]:
        """Get cached vendors"""
        key = self._generate_key(**query_params)
        async with self._lock:
            if key not in self._cache:
                return None
                
            # Check TTL
            if datetime.now() - self._timestamps[key] > timedelta(seconds=self._ttl):
                del self._cache[key]
                del self._timestamps[key]
                return None                
            log.info(f"Cache HIT for key: {key}")
            return self._cache[key]    
    async def set(self, vendors_data: List[Dict], **query_params):
        """Cache vendors data"""
        key = self._generate_key(**query_params)
        async with self._lock:
            self._cache[key] = vendors_data
            self._timestamps[key] = datetime.now()
            log.info(f"Cache SET for key: {key}, {len(vendors_data)} vendors")
    
    async def invalidate_all(self):
        """Clear entire cache"""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            log.info("Cache cleared")

# === LLM RESPONSE CACHE ===
class LLMCache:
    """Cache LLM responses to avoid redundant API calls"""    
    def __init__(self, ttl_seconds: int = 7200):  # 2 hours TTL
        self._cache: Dict[str, Dict] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()
    
    def _generate_key(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        """Generate cache key for LLM request"""
        content = f"{prompt}|{system or ''}|{temperature}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def get(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> Optional[str]:
        """Get cached LLM response"""
        key = self._generate_key(prompt, system, temperature)
        async with self._lock:
            if key not in self._cache:
                return None
                
            # Check TTL
            if datetime.now() - self._timestamps[key] > timedelta(seconds=self._ttl):
                del self._cache[key]
                del self._timestamps[key]
                return None
                
            log.info(f"LLM Cache HIT for key: {key}")
            return self._cache[key]["response"]
    
    async def set(self, response: str, prompt: str, system: Optional[str] = None, temperature: float = 0.2):
        """Cache LLM response"""
        key = self._generate_key(prompt, system, temperature)
        async with self._lock:
            self._cache[key] = {"response": response}
            self._timestamps[key] = datetime.now()
            log.info(f"LLM Cache SET for key: {key}")
    
    async def invalidate_all(self):
        """Clear entire cache"""
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            log.info("LLM Cache cleared")

# Initialize caches
vendor_cache = VendorCache(ttl_seconds=3600)  # 1 hour
llm_cache = LLMCache(ttl_seconds=3600)  # 2 hours

class Settings(BaseSettings):
    API_KEY: Optional[str] = None
    DATABASE_URL: str = "postgresql+asyncpg://postgres:Admin@localhost:5432/agentspace"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    SEED_DATA: bool = True
    CORS_ORIGINS: List[str] = ["*"]
    RATE_LIMIT_RPS: float = 5.0
    
    # Performance settings
    ENABLE_VENDOR_CACHE: bool = True
    ENABLE_LLM_CACHE: bool = True
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    LLM_TIMEOUT: int = 30
    PARALLEL_PROCESSING: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

settings = Settings()

class Base(DeclarativeBase):
    pass

class Vendor(Base):
    __tablename__ = "vendors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True) 
    service_type: Mapped[str] = mapped_column(String(100), index=True) 
    city: Mapped[str] = mapped_column(String(100), index=True)  
    price_min: Mapped[int] = mapped_column(Integer, index=True)  
    price_max: Mapped[int] = mapped_column(Integer, index=True)  
    available_date: Mapped[Optional[date]] = mapped_column(SQLDate, nullable=True, index=True)  
    contact: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

# Optimized connection pool
engine = create_async_engine(
    settings.DATABASE_URL, 
    echo=False, 
    future=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=3600,
    pool_pre_ping=True,
    pool_reset_on_return='commit',
    connect_args={
        "command_timeout": 10,
        "server_settings": {
            "jit": "off",  
        }
    }
)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def test_db_connection():
    """Test PostgreSQL connection"""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version();"))
            version = result.fetchone()
            log.info(f"✅ PostgreSQL connected: {version[0]}")
            return True
    except Exception as e:
        log.error(f"❌ Database connection failed: {e}")
        return False

# === OPTIMIZED LLM CLIENT ===
class OptimizedLLMClient:
    """Optimized LLM client with caching, connection pooling, and timeout handling"""
    
    def __init__(self):
        # Persistent HTTP client with connection pooling
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        self.client = httpx.AsyncClient(
            timeout=settings.LLM_TIMEOUT,
            limits=limits,
            headers={"Connection": "keep-alive"}
        )

    async def call_llm(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
        """Optimized LLM call with caching"""
        
        # Check cache first
        if settings.ENABLE_LLM_CACHE:
            cached_response = await llm_cache.get(prompt, system, temperature)
            if cached_response:
                return cached_response
        
        if not settings.OPENAI_API_KEY: 
            # Faster stub response
            stub_response = f"[LLM-STUB] {prompt[:100]}..."
            if settings.ENABLE_LLM_CACHE:
                await llm_cache.set(stub_response, prompt, system, temperature)
            return stub_response
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
        body = {
            "model": settings.OPENAI_MODEL,
            "temperature": temperature,
            "max_tokens": 1000,
            "messages": ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ],
        }
        
        try:
            r = await self.client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            response = data["choices"][0]["message"]["content"].strip()
            
            # Cache the response
            if settings.ENABLE_LLM_CACHE:
                await llm_cache.set(response, prompt, system, temperature)
            
            return response
            
        except httpx.TimeoutException:
            log.error("LLM request timed out")
            fallback = "[TIMEOUT] Unable to process request quickly enough"
            if settings.ENABLE_LLM_CACHE:
                await llm_cache.set(fallback, prompt, system, temperature)
            return fallback
        except Exception as e:
            log.error(f"LLM request failed: {e}")
            raise

llm_client = OptimizedLLMClient()

# ============================================================================
# 🆕 NEW: AI-POWERED SERVICE TYPE MAPPING
# ============================================================================
class AIServiceMapper:
    """Uses LLM to intelligently map user service requests to database service types"""
    
    def __init__(self):
        self._db_service_types: Set[str] = set()
        self._initialized = False
        self._mapping_cache: Dict[str, List[str]] = {}  
    
    async def initialize(self, engine):
        """Fetch distinct service types from database"""
        if self._initialized:
            return
            
        try:
            async with AsyncSessionLocal() as ses:
                result = await ses.execute(
                    text("SELECT DISTINCT service_type FROM vendors WHERE service_type IS NOT NULL")
                )
                db_types = [row[0].lower() for row in result.fetchall()]
                self._db_service_types = set(db_types)
                
                self._initialized = True
                
                log.info(f"✅ Loaded {len(self._db_service_types)} service types from DB: {sorted(self._db_service_types)}")
                
        except Exception as e:
            log.error(f"Failed to load service types from DB: {e}")
            # Fallback to common types
            self._db_service_types = {"camera", "food", "decoration", "cleaning", "makeup", "photography"}
    
    async def normalize_service_type(self, user_input: str) -> List[str]:
        """
        Use AI to map user input to database service types
        Returns list of matching database service types
        """
        if not user_input:
            return []
        
        user_input_lower = user_input.strip().lower()
        
        # Check cache first
        if user_input_lower in self._mapping_cache:
            log.info(f"🎯 Cache hit for service mapping: '{user_input}' -> {self._mapping_cache[user_input_lower]}")
            return self._mapping_cache[user_input_lower]
        
        # 🔥 ALWAYS use AI mapping to find ALL related types
        # Don't just return exact match - AI will find synonyms too!
        try:
            ai_mapping = await self._ai_map_service(user_input, list(self._db_service_types))
            
            # Validate AI response - only use types that exist in DB
            validated_types = [
                t.lower() for t in ai_mapping 
                if t.lower() in self._db_service_types
            ]
            print(validated_types, "<<<<<<<---- Validated Types from _ai_map_service")
            if validated_types:
                log.info(f"🤖 AI mapped '{user_input}' -> {validated_types}")
                self._mapping_cache[user_input_lower] = validated_types
                return validated_types
            else:
                log.warning(f"⚠️ AI returned no valid mappings for '{user_input}'")
                # Fallback: if exact match exists, use it
                if user_input_lower in self._db_service_types:
                    self._mapping_cache[user_input_lower] = [user_input_lower]
                    return [user_input_lower]
                # Otherwise return original input
                self._mapping_cache[user_input_lower] = [user_input_lower]
                return [user_input_lower]
                
        except Exception as e:
            log.error(f"❌ AI mapping failed for '{user_input}': {e}")
            # Fallback to exact match if exists
            if user_input_lower in self._db_service_types:
                return [user_input_lower]
            return [user_input_lower]
    
    async def _ai_map_service(self, user_service: str, available_types: List[str]) -> List[str]:
        """
        Use LLM to intelligently map user service request to database service types
        """
        system_prompt = """You are a service type mapping expert for event planning.
Your job is to map what users ask for to ALL relevant service types available in the database.

CRITICAL RULES:
1. Return ALL relevant database service types that match the user's request
2. If "photography" and "camera" both exist, return BOTH for photography-related requests
3. Consider synonyms, related services, and common variations
4. Be GENEROUS - include all related types
5. Return ONLY a JSON array of matching service types
6. If no good match exists, return an empty array

Photography/Camera Examples:
- "photography" → ["camera", "photography"] (return BOTH if both exist)
- "photo" → ["camera", "photography"]
- "photographer" → ["camera", "photography"]
- "cameraman" → ["camera", "photography"]
- "camera" → ["camera", "photography"] (return BOTH if both exist)

Other Examples:
- "catering" → ["food"]
- "meals" → ["food"]
- "decor" → ["decoration"]
- "flowers" → ["decoration"]
- "beautician" → ["makeup"]
"""

        user_prompt = f"""User requested: "{user_service}"

Available service types in database: {json.dumps(sorted(available_types))}

Return a JSON array of ALL matching service types from the available list.
If the user asks for photography-related services and BOTH "camera" and "photography" exist in the database, return BOTH.

Return ONLY the JSON array, nothing else.

Example format: ["service1", "service2"]"""

        try:
            response = await llm_client.call_llm(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.0  
            )
            
            response_clean = response.strip()
                        
            if response_clean.startswith("```"):
                start = response_clean.find("[")
                end = response_clean.rfind("]") + 1
                if start != -1 and end > start:
                    response_clean = response_clean[start:end]        
            # Parse JSON
            mapped_types = json.loads(response_clean)
            print(mapped_types, "<<<<<-- AI mapped service types for " , user_service)
            
            if isinstance(mapped_types, list):
                log.info(f"🤖 AI raw response for '{user_service}': {mapped_types}")
                return mapped_types
            else:
                log.warning(f"AI returned non-list response: {mapped_types}")
                return []
                
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse AI response as JSON: {response[:200]} - {e}")
            return []
        except Exception as e:
            log.error(f"AI mapping request failed: {e}")
            raise
    
    def get_all_mappings(self) -> Dict[str, List[str]]:
        """Get all cached mappings for debugging"""
        return dict(self._mapping_cache)

# 🆕 Initialize the AI mapper (REPLACES old service_mapper)
service_mapper = AIServiceMapper()

# === FAST INTENT EXTRACTION ===
INTENT_SYS = (
    "You classify event-planning queries quickly. "
    "Return ONLY a JSON object (no markdown, no code blocks, no explanation) with keys: "
    "intent (QUERY_VENDORS|PLAN_EVENT|GENERAL_Q|CLARIFY|VENDOR_INFO|GREETING) and "
    "slots {city,date,service_type,event_type,budget,vendor_name} where available. "
    "Use VENDOR_INFO intent when user asks about specific vendor details/contact. "
    "Dates in YYYY-MM-DD format, budget as integer. "
    "Normalize city names: bangalore/bengaluru -> Bangalore, chennai -> Chennai, mumbai -> Mumbai."
    "Return pure JSON only, no other text."
)

import re
CITY_PATTERNS = {
    "chennai": "Chennai",
    "mumbai": "Mumbai", 
    "delhi": "Delhi",
    "bengaluru": "Bangalore",
    "bangalore": "Bangalore"
}

SERVICE_PATTERNS = {
    "food": "food",
    "camera": "camera", 
    "photography": "camera",
    "photo": "camera",
    "photographer": "camera",
    "decoration": "decoration",
    "decor": "decoration",
    "cleaning": "cleaning",
    "clean": "cleaning",
    "makeup": "makeup"
}

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
BUDGET_PATTERN = re.compile(r"(\d{4,7})")

async def extract_intent_fast(user_text: str) -> Dict[str, Any]:
    """Optimized intent extraction with faster fallbacks"""
    try:
        # Try LLM extraction (cached)        
        raw = await llm_client.call_llm(            
            f"User message:\n{user_text}\n\nReturn JSON only.",
            system=INTENT_SYS,
            temperature=0.0,
        )
        
        # Fast JSON parsing
        json_text = raw.strip()
        if json_text.startswith("```"):
            start = json_text.find("{")
            end = json_text.rfind("}")
            if start != -1 and end != -1:
                json_text = json_text[start:end+1]
        
        parsed_result = json.loads(json_text)
        return parsed_result
        
    except Exception as e:
        log.warning(f"LLM intent extraction failed, using fast fallback: {e}")
        # Super fast regex-based fallback
        return fast_intent_fallback(user_text)

def fast_intent_fallback(user_text: str) -> Dict[str, Any]:
    """Lightning-fast regex-based intent extraction"""
    t = user_text.lower()
    
    # Fast intent classification
    if any(w in t for w in ["contact", "phone", "details", "info"]):
        intent = "VENDOR_INFO"
    elif any(w in t for w in ["list", "available", "show", "find", "vendors", "get", "fetch"]):
        intent = "QUERY_VENDORS"
    elif any(w in t for w in ["budget", "price", "cost"]):
        intent = "GENERAL_Q"
    else:
        intent = "CLARIFY"
    
    slots: Dict[str, Any] = {}
    
    # Fast pattern matching
    for pattern, normalized in CITY_PATTERNS.items():
        if pattern in t:
            slots["city"] = normalized
            break
    
    for pattern, normalized in SERVICE_PATTERNS.items():
        if pattern in t:
            slots["service_type"] = normalized
            break
    
    # Fast regex matching
    date_match = DATE_PATTERN.search(t)
    if date_match:
        slots["date"] = date_match.group(1)
    
    budget_match = BUDGET_PATTERN.search(t)
    if budget_match:
        slots["budget"] = int(budget_match.group(1))
    
    return {"intent": intent, "slots": slots}

async def fetch_vendors_optimized(slots: Dict[str, Any]) -> List[Vendor]:
    """Optimized vendor fetching with AI-powered service type mapping"""
    
    # Parse and normalize service types using AI mapping
    service_types = []
    if slots.get("service_type"):
        raw_services = slots["service_type"]
        
        # Split by comma if multiple services
        if isinstance(raw_services, str) and "," in raw_services:
            service_list = [s.strip() for s in raw_services.split(",")]
        else:
            service_list = [str(raw_services).strip()]
        
        # Use AI mapper to normalize each service
        for service in service_list:
            normalized_list = await service_mapper.normalize_service_type(service)
            for normalized in normalized_list:
                if normalized not in service_types:
                    service_types.append(normalized)
        
        log.info(f"🔍 AI normalized services: {service_list} -> {service_types}")
    
    # Check cache first (use sorted services for consistent cache key)
    if settings.ENABLE_VENDOR_CACHE:
        cache_key_params = {
            "city": slots.get("city"),
            "service_type": ",".join(sorted(service_types)) if service_types else None,
            "date": slots.get("date"),
            "vendor_name": slots.get("vendor_name"),
            "budget": slots.get("budget")
        }
        cached_data = await vendor_cache.get(**cache_key_params)
        if cached_data:
            vendors = [Vendor(**data) for data in cached_data]
            log.info(f"✅ Cache HIT: Found {len(vendors)} vendors")
            return vendors
    
    async with AsyncSessionLocal() as ses:
        stmt = select(Vendor)
        filters = []
        
        if slots.get("vendor_name"):
            from sqlalchemy import func
            vendor_name = slots["vendor_name"].lower()
            filters.append(func.lower(Vendor.name).contains(vendor_name))
        else:
            if slots.get("city"):
                from sqlalchemy import func
                filters.append(func.lower(Vendor.city) == slots["city"].lower())
            
            # For date matching
            if slots.get("date"):
                try:
                    y, m, d = map(int, slots["date"].split("-"))
                    target_date = date(y, m, d)
                    filters.append(Vendor.available_date == target_date)
                except Exception as e:
                    log.warning(f"Invalid date format: {slots.get('date')} - {e}")
            
            # Use AI-mapped service types (case-insensitive)
            if service_types:
                from sqlalchemy import or_, func
                service_filters = [
                    func.lower(Vendor.service_type) == st.lower() 
                    for st in service_types
                ]
                filters.append(or_(*service_filters))
                log.info(f"🔎 Searching for service types: {service_types}")
        
        if filters:
            from sqlalchemy import and_
            stmt = stmt.where(and_(*filters))
        
        # Execute query
        result = await ses.execute(stmt)
        vendors_list = result.scalars().all()
        
        log.info(f"🎯 Database query found {len(vendors_list)} vendors for AI-mapped services: {service_types}")
        
        # If no results and we had service filters, log for debugging
        if not vendors_list and service_types:
            log.warning(f"⚠️ No vendors found for mapped services {service_types}. Original input: {slots.get('service_type')}")
        
        # Cache the results
        if settings.ENABLE_VENDOR_CACHE:
            vendors_data = [
                {
                    "id": v.id,
                    "name": v.name,
                    "service_type": v.service_type,
                    "city": v.city,
                    "price_min": v.price_min,
                    "price_max": v.price_max,
                    "available_date": v.available_date,
                    "contact": v.contact
                }
                for v in vendors_list
            ]
            await vendor_cache.set(vendors_data, **cache_key_params)
        
        return list(vendors_list)

async def find_best_vendor_combination(vendors: List[Vendor], requested_services: List[str], budget: int) -> Tuple[List[Vendor], int, bool]:
    """
    Find the best combination of vendors that:
    1. Covers all requested services (one vendor per service)
    2. Fits within the budget
    3. Minimizes total cost while maximizing quality
    
    Returns: (selected_vendors, total_cost, fits_budget)
    """
    from itertools import product    
    # Group vendors by service type
    vendors_by_service = {}
    for service in requested_services:
        vendors_by_service[service] = [v for v in vendors if v.service_type == service]    
    # Check if we have vendors for all services
    missing_services = [s for s in requested_services if not vendors_by_service.get(s)]
    if missing_services:
        log.warning(f"Missing vendors for services: {missing_services}")
        # Return best available
        available_vendors = [v for v in vendors if v.service_type in requested_services]
        total = sum(v.price_min for v in available_vendors)
        return available_vendors, total, False
    
    service_lists = [vendors_by_service[service] for service in requested_services]
    all_combinations = list(product(*service_lists))
    
    log.info(f"Evaluating {len(all_combinations)} vendor combinations for {len(requested_services)} services")
    
    valid_combinations = []
    for combo in all_combinations:
        min_cost = sum(v.price_min for v in combo)
        max_cost = sum(v.price_max for v in combo)
        
        if min_cost <= budget:
            # Score: prefer combinations where max_cost is also close to budget
            score = budget - min_cost  # Higher score = more budget left
            valid_combinations.append({
                "vendors": combo,
                "min_cost": min_cost,
                "max_cost": max_cost,
                "score": score
            })
    
    if not valid_combinations:
        # No combination fits budget - return cheapest option anyway
        log.warning(f"No combinations fit budget {budget}, returning cheapest")
        cheapest_combo = min(all_combinations, key=lambda c: sum(v.price_min for v in c))
        total_cost = sum(v.price_min for v in cheapest_combo)
        return list(cheapest_combo), total_cost, False
    
    # Sort by score (prefer leaving some buffer in budget)
    def combination_quality(combo_dict):
        # Prefer combinations where even max price fits budget
        max_fits = combo_dict["max_cost"] <= budget
        return (max_fits, combo_dict["score"])
    
    valid_combinations.sort(key=combination_quality, reverse=True)
    best_combo = valid_combinations[0]
    
    log.info(f"Best combination: min={best_combo['min_cost']}, max={best_combo['max_cost']}, budget={budget}")
    
    return list(best_combo["vendors"]), best_combo["min_cost"], True

# === PARALLEL PROCESSING FOR AI MATCHING ===
async def ai_semantic_vendor_match_optimized(vendors: List[Vendor], slots: Dict[str, Any]) -> List[Vendor]:
    """Optimized AI matching with timeouts and fallbacks"""
    if not slots.get("service_type") or len(vendors) <= 3:
        return vendors
    
    try:
        vendor_data = [
            {
                "id": v.id,
                "name": v.name,
                "service_type": v.service_type,
                "city": v.city,
                "price_min": v.price_min,
                "price_max": v.price_max
            } for v in vendors
        ]
        
        ai_prompt = f"""
Match vendors to user request quickly. Return top 5 matches only.

Vendors: {json.dumps(vendor_data[:20], separators=(',', ':'))}
User wants: {slots.get('service_type')}

Return JSON: {{"matched_vendors": [{{"vendor_id": int, "score": int}}]}}
No explanations, just JSON.
"""
        
        ai_response = await llm_client.call_llm(ai_prompt, temperature=0.0)
        ai_result = json.loads(ai_response.strip())
        
        matched_data = ai_result.get("matched_vendors", [])
        vendor_lookup = {v.id: v for v in vendors}
        
        result_vendors = []
        for match in matched_data[:5]:
            vendor_id = match.get("vendor_id")
            if vendor_id in vendor_lookup:
                result_vendors.append(vendor_lookup[vendor_id])
        
        return result_vendors or vendors[:5]
        
    except Exception as e:
        log.warning(f"AI matching failed, using fallback: {e}")
        return strict_fallback_filter_fast(vendors, slots)

def strict_fallback_filter_fast(vendors: List[Vendor], slots: Dict[str, Any]) -> List[Vendor]:
    """Ultra-fast fallback filtering"""
    requested_service = slots.get("service_type", "").lower()
    if not requested_service:
        return vendors[:5]
    
    matched = []
    for vendor in vendors:
        if vendor.service_type.lower() == requested_service:
            matched.append(vendor)
            if len(matched) >= 5:
                break
    
    return matched or vendors[:3]

# === SIMPLIFIED CREWAI SETUP ===
from crewai import Agent, Task, Crew, LLM as CrewLLM

crewai_llm = None

def get_crewai_components():
    """Lazy initialization of CrewAI components"""
    global crewai_llm
    if crewai_llm is None:
        crewai_llm = CrewLLM(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
    
    planner_agent = Agent(
        role="Event Planner",
        goal="Quick event planning",
        backstory="Expert planner",
        llm=crewai_llm,
    )
    
    recommend_task = Task(
        description=(
            "Vendors: {vendors_json}\n"
            "User: {user_query}\n"
            "Pick 3 best vendors. Return JSON: "
            '{"recommendations": [{"vendor_id": int, "name": str, "service_type": str}], "message": str}'
        ),
        agent=planner_agent,
        expected_output="JSON with recommendations and message"
    )
    
    crew = Crew(
        agents=[planner_agent],
        tasks=[recommend_task],
        llm=crewai_llm,
        verbose=False,
    )
    
    return crew

app = FastAPI(title="AI Event Planner with AI Service Mapping", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === AUTH & RATE LIMITING ===
async def api_key_auth(x_api_key: Optional[str] = Header(default=None)):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

RATE_BUCKETS = {}

async def rate_guard(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    if ip not in RATE_BUCKETS:
        RATE_BUCKETS[ip] = {"tokens": 5, "last": now}
        return
    
    bucket = RATE_BUCKETS[ip]
    elapsed = now - bucket["last"]
    bucket["tokens"] = min(5, bucket["tokens"] + elapsed * 2)
    bucket["last"] = now
    
    if bucket["tokens"] >= 1:
        bucket["tokens"] -= 1
    else:
        raise HTTPException(429, "Rate limit exceeded")

# === SCHEMAS ===
class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    intent: Dict[str, Any]
    slots: Dict[str, Any]
    recommendations: Optional[List[Dict[str, Any]]] = None
    processing_time: float

class VendorOut(BaseModel):
    id: int
    name: str
    service_type: str
    city: str
    price_min: int
    price_max: int
    available_date: Optional[date] = None
    contact: Optional[str] = None

# === ENDPOINTS ===
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok", 
        "cache_enabled": settings.ENABLE_VENDOR_CACHE,
        "ai_service_mapper": "enabled",
        "db_service_types": sorted(list(service_mapper._db_service_types)) if service_mapper._initialized else []
    }

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "vendor_cache_size": len(vendor_cache._cache),
        "llm_cache_size": len(llm_cache._cache),
        "service_mapping_cache_size": len(service_mapper._mapping_cache),
        "settings": {
            "vendor_cache_enabled": settings.ENABLE_VENDOR_CACHE,
            "llm_cache_enabled": settings.ENABLE_LLM_CACHE,
        }
    }

@app.post("/cache/clear")
async def clear_caches():
    """Clear all caches"""
    await vendor_cache.invalidate_all()
    await llm_cache.invalidate_all()
    service_mapper._mapping_cache.clear()
    return {"status": "success", "message": "All caches cleared (vendor, LLM, and service mapping)"}

@app.post("/chat", response_model=ChatOut, dependencies=[Depends(rate_guard), Depends(api_key_auth)])
async def chat_optimized(body: ChatIn):
    """Enhanced chat endpoint with AI service mapping and smart budget combination matching"""
    start_time = time.time()    
    try:
        intent_obj = await extract_intent_fast(body.message)
        log.info(f"Level 1 Extraction:{intent_obj}")
        slots = intent_obj.get("slots", {}) or {}        
        intent = intent_obj.get("intent", "CLARIFY")
        if intent == "GREETING" or intent == "GENERAL_Q":
            reply = (
                "Hi! I'm Zing — your event vendor assistant. 🤖\n\n"
                "I can help you find vendors (photography, catering, decoration, makeup, cleaning, etc.) "
                "and compare prices. Tell me your city, date (YYYY-MM-DD), service type, and budget, "
                "and I'll find suitable vendors and recommendations.\n\n"
                "Examples:\n"
                "• \"Find a photographer in Chennai on 2025-10-24\"\n"
                "• \"Catering in Mumbai for 20000\"\n\n"
                "How can I help you today?"
            )
            processing_time = time.time() - start_time
            return ChatOut(
                reply=reply,
                intent=intent_obj,
                slots=slots,
                recommendations=[],
                processing_time=round(processing_time, 3)
            )
        service_types = []
        if slots.get("service_type"):
            raw_services = slots["service_type"]
            if isinstance(raw_services, str) and "," in raw_services:
                service_list = [s.strip().lower() for s in raw_services.split(",")]
            else:
                service_list = [str(raw_services).strip().lower()]
            
            service_types = service_list
        
        vendors_list = await fetch_vendors_optimized(slots)
        budget = slots.get("budget")
        
        if intent == "VENDOR_INFO" and vendors_list:
            vendor = vendors_list[0]
            contact_info = f"Contact: {vendor.contact}" if vendor.contact else "Contact information not available"
            available_date = f"Available: {vendor.available_date.strftime('%Y-%m-%d')}" if vendor.available_date else "Availability not specified"
            
            reply = f"Here are the details for {vendor.name}:\n"
            reply += f"Service: {vendor.service_type.title()}\n"
            reply += f"Location: {vendor.city}\n"
            reply += f"Price Range: ₹{vendor.price_min:,} - ₹{vendor.price_max:,}\n"
            reply += f"{available_date}\n"
            reply += contact_info
            
            recommendations = [{
                "vendor_id": vendor.id,
                "name": vendor.name,
                "service_type": vendor.service_type,
                "contact": vendor.contact,
                "price_range": f"₹{vendor.price_min:,} - ₹{vendor.price_max:,}"
            }]
            
        elif vendors_list and len(service_types) > 1 and budget:
            log.info(f"Finding best combination for {service_types} within budget {budget}")
            
            mapped_services = []
            for service in service_types:
                mapped = await service_mapper.normalize_service_type(service)
                mapped_services.extend(mapped)
            
            selected_vendors, total_cost, fits_budget = await find_best_vendor_combination(
                vendors_list, list(set(mapped_services)), budget
            )
            
            if fits_budget:
                reply = f"✅ Perfect! I found a great combination within your ₹{budget:,} budget:\n\n"
                
                summary_lines = []
                for v in selected_vendors:
                    summary_lines.append(
                        f"• {v.name} ({v.service_type.title()})\n"
                        f"  Price: ₹{v.price_min:,} - ₹{v.price_max:,}\n"
                        f"  Contact: {v.contact or 'Not provided'}"
                    )
                
                reply += "\n\n".join(summary_lines)
                reply += f"\n\n💰 Total Cost Range: ₹{total_cost:,} - ₹{sum(v.price_max for v in selected_vendors):,}"
                reply += f"\n📊 Budget Utilization: ₹{total_cost:,} / ₹{budget:,} ({int(total_cost/budget*100)}% of budget)"
                
                max_total = sum(v.price_max for v in selected_vendors)
                if max_total <= budget:
                    reply += f"\n✨ Great news! Even at maximum prices (₹{max_total:,}), you're within budget!"
                else:
                    reply += f"\n⚠️ Note: Maximum prices (₹{max_total:,}) exceed budget. Negotiate for better rates!"
                
            else:
                reply = f"⚠️ I found vendors for your services, but they exceed your ₹{budget:,} budget:\n\n"
                
                summary_lines = []
                for v in selected_vendors:
                    summary_lines.append(
                        f"• {v.name} ({v.service_type.title()})\n"
                        f"  Price: ₹{v.price_min:,} - ₹{v.price_max:,}\n"
                        f"  Contact: {v.contact or 'Not provided'}"
                    )
                
                reply += "\n\n".join(summary_lines)
                reply += f"\n\n💰 Minimum Total: ₹{total_cost:,}"
                reply += f"\n📊 Over Budget By: ₹{total_cost - budget:,}"
                reply += f"\n\n💡 Suggestions:\n"
                reply += f"  • Increase budget to ₹{total_cost:,}\n"
                reply += f"  • Negotiate prices with vendors\n"
                reply += f"  • Consider fewer services"
            
            recommendations = [{
                "vendor_id": v.id,
                "name": v.name,
                "service_type": v.service_type,
                "contact": v.contact,
                "price_range": f"₹{v.price_min:,} - ₹{v.price_max:,}",
                "price_min": v.price_min,
                "price_max": v.price_max
            } for v in selected_vendors]
            
            # Add budget summary to recommendations
            recommendations.append({
                "summary": True,
                "total_min": total_cost,
                "total_max": sum(v.price_max for v in selected_vendors),
                "budget": budget,
                "fits_budget": fits_budget
            })
            
        elif vendors_list and budget:
            affordable_vendors = [v for v in vendors_list if v.price_min <= budget]
            
            if affordable_vendors:
                reply = f"Found {len(affordable_vendors)} vendor(s) within your ₹{budget:,} budget:\n\n"
                
                summary_lines = []
                for v in affordable_vendors[:5]:
                    summary_lines.append(
                        f"• {v.name} ({v.service_type.title()})\n"
                        f"  Price: ₹{v.price_min:,} - ₹{v.price_max:,}\n"
                        f"  Contact: {v.contact or 'Not provided'}"
                    )
                
                reply += "\n\n".join(summary_lines)
            else:
                cheapest = min(vendors_list, key=lambda v: v.price_min)
                reply = f"⚠️ No vendors found within ₹{budget:,} budget.\n\n"
                reply += f"Cheapest option:\n"
                reply += f"• {cheapest.name} - ₹{cheapest.price_min:,} - ₹{cheapest.price_max:,}\n"
                reply += f"  (₹{cheapest.price_min - budget:,} over budget)"
                affordable_vendors = [cheapest]
            
            recommendations = [{
                "vendor_id": v.id,
                "name": v.name,
                "service_type": v.service_type,
                "contact": v.contact,
                "price_range": f"₹{v.price_min:,} - ₹{v.price_max:,}"
            } for v in affordable_vendors[:5]]
            
        elif vendors_list:
            # NO BUDGET - Just list vendors
            reply = f"Found {len(vendors_list)} vendor(s):\n\n"
            
            summary_lines = []
            for v in vendors_list[:5]:
                summary_lines.append(
                    f"• {v.name} ({v.service_type.title()})\n"
                    f"  Price: ₹{v.price_min:,} - ₹{v.price_max:,}\n"
                    f"  Contact: {v.contact or 'Not provided'}"
                )
            
            reply += "\n\n".join(summary_lines)
            
            recommendations = [{
                "vendor_id": v.id,
                "name": v.name,
                "service_type": v.service_type,
                "contact": v.contact,
                "price_range": f"₹{v.price_min:,} - ₹{v.price_max:,}"
            } for v in vendors_list[:5]]
            
        else:
            reply = "I couldn't find matching vendors. Please specify city, date (YYYY-MM-DD), or service type."
            recommendations = []
        
        processing_time = time.time() - start_time
        
        return ChatOut(
            reply=reply,
            intent=intent_obj,
            slots=slots,
            recommendations=recommendations,
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        log.error(f"Chat endpoint error: {e}", exc_info=True)
        processing_time = time.time() - start_time
        
        return ChatOut(
            reply="I encountered an error processing your request. Please try again.",
            intent={"intent": "ERROR", "slots": {}},
            slots={},
            recommendations=[],
            processing_time=round(processing_time, 3)
        )

@app.get("/vendors/fast", response_model=List[VendorOut])
async def vendors_fast(
    city: Optional[str] = None,
    service_type: Optional[str] = None,
    limit: int = 10
):
    """Ultra-fast vendor endpoint with minimal processing"""
    slots = {}
    if city: slots["city"] = city
    if service_type: slots["service_type"] = service_type
    
    vendors = await fetch_vendors_optimized(slots)
    limited_vendors = vendors[:limit]
    
    return [
        VendorOut(
            id=v.id, name=v.name, service_type=v.service_type, city=v.city,
            price_min=v.price_min, price_max=v.price_max,
            available_date=v.available_date, contact=v.contact
        ) for v in limited_vendors
    ]

@app.get("/performance/metrics")
async def performance_metrics():
    """Get performance metrics"""
    return {
        "cache_stats": {
            "vendor_cache_size": len(vendor_cache._cache),
            "llm_cache_size": len(llm_cache._cache),
            "service_mapping_cache_size": len(service_mapper._mapping_cache),
        },
        "settings": {
            "vendor_cache_enabled": settings.ENABLE_VENDOR_CACHE,
            "llm_cache_enabled": settings.ENABLE_LLM_CACHE,
            "parallel_processing": settings.PARALLEL_PROCESSING,
            "db_pool_size": settings.DB_POOL_SIZE,
            "llm_timeout": settings.LLM_TIMEOUT,
        },
        "ai_service_mapper": {
            "initialized": service_mapper._initialized,
            "db_service_types": sorted(list(service_mapper._db_service_types)),
            "cached_mappings_count": len(service_mapper._mapping_cache),
        },
        "recommendations": {
            "enable_caching": True,
            "use_database_indexes": True,
            "ai_service_mapping": "enabled",
            "consider_redis": "For production scale",
            "monitor_llm_costs": True,
        }
    }


@app.get("/service-types/mappings")
async def get_service_mappings():
    """Get all available service types and AI mapping cache"""
    return {
        "database_service_types": sorted(list(service_mapper._db_service_types)),
        "ai_cached_mappings": service_mapper.get_all_mappings(),
        "total_cached_mappings": len(service_mapper._mapping_cache),
        "example_usage": {
            "description": "AI will map user inputs to these DB types automatically",
            "test_endpoint": "POST /service-types/test-mapping"
        }
    }

@app.post("/service-types/test-mapping")
async def test_service_mapping(body: dict):
    """Test AI mapping for a specific service"""
    service_input = body.get("service")
    if not service_input:
        raise HTTPException(400, "Provide 'service' in request body")
    
    start_time = time.time()
    mapped_types = await service_mapper.normalize_service_type(service_input)
    processing_time = time.time() - start_time
    
    return {
        "input": service_input,
        "mapped_to": mapped_types,
        "available_types": sorted(list(service_mapper._db_service_types)),
        "processing_time": round(processing_time, 3),
        "cached": service_input.lower() in service_mapper._mapping_cache,
        "explanation": f"User input '{service_input}' was mapped to {len(mapped_types)} database service type(s)"
    }

@app.post("/service-types/batch-test")
async def batch_test_mappings(body: dict):
    """Test multiple service mappings at once"""
    services = body.get("services", [])
    if not services or not isinstance(services, list):
        raise HTTPException(400, "Provide 'services' as array in request body")
    
    results = {}
    start_time = time.time()
    
    for service in services:
        try:
            mapped = await service_mapper.normalize_service_type(service)
            results[service] = {
                "mapped_to": mapped,
                "success": True,
                "cached": service.lower() in service_mapper._mapping_cache
            }
        except Exception as e:
            results[service] = {
                "mapped_to": [],
                "success": False,
                "error": str(e)
            }
    
    total_time = time.time() - start_time
    
    return {
        "results": results,
        "total_processing_time": round(total_time, 3),
        "average_time_per_service": round(total_time / len(services), 3) if services else 0,
        "services_tested": len(services)
    }

@app.post("/service-types/clear-cache")
async def clear_service_mapping_cache():
    """Clear the AI service mapping cache"""
    cache_size = len(service_mapper._mapping_cache)
    service_mapper._mapping_cache.clear()
    return {
        "status": "success",
        "message": "AI service mapping cache cleared",
        "mappings_cleared": cache_size
    }

@app.on_event("startup")
async def startup_optimized():    
    connection_ok = await test_db_connection()
    if not connection_ok:
        raise Exception("Failed to connect to PostgreSQL database")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_vendors_service_city ON vendors(LOWER(service_type), LOWER(city));",
            "CREATE INDEX IF NOT EXISTS idx_vendors_date_city ON vendors(available_date, LOWER(city));", 
            "CREATE INDEX IF NOT EXISTS idx_vendors_price_range ON vendors(price_min, price_max);",
            "CREATE INDEX IF NOT EXISTS idx_vendors_service_type_lower ON vendors(LOWER(service_type));"
        ]
        
        for index_sql in index_statements:
            try:
                await conn.execute(text(index_sql))
                log.info(f"✅ Created index: {index_sql.split()[5] if len(index_sql.split()) > 5 else 'index'}")
            except Exception as e:
                log.warning(f"⚠️ Index creation failed (may already exist): {e}")
    
    # 🆕 CRITICAL: Initialize the AI service mapper
    await service_mapper.initialize(engine)
    
    # 🆕 Pre-warm AI mappings for common queries
    if settings.ENABLE_LLM_CACHE:
        log.info("🤖 Pre-warming AI service mappings...")
        common_services = [
            "photo", "photographer", "cameraman",  
            "catering", "caterer", "meals",
            "decor", "flowers",
            "cleaner",
            "beauty"
        ]
        
        for service in common_services:
            try:
                await service_mapper.normalize_service_type(service)
            except Exception as e:
                log.warning(f"Failed to pre-warm mapping for '{service}': {e}")
        
        log.info("✅ AI mapping pre-warming completed")
    
    if settings.ENABLE_VENDOR_CACHE:
        log.info("🔥 Pre-warming vendor cache...")
        common_queries = [
            {"city": "Chennai"},
            {"city": "Mumbai"},
            {"city": "Delhi"},
            {"service_type": "photography"},
            {"service_type": "camera"},
            {"service_type": "food"},
            {"service_type": "decoration"},
        ]
        
        for query in common_queries:
            try:
                await fetch_vendors_optimized(query)
            except Exception as e:
                log.warning(f"Cache pre-warming failed for {query}: {e}")
        
        log.info("✅ Cache pre-warming completed")
    
    log.info("🚀 Optimized service started with AI-POWERED service type mapping")

@app.on_event("shutdown")
async def shutdown_optimized():
    """Clean shutdown with resource cleanup"""
    await llm_client.client.aclose()
    await engine.dispose()
    log.info("🛑 Service shutdown completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        loop="uvloop",
        http="httptools",
        access_log=False,
    )