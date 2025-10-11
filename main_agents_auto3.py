import os
import json
import time
import logging
import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
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

# === Structured logging ===
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("crewai-event-planner")

# === CACHING LAYER ===
class VendorCache:
    """In-memory vendor cache with TTL and smart invalidation"""
   
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour TTL
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
            print("*******cache updated************")
    
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

# Initialize caches
vendor_cache = VendorCache(ttl_seconds=3600)  # 1 hour
llm_cache = LLMCache(ttl_seconds=7200)  # 2 hours

# === Settings (12-factor) ===
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
        extra = "allow"  # Allow extra fields from environment variables

settings = Settings()

# === DB: SQLAlchemy 2.0 (async) with optimized connection pool ===
class Base(DeclarativeBase):
    pass

class Vendor(Base):
    __tablename__ = "vendors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)  # Add index for faster searches
    service_type: Mapped[str] = mapped_column(String(100), index=True)  # Add index
    city: Mapped[str] = mapped_column(String(100), index=True)  # Add index
    price_min: Mapped[int] = mapped_column(Integer, index=True)  # Add index for price filtering
    price_max: Mapped[int] = mapped_column(Integer, index=True)  # Add index for price filtering
    available_date: Mapped[Optional[date]] = mapped_column(SQLDate, nullable=True, index=True)  # Add index
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
            "jit": "off",  # Disable JIT for faster simple queries
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
            "max_tokens": 1000,  # Limit tokens for faster response
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

# Initialize optimized LLM client
llm_client = OptimizedLLMClient()

# === FAST INTENT EXTRACTION ===
INTENT_SYS = (
    "You classify event-planning queries quickly. "
    "Return ONLY a JSON object (no markdown, no code blocks, no explanation) with keys: "
    "intent (QUERY_VENDORS|PLAN_EVENT|GENERAL_Q|CLARIFY|VENDOR_INFO) and "
    "slots {city,date,service_type,event_type,budget,vendor_name} where available. "
    "Use VENDOR_INFO intent when user asks about specific vendor details/contact. "
    "Dates in YYYY-MM-DD format, budget as integer. "
    "Normalize city names: bangalore/bengaluru -> Bangalore, chennai -> Chennai, mumbai -> Mumbai."
    "Return pure JSON only, no other text."
)

# Precompiled patterns for faster fallback parsing
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

# === OPTIMIZED VENDOR FETCHING ===
async def fetch_vendors_optimized(slots: Dict[str, Any]) -> List[Vendor]:
    """Optimized vendor fetching with caching and efficient queries"""
    
    # Normalize service type for better matching
    if slots.get("service_type"):
        service_type = slots["service_type"].lower()
        # Map common synonyms to database values
        if service_type in ["photography", "photo", "photographer"]:
            slots["service_type"] = "camera"
        elif service_type in ["decor", "decorations"]:
            slots["service_type"] = "decoration"
        elif service_type in ["clean", "cleaner"]:
            slots["service_type"] = "cleaning"
    
    # Check cache first
    if settings.ENABLE_VENDOR_CACHE:
        cache_key_params = {
            "city": slots.get("city"),
            "service_type": slots.get("service_type"), 
            "date": slots.get("date"),
            "vendor_name": slots.get("vendor_name")
        }
        
        cached_data = await vendor_cache.get(**cache_key_params)
        if cached_data:
            # Convert cached data back to Vendor objects
            vendors = [Vendor(**data) for data in cached_data]
            log.info(f"Cache HIT: Found {len(vendors)} vendors")
            return vendors
    
    # Database query with optimized indexes
    async with AsyncSessionLocal() as ses:
        stmt = select(Vendor)
        
        # Build efficient query with indexed columns
        filters = []
        
        if slots.get("vendor_name"):
            from sqlalchemy import func
            vendor_name = slots["vendor_name"].lower()
            filters.append(func.lower(Vendor.name).contains(vendor_name))
        else:
            if slots.get("city"):
                filters.append(Vendor.city == slots["city"])
            
            # For date matching, be more flexible - match if available_date is close
            if slots.get("date"):
                try:
                    y, m, d = map(int, slots["date"].split("-"))
                    target_date = date(y, m, d)
                    # Match exact date OR if no specific date restriction
                    filters.append(Vendor.available_date == target_date)
                except Exception:
                    pass
            
            if slots.get("service_type"):
                filters.append(Vendor.service_type == slots["service_type"])
        
        if filters:
            from sqlalchemy import and_
            stmt = stmt.where(and_(*filters))
        
        # Execute query
        result = await ses.execute(stmt)
        vendors_list = result.scalars().all()
        
        log.info(f"Database query found {len(vendors_list)} vendors for filters: {filters}")
        
        # If no exact matches and we have date filter, try without date constraint
        if not vendors_list and slots.get("date"):
            log.info("No exact date matches, trying without date constraint...")
            filters_no_date = [f for f in filters if "available_date" not in str(f)]
            
            if filters_no_date:
                stmt_no_date = select(Vendor).where(and_(*filters_no_date))
            else:
                stmt_no_date = select(Vendor)
                if slots.get("city"):
                    stmt_no_date = stmt_no_date.where(Vendor.city == slots["city"])
                if slots.get("service_type"):
                    stmt_no_date = stmt_no_date.where(Vendor.service_type == slots["service_type"])
            
            result = await ses.execute(stmt_no_date)
            vendors_list = result.scalars().all()
            log.info(f"Without date constraint found {len(vendors_list)} vendors")
        
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

# === PARALLEL PROCESSING FOR AI MATCHING ===
async def ai_semantic_vendor_match_optimized(vendors: List[Vendor], slots: Dict[str, Any]) -> List[Vendor]:
    """Optimized AI matching with timeouts and fallbacks"""
    if not slots.get("service_type") or len(vendors) <= 3:
        return vendors
    
    try:
        # Fast AI matching with timeout
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

Vendors: {json.dumps(vendor_data[:20], separators=(',', ':'))}  # Limit data for speed
User wants: {slots.get('service_type')}

Return JSON: {{"matched_vendors": [{{"vendor_id": int, "score": int}}]}}
No explanations, just JSON.
"""
        
        # Use lower temperature and shorter prompt for speed
        ai_response = await llm_client.call_llm(ai_prompt, temperature=0.0)
        ai_result = json.loads(ai_response.strip())
        
        matched_data = ai_result.get("matched_vendors", [])
        vendor_lookup = {v.id: v for v in vendors}
        
        result_vendors = []
        for match in matched_data[:5]:  # Limit to top 5
            vendor_id = match.get("vendor_id")
            if vendor_id in vendor_lookup:
                result_vendors.append(vendor_lookup[vendor_id])
        
        return result_vendors or vendors[:5]  # Fallback to first 5
        
    except Exception as e:
        log.warning(f"AI matching failed, using fallback: {e}")
        return strict_fallback_filter_fast(vendors, slots)

def strict_fallback_filter_fast(vendors: List[Vendor], slots: Dict[str, Any]) -> List[Vendor]:
    """Ultra-fast fallback filtering"""
    requested_service = slots.get("service_type", "").lower()
    if not requested_service:
        return vendors[:5]  # Return first 5
    
    # Fast matching
    matched = []
    for vendor in vendors:
        if vendor.service_type.lower() == requested_service:
            matched.append(vendor)
            if len(matched) >= 5:  # Limit results
                break
    
    return matched or vendors[:3]  # Fallback to first 3

# === SIMPLIFIED CREWAI SETUP ===
from crewai import Agent, Task, Crew, LLM as CrewLLM

# Only create CrewAI components when needed
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

# === FASTAPI APP ===
app = FastAPI(title="Optimized AI Event Planner", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === AUTH & RATE LIMITING (Simplified) ===
async def api_key_auth(x_api_key: Optional[str] = Header(default=None)):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Simplified rate limiting
RATE_BUCKETS = {}

async def rate_guard(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    if ip not in RATE_BUCKETS:
        RATE_BUCKETS[ip] = {"tokens": 5, "last": now}
        return
    
    bucket = RATE_BUCKETS[ip]
    elapsed = now - bucket["last"]
    bucket["tokens"] = min(5, bucket["tokens"] + elapsed * 2)  # 2 tokens per second
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
    return {"status": "ok", "cache_enabled": settings.ENABLE_VENDOR_CACHE}

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "vendor_cache_size": len(vendor_cache._cache),
        "llm_cache_size": len(llm_cache._cache),
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
    return {"status": "success", "message": "All caches cleared"}

# === OPTIMIZED CHAT ENDPOINT ===
@app.post("/chat", response_model=ChatOut, dependencies=[Depends(rate_guard), Depends(api_key_auth)])
async def chat_optimized(body: ChatIn):
    """Optimized chat endpoint with performance monitoring"""
    start_time = time.time()
    
    try:
        # 1) Fast intent extraction
        intent_obj = await extract_intent_fast(body.message)
        slots = intent_obj.get("slots", {}) or {}
        intent = intent_obj.get("intent", "CLARIFY")
        
        # 2) Optimized vendor fetching
        vendors_list = await fetch_vendors_optimized(slots)
        
        # 3) Fast response generation based on intent
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
            
        elif intent == "QUERY_VENDORS" and vendors_list:
            # Fast vendor listing without AI processing for simple queries
            reply = f"Found {len(vendors_list)} vendor(s):\n\n"
            
            summary_lines = []
            for v in vendors_list[:5]:  # Limit to 5 for speed
                summary_lines.append(
                    f"• {v.name} - {v.service_type.title()}\n"
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
            
        elif vendors_list:
            # Complex planning - use optimized AI matching
            if settings.PARALLEL_PROCESSING and len(vendors_list) > 5:
                # Use parallel processing for large vendor lists
                matched_vendors = await ai_semantic_vendor_match_optimized(vendors_list, slots)
            else:
                matched_vendors = vendors_list[:3]
            
            summary = "\n".join(
                f"• {v.name} ({v.service_type}) in {v.city} – ₹{v.price_min:,}–₹{v.price_max:,}"
                for v in matched_vendors[:3]
            )
            reply = f"Here are the best options for your event:\n{summary}"
            
            recommendations = [{
                "vendor_id": v.id,
                "name": v.name,
                "service_type": v.service_type,
                "price_range": f"₹{v.price_min:,} - ₹{v.price_max:,}"
            } for v in matched_vendors[:3]]
            
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
        log.error(f"Chat endpoint error: {e}")
        processing_time = time.time() - start_time
        
        return ChatOut(
            reply="I encountered an error processing your request. Please try again.",
            intent={"intent": "ERROR", "slots": {}},
            slots={},
            recommendations=[],
            processing_time=round(processing_time, 3)
        )

# === STARTUP WITH OPTIMIZATIONS ===
@app.on_event("startup")
async def startup_optimized():    
    # Test database
    connection_ok = await test_db_connection()
    if not connection_ok:
        raise Exception("Failed to connect to PostgreSQL database")
    
    # Initialize database with optimized schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Create additional indexes for performance (one at a time)
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_vendors_service_city ON vendors(service_type, city);",
            "CREATE INDEX IF NOT EXISTS idx_vendors_date_city ON vendors(available_date, city);", 
            "CREATE INDEX IF NOT EXISTS idx_vendors_price_range ON vendors(price_min, price_max);"
        ]
        
        for index_sql in index_statements:
            try:
                await conn.execute(text(index_sql))
                log.info(f"Created index: {index_sql.split()[5]}")
            except Exception as e:
                log.warning(f"Index creation failed (may already exist): {e}")
    
    # Seed data
    if settings.SEED_DATA:
        async with AsyncSessionLocal() as ses:
            existing = (await ses.execute(select(Vendor))).scalars().first()
            if not existing:
                vendors = [
                    Vendor(name="Royal Caterers", service_type="food", city="Chennai",
                           price_min=15000, price_max=60000, available_date=date(2025, 10, 24), contact="99999 11111"),
                    Vendor(name="Elite Photography", service_type="camera", city="Chennai",
                           price_min=12000, price_max=45000, available_date=date(2025, 10, 24), contact="99999 22222"),
                    Vendor(name="Floral Decors", service_type="decoration", city="Chennai",
                           price_min=8000, price_max=40000, available_date=date(2025, 10, 24), contact="99999 33333"),
                    Vendor(name="Sparkle Cleaners", service_type="cleaning", city="Chennai",
                           price_min=3000, price_max=12000, available_date=date(2025, 10, 24), contact="99999 44444"),
                    Vendor(name="GlamUp Makeup", service_type="makeup", city="Chennai",
                           price_min=5000, price_max=25000, available_date=date(2025, 10, 24), contact="99999 55555"),
                    Vendor(name="Mumbai Delights", service_type="food", city="Mumbai",
                           price_min=20000, price_max=80000, available_date=date(2025, 10, 24), contact="99999 66666"),
                    Vendor(name="Pixel Perfect Studios", service_type="camera", city="Mumbai",
                           price_min=15000, price_max=50000, available_date=date(2025, 10, 24), contact="99999 77777"),
                    Vendor(name="Bloom & Blossom", service_type="decoration", city="Bangalore",
                           price_min=10000, price_max=45000, available_date=date(2025, 10, 24), contact="99999 88888"),
                    # Additional vendors for Delhi
                    Vendor(name="Luxe Photography", service_type="camera", city="Delhi",
                           price_min=20000, price_max=60000, available_date=date(2025, 10, 30), contact="99999 99999"),
                    Vendor(name="Capital Events", service_type="food", city="Delhi",
                           price_min=25000, price_max=75000, available_date=date(2025, 10, 30), contact="99999 10101"),
                    Vendor(name="Delhi Decorators", service_type="decoration", city="Delhi",
                           price_min=15000, price_max=50000, available_date=date(2025, 10, 30), contact="99999 10102"),
                ]
                ses.add_all(vendors)
                await ses.commit()
                log.info("✅ Seed data added with optimized indexes")
    
    # Pre-warm caches if enabled
    if settings.ENABLE_VENDOR_CACHE:
        log.info("🔥 Pre-warming vendor cache...")
        common_queries = [
            {"city": "Chennai"},
            {"city": "Mumbai"},
            {"city": "Delhi"},
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
    
    log.info("🚀 Optimized service started with performance enhancements")

@app.on_event("shutdown")
async def shutdown_optimized():
    """Clean shutdown with resource cleanup"""
    await llm_client.client.aclose()
    await engine.dispose()
    log.info("🛑 Service shutdown completed")

# === ADDITIONAL OPTIMIZATION ENDPOINTS ===
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
        },
        "settings": {
            "vendor_cache_enabled": settings.ENABLE_VENDOR_CACHE,
            "llm_cache_enabled": settings.ENABLE_LLM_CACHE,
            "parallel_processing": settings.PARALLEL_PROCESSING,
            "db_pool_size": settings.DB_POOL_SIZE,
            "llm_timeout": settings.LLM_TIMEOUT,
        },
        "recommendations": {
            "enable_caching": True,
            "use_database_indexes": True,
            "consider_redis": "For production scale",
            "monitor_llm_costs": True,
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Single worker with async
        loop="uvloop",  # Faster event loop (install: pip install uvloop)
        http="httptools",  # Faster HTTP parser (install: pip install httptools)
        access_log=False,  # Disable access logs for performance
    )