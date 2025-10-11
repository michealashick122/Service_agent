import os
import json
import time
import logging
import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
import hashlib
import uuid

import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Date as SQLDate, select, text, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

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

# New models based on your Prisma schema
class VendorProfile(Base):
    __tablename__ = "vendor_profiles"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # Add other vendor profile fields as needed
    name: Mapped[str] = mapped_column(String(255), index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    
    # Relationships
    vendor_requested_subcategories: Mapped[List["VendorRequestedSubcategory"]] = relationship(
        "VendorRequestedSubcategory", back_populates="vendor"
    )

class ServiceGroup(Base):
    __tablename__ = "service_groups"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Relationships
    vendor_requested_subcategories: Mapped[List["VendorRequestedSubcategory"]] = relationship(
        "VendorRequestedSubcategory", back_populates="subcategory"
    )

class VendorRequestedSubcategory(Base):
    __tablename__ = "vendor_requested_subcategories"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    vendor_id: Mapped[str] = mapped_column(String, ForeignKey("vendor_profiles.id"), index=True)
    subcategory_id: Mapped[str] = mapped_column(String, ForeignKey("service_groups.id"), index=True)
    approved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    base_charge_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    base_charge_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    hourly_charge_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hourly_charge_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    location: Mapped[str] = mapped_column(String(255), index=True)
    
    # Relationships
    vendor: Mapped["VendorProfile"] = relationship("VendorProfile", back_populates="vendor_requested_subcategories")
    subcategory: Mapped["ServiceGroup"] = relationship("ServiceGroup", back_populates="vendor_requested_subcategories")

# Keep the old Vendor model for backward compatibility
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
    # Additional optimizations
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

# === ENHANCED VENDOR FETCHING FOR NEW MODEL ===
async def fetch_vendors_from_new_model(slots: Dict[str, Any]) -> List[VendorRequestedSubcategory]:
    """Fetch vendors from the new VendorRequestedSubcategory model"""
    
    # Check cache first
    if settings.ENABLE_VENDOR_CACHE:
        cache_key_params = {
            "city": slots.get("city"),
            "service_type": slots.get("service_type"),
            "approved": True,  # Only approved vendors
            "vendor_name": slots.get("vendor_name")
        }
        
        cached_data = await vendor_cache.get(**cache_key_params)
        if cached_data:
            # Convert cached data back to VendorRequestedSubcategory objects
            vendors = [VendorRequestedSubcategory(**data) for data in cached_data]
            log.info(f"Cache HIT: Found {len(vendors)} approved vendor services")
            return vendors
    
    # Database query with optimized joins
    async with AsyncSessionLocal() as session:
        stmt = (
            select(VendorRequestedSubcategory)
            .join(VendorProfile, VendorRequestedSubcategory.vendor_id == VendorProfile.id)
            .join(ServiceGroup, VendorRequestedSubcategory.subcategory_id == ServiceGroup.id)
            .where(VendorRequestedSubcategory.approved == True)
        )
        
        # Build efficient query filters
        filters = []
        
        if slots.get("vendor_name"):
            from sqlalchemy import func
            vendor_name = slots["vendor_name"].lower()
            filters.append(func.lower(VendorProfile.name).contains(vendor_name))
        
        if slots.get("city"):
            # Check both vendor location and service location
            from sqlalchemy import or_
            city_filter = or_(
                VendorProfile.city == slots["city"],
                VendorRequestedSubcategory.location.ilike(f"%{slots['city']}%")
            )
            filters.append(city_filter)
        
        if slots.get("service_type"):
            # Map service type to service group name
            service_type = slots["service_type"]
            if service_type == "camera":
                service_type = "photography"
            filters.append(ServiceGroup.name.ilike(f"%{service_type}%"))
        
        if slots.get("budget"):
            budget = float(slots["budget"])
            # Filter by base charge or actual price
            from sqlalchemy import and_, or_
            budget_filter = or_(
                and_(
                    VendorRequestedSubcategory.base_charge_min <= budget,
                    VendorRequestedSubcategory.base_charge_max >= budget
                ),
                VendorRequestedSubcategory.actual_price <= budget
            )
            filters.append(budget_filter)
        
        if filters:
            from sqlalchemy import and_
            stmt = stmt.where(and_(*filters))
        
        # Order by recent requests and price
        stmt = stmt.order_by(
            VendorRequestedSubcategory.requested_at.desc(),
            VendorRequestedSubcategory.base_charge_min.asc()
        )
        
        result = await session.execute(stmt)
        vendors_list = result.scalars().all()
        
        log.info(f"Database query found {len(vendors_list)} approved vendor services")
        
        # Cache the results
        if settings.ENABLE_VENDOR_CACHE:
            vendors_data = [
                {
                    "id": v.id,
                    "vendor_id": v.vendor_id,
                    "subcategory_id": v.subcategory_id,
                    "approved": v.approved,
                    "requested_at": v.requested_at,
                    "approved_at": v.approved_at,
                    "base_charge_max": v.base_charge_max,
                    "base_charge_min": v.base_charge_min,
                    "hourly_charge_max": v.hourly_charge_max,
                    "hourly_charge_min": v.hourly_charge_min,
                    "actual_price": v.actual_price,
                    "title": v.title,
                    "location": v.location
                }
                for v in vendors_list
            ]
            await vendor_cache.set(vendors_data, **cache_key_params)
        
        return list(vendors_list)

# Keep the old optimized vendor fetching for backward compatibility
async def fetch_vendors_optimized(slots: Dict[str, Any]) -> List[Vendor]:
    """Optimized vendor fetching with caching and efficient queries (legacy)"""
    
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
app = FastAPI(title="Optimized AI Event Planner", version="0.4.0")

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
    use_new_model: bool = True  # Toggle between old and new vendor model

class ChatOut(BaseModel):
    reply: str
    intent: Dict[str, Any]
    slots: Dict[str, Any]
    recommendations: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    model_used: str  # Indicate which model was used

class VendorOut(BaseModel):
    id: int
    name: str
    service_type: str
    city: str
    price_min: int
    price_max: int
    available_date: Optional[date] = None
    contact: Optional[str] = None

class VendorRequestedSubcategoryOut(BaseModel):
    id: str
    vendor_id: str
    vendor_name: Optional[str] = None
    subcategory_id: str
    subcategory_name: Optional[str] = None
    title: Optional[str] = None
    location: str
    base_charge_min: Optional[float] = None
    base_charge_max: Optional[float] = None
    hourly_charge_min: Optional[float] = None
    hourly_charge_max: Optional[float] = None
    actual_price: Optional[float] = None
    approved: bool
    requested_at: datetime
    approved_at: Optional[datetime] = None

class VendorRequestedSubcategoryCreate(BaseModel):
    vendor_id: str
    subcategory_id: str
    title: Optional[str] = None
    location: str
    base_charge_min: Optional[float] = None
    base_charge_max: Optional[float] = None
    hourly_charge_min: Optional[float] = None
    hourly_charge_max: Optional[float] = None
    actual_price: Optional[float] = None

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

# === NEW MODEL ENDPOINTS ===
@app.get("/vendor-services", response_model=List[VendorRequestedSubcategoryOut])
async def get_vendor_services(
    city: Optional[str] = None,
    service_type: Optional[str] = None,
    approved_only: bool = True,
    limit: int = 10
):
    """Get vendor services from the new model"""
    slots = {}
    if city: slots["city"] = city
    if service_type: slots["service_type"] = service_type
    
    # Use the new model fetching
    vendors = await fetch_vendors_from_new_model(slots)
    limited_vendors = vendors[:limit]
    
    # Get related data for better response
    async with AsyncSessionLocal() as session:
        result = []
        for v in limited_vendors:
            # Get vendor and subcategory names
            vendor_stmt = select(VendorProfile).where(VendorProfile.id == v.vendor_id)
            subcategory_stmt = select(ServiceGroup).where(ServiceGroup.id == v.subcategory_id)
            
            vendor_result = await session.execute(vendor_stmt)
            subcategory_result = await session.execute(subcategory_stmt)
            
            vendor_profile = vendor_result.scalar_one_or_none()
            subcategory = subcategory_result.scalar_one_or_none()
            
            result.append(VendorRequestedSubcategoryOut(
                id=v.id,
                vendor_id=v.vendor_id,
                vendor_name=vendor_profile.name if vendor_profile else None,
                subcategory_id=v.subcategory_id,
                subcategory_name=subcategory.name if subcategory else None,
                title=v.title,
                location=v.location,
                base_charge_min=v.base_charge_min,
                base_charge_max=v.base_charge_max,
                hourly_charge_min=v.hourly_charge_min,
                hourly_charge_max=v.hourly_charge_max,
                actual_price=v.actual_price,
                approved=v.approved,
                requested_at=v.requested_at,
                approved_at=v.approved_at
            ))
        
        return result

@app.post("/vendor-services", response_model=VendorRequestedSubcategoryOut)
async def create_vendor_service(
    service_data: VendorRequestedSubcategoryCreate,
    dependencies=[Depends(api_key_auth)]
):
    """Create a new vendor service request"""
    async with AsyncSessionLocal() as session:
        # Check if vendor and subcategory exist
        vendor_stmt = select(VendorProfile).where(VendorProfile.id == service_data.vendor_id)
        subcategory_stmt = select(ServiceGroup).where(ServiceGroup.id == service_data.subcategory_id)
        
        vendor_result = await session.execute(vendor_stmt)
        subcategory_result = await session.execute(subcategory_stmt)
        
        vendor_profile = vendor_result.scalar_one_or_none()
        subcategory = subcategory_result.scalar_one_or_none()
        
        if not vendor_profile:
            raise HTTPException(status_code=404, detail="Vendor not found")
        if not subcategory:
            raise HTTPException(status_code=404, detail="Service category not found")
        
        # Create new vendor service
        new_service = VendorRequestedSubcategory(
            vendor_id=service_data.vendor_id,
            subcategory_id=service_data.subcategory_id,
            title=service_data.title,
            location=service_data.location,
            base_charge_min=service_data.base_charge_min,
            base_charge_max=service_data.base_charge_max,
            hourly_charge_min=service_data.hourly_charge_min,
            hourly_charge_max=service_data.hourly_charge_max,
            actual_price=service_data.actual_price
        )
        
        session.add(new_service)
        await session.commit()
        await session.refresh(new_service)
        
        # Clear cache to ensure fresh data
        if settings.ENABLE_VENDOR_CACHE:
            await vendor_cache.invalidate_all()
        
        return VendorRequestedSubcategoryOut(
            id=new_service.id,
            vendor_id=new_service.vendor_id,
            vendor_name=vendor_profile.name,
            subcategory_id=new_service.subcategory_id,
            subcategory_name=subcategory.name,
            title=new_service.title,
            location=new_service.location,
            base_charge_min=new_service.base_charge_min,
            base_charge_max=new_service.base_charge_max,
            hourly_charge_min=new_service.hourly_charge_min,
            hourly_charge_max=new_service.hourly_charge_max,
            actual_price=new_service.actual_price,
            approved=new_service.approved,
            requested_at=new_service.requested_at,
            approved_at=new_service.approved_at
        )

@app.patch("/vendor-services/{service_id}/approve")
async def approve_vendor_service(
    service_id: str,
    dependencies=[Depends(api_key_auth)]
):
    """Approve a vendor service request"""
    async with AsyncSessionLocal() as session:
        stmt = select(VendorRequestedSubcategory).where(VendorRequestedSubcategory.id == service_id)
        result = await session.execute(stmt)
        service = result.scalar_one_or_none()
        
        if not service:
            raise HTTPException(status_code=404, detail="Vendor service not found")
        
        service.approved = True
        service.approved_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(service)
        
        # Clear cache to ensure fresh data
        if settings.ENABLE_VENDOR_CACHE:
            await vendor_cache.invalidate_all()
        
        return {"status": "success", "message": "Vendor service approved", "service_id": service_id}

# === OPTIMIZED CHAT ENDPOINT ===
@app.post("/chat", response_model=ChatOut, dependencies=[Depends(rate_guard), Depends(api_key_auth)])
async def chat_optimized(body: ChatIn):
    """Optimized chat endpoint with performance monitoring"""
    start_time = time.time()
    model_used = "new_model" if body.use_new_model else "legacy_model"
    
    try:
        # 1) Fast intent extraction
        intent_obj = await extract_intent_fast(body.message)
        slots = intent_obj.get("slots", {}) or {}
        intent = intent_obj.get("intent", "CLARIFY")
        
        # 2) Choose which model to use for vendor fetching
        if body.use_new_model:
            # Use new VendorRequestedSubcategory model
            vendor_services = await fetch_vendors_from_new_model(slots)
            
            if intent == "VENDOR_INFO" and vendor_services:
                service = vendor_services[0]
                
                # Get vendor and subcategory details
                async with AsyncSessionLocal() as session:
                    vendor_stmt = select(VendorProfile).where(VendorProfile.id == service.vendor_id)
                    subcategory_stmt = select(ServiceGroup).where(ServiceGroup.id == service.subcategory_id)
                    
                    vendor_result = await session.execute(vendor_stmt)
                    subcategory_result = await session.execute(subcategory_stmt)
                    
                    vendor_profile = vendor_result.scalar_one_or_none()
                    subcategory = subcategory_result.scalar_one_or_none()
                
                vendor_name = vendor_profile.name if vendor_profile else "Unknown Vendor"
                service_name = subcategory.name if subcategory else "Unknown Service"
                
                reply = f"Here are the details for {vendor_name}:\n"
                reply += f"Service: {service.title or service_name}\n"
                reply += f"Location: {service.location}\n"
                
                if service.base_charge_min and service.base_charge_max:
                    reply += f"Base Price: ₹{service.base_charge_min:,.0f} - ₹{service.base_charge_max:,.0f}\n"
                elif service.actual_price:
                    reply += f"Price: ₹{service.actual_price:,.0f}\n"
                
                if service.hourly_charge_min and service.hourly_charge_max:
                    reply += f"Hourly Rate: ₹{service.hourly_charge_min:,.0f} - ₹{service.hourly_charge_max:,.0f}/hour\n"
                
                reply += f"Status: {'Approved' if service.approved else 'Pending Approval'}"
                
                recommendations = [{
                    "service_id": service.id,
                    "vendor_name": vendor_name,
                    "service_name": service_name,
                    "title": service.title,
                    "location": service.location,
                    "price_info": {
                        "base_min": service.base_charge_min,
                        "base_max": service.base_charge_max,
                        "hourly_min": service.hourly_charge_min,
                        "hourly_max": service.hourly_charge_max,
                        "actual": service.actual_price
                    }
                }]
                
            elif intent == "QUERY_VENDORS" and vendor_services:
                reply = f"Found {len(vendor_services)} service(s):\n\n"
                
                # Get vendor and subcategory details for each service
                async with AsyncSessionLocal() as session:
                    summary_lines = []
                    recommendations = []
                    
                    for service in vendor_services[:5]:
                        vendor_stmt = select(VendorProfile).where(VendorProfile.id == service.vendor_id)
                        subcategory_stmt = select(ServiceGroup).where(ServiceGroup.id == service.subcategory_id)
                        
                        vendor_result = await session.execute(vendor_stmt)
                        subcategory_result = await session.execute(subcategory_stmt)
                        
                        vendor_profile = vendor_result.scalar_one_or_none()
                        subcategory = subcategory_result.scalar_one_or_none()
                        
                        vendor_name = vendor_profile.name if vendor_profile else "Unknown Vendor"
                        service_name = subcategory.name if subcategory else "Unknown Service"
                        
                        price_info = ""
                        if service.base_charge_min and service.base_charge_max:
                            price_info = f"₹{service.base_charge_min:,.0f} - ₹{service.base_charge_max:,.0f}"
                        elif service.actual_price:
                            price_info = f"₹{service.actual_price:,.0f}"
                        
                        summary_lines.append(
                            f"• {vendor_name} - {service.title or service_name}\n"
                            f"  Price: {price_info}\n"
                            f"  Location: {service.location}"
                        )
                        
                        recommendations.append({
                            "service_id": service.id,
                            "vendor_name": vendor_name,
                            "service_name": service_name,
                            "title": service.title,
                            "location": service.location,
                            "price_info": {
                                "base_min": service.base_charge_min,
                                "base_max": service.base_charge_max,
                                "actual": service.actual_price
                            }
                        })
                
                reply += "\n\n".join(summary_lines)
                
            else:
                reply = "I couldn't find matching services. Please specify city, service type, or vendor name."
                recommendations = []
        
        else:
            # Use legacy Vendor model
            vendors_list = await fetch_vendors_optimized(slots)
            
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
                reply = f"Found {len(vendors_list)} vendor(s):\n\n"
                
                summary_lines = []
                for v in vendors_list[:5]:
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
                
            else:
                reply = "I couldn't find matching vendors. Please specify city, date (YYYY-MM-DD), or service type."
                recommendations = []
        
        processing_time = time.time() - start_time
        
        return ChatOut(
            reply=reply,
            intent=intent_obj,
            slots=slots,
            recommendations=recommendations,
            processing_time=round(processing_time, 3),
            model_used=model_used
        )
        
    except Exception as e:
        log.error(f"Chat endpoint error: {e}")
        processing_time = time.time() - start_time
        
        return ChatOut(
            reply="I encountered an error processing your request. Please try again.",
            intent={"intent": "ERROR", "slots": {}},
            slots={},
            recommendations=[],
            processing_time=round(processing_time, 3),
            model_used=model_used
        )

# === LEGACY ENDPOINTS (for backward compatibility) ===
@app.get("/vendors/fast", response_model=List[VendorOut])
async def vendors_fast(
    city: Optional[str] = None,
    service_type: Optional[str] = None,
    limit: int = 10
):
    """Ultra-fast vendor endpoint with minimal processing (legacy)"""
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
        "models": {
            "legacy_vendor_model": "Available",
            "new_vendor_requested_subcategory_model": "Available",
            "default_model": "new_model"
        },
        "recommendations": {
            "enable_caching": True,
            "use_database_indexes": True,
            "consider_redis": "For production scale",
            "monitor_llm_costs": True,
        }
    }

# === DATABASE TABLE CREATION QUERIES ===
CREATE_TABLE_QUERIES = {
    "vendor_profiles": """
    CREATE TABLE IF NOT EXISTS vendor_profiles (
        id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255),
        phone VARCHAR(50),
        city VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_vendor_profiles_name ON vendor_profiles(name);
    CREATE INDEX IF NOT EXISTS idx_vendor_profiles_city ON vendor_profiles(city);
    CREATE INDEX IF NOT EXISTS idx_vendor_profiles_email ON vendor_profiles(email);
    """,
    
    "service_groups": """
    CREATE TABLE IF NOT EXISTS service_groups (
        id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
        name VARCHAR(255) NOT NULL,
        description VARCHAR(500),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_service_groups_name ON service_groups(name);
    """,
    
    "vendor_requested_subcategories": """
    CREATE TABLE IF NOT EXISTS vendor_requested_subcategories (
        id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
        vendor_id VARCHAR NOT NULL REFERENCES vendor_profiles(id) ON DELETE CASCADE,
        subcategory_id VARCHAR NOT NULL REFERENCES service_groups(id) ON DELETE CASCADE,
        approved BOOLEAN DEFAULT FALSE,
        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        approved_at TIMESTAMP,
        base_charge_max DECIMAL(10,2),
        base_charge_min DECIMAL(10,2),
        hourly_charge_max DECIMAL(10,2),
        hourly_charge_min DECIMAL(10,2),
        actual_price DECIMAL(10,2),
        title VARCHAR(255),
        location VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Unique constraints
    ALTER TABLE vendor_requested_subcategories 
    ADD CONSTRAINT vendor_subcategory_unique UNIQUE (vendor_id, subcategory_id);
    
    ALTER TABLE vendor_requested_subcategories 
    ADD CONSTRAINT vendor_title_unique UNIQUE (vendor_id, title);
    
    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_vrs_vendor_id ON vendor_requested_subcategories(vendor_id);
    CREATE INDEX IF NOT EXISTS idx_vrs_subcategory_id ON vendor_requested_subcategories(subcategory_id);
    CREATE INDEX IF NOT EXISTS idx_vrs_approved ON vendor_requested_subcategories(approved);
    CREATE INDEX IF NOT EXISTS idx_vrs_requested_at ON vendor_requested_subcategories(requested_at);
    CREATE INDEX IF NOT EXISTS idx_vrs_location ON vendor_requested_subcategories(location);
    CREATE INDEX IF NOT EXISTS idx_vrs_title ON vendor_requested_subcategories(title);
    CREATE INDEX IF NOT EXISTS idx_vrs_base_charge_min ON vendor_requested_subcategories(base_charge_min);
    CREATE INDEX IF NOT EXISTS idx_vrs_actual_price ON vendor_requested_subcategories(actual_price);
    
    -- Composite indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_vrs_approved_location ON vendor_requested_subcategories(approved, location);
    CREATE INDEX IF NOT EXISTS idx_vrs_approved_subcategory ON vendor_requested_subcategories(approved, subcategory_id);
    CREATE INDEX IF NOT EXISTS idx_vrs_location_price ON vendor_requested_subcategories(location, base_charge_min);
    """
}

@app.get("/db/create-tables")
async def create_database_tables(dependencies=[Depends(api_key_auth)]):
    """Create database tables with proper indexes"""
    try:
        async with engine.begin() as conn:
            # Create tables in order due to foreign key dependencies
            for table_name, query in CREATE_TABLE_QUERIES.items():
                try:
                    await conn.execute(text(query))
                    log.info(f"✅ Created/verified table: {table_name}")
                except Exception as e:
                    log.error(f"❌ Error creating table {table_name}: {e}")
                    raise
            
            log.info("✅ All database tables created successfully")
            
        return {
            "status": "success", 
            "message": "Database tables created successfully",
            "tables_created": list(CREATE_TABLE_QUERIES.keys())
        }
        
    except Exception as e:
        log.error(f"Database table creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/db/seed-new-data")
async def seed_new_model_data(dependencies=[Depends(api_key_auth)]):
    """Seed sample data for the new model"""
    try:
        async with AsyncSessionLocal() as session:
            # Check if data already exists
            existing_vendors = (await session.execute(select(VendorProfile))).scalars().first()
            if existing_vendors:
                return {"status": "info", "message": "Sample data already exists"}
            
            # Create sample vendor profiles
            vendors = [
                VendorProfile(name="Royal Caterers", city="Chennai", email="royal@catering.com", phone="+91-99999-11111"),
                VendorProfile(name="Elite Photography", city="Chennai", email="elite@photo.com", phone="+91-99999-22222"),
                VendorProfile(name="Floral Decors", city="Chennai", email="floral@decor.com", phone="+91-99999-33333"),
                VendorProfile(name="Mumbai Delights", city="Mumbai", email="mumbai@food.com", phone="+91-99999-66666"),
                VendorProfile(name="Pixel Perfect Studios", city="Mumbai", email="pixel@studio.com", phone="+91-99999-77777"),
                VendorProfile(name="Luxe Photography", city="Delhi", email="luxe@photo.com", phone="+91-99999-99999"),
            ]
            
            session.add_all(vendors)
            await session.flush()  # Get IDs without committing
            
            # Create sample service groups
            service_groups = [
                ServiceGroup(name="Catering & Food", description="Food and catering services"),
                ServiceGroup(name="Photography", description="Photography and videography services"),
                ServiceGroup(name="Decoration", description="Event decoration services"),
                ServiceGroup(name="Cleaning", description="Cleaning and maintenance services"),
                ServiceGroup(name="Makeup & Beauty", description="Makeup and beauty services"),
            ]
            
            session.add_all(service_groups)
            await session.flush()  # Get IDs without committing
            
            # Create sample vendor requested subcategories
            vendor_services = [
                # Royal Caterers - Food
                VendorRequestedSubcategory(
                    vendor_id=vendors[0].id,
                    subcategory_id=service_groups[0].id,
                    title="Wedding Catering Package",
                    location="Chennai, Tamil Nadu",
                    base_charge_min=15000.0,
                    base_charge_max=60000.0,
                    approved=True,
                    approved_at=datetime.utcnow()
                ),
                # Elite Photography - Photography
                VendorRequestedSubcategory(
                    vendor_id=vendors[1].id,
                    subcategory_id=service_groups[1].id,
                    title="Wedding Photography",
                    location="Chennai, Tamil Nadu",
                    base_charge_min=12000.0,
                    base_charge_max=45000.0,
                    hourly_charge_min=2000.0,
                    hourly_charge_max=5000.0,
                    approved=True,
                    approved_at=datetime.utcnow()
                ),
                # Floral Decors - Decoration
                VendorRequestedSubcategory(
                    vendor_id=vendors[2].id,
                    subcategory_id=service_groups[2].id,
                    title="Floral Decoration Package",
                    location="Chennai, Tamil Nadu",
                    base_charge_min=8000.0,
                    base_charge_max=40000.0,
                    approved=True,
                    approved_at=datetime.utcnow()
                ),
                # Mumbai Delights - Food
                VendorRequestedSubcategory(
                    vendor_id=vendors[3].id,
                    subcategory_id=service_groups[0].id,
                    title="Mumbai Special Catering",
                    location="Mumbai, Maharashtra",
                    base_charge_min=20000.0,
                    base_charge_max=80000.0,
                    approved=True,
                    approved_at=datetime.utcnow()
                ),
                # Pixel Perfect - Photography
                VendorRequestedSubcategory(
                    vendor_id=vendors[4].id,
                    subcategory_id=service_groups[1].id,
                    title="Corporate Photography",
                    location="Mumbai, Maharashtra",
                    base_charge_min=15000.0,
                    base_charge_max=50000.0,
                    hourly_charge_min=2500.0,
                    hourly_charge_max=6000.0,
                    approved=True,
                    approved_at=datetime.utcnow()
                ),
                # Luxe Photography - Photography
                VendorRequestedSubcategory(
                    vendor_id=vendors[5].id,
                    subcategory_id=service_groups[1].id,
                    title="Luxury Event Photography",
                    location="Delhi, NCR",
                    base_charge_min=20000.0,
                    base_charge_max=60000.0,
                    hourly_charge_min=3000.0,
                    hourly_charge_max=8000.0,
                    approved=True,
                    approved_at=datetime.utcnow()
                ),
                # Pending approval example
                VendorRequestedSubcategory(
                    vendor_id=vendors[0].id,
                    subcategory_id=service_groups[2].id,
                    title="Catering + Decoration Combo",
                    location="Chennai, Tamil Nadu",
                    actual_price=75000.0,
                    approved=False
                ),
            ]
            
            session.add_all(vendor_services)
            await session.commit()
            
            log.info("✅ Sample data seeded for new model")
            
        # Clear cache to ensure fresh data
        if settings.ENABLE_VENDOR_CACHE:
            await vendor_cache.invalidate_all()
            
        return {
            "status": "success", 
            "message": "Sample data seeded successfully",
            "data_created": {
                "vendors": len(vendors),
                "service_groups": len(service_groups),
                "vendor_services": len(vendor_services)
            }
        }
        
    except Exception as e:
        log.error(f"Data seeding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Seeding error: {str(e)}")

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
    
    # Seed legacy data if needed
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
                log.info("✅ Legacy seed data added with optimized indexes")
    
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
                # Pre-warm both models
                await fetch_vendors_optimized(query)
                await fetch_vendors_from_new_model(query)
            except Exception as e:
                log.warning(f"Cache pre-warming failed for {query}: {e}")
        
        log.info("✅ Cache pre-warming completed")
    
    log.info("🚀 Optimized service started with new VendorRequestedSubcategory model")

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
        workers=1,  # Single worker with async
        loop="uvloop",  # Faster event loop (install: pip install uvloop)
        http="httptools",  # Faster HTTP parser (install: pip install httptools)
        access_log=False,  # Disable access logs for performance
    )