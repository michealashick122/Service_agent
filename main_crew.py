import os
import json
import time
import logging
import asyncio
from datetime import date, datetime
from typing import Optional, List, Dict, Any
import hashlib
import re

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Date as SQLDate, select, func, Text
from cachetools import TTLCache
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_openai import ChatOpenAI
import openai

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s","trace_id":"%(trace_id)s"}',
)
log = logging.getLogger("crewai-event-planner")

# === SETTINGS ===
class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:Admin@localhost:5432/agentspace"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    CORS_ORIGINS: List[str] = ["*"]
    CACHE_TTL: int = 300
    MAX_CACHE_SIZE: int = 1000
    
    API_KEY: Optional[str] = None
    RATE_LIMIT_RPS: float = 5.0
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    
    # AI Guardrails
    MAX_MESSAGE_LENGTH: int = 2000
    MAX_TOKENS_PER_REQUEST: int = 1500
    CONVERSATION_TIMEOUT: int = 30
    MAX_FUNCTION_CALLS: int = 5
    ENABLE_CONTENT_FILTERING: bool = True
    MAX_BUDGET: int = 10000000  # 1 crore max
    
    # Security
    ENABLE_RATE_LIMITING: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 30
    SESSION_TIMEOUT_MINUTES: int = 60
    
    # Monitoring
    ENABLE_METRICS: bool = True
    ALERT_EMAIL: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()

if settings.OPENAI_API_KEY:
    openai.api_key = settings.OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# === CACHING ===
vendor_cache = TTLCache(maxsize=settings.MAX_CACHE_SIZE, ttl=settings.CACHE_TTL)
rate_limit_cache = TTLCache(maxsize=10000, ttl=60)

def cache_key(**kwargs) -> str:
    key_data = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()

# === DATABASE ===
class Base(DeclarativeBase):
    pass

class Vendor(Base):
    __tablename__ = "vendors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    service_type: Mapped[str] = mapped_column(String(100), index=True)
    city: Mapped[str] = mapped_column(String(100), index=True)
    price_min: Mapped[int] = mapped_column(Integer)
    price_max: Mapped[int] = mapped_column(Integer)
    available_date: Mapped[Optional[date]] = mapped_column(SQLDate, nullable=True)
    contact: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    session_id: Mapped[str] = mapped_column(String(255), index=True)
    user_message: Mapped[str] = mapped_column(Text)
    ai_response: Mapped[str] = mapped_column(Text)
    function_calls: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processing_time: Mapped[float] = mapped_column()
    success: Mapped[bool] = mapped_column()
    trace_id: Mapped[str] = mapped_column(String(64), index=True)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=3600,
    pool_pre_ping=True,
)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# === CONVERSATION MEMORY ===
conversations: Dict[str, Dict] = {}

# === CONTENT FILTERING ===
class ContentFilter:
    """Filter inappropriate content and enforce guardrails"""
    
    BLOCKED_PATTERNS = [
        r'\b(hack|exploit|vulnerability|sql injection|xss)\b',
        r'\b(credit card|ssn|social security)\b',
        r'\b(illegal|fraud|scam)\b',
    ]
    
    SPAM_PATTERNS = [
        r'(.)\1{10,}',  # Repeated characters
        r'(https?://[^\s]+){3,}',  # Multiple URLs
    ]
    
    @staticmethod
    def is_safe(text: str) -> tuple[bool, Optional[str]]:
        """Check if content is safe"""
        if not text or len(text.strip()) == 0:
            return False, "Empty message"
        
        if len(text) > settings.MAX_MESSAGE_LENGTH:
            return False, f"Message too long (max {settings.MAX_MESSAGE_LENGTH} chars)"
        
        text_lower = text.lower()
        
        # Check blocked patterns
        for pattern in ContentFilter.BLOCKED_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, "Message contains prohibited content"
        
        # Check spam patterns
        for pattern in ContentFilter.SPAM_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Message appears to be spam"
        
        return True, None
    
    @staticmethod
    def sanitize_output(text: str) -> str:
        """Sanitize AI output"""
        # Remove potential PII patterns
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', text)
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED]', text)
        return text

# === RATE LIMITING ===
class RateLimiter:
    """Rate limiting for API endpoints"""
    
    @staticmethod
    def check_rate_limit(session_id: str) -> tuple[bool, Optional[str]]:
        """Check if request is within rate limit"""
        if not settings.ENABLE_RATE_LIMITING:
            return True, None
        
        current_time = time.time()
        key = f"rate_{session_id}"
        
        if key in rate_limit_cache:
            requests = rate_limit_cache[key]
            # Filter requests in last minute
            recent = [t for t in requests if current_time - t < 60]
            
            if len(recent) >= settings.MAX_REQUESTS_PER_MINUTE:
                return False, "Rate limit exceeded. Please wait before sending more messages."
            
            recent.append(current_time)
            rate_limit_cache[key] = recent
        else:
            rate_limit_cache[key] = [current_time]
        
        return True, None

# === DATABASE FUNCTIONS ===
async def search_vendors(
    city: Optional[str] = None,
    service_types: Optional[List[str]] = None,
    budget: Optional[int] = None
) -> List[Dict]:
    """Search vendors with caching and validation"""
    
    # Input validation
    if budget and (budget < 0 or budget > settings.MAX_BUDGET):
        raise ValueError(f"Budget must be between 0 and {settings.MAX_BUDGET}")
    
    if city and len(city) > 100:
        raise ValueError("City name too long")
    
    cache_k = cache_key(city=city, services=service_types, budget=budget)
    
    if cache_k in vendor_cache:
        log.info(f"✅ Cache HIT")
        return vendor_cache[cache_k]
    
    log.info(f"🔍 DB Search: {city}, {service_types}, {budget}")
    
    async with AsyncSessionLocal() as session:
        stmt = select(Vendor)
        
        if city:
            stmt = stmt.where(func.lower(Vendor.city) == city.lower())
        
        if service_types:
            from sqlalchemy import or_
            filters = [func.lower(Vendor.service_type) == s.lower() for s in service_types]
            stmt = stmt.where(or_(*filters))
        
        if budget:
            stmt = stmt.where(Vendor.price_min <= budget)
        
        stmt = stmt.limit(50)
        result = await session.execute(stmt)
        vendors = result.scalars().all()
        
        vendor_list = [
            {
                "name": v.name,
                "service_type": v.service_type,
                "city": v.city,
                "price_min": v.price_min,
                "price_max": v.price_max,
                "contact": v.contact,
                "available_date": v.available_date.isoformat() if v.available_date else None
            }
            for v in vendors
        ]
        
        vendor_cache[cache_k] = vendor_list
        return vendor_list

async def get_available_service_types(city: Optional[str] = None) -> List[str]:
    """Get distinct service types with caching"""
    cache_k = f"service_types_{city or 'all'}"
    if cache_k in vendor_cache:
        return vendor_cache[cache_k]
    
    async with AsyncSessionLocal() as session:
        stmt = select(Vendor.service_type).distinct()
        if city:
            stmt = stmt.where(func.lower(Vendor.city) == city.lower())
        result = await session.execute(stmt)
        service_types = [row[0] for row in result.fetchall()]
        vendor_cache[cache_k] = service_types
        return service_types

async def log_interaction(
    session_id: str,
    user_message: str,
    ai_response: str,
    function_calls: Optional[str],
    processing_time: float,
    success: bool,
    trace_id: str
):
    """Log interaction to database for audit trail"""
    try:
        async with AsyncSessionLocal() as session:
            audit = AuditLog(
                session_id=session_id,
                user_message=user_message[:1000],  # Truncate if too long
                ai_response=ai_response[:2000],
                function_calls=function_calls,
                processing_time=processing_time,
                success=success,
                trace_id=trace_id
            )
            session.add(audit)
            await session.commit()
    except Exception as e:
        log.error(f"Failed to log interaction: {e}")

# === CREWAI TOOLS ===
@tool("Search Vendors")
async def search_vendors_tool(
    city: Optional[str] = None,
    service_types: Optional[List[str]] = None,
    budget: Optional[int] = None
) -> str:
    """
    Search for event vendors in the database by city, service type, and budget.
    
    Args:
        city: City name (e.g., Chennai, Mumbai, Delhi, Bangalore)
        service_types: List of service types (camera, videography, food, decoration, etc.)
        budget: Maximum budget in rupees
    
    Returns:
        JSON string with vendor list and count
    """
    try:
        vendors = await search_vendors(city, service_types, budget)
        return json.dumps({"vendors": vendors, "count": len(vendors)})
    except Exception as e:
        log.error(f"Search vendors tool error: {e}")
        return json.dumps({"error": str(e), "vendors": [], "count": 0})

@tool("Get Service Types")
async def get_service_types_tool(city: Optional[str] = None) -> str:
    """
    Get list of available service types/vendor categories.
    
    Args:
        city: Optional city to filter service types
    
    Returns:
        JSON string with available service types
    """
    try:
        service_types = await get_available_service_types(city)
        return json.dumps({"service_types": service_types})
    except Exception as e:
        log.error(f"Get service types tool error: {e}")
        return json.dumps({"error": str(e), "service_types": []})

# === CREWAI AGENTS ===
class EventPlannerCrew:
    """CrewAI-based event planning system with multiple specialized agents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            max_tokens=settings.MAX_TOKENS_PER_REQUEST
        )
        
        # Requirement Analyst Agent
        self.requirements_agent = Agent(
            role="Event Requirements Analyst",
            goal="Extract and clarify event planning requirements from user conversations",
            backstory="""You are an expert at understanding customer needs for Indian weddings 
            and events. You ask clarifying questions when needed and extract key details like 
            city, budget, date, number of guests, and required services.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Vendor Search Agent
        self.vendor_agent = Agent(
            role="Vendor Search Specialist",
            goal="Find and recommend the best vendors based on requirements",
            backstory="""You are a vendor search expert who uses the search_vendors_tool to 
            find suitable vendors. You ONLY recommend vendors that actually exist in the database. 
            You provide detailed information including names, prices, and contact details.""",
            llm=self.llm,
            verbose=True,
            tools=[search_vendors_tool, get_service_types_tool],
            allow_delegation=False
        )
        
        # Event Coordinator Agent
        self.coordinator_agent = Agent(
            role="Event Coordination Expert",
            goal="Provide comprehensive event planning advice and vendor coordination",
            backstory="""You are an experienced event planner who helps coordinate all aspects 
            of weddings and events. You suggest vendor combinations, help with budget allocation, 
            and provide timeline recommendations.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    async def process_message(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process user message through CrewAI agents"""
        
        try:
            # Create tasks
            requirement_task = Task(
                description=f"""
                Analyze this user message and extract event planning requirements:
                
                User message: {message}
                Previous context: {json.dumps(context)}
                
                Identify:
                - City/location
                - Event type (wedding, birthday, corporate, etc.)
                - Budget
                - Required services
                - Number of guests
                - Date/timeline
                - Any specific preferences
                
                If critical information is missing, formulate 1-2 clarifying questions.
                """,
                agent=self.requirements_agent,
                expected_output="Structured analysis of requirements and any clarifying questions needed"
            )
            
            vendor_search_task = Task(
                description=f"""
                Based on the requirements analysis, search for suitable vendors.
                
                Use the search_vendors_tool and get_service_types_tool as needed.
                
                CRITICAL RULES:
                1. ONLY mention vendors returned by the search tools
                2. NEVER invent vendor names or details
                3. Provide specific names, price ranges, and contact information
                4. If no vendors found, explain why and suggest alternatives
                
                User message: {message}
                Context: {json.dumps(context)}
                """,
                agent=self.vendor_agent,
                expected_output="List of actual vendors with complete details",
                context=[requirement_task]
            )
            
            coordination_task = Task(
                description=f"""
                Provide a comprehensive, helpful response to the user.
                
                Combine the requirements analysis and vendor search results to:
                1. Answer the user's question directly
                2. Provide specific vendor recommendations with details
                3. Offer additional helpful advice
                4. Ask clarifying questions if needed
                
                Be conversational, friendly, and professional.
                Focus on being genuinely helpful.
                
                User message: {message}
                """,
                agent=self.coordinator_agent,
                expected_output="Complete, conversational response to the user",
                context=[requirement_task, vendor_search_task]
            )
            
            # Create and run crew
            crew = Crew(
                agents=[
                    self.requirements_agent,
                    self.vendor_agent,
                    self.coordinator_agent
                ],
                tasks=[
                    requirement_task,
                    vendor_search_task,
                    coordination_task
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(crew.kickoff),
                timeout=settings.CONVERSATION_TIMEOUT
            )
            
            return {
                "reply": str(result),
                "success": True,
                "agents_used": ["requirements", "vendor_search", "coordinator"]
            }
            
        except asyncio.TimeoutError:
            return {
                "reply": "I'm taking longer than expected to process your request. Could you try rephrasing?",
                "success": False
            }
        except Exception as e:
            log.error(f"CrewAI processing error: {e}", exc_info=True)
            return {
                "reply": "I encountered an error while processing your request. Please try again.",
                "success": False
            }

# === MAIN EVENT PLANNER ===
planner_crew = EventPlannerCrew()

# === FASTAPI APP ===
app = FastAPI(
    title="CrewAI Event Planner",
    version="6.0.0",
    description="Production-ready AI event planner with CrewAI and comprehensive guardrails"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.MAX_MESSAGE_LENGTH)
    session_id: Optional[str] = "default"
    
    @validator('message')
    def validate_message(cls, v):
        is_safe, error = ContentFilter.is_safe(v)
        if not is_safe:
            raise ValueError(error)
        return v

class ChatOut(BaseModel):
    reply: str
    processing_time: float
    success: bool = True
    trace_id: str
    agents_used: Optional[List[str]] = None

# Middleware for logging and monitoring
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = hashlib.md5(
        f"{time.time()}{request.client.host}".encode()
    ).hexdigest()
    
    # Add to logging context
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.trace_id = trace_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    
    logging.setLogRecordFactory(old_factory)
    return response

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "success": False}
    )

@app.get("/healthz")
async def healthz():
    """Health check endpoint"""
    return {
        "status": "ok",
        "cache_size": len(vendor_cache),
        "active_sessions": len(conversations),
        "model": settings.OPENAI_MODEL,
        "version": "6.0.0",
        "guardrails_enabled": settings.ENABLE_CONTENT_FILTERING
    }

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn, request: Request):
    """AI-powered chat endpoint with CrewAI"""
    start = time.time()
    trace_id = request.headers.get("X-Trace-ID", "unknown")
    
    session_id = body.session_id or "default"
    
    # Rate limiting
    can_proceed, error = RateLimiter.check_rate_limit(session_id)
    if not can_proceed:
        raise HTTPException(status_code=429, detail=error)
    
    # Content filtering
    if settings.ENABLE_CONTENT_FILTERING:
        is_safe, error = ContentFilter.is_safe(body.message)
        if not is_safe:
            raise HTTPException(status_code=400, detail=error)
    
    try:
        # Initialize or get conversation context
        if session_id not in conversations:
            conversations[session_id] = {
                "context": {},
                "message_count": 0,
                "created_at": time.time()
            }
        
        conv = conversations[session_id]
        conv["message_count"] += 1
        
        # Session timeout check
        if time.time() - conv["created_at"] > settings.SESSION_TIMEOUT_MINUTES * 60:
            del conversations[session_id]
            conversations[session_id] = {
                "context": {},
                "message_count": 1,
                "created_at": time.time()
            }
            conv = conversations[session_id]
        
        # Process with CrewAI
        result = await planner_crew.process_message(
            body.message,
            conv["context"]
        )
        
        elapsed = time.time() - start
        
        # Sanitize output
        reply = ContentFilter.sanitize_output(result["reply"])
        
        # Log interaction
        await log_interaction(
            session_id=session_id,
            user_message=body.message,
            ai_response=reply,
            function_calls=json.dumps(result.get("agents_used")),
            processing_time=elapsed,
            success=result["success"],
            trace_id=trace_id
        )
        
        return ChatOut(
            reply=reply,
            processing_time=round(elapsed, 2),
            success=result["success"],
            trace_id=trace_id,
            agents_used=result.get("agents_used")
        )
        
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        elapsed = time.time() - start
        
        await log_interaction(
            session_id=session_id,
            user_message=body.message,
            ai_response="Error occurred",
            function_calls=None,
            processing_time=elapsed,
            success=False,
            trace_id=trace_id
        )
        
        return ChatOut(
            reply="I encountered an error while processing your request. Please try again.",
            processing_time=round(elapsed, 2),
            success=False,
            trace_id=trace_id
        )

@app.post("/reset-session")
async def reset_session(session_id: str):
    """Reset a conversation session"""
    if session_id in conversations:
        del conversations[session_id]
    return {"message": f"Session {session_id} reset", "success": True}

@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches (admin endpoint)"""
    vendor_cache.clear()
    rate_limit_cache.clear()
    return {"message": "All caches cleared", "success": True}

@app.get("/service-types")
async def get_service_types(city: Optional[str] = None):
    """Get available service types"""
    service_types = await get_available_service_types(city)
    return {"service_types": service_types, "city": city}

@app.get("/metrics")
async def get_metrics():
    """Get system metrics (monitoring endpoint)"""
    if not settings.ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return {
        "active_sessions": len(conversations),
        "cache_size": len(vendor_cache),
        "rate_limit_entries": len(rate_limit_cache),
        "total_conversations": sum(
            conv["message_count"] for conv in conversations.values()
        )
    }

@app.on_event("startup")
async def startup():
    """Startup tasks"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    log.info("🚀 CrewAI Event Planner started with AI guardrails enabled")
    log.info(f"Content filtering: {settings.ENABLE_CONTENT_FILTERING}")
    log.info(f"Rate limiting: {settings.ENABLE_RATE_LIMITING}")
    log.info(f"Metrics: {settings.ENABLE_METRICS}")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await engine.dispose()
    log.info("👋 Shutting down CrewAI Event Planner")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": '{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
                }
            }
        }
    )