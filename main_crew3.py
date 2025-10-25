import os
import json
import time
import logging
import asyncio
from datetime import date, datetime
from typing import Optional, List, Dict, Any
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Date as SQLDate, select, func
from cachetools import TTLCache
import openai

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("event-planner")

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:Admin@localhost:5432/agentspace"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    CORS_ORIGINS: List[str] = ["*"]
    CACHE_TTL: int = 300
    MAX_CACHE_SIZE: int = 1000
    
    API_KEY: Optional[str] = None
    SEED_DATA: bool = True
    RATE_LIMIT_RPS: float = 5.0
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()

if settings.OPENAI_API_KEY:
    openai.api_key = settings.OPENAI_API_KEY

vendor_cache = TTLCache(maxsize=settings.MAX_CACHE_SIZE, ttl=settings.CACHE_TTL)

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

SERVICE_MAP = {
    "photographer": "camera", "photography": "camera", "photo": "camera",
    "cameraman": "camera", "photos": "camera", "picture": "camera",
    "videographer": "videography", "video": "videography",
    "catering": "food", "caterer": "food", "food": "food", "meals": "food",
    "decoration": "decoration", "decor": "decoration", "flowers": "decoration",
    "cleaning": "cleaning", "cleaner": "cleaning",
    "makeup": "makeup", "beauty": "makeup",
    "event planner": "event planning", "planner": "event planning"
}

# === DATABASE FUNCTIONS ===
async def search_vendors(
    city: Optional[str] = None,
    service_types: Optional[List[str]] = None,
    budget: Optional[int] = None
) -> List[Dict]:
    """Search vendors with caching"""
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
                "name": v.name, "service_type": v.service_type,
                "city": v.city, "price_min": v.price_min, "price_max": v.price_max,
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_vendors",
            "description": "Search for event vendors in the database by city, service type, and budget",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Chennai, Mumbai, Delhi, Bangalore)"
                    },
                    "service_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Service types: camera, videography, food, decoration, cleaning, makeup, event planning"
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Maximum budget in rupees"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_service_types",
            "description": "Get list of available service types/vendor categories, optionally filtered by city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Optional city to filter service types"
                    }
                }
            }
        }
    }
]

async def execute_function(name: str, arguments: Dict) -> str:
    """Execute function calls from AI"""
    try:
        if name == "search_vendors":
            vendors = await search_vendors(
                city=arguments.get("city"),
                service_types=arguments.get("service_types"),
                budget=arguments.get("budget")
            )
            return json.dumps({"vendors": vendors, "count": len(vendors)})
        
        elif name == "get_available_service_types":
            service_types = await get_available_service_types(
                city=arguments.get("city")
            )
            return json.dumps({"service_types": service_types})
        
        return json.dumps({"error": "Unknown function"})
    except Exception as e:
        log.error(f"Function execution error: {e}")
        return json.dumps({"error": str(e)})

class IntelligentEventPlanner:
    """AI-powered conversational event planner with function calling"""
    
    def __init__(self):
        self.system_prompt_base = """You are an expert event planning assistant specializing in Indian weddings and events. 

Your role:
- Help users plan events by understanding their needs
- Search for vendors using the available functions
- Ask follow-up questions when information is missing
- Provide detailed, helpful recommendations based on actual vendor data
- Be conversational, friendly, and professional

CRITICAL RULES:
1. ONLY mention vendors returned by the search_vendors function - NEVER invent names
2. When users ask about vendors, ALWAYS call search_vendors to get real data
3. If city is missing, ask for it naturally in conversation
4. If service type is unclear, ask or use get_available_service_types to show options
5. For multi-day weddings, explain what vendors are typically needed
6. Always provide specific vendor details: name, price range, and contact when available

For comprehensive event planning:
- Ask about: city, date, number of guests, budget, specific services needed
- Suggest typical vendor combinations for events
- Help prioritize based on budget

Available vendor service types: {service_types}

Be conversational and helpful like a real event planner would be!"""
        self.cached_system_prompt = None
    
    async def get_system_prompt(self) -> str:
        """Get system prompt with dynamic service types"""
        if self.cached_system_prompt:
            return self.cached_system_prompt
        
        service_types = await get_available_service_types()
        service_list = "\n".join([f"- {st}" for st in service_types])
        
        self.cached_system_prompt = self.system_prompt_base.format(
            service_types=service_list
        )
        return self.cached_system_prompt
    
    async def process(self, message: str, session_id: str = None) -> Dict:
        """Process conversation with AI function calling"""
        
        if session_id not in conversations:
            system_prompt = await self.get_system_prompt()
            conversations[session_id] = {
                "messages": [{"role": "system", "content": system_prompt}],
                "context": {}
            }
        
        conv = conversations[session_id]
        
        conv["messages"].append({"role": "user", "content": message})
                
        if len(conv["messages"]) > 15:
            conv["messages"] = [conv["messages"][0]] + conv["messages"][-14:]
        
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=conv["messages"],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000,
                timeout=10
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                # Add assistant message with function call
                conv["messages"].append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute functions
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    log.info(f"🔧 Calling {func_name} with {func_args}")
                    
                    result = await execute_function(func_name, func_args)
                    
                    # Add function result
                    conv["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                # Get final response after function execution
                final_response = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model=settings.OPENAI_MODEL,
                    messages=conv["messages"],
                    temperature=0.7,
                    max_tokens=1000,
                    timeout=10
                )
                
                final_message = final_response.choices[0].message.content
                conv["messages"].append({"role": "assistant", "content": final_message})
                
                return {
                    "reply": final_message,
                    "success": True,
                    "function_called": True
                }
            
            else:
                # No function call, direct response
                reply = assistant_message.content
                conv["messages"].append({"role": "assistant", "content": reply})
                
                return {
                    "reply": reply,
                    "success": True,
                    "function_called": False
                }
        
        except asyncio.TimeoutError:
            return {
                "reply": "I'm taking longer than expected. Could you rephrase that?",
                "success": False
            }
        except Exception as e:
            log.error(f"AI error: {e}", exc_info=True)
            return {
                "reply": "I encountered an error. Please try again.",
                "success": False
            }

planner = IntelligentEventPlanner()

app = FastAPI(title="AI Event Planner", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatOut(BaseModel):
    reply: str
    processing_time: float
    success: bool = True

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "cache_size": len(vendor_cache),
        "active_sessions": len(conversations),
        "model": settings.OPENAI_MODEL
    }

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    """AI-powered chat endpoint"""
    start = time.time()
    
    try:
        result = await planner.process(body.message, body.session_id)
        elapsed = time.time() - start
        
        return ChatOut(
            reply=result["reply"],
            processing_time=round(elapsed, 2),
            success=result["success"]
        )
        
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        elapsed = time.time() - start
        
        return ChatOut(
            reply="I encountered an error. Please try again.",
            processing_time=round(elapsed, 2),
            success=False
        )

@app.post("/reset-session")
async def reset_session(session_id: str):
    """Reset a conversation session"""
    if session_id in conversations:
        del conversations[session_id]
    planner.cached_system_prompt = None
    return {"message": f"Session {session_id} reset"}

@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches"""
    vendor_cache.clear()
    planner.cached_system_prompt = None
    return {"message": "All caches cleared"}

@app.get("/service-types")
async def get_service_types(city: Optional[str] = None):
    """Get available service types"""
    service_types = await get_available_service_types(city)
    return {"service_types": service_types, "city": city}

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Warm up cache
    cities = ["Chennai", "Mumbai", "Delhi", "Bangalore"]
    services = ["camera", "food", "decoration", "makeup", "videography"]
    
    for city in cities:
        for service in services:
            try:
                await search_vendors(city=city, service_types=[service])
            except:
                pass
    
    log.info("🚀 AI Event Planner started with function calling")

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)