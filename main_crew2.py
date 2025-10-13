import os
import json
import time
import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, String, Integer, Date as SQLDate, select, func, or_, and_
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column, Session
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("fast-planner")

# === SETTINGS ===
class Settings(BaseSettings):
    API_KEY: Optional[str] = None
    DATABASE_URL: str = "postgresql://postgres:Admin@localhost:5432/agentspace"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()

# Convert async URL to sync if needed
SYNC_DB_URL = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
log.info(f"Using database: {SYNC_DB_URL}")

# === DATABASE (SYNC ONLY) ===
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

engine = create_engine(SYNC_DB_URL, echo=False, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(engine, expire_on_commit=False)

# === DIRECT VENDOR SEARCH (NO TOOLS) ===
def search_vendors(session: Session, city: str = None, service_type: str = None, 
                   date_str: str = None, budget: int = None) -> List[Dict]:
    """Direct database search - FAST"""
    stmt = select(Vendor)
    filters = []
    
    if city:
        filters.append(func.lower(Vendor.city) == city.lower())
    
    if service_type:
        services = [s.strip().lower() for s in service_type.split(",")]
        expanded = set(services)
        
        # Map synonyms
        for svc in services:
            if svc in ["photo", "photographer", "photography"]:
                expanded.update(["camera", "photography"])
            elif svc in ["catering", "caterer", "food"]:
                expanded.add("food")
            elif svc in ["decor", "decoration"]:
                expanded.add("decoration")
        
        service_filters = [func.lower(Vendor.service_type) == s for s in expanded]
        filters.append(or_(*service_filters))
    
    if date_str:
        try:
            y, m, d = map(int, date_str.split("-"))
            target = datetime(y, m, d).date()
            filters.append(Vendor.available_date == target)
        except:
            pass
    
    if budget:
        filters.append(Vendor.price_min <= budget)
    
    if filters:
        stmt = stmt.where(and_(*filters))
    
    result = session.execute(stmt)
    vendors = result.scalars().all()
    
    return [{
        "id": v.id,
        "name": v.name,
        "service_type": v.service_type,
        "city": v.city,
        "price_min": v.price_min,
        "price_max": v.price_max,
        "available_date": v.available_date.isoformat() if v.available_date else None,
        "contact": v.contact
    } for v in vendors]

# === SIMPLE LLM PROCESSOR ===
class FastPlanner:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.model = settings.OPENAI_MODEL
    
    def process(self, user_message: str, session: Session) -> Dict[str, Any]:
        """ONE LLM call to extract params, then direct DB query"""
        
        if not self.client:
            return {
                "reply": "OpenAI API key not configured",
                "recommendations": [],
                "params_used": {}
            }
        
        # Step 1: Extract parameters with ONE LLM call
        system_prompt = """Extract event planning parameters from user message.
Return ONLY valid JSON with these fields:
{
  "city": "string or null",
  "service_type": "comma-separated services or null",
  "date": "YYYY-MM-DD or null",
  "budget": number or null
}

Service type mappings:
- photo/photographer/photography -> "photography"
- catering/caterer/food -> "food"
- decor/decoration -> "decoration"

City normalization:
- Bengaluru -> bangalore
- All lowercase

Examples:
"photographer in Chennai" -> {"city": "chennai", "service_type": "photography", "date": null, "budget": null}
"catering for wedding in Mumbai on 2025-12-15" -> {"city": "mumbai", "service_type": "food", "date": "2025-12-15", "budget": null}"""

        try:
            # Single LLM call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            params_text = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if "```json" in params_text:
                params_text = params_text.split("```json")[1].split("```")[0].strip()
            elif "```" in params_text:
                params_text = params_text.split("```")[1].split("```")[0].strip()
            
            params = json.loads(params_text)
            
            log.info(f"Extracted params: {params}")
            
            # Step 2: Direct database query (NO LLM)
            vendors = search_vendors(
                session,
                city=params.get("city"),
                service_type=params.get("service_type"),
                date_str=params.get("date"),
                budget=params.get("budget")
            )
            
            # Step 3: Format response
            if vendors:
                reply = f"Found {len(vendors)} vendor(s) matching your criteria:\n\n"
                for v in vendors[:5]:  # Limit to 5
                    reply += f"• {v['name']} - {v['service_type']} in {v['city']}\n"
                    reply += f"  Price: ₹{v['price_min']:,} - ₹{v['price_max']:,}\n"
                    if v['contact']:
                        reply += f"  Contact: {v['contact']}\n"
                    reply += "\n"
            else:
                reply = f"No vendors found for your criteria. Try adjusting your filters."
            
            return {
                "reply": reply.strip(),
                "recommendations": vendors,
                "params_used": params
            }
            
        except Exception as e:
            log.error(f"Processing error: {e}", exc_info=True)
            return {
                "reply": f"Error processing request: {str(e)}",
                "recommendations": [],
                "params_used": {}
            }

planner = FastPlanner()

# === FASTAPI ===
app = FastAPI(title="Fast Event Planner", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    recommendations: Optional[List[Dict[str, Any]]] = None
    processing_time: float

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "mode": "direct_llm", "agents": 0}

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    """Fast chat - 1 LLM call + 1 DB query"""
    start = time.time()
    
    with SessionLocal() as session:
        result = planner.process(body.message, session)
        elapsed = time.time() - start
        
        log.info(f"✅ Processed in {elapsed:.2f}s")
        
        return ChatOut(
            reply=result["reply"],
            recommendations=result.get("recommendations", []),
            processing_time=round(elapsed, 3)
        )

@app.on_event("startup")
async def startup():
    # Create tables synchronously
    Base.metadata.create_all(engine)
    log.info("🚀 Fast planner started - SYNC mode")

@app.on_event("shutdown")
async def shutdown():
    engine.dispose()
    log.info("👋 Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)