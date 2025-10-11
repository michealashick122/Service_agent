# main_crewai_enterprise.py
# Python 3.11+ recommended

import os
import json
import time
import logging
from datetime import date
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Date as SQLDate, select

# === Structured logging ===
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("crewai-event-planner")

# === Settings (12-factor) ===
class Settings(BaseSettings):
    API_KEY: Optional[str] = None  # header: X-API-Key
    DATABASE_URL: str = "sqlite+aiosqlite:///./event_planner.db"  # swap to postgres+asyncpg://...
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    SEED_DATA: bool = True
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# === DB: SQLAlchemy 2.0 (async) ===
class Base(DeclarativeBase):
    pass

class Vendor(Base):
    __tablename__ = "vendors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    service_type: Mapped[str] = mapped_column(String(100))  # food, camera, decoration, cleaning, makeup
    city: Mapped[str] = mapped_column(String(100))
    price_min: Mapped[int] = mapped_column(Integer)
    price_max: Mapped[int] = mapped_column(Integer)
    available_date: Mapped[Optional[date]] = mapped_column(SQLDate, nullable=True)
    contact: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db(seed: bool = True):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    if seed:
        async with AsyncSessionLocal() as ses:
            # seed once
            existing = (await ses.execute(select(Vendor))).scalars().first()
            if not existing:
                ses.add_all([
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
                ])
                await ses.commit()

# === Simple API key auth (swap to JWT later) ===
async def api_key_auth(x_api_key: Optional[str] = Header(default=None)):
    if settings.API_KEY:
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return True

# === Super-light rate limiting (per-IP token bucket) ===
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "5"))
_BUCKETS: Dict[str, Dict[str, float]] = {}

def _allow(ip: str) -> bool:
    now = time.time()
    b = _BUCKETS.get(ip)
    if not b:
        _BUCKETS[ip] = {"tokens": RATE_LIMIT_RPS, "ts": now}
        return True
    elapsed = now - b["ts"]
    b["tokens"] = min(RATE_LIMIT_RPS, b["tokens"] + elapsed * RATE_LIMIT_RPS)
    b["ts"] = now
    if b["tokens"] >= 1.0:
        b["tokens"] -= 1.0
        return True
    return False

async def rate_guard(request: Request):
    ip = request.client.host if request.client else "unknown"
    print("Client ip ==>>>>>>>", ip)
    if not _allow(ip):
        raise HTTPException(429, "Rate limit exceeded")

# === Minimal LLM client (OpenAI chat) ===
async def call_llm(prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> str:
    print("Calling crew ai")
    if not settings.OPENAI_API_KEY:
        print("*****Going inside stub*******")
        # offline stub so you can POC without external calls
        return f"[LLM-STUB] {prompt[:240]}"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    body = {
        "model": settings.OPENAI_MODEL,
        "temperature": temperature,
        "messages": ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        print(data, "<---- data inside extract")
        return data["choices"][0]["message"]["content"].strip()

# === Enhanced intent extraction to handle specific vendor queries ===
INTENT_SYS = (
    "You classify event-planning queries. "
    "Return strict JSON with keys: intent (QUERY_VENDORS|PLAN_EVENT|GENERAL_Q|CLARIFY|VENDOR_INFO) and "
    "slots {city,date,service_type,event_type,budget,vendor_name} where available. "
    "Use VENDOR_INFO intent when user asks about specific vendor details/contact. "
    "Dates in YYYY-MM-DD, budget in integer."
)

async def extract_intent(user_text: str) -> Dict[str, Any]:
    try:
        raw = await call_llm(            
            f"User message:\n{user_text}\n\nReturn JSON only.",
            system=INTENT_SYS,
            temperature=0.0,
        )
        print("LLM Call for feature extraction****************************************")
        print("Called LLM for extract_intent")
        print(raw,"<--- raw from call LLM")
        return json.loads(raw)
    except Exception:
        print("Exception in extract intent")
        t = user_text.lower()
        
        # Check for specific vendor name mentions
        vendor_names = ["elite photography", "royal caterers", "floral decors", "sparkle cleaners", "glamup makeup"]
        vendor_name = None
        for name in vendor_names:
            if name in t:
                vendor_name = name
                break
        
        # Determine intent
        if vendor_name or any(w in t for w in ["contact", "phone", "details", "info"]):
            intent = "VENDOR_INFO"
        elif any(w in t for w in ["list", "available", "show", "find"]):
            intent = "QUERY_VENDORS"
        elif any(w in t for w in ["budget", "price", "cost"]):
            intent = "GENERAL_Q"
        else:
            intent = "CLARIFY"
        
        slots: Dict[str, Any] = {}
        
        if vendor_name:
            slots["vendor_name"] = vendor_name
        
        for city in ["chennai", "mumbai", "delhi", "bengaluru", "bangalore"]:
            if city in t:
                slots["city"] = "Bengaluru" if city in ["bengaluru", "bangalore"] else city.capitalize()
        for s in ["food", "camera", "decoration", "cleaning", "makeup", "photography"]:
            if s in t:
                slots["service_type"] = "camera" if s == "photography" else s
        
        # very light budget parse
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2})", t)
        if m:
            slots["date"] = m.group(1)
        m2 = re.search(r"(\d{4,7})", t)
        if m2:
            slots["budget"] = int(m2.group(1))
        return {"intent": intent, "slots": slots}

# === Enhanced fetch_vendors to handle specific vendor queries ===
async def fetch_vendors(slots: Dict[str, Any]) -> List[Vendor]:
    """
    Enhanced vendor fetching that handles specific vendor name queries
    """
    async with AsyncSessionLocal() as ses:
        stmt = select(Vendor)
        
        # Handle specific vendor name queries first
        if slots.get("vendor_name"):
            vendor_name = slots["vendor_name"].lower()
            from sqlalchemy import func
            stmt = stmt.where(func.lower(Vendor.name).contains(vendor_name.replace(" ", " ")))
        else:
            # Apply geographic/temporal/service constraints
            filters = []
            from sqlalchemy import func, and_
            
            # City filtering
            if slots.get("city"):
                filters.append(func.lower(Vendor.city) == slots["city"].lower())
            
            # Date filtering 
            if slots.get("date"):
                try:
                    y, m, d = map(int, slots["date"].split("-"))
                    filters.append(Vendor.available_date == date(y, m, d))
                except Exception:
                    pass
            
            if filters:
                stmt = stmt.where(and_(*filters))
            
        res = await ses.execute(stmt)
        candidate_vendors = res.scalars().all()

    if not candidate_vendors:
        return []

    # If it's a specific vendor query, return directly
    if slots.get("vendor_name"):
        return list(candidate_vendors)

    # Use AI for semantic matching only for general queries
    return await ai_semantic_vendor_match(candidate_vendors, slots)

async def ai_semantic_vendor_match(vendors: List[Vendor], slots: Dict[str, Any]) -> List[Vendor]:
    """
    Pure AI-driven semantic matching - STRICT mode: only match what user actually requested
    """
    # Prepare vendor data
    vendor_data = []
    for v in vendors:
        vendor_data.append({
            "id": v.id,
            "name": v.name,
            "service_type": v.service_type,
            "city": v.city,
            "price_min": v.price_min,
            "price_max": v.price_max
        })
    
    # Create STRICT AI prompt
    user_context = {
        "requested_services": slots.get("service_type", []),
        "budget": slots.get("budget"),
        "event_type": slots.get("event_type"),
        "city": slots.get("city"),
        "date": slots.get("date")
    }
    
    ai_prompt = f"""
You are a precise vendor matching system. Your job is to match ONLY the services the user explicitly requested.

Available Vendors:
{json.dumps(vendor_data, indent=2)}

User Requirements:
{json.dumps(user_context, indent=2)}

STRICT MATCHING RULES:
1. Only match vendors whose service_type semantically matches the user's requested_services
2. DO NOT apply budget filtering unless explicitly requested
3. If user asks for "camera service" → only match vendors with "camera" service_type
4. If user asks for "decoration" → only match vendors with "decoration" service_type  
5. If user asks for multiple services → match vendors for each requested service

Service Matching Examples:
- User: "camera service" → Match: vendors with service_type="camera" 
- User: "photography" → Match: vendors with service_type="camera"
- User: "decoration" → Match: vendors with service_type="decoration"
- User: "makeup service" → Match: vendors with service_type="makeup"
- User: "food service" → Match: vendors with service_type="food"

IMPORTANT: 
- Score vendors 0-100 based on how well their service_type matches requested_services
- Include vendors with score > 50 (moderate confidence matches)
- DO NOT filter by budget unless user specifically mentions budget constraints

Return ONLY a JSON object:
{{
    "matched_vendors": [
        {{
            "vendor_id": 2,
            "relevance_score": 95,
            "match_reason": "Camera service directly matches user request for camera service"
        }}
    ]
}}

Sort by relevance_score (highest first). Be inclusive rather than exclusive.
Return ONLY the JSON, no other text.
"""

    try:
        # Get AI analysis
        ai_response = await call_llm(ai_prompt, temperature=0.1)
        
        # Parse response
        ai_result = json.loads(ai_response.strip())
        matched_data = ai_result.get("matched_vendors", [])
        
        # Build result list maintaining AI's ranking
        result_vendors = []
        vendor_lookup = {v.id: v for v in vendors}
        
        for match in matched_data:
            vendor_id = match.get("vendor_id")
            if vendor_id in vendor_lookup:
                result_vendors.append(vendor_lookup[vendor_id])
                
        return result_vendors
        
    except Exception as e:
        log.error(f"AI semantic matching failed: {e}")
        # Fallback: return only vendors that match requested service types
        return strict_fallback_filter(vendors, slots)

def strict_fallback_filter(vendors: List[Vendor], slots: Dict[str, Any]) -> List[Vendor]:
    """
    Strict fallback that only returns exact service matches
    """
    print("*********fallbckkk response*************************")
    requested_services = slots.get("service_type", [])
    if not requested_services:
        return vendors
    
    if isinstance(requested_services, str):
        requested_services = [requested_services]
    
    # Very strict matching
    matched_vendors = []
    for vendor in vendors:
        vendor_service = vendor.service_type.lower()
        
        for requested in requested_services:
            requested_lower = str(requested).lower()
            
            # Only exact semantic matches
            if ("camera" in requested_lower and vendor_service == "camera") or \
               ("photo" in requested_lower and vendor_service == "camera") or \
               ("decoration" in requested_lower and vendor_service == "decoration") or \
               ("makeup" in requested_lower and vendor_service == "makeup") or \
               ("food" in requested_lower and vendor_service == "food") or \
               ("clean" in requested_lower and vendor_service == "cleaning"):
                matched_vendors.append(vendor)
                break
    
    return matched_vendors

# === CrewAI: Agents & Tasks ===
from crewai import Agent, Task, Crew, LLM as CrewLLM

# Use CrewAI's LLM wrapper to keep the interface clean
crewai_llm = CrewLLM(
    model=settings.OPENAI_MODEL,
    api_key=settings.OPENAI_API_KEY
)

planner_agent = Agent(
    role="Event Planner",
    goal="Interpret user needs and extract planning constraints precisely.",
    backstory="Expert event planner who structures requirements into clear constraints and next steps.",
    llm=crewai_llm,
)

budget_agent = Agent(
    role="Budget Optimizer",
    goal="Distribute or validate the user's budget across requested services.",
    backstory="Skilled at maximizing value within constraints.",
    llm=crewai_llm,
)

recommender_agent = Agent(
    role="Grounded Service Recommender",
    goal="Recommend vendors strictly from the provided vendor list and provide contact information when requested.",
    backstory="Understands the local market and only proposes from given inventory. Expert at providing detailed vendor information.",
    llm=crewai_llm,
)

# Enhanced task for vendor information queries
vendor_info_task = Task(
    description=(
        "User query: {user_query}\n"
        "Available vendors: {vendors_json}\n"
        "The user is asking for specific vendor information. "
        "Provide detailed information including contact details from the vendor data. "
        "Return JSON with: vendor_info: [{name,service_type,city,price_range,contact,available_date}] and message."
    ),
    agent=recommender_agent,
    expected_output="JSON with vendor_info[] and user-friendly message"
)

# Task templates: they use inputs passed at kickoff()
plan_task = Task(
    description=(
        "User query: {user_query}\n"
        "Extract a structured view of constraints (city, date, event_type, services, budget). "
        "Return a compact JSON object with keys: city, date, event_type, services[], budget."
    ),
    agent=planner_agent,
    expected_output="JSON with keys: city, date, event_type, services, budget"
)

budget_task = Task(
    description=(
        "Given constraints: {constraints_json}\n"
        "Validate/adjust the budget and allocate suggested amounts per service. "
        "Return JSON: { total_budget, allocations: [{service_type, min, max}] }."
    ),
    agent=budget_agent,
    expected_output="JSON with total_budget and allocations"
)

recommend_task = Task(
    description=(
        "Inventory (strict, do not invent): {vendors_json}\n"
        "Constraints: {constraints_json}\n"
        "Budget plan: {budget_json}\n"
        "Pick the best 3 vendors (if available) across requested services within budget. "
        "Return JSON: recommendations: [{vendor_id,name,service_type,est_price}], "
        "and a short user-friendly message."
    ),
    agent=recommender_agent,
    expected_output="JSON with recommendations[] and message"
)

# === FastAPI app ===
app = FastAPI(title="AI Event Planner (CrewAI E2E)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    intent: Dict[str, Any]
    slots: Dict[str, Any]
    recommendations: Optional[List[Dict[str, Any]]] = None

class VendorOut(BaseModel):
    id: int
    name: str
    service_type: str
    city: str
    price_min: int
    price_max: int
    available_date: Optional[date] = None
    contact: Optional[str] = None

# Health
@app.get("/healthz")
async def healthz():
    return {"status": "ok ashick"}

@app.get("/readyz")
async def readyz():
    try:
        async with engine.begin() as _:
            pass
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(500, f"db not ready: {e}")

# Grounded vendor search
@app.get("/vendors", response_model=List[VendorOut], dependencies=[Depends(rate_guard), Depends(api_key_auth)])
async def vendors(
    city: Optional[str] = None,
    service_type: Optional[str] = None,
    date_str: Optional[str] = None,
    max_price: Optional[int] = None,
):
    slots: Dict[str, Any] = {}
    if city: slots["city"] = city
    if service_type: slots["service_type"] = service_type
    if date_str: slots["date"] = date_str
    res = await fetch_vendors(slots)
    if max_price is not None:
        res = [v for v in res if v.price_min <= max_price or v.price_max <= max_price]
    return [
        VendorOut(
            id=v.id, name=v.name, service_type=v.service_type, city=v.city,
            price_min=v.price_min, price_max=v.price_max,
            available_date=v.available_date, contact=v.contact
        ) for v in res
    ]

# Enhanced CrewAI-driven chat with vendor info handling
@app.post("/chat", response_model=ChatOut, dependencies=[Depends(rate_guard), Depends(api_key_auth)])
async def chat(body: ChatIn):
    # 1) Intent extraction (LLM + fallback)
    intent_obj = await extract_intent(body.message)
    print(intent_obj,"<<<<===== intent obj")
    slots = intent_obj.get("slots", {}) or {}
    print(slots,"<<<=== slots extracted")
    intent = intent_obj.get("intent", "CLARIFY")

    # 2) Query DB (ground truth)
    vendors_list = await fetch_vendors(slots)
    print(vendors_list,"<<<<<== vendor list main")
    
    if not vendors_list and intent == "VENDOR_INFO":
        return ChatOut(
            reply="I couldn't find any vendor matching that name. Please check the spelling or try a different search.",
            intent=intent_obj,
            slots=slots,
            recommendations=[]
        )
    
    vendors_payload = [
        {
            "id": v.id,
            "name": v.name,
            "service_type": v.service_type,
            "city": v.city,
            "price_min": v.price_min,
            "price_max": v.price_max,
            "available_date": v.available_date.isoformat() if v.available_date else None,
            "contact": v.contact
        } for v in vendors_list
    ]

    # 3) Handle VENDOR_INFO intent specially
    if intent == "VENDOR_INFO" and vendors_list:
        # Direct response for vendor information queries
        vendor = vendors_list[0]  # Take the first matched vendor
        contact_info = f"Contact: {vendor.contact}" if vendor.contact else "Contact information not available"
        available_date = f"Available: {vendor.available_date.strftime('%Y-%m-%d')}" if vendor.available_date else "Availability not specified"
        
        reply = f"Here are the details for {vendor.name}:\n"
        reply += f"Service: {vendor.service_type.title()}\n"
        reply += f"Location: {vendor.city}\n"
        reply += f"Price Range: ₹{vendor.price_min:,} - ₹{vendor.price_max:,}\n"
        reply += f"{available_date}\n"
        reply += contact_info
        
        return ChatOut(
            reply=reply,
            intent=intent_obj,
            slots=slots,
            recommendations=[{
                "vendor_id": vendor.id,
                "name": vendor.name,
                "service_type": vendor.service_type,
                "contact": vendor.contact,
                "price_range": f"₹{vendor.price_min:,} - ₹{vendor.price_max:,}"
            }]
        )

    # Override intent if it's clearly a vendor query but was misclassified
    user_message_lower = body.message.lower()
    if intent == "CLARIFY" and any(word in user_message_lower for word in ["fetch", "get", "show", "list", "find"]) and any(service in user_message_lower for service in ["vendor", "photography", "camera", "food", "decoration", "makeup", "cleaning"]):
        intent = "QUERY_VENDORS"
        intent_obj["intent"] = "QUERY_VENDORS"
        print(f"DEBUG: Overrode intent from CLARIFY to QUERY_VENDORS based on message content")

    # 4) CrewAI orchestration with grounded inputs for other intents
    # Skip CrewAI for simple vendor listing queries to avoid hallucinations
    if intent == "QUERY_VENDORS" and vendors_list and not slots.get("budget"):
        # Direct response for simple vendor queries
        summary_lines = []
        for v in vendors_list[:5]:  # Show up to 5 vendors
            summary_lines.append(
                f"• {v.name} - {v.service_type.title()} service in {v.city}"
                f"\n  Price Range: ₹{v.price_min:,} - ₹{v.price_max:,}"
                f"\n  Contact: {v.contact or 'Not provided'}"
                f"\n  Available: {v.available_date.strftime('%Y-%m-%d') if v.available_date else 'Not specified'}\n"
            )
        
        reply = f"Found {len(vendors_list)} photography vendor(s) in Chennai:\n\n" + "\n".join(summary_lines)
        
        return ChatOut(
            reply=reply,
            intent=intent_obj,
            slots=slots,
            recommendations=[{
                "vendor_id": v.id,
                "name": v.name,
                "service_type": v.service_type,
                "contact": v.contact,
                "price_range": f"₹{v.price_min:,} - ₹{v.price_max:,}"
            } for v in vendors_list]
        )
    
    # Use CrewAI only for complex planning queries
    crew = Crew(
        agents=[planner_agent, budget_agent, recommender_agent],
        tasks=[plan_task, budget_task, recommend_task],
        llm=crewai_llm,
        verbose=False,
    )

    constraints_json = {
        "city": slots.get("city"),
        "date": slots.get("date"),
        "event_type": intent_obj.get("event_type") or slots.get("event_type"),
        "services": [slots.get("service_type")] if slots.get("service_type") else [],
        "budget": slots.get("budget"),
    }

    try:
        crew_result = crew.kickoff(inputs={
            "user_query": body.message,
            "constraints_json": json.dumps(constraints_json),
            "vendors_json": json.dumps(vendors_payload),
            "budget_json": json.dumps({"total_budget": slots.get("budget"), "allocations": []}),
        })
        
        # crew_result is usually a string; we expect recommend_task to output JSON
        txt = str(crew_result)
        parsed = None
        # simple parse attempt
        try:
            start = txt.rfind("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(txt[start:end+1])
        except Exception:
            parsed = None

        if parsed and isinstance(parsed, dict):
            recs = parsed.get("recommendations")
            msg = parsed.get("message") or "Here are some options for your event."
            return ChatOut(
                reply=msg,
                intent=intent_obj,
                slots=slots,
                recommendations=recs
            )
            
        # Fallback responder if parsing failed
        if vendors_list:
            summary = "\n".join(
                f"- {v.name} ({v.service_type}) in {v.city} – ₹{v.price_min:,}–₹{v.price_max:,}"
                for v in vendors_list[:3]
            )
            return ChatOut(
                reply=f"Here are some options I can recommend:\n{summary}",
                intent=intent_obj,
                slots=slots,
                recommendations=[{"vendor_id": v.id, "name": v.name, "service_type": v.service_type} for v in vendors_list[:3]]
            )
        else:
            return ChatOut(
                reply="I couldn't find matching vendors. Please share city, date (YYYY-MM-DD), service types, and budget.",
                intent=intent_obj,
                slots=slots,
                recommendations=[]
            )

    except Exception as e:
        log.error(f"crew error: {e}")
        # Final fallback: grounded, no-LLM
        if vendors_list:
            summary = "\n".join(
                f"- {v.name} ({v.service_type}) in {v.city} – ₹{v.price_min:,}–₹{v.price_max:,}"
                for v in vendors_list[:3]
            )
            return ChatOut(
                reply=f"[Fallback] Here are some options:\n{summary}",
                intent=intent_obj,
                slots=slots,
                recommendations=[{"vendor_id": v.id, "name": v.name, "service_type": v.service_type} for v in vendors_list[:3]]
            )
        raise HTTPException(500, "CrewAI orchestration failed")

# Startup
@app.on_event("startup")
async def _startup():
    await init_db(seed=settings.SEED_DATA)
    log.info("service started (CrewAI E2E)")