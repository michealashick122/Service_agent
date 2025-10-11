"""
main.py

Run:
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt

Set env (PowerShell):
  $env:LLM_PROVIDER="openai"
  $env:OPENAI_API_KEY="sk-..."

Or create a .env file with:
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-...

Run server:
  uvicorn main:app --reload --port 8000

Open Swagger:
  http://127.0.0.1:8000/docs
"""

import os
import json
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field as SQLField, create_engine, Session, select
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Config / LLM client setup
# -------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Use modern OpenAI client when requested
OPENAI_AVAILABLE = False
if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    try:
        # openai >=1.0 client
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_AVAILABLE = True
    except Exception as e:
        openai_client = None
        OPENAI_AVAILABLE = False
else:
    openai_client = None

# CrewAI optional stub (do NOT auto-import heavy libs)
try:
    import crewai  # optional; only used if user installs it
    HAS_CREWAI = True
except Exception:
    crewai = None
    HAS_CREWAI = False

# -------------------------
# DB models & seed
# -------------------------
class Vendor(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    location: str
    rating: float = 4.5

class Service(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    vendor_id: str
    category: str
    title: str
    description: Optional[str] = None
    price_min: float = 0.0
    price_max: float = 0.0
    tags: Optional[str] = None

DATABASE_URL = "sqlite:///./agent_demo.db"
engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_seed():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        existing = session.exec(select(Vendor)).first()
        if existing:
            return
        v1 = Vendor(name="PixelStories Photography", location="Bangalore", rating=4.7)
        v2 = Vendor(name="GrandCaterers", location="Bangalore", rating=4.4)
        v3 = Vendor(name="Spark Decor", location="Koramangala", rating=4.6)
        s1 = Service(vendor_id=v1.id, category="photography", title="Wedding Photography - Full day",
                     price_min=30000, price_max=60000, tags=json.dumps(["wedding","candid"]))
        s2 = Service(vendor_id=v1.id, category="photography", title="Portraits & Pre-wedding",
                     price_min=10000, price_max=20000, tags=json.dumps(["pre-wedding"]))
        s3 = Service(vendor_id=v2.id, category="catering", title="Veg Catering (100 pax)",
                     price_min=40000, price_max=70000, tags=json.dumps(["veg","buffet"]))
        s4 = Service(vendor_id=v3.id, category="decor", title="Wedding Decor Basic",
                     price_min=25000, price_max=60000, tags=json.dumps(["flowers","stage"]))
        session.add_all([v1, v2, v3, s1, s2, s3, s4])
        session.commit()

create_db_and_seed()

# -------------------------
# Pydantic models (request/response)
# -------------------------
class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    event_type: Optional[str] = Field(None, description="wedding / birthday / corporate")
    services: Optional[List[str]] = Field(default_factory=list)
    budget: Optional[float] = None
    date: Optional[str] = None
    location: Optional[str] = None
    extras: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # Accept raw text query as alternative input
    raw_query: Optional[str] = Field(None, description="Raw user sentence to be parsed")

class CandidateOut(BaseModel):
    service_id: str
    vendor_name: str
    title: str
    avg_price: float
    score: float
    price_min: float
    price_max: float
    reason: Optional[str] = None

class RecommendResponse(BaseModel):
    query_id: str
    recommendations: List[CandidateOut]
    assistant_text: str
    assistant_json: Optional[Dict[str, Any]] = None

# -------------------------
# Utilities: parse extraction + LLM wrapper
# -------------------------
def try_parse_json_from_text(text: str):
    import re, json
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])\s*$", text.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None

def mock_local_llm(prompt: str, system: Optional[str] = None):
    # Very small heuristic mock - extracts basic categories and budget numbers if present.
    lower = prompt.lower()
    services = []
    for tok in ["photograph", "photo", "camera", "catering", "food", "decor", "decoration", "makeup", "dj", "music"]:
        if tok in lower and "photograph" not in services:
            if "photo" in tok or "photograph" in tok or "camera" in tok:
                services.append("photography")
            elif "cater" in tok or "food" in tok:
                services.append("catering")
            elif "decor" in tok:
                services.append("decoration")
            elif "makeup" in tok:
                services.append("makeup")
            elif "dj" in tok or "music" in tok:
                services.append("music")
    # find numbers like '5 lakh' or numbers with commas
    import re
    budget = None
    m = re.search(r"(\d+[,\d]*)(?:\s*(?:rs|inr|rupees))?", prompt, re.I)
    if m:
        val = m.group(1).replace(",", "")
        try:
            budget = float(val)
        except:
            budget = None
    # find 'lakh' or 'lac' words
    m2 = re.search(r"(\d+(\.\d+)?)\s*(lakh|lac|lacs|lakhs)", prompt, re.I)
    if m2:
        budget = float(m2.group(1)) * 100000
    # rudimentary date extraction (not perfect)
    date = None
    m3 = re.search(r"(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\s*(\d{4})?", prompt, re.I)
    if m3:
        date = m3.group(0)
    # location guess
    location = None
    # if user mentions known cities from DB
    with Session(engine) as session:
        vendors = session.exec(select(Vendor)).all()
        for v in vendors:
            if v.location.lower() in prompt.lower():
                location = v.location
                break
    # event type guess
    event_type = None
    if "wedding" in prompt.lower():
        event_type = "wedding"
    elif "birthday" in prompt.lower():
        event_type = "birthday"
    # Build structured output
    out = {
        "user_id": "anonymous",
        "event_type": event_type,
        "services": services,
        "budget": budget,
        "date": date,
        "location": location,
        "extras": {}
    }
    text = "[MOCK PARSE] " + json.dumps(out)
    return {"text": text, "json": out}

def llm_extract_structured(raw_query: str) -> Dict[str, Any]:
    print("Came inside llm_extract")
    """
    Use LLM to extract structured fields from raw_query.
    Falls back to mock parser if OpenAI not available.
    """
    system = (
        "You are a helpful extractor. Extract the following fields from the user's single-sentence query:\n"
        "user_id (string, optional), event_type (wedding/birthday/corporate/etc), services (list of categories),\n"
        "budget (numeric, INR), date (ISO or human readable), location (city), extras (dictionary).\n"
        "Return ONLY a single JSON object (no surrounding text) with keys: user_id, event_type, services, budget, date, location, extras.\n"
        "If a field is missing, set its value to null or an empty list/dict as appropriate."
    )
    prompt = f"User query: {raw_query}\n\nProvide the JSON now."
    if OPENAI_AVAILABLE:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # try gpt-4o-mini; fallback handled below
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            text = resp.choices[0].message.content
            parsed = try_parse_json_from_text(text)
            if parsed:
                return parsed
            # fallback to parse raw text if JSON not present
            try:
                return json.loads(text)
            except Exception:
                return {"user_id": None, "event_type": None, "services": [], "budget": None, "date": None, "location": None, "extras": {}}
        except Exception as e:
            # fallback - try gpt-3.5-turbo style
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=400,
                )
                text = resp.choices[0].message.content
                parsed = try_parse_json_from_text(text)
                if parsed:
                    return parsed
            except Exception:
                pass
            # last resort: local mock
            out = mock_local_llm(raw_query)
            return out["json"]
    else:
        out = mock_local_llm(raw_query)
        print(out)
        return out["json"]

# -------------------------
# Simple retrieval & ranking
# -------------------------
def retrieve_candidates(category: str, location: Optional[str], budget: Optional[float], limit: int = 6):
    with Session(engine) as session:
        q = select(Service, Vendor).join(Vendor, Service.vendor_id == Vendor.id).where(Service.category == category)
        results = session.exec(q).all()
        candidates = []
        for svc, vend in results:
            if budget and svc.price_min > (budget * 1.2):
                continue
            score = 0.0
            if location and vend.location and location.lower() in vend.location.lower():
                score += 30
            score += vend.rating * 10
            if budget:
                avg_price = (svc.price_min + svc.price_max) / 2.0
                diff = abs(avg_price - budget)
                score += max(0, 30 - (diff / max(1, budget) * 30))
            score += max(0, 10 - (svc.price_min / max(1, budget or svc.price_min)))
            candidates.append({
                "service": svc,
                "vendor": vend,
                "score": round(score,2),
                "avg_price": (svc.price_min + svc.price_max)/2.0
            })
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Agentic Event Planner — Parser + Recommend")

@app.post("/parse-query")
def parse_query(raw_query: str = Body(..., embed=True)):
    """
    Parse a raw user sentence into structured fields.
    """
    parsed = llm_extract_structured(raw_query)
    # normalize fields
    parsed.setdefault("user_id", parsed.get("user_id") or "anonymous")
    parsed.setdefault("services", parsed.get("services") or [])
    parsed.setdefault("extras", parsed.get("extras") or {})
    return parsed

@app.post("/agent/recommend", response_model=RecommendResponse)
def agent_recommend(req: RecommendRequest):
    """
    Accepts either:
      - structured fields (event_type, services, budget, date, location), OR
      - raw_query string (to be parsed)
    """
    # If raw_query provided, parse it first
    if req.raw_query:
        print("Parsed")
        parsed = llm_extract_structured(req.raw_query)
        # populate missing fields in req
        req.user_id = req.user_id or parsed.get("user_id") or "anonymous"
        print(req.user_id, "<---")
        req.event_type = req.event_type or parsed.get("event_type")
        req.services = req.services or parsed.get("services") or []
        req.budget = req.budget or parsed.get("budget")
        req.date = req.date or parsed.get("date")
        req.location = req.location or parsed.get("location")
        req.extras = req.extras or parsed.get("extras") or {}

    # Basic validation
    if not req.services:
        raise HTTPException(status_code=400, detail="No services specified (either pass services[] or raw_query).")

    all_candidates = []
    for cat in req.services:
        cands = retrieve_candidates(cat, req.location, req.budget, limit=6)
        for c in cands:
            svc = c["service"]
            vend = c["vendor"]
            all_candidates.append({
                "service_id": svc.id,
                "vendor_id": vend.id,
                "vendor_name": vend.name,
                "category": svc.category,
                "title": svc.title,
                "avg_price": c["avg_price"],
                "price_min": svc.price_min,
                "price_max": svc.price_max,
                "score": c["score"],
            })

    if not all_candidates:
        print("except")
        raise HTTPException(status_code=404, detail="No candidates found for requested services")


    # Simple fallback recommendation: top candidate per category
    recommendations = []
    seen_cats = set()
    for c in sorted(all_candidates, key=lambda x: x["score"], reverse=True):
        if c["category"] not in seen_cats:
            recommendations.append(CandidateOut(
                service_id=c["service_id"],
                vendor_name=c["vendor_name"],
                title=c["title"],
                avg_price=c["avg_price"],
                score=c["score"],
                price_min=c["price_min"],
                price_max=c["price_max"],
                reason="Matched by score & location"
            ))
            seen_cats.add(c["category"])
        if len(seen_cats) >= len(req.services):
            break

    assistant_text = "Local recommender returned top matches."
    assistant_json = {"candidates": [r.dict() for r in recommendations]}

    return RecommendResponse(
        query_id=str(uuid.uuid4()),
        recommendations=recommendations,
        assistant_text=assistant_text,
        assistant_json=assistant_json
    )

@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": LLM_PROVIDER, "openai_available": OPENAI_AVAILABLE, "has_crewai": HAS_CREWAI}
