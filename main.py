"""
main.py

Run:
  pip install -r requirements.txt
  export OPENAI_API_KEY="sk-..."    # OR use .env
  python main.py

Then try:
  POST http://127.0.0.1:8000/agent/recommend
  POST http://127.0.0.1:8000/agent/negotiate
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field as SQLField, create_engine, Session, select
import os
import json
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

# --- LLM provider selection ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "local"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Try to import crewai if user wants to wire it later (optional)
try:
    import crewai  # noqa: F401
    HAS_CREWAI = True
except Exception:
    HAS_CREWAI = False

# If using OpenAI
if LLM_PROVIDER == "openai":
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        openai = None


# --------------------------
# Database models (SQLModel)
# --------------------------
class Vendor(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    location: str  # city
    rating: float = 4.5


class Service(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    vendor_id: str
    category: str  # photography, catering, decor ...
    title: str
    description: Optional[str] = None
    price_min: float = 0.0
    price_max: float = 0.0
    tags: Optional[str] = None  # JSON list as string for simplicity


DATABASE_URL = "sqlite:///./agent_demo.db"
engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_seed():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        vendors = session.exec(select(Vendor)).all()
        if vendors:
            return  # already seeded

        v1 = Vendor(name="PixelStories Photography", location="Indira Nagar", rating=4.7)
        v2 = Vendor(name="GrandCaterers", location="Indira Nagar", rating=4.4)
        v3 = Vendor(name="Spark Decor", location="Koramangala", rating=4.6)

        s1 = Service(
            vendor_id=v1.id,
            category="photography",
            title="Wedding Photography - Full day",
            description="Candid + traditional wedding coverage. 2 photographers.",
            price_min=30000,
            price_max=60000,
            tags=json.dumps(["wedding", "candid", "album"]),
        )
        s2 = Service(
            vendor_id=v1.id,
            category="photography",
            title="Portraits & Pre-wedding",
            description="Pre-wedding shoot. 1 photographer, 3 locations.",
            price_min=10000,
            price_max=20000,
            tags=json.dumps(["pre-wedding", "portraits"]),
        )
        s3 = Service(
            vendor_id=v2.id,
            category="catering",
            title="Veg Catering",
            description="Buffet for 100 pax, vegetarian menu.",
            price_min=40000,
            price_max=70000,
            tags=json.dumps(["veg", "buffet"]),
        )
        s4 = Service(
            vendor_id=v3.id,
            category="decor",
            title="Wedding Decor Basic",
            description="Stage + table decor for 200 pax.",
            price_min=25000,
            price_max=60000,
            tags=json.dumps(["flowers", "stage"]),
        )

        session.add_all([v1, v2, v3, s1, s2, s3, s4])
        session.commit()


# --------------------------
# Helper: simple retrieval & ranking
# --------------------------
def retrieve_candidates(category: str, location: Optional[str], date: Optional[str], budget: Optional[float], limit: int = 6):
    """
    Query the DB for matching services. This is a simple example:
      - match by category
      - prefer same city (location)
      - filter price_max <= budget * 1.2 (allow small overshoot)
    """
    print("Accessing session")
    with Session(engine) as session:
        q = select(Service, Vendor).join(Vendor, Service.vendor_id == Vendor.id).where(Service.category == category)
        results = session.exec(q).all()
        print(results)
        candidates = []
        for service, vendor in results:
            # Simple budget filter
            if budget:
                # allow up to 20% overshoot
                if service.price_min > (budget * 1.2):
                    continue
            # compute base score
            score = 0.0
            # prefer same location
            if location and vendor.location and location.lower() in vendor.location.lower():
                score += 30
            # rating
            score += vendor.rating * 10
            # budget proximity (closer price to budget is better)
            if budget:
                avg_price = (service.price_min + service.price_max) / 2.0
                # smaller difference -> higher score
                diff = abs(avg_price - budget)
                # normalize
                score += max(0, 30 - (diff / max(1, budget) * 30))
            # small preference for lower min price
            score += max(0, 10 - (service.price_min / max(1, budget or service.price_min)))
            candidates.append({
                "service": service,
                "vendor": vendor,
                "score": round(score, 2),
                "avg_price": (service.price_min + service.price_max) / 2.0
            })
        # sort by score desc
        candidates_sorted = sorted(candidates, key=lambda c: c["score"], reverse=True)
        return candidates_sorted[:limit]


# --------------------------
# LLM wrapper (pluggable)
# --------------------------
class LLMResponse(BaseModel):
    text: str
    structured: Optional[Dict[str, Any]] = None


def llm_call(prompt: str, system: Optional[str] = None, max_tokens: int = 300) -> LLMResponse:
    """
    Pluggable LLM call. Uses:
      - CrewAI if present & configured (optional)
      - OpenAI (if LLM_PROVIDER=openai)
      - Local mock LLM otherwise (for offline testing)
    The function returns a text and optional JSON-structured payload if the model included JSON at the end.
    """
    # Optionally route via CrewAI if it's available and you want to use it here.
    if HAS_CREWAI and LLM_PROVIDER == "crewai":
        # NOTE: This is illustrative. Replace with actual crewai usage per your installed version.
        # For now we will just fall through (crewai specific usage depends on the SDK).
        pass

    if LLM_PROVIDER == "openai" and openai is not None:
        # Use ChatCompletion (gpt-3.5/4 family) style
        system_msg = {"role": "system", "content": system or "You are a helpful event planning assistant."}
        user_msg = {"role": "user", "content": prompt}
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list().__dict__ else "gpt-4o-mini",
                messages=[system_msg, user_msg],
                max_tokens=max_tokens,
                temperature=0.25,
            )
        except Exception:
            # Fallback to a safer call (not all envs include model list)
            print("*********************************")
            print("failed")
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[system_msg, user_msg],
                max_tokens=max_tokens,
                temperature=0.25,
            )
        text = resp["choices"][0]["message"]["content"]
        # Try to extract trailing JSON if present
        structured = try_parse_json_from_text(text)
        return LLMResponse(text=text, structured=structured)
    else:
        # Local mock LLM for offline use
        return mock_local_llm(prompt)


def try_parse_json_from_text(text: str):
    # Attempts to parse last {...} or [...] blob in the text
    text = text.strip()
    for start in (text.rfind("{"), text.rfind("[")):
        if start != -1:
            maybe = text[start:]
            try:
                return json.loads(maybe)
            except Exception:
                continue
    return None


def mock_local_llm(prompt: str) -> LLMResponse:
    """
    A small deterministic mock LLM to let you test the pipeline offline.
    It looks at the prompt and returns structured JSON with recommended options.
    """
    # Very tiny parser to detect category from prompt
    lower = prompt.lower()
    category = "photography" if "photograph" in lower else ("catering" if "cater" in lower else "decor")
    sample = {
        "recommendations": [
            {"service_id": "local-svc-1", "title": f"Mock {category} A", "price": 30000, "reason": "Good fit in your city."},
            {"service_id": "local-svc-2", "title": f"Mock {category} B", "price": 22000, "reason": "Lower price, good reviews."}
        ],
        "explain": f"Found 2 {category} options that match your budget and location. You can negotiate the price or confirm booking."
    }
    text = "Recommendations:\n" + json.dumps(sample, indent=2)
    return LLMResponse(text=text, structured=sample)


# --------------------------
# FastAPI app + endpoints
# --------------------------
app = FastAPI(title="Event Agent Service (Recommend + Negotiate)")

# Seed DB
create_db_and_seed()


class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    event_type: str = Field(..., description="wedding / birthday / corporate")
    services: List[str] = Field(..., description="list of required service categories e.g. ['photography', 'catering']")
    budget: Optional[float] = None
    date: Optional[str] = None
    location: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None


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
    
@app.post("/agent/recommend", response_model=RecommendResponse)
def agent_recommend(req:RecommendRequest):
    """
    1. Retrieve candidates from DB for each requested service category
    2. Rank them using a simple heuristic
    3. Build a structured prompt for the LLM and call it
    4. Return LLM's suggestion + structured candidate list
    """    
    all_candidates = []
    for cat in req.services:
        cands = retrieve_candidates(cat, req.location, req.date, req.budget, limit=6)
        for c in cands:
            s: Service = c["service"]
            v: Vendor = c["vendor"]
            all_candidates.append({
                "service_id": s.id,
                "vendor_id": v.id,
                "vendor_name": v.name,
                "category": s.category,
                "title": s.title,
                "avg_price": c["avg_price"],
                "price_min": s.price_min,
                "price_max": s.price_max,
                "score": c["score"],
            })

    if not all_candidates:
        raise HTTPException(status_code=404, detail="No candidates found for requested services")

    # Build a short structured summary for the LLM
    summary_candidates = [
        {
            "service_id": c["service_id"],
            "vendor_name": c["vendor_name"],
            "category": c["category"],
            "title": c["title"],
            "avg_price": c["avg_price"],
            "price_min": c["price_min"],
            "price_max": c["price_max"],
            "score": c["score"],
        }
        for c in all_candidates
    ][:8]

    prompt = build_recommend_prompt(req, summary_candidates)
    llm_resp = llm_call(prompt, system="You are an expert event planning assistant. Provide user-friendly recommendations and a JSON array 'recommendations' with chosen service_ids, proposed price, and short reason.")

    # Map LLM JSON to our output if available, otherwise use DB candidates
    recommendations_out: List[CandidateOut] = []
    if llm_resp.structured and "recommendations" in llm_resp.structured:
        for rec in llm_resp.structured["recommendations"]:
            # try to locate service by id in candidates
            found = next((c for c in summary_candidates if c["service_id"] == rec.get("service_id")), None)
            if found:
                recommendations_out.append(CandidateOut(
                    service_id=found["service_id"],
                    vendor_name=found["vendor_name"],
                    title=found["title"],
                    avg_price=found["avg_price"],
                    score=found["score"],
                    price_min=found["price_min"],
                    price_max=found["price_max"],
                    reason=rec.get("reason")
                ))
            else:
                # If LLM returned custom ids (e.g., mock), return minimal
                recommendations_out.append(CandidateOut(
                    service_id=rec.get("service_id", "unknown"),
                    vendor_name=rec.get("vendor_name", "unknown"),
                    title=rec.get("title", "unknown"),
                    avg_price=rec.get("price", 0.0),
                    score=0.0,
                    price_min=0.0,
                    price_max=0.0,
                    reason=rec.get("reason")
                ))
    else:
        # fallback: pick top 3 DB candidates
        top = sorted(summary_candidates, key=lambda x: x["score"], reverse=True)[:3]
        for c in top:
            recommendations_out.append(CandidateOut(
                service_id=c["service_id"],
                vendor_name=c["vendor_name"],
                title=c["title"],
                avg_price=c["avg_price"],
                score=c["score"],
                price_min=c["price_min"],
                price_max=c["price_max"],
                reason="Good match based on location & price."
            ))

    return RecommendResponse(
        query_id=str(uuid.uuid4()),
        recommendations=recommendations_out,
        assistant_text=llm_resp.text,
        assistant_json=llm_resp.structured
    )


def build_recommend_prompt(req: RecommendRequest, summary_candidates: List[Dict[str, Any]]):
    user_req = {
        "event_type": req.event_type,
        "services": req.services,
        "budget": req.budget,
        "date": req.date,
        "location": req.location,
        "extras": req.extras
    }
    prompt = (
        "User request:\n" + json.dumps(user_req, indent=2) + "\n\n"
        "Available candidate services (JSON array):\n" + json.dumps(summary_candidates, indent=2) + "\n\n"
        "Task: Recommend the best-fit services for the user's needs. For each recommendation, include:\n"
        "  - service_id (must match a candidate service_id)\n"
        "  - recommended_price (a single number the user can pay, prefer the lower end but reasonable)\n"
        "  - reason (short)\n\n"
        "Return both a short natural-language explanation and a JSON object with a top-level key 'recommendations'.\n"
        "Be concise and present the JSON blob at the end of your reply so it can be parsed."
    )
    return prompt


# --------------------------
# Negotiation endpoint
# --------------------------
class NegotiateRequest(BaseModel):
    user_id: Optional[str] = None
    service_id: str
    user_offer: float
    context: Optional[Dict[str, Any]] = None


class NegotiateResponse(BaseModel):
    service_id: str
    original_price_min: float
    original_price_max: float
    vendor_name: str
    offered_price: float
    vendor_response: str
    accepted: bool
    explanation: Optional[str] = None


@app.post("/agent/negotiate", response_model=NegotiateResponse)
def agent_negotiate(req: NegotiateRequest):
    """
    Negotiation logic:
      - Find service
      - Determine vendor min/acceptable threshold (e.g., they accept if offer >= price_min * acceptance_factor)
      - Optionally consult LLM to craft a negotiation explanation/justification
      - Return accepted: true/false and explanation.
    """
    with Session(engine) as session:
        stmt = select(Service, Vendor).join(Vendor, Service.vendor_id == Vendor.id).where(Service.id == req.service_id)
        res = session.exec(stmt).first()
        if not res:
            raise HTTPException(status_code=404, detail="Service not found")
        svc, vendor = res

    # business rule: vendor will accept if offer >= vendor_accept_threshold
    # vendor_accept_threshold = price_min + 20% of (price_max - price_min) as buffer
    buffer = 0.2 * (svc.price_max - svc.price_min)
    vendor_accept_threshold = svc.price_min + buffer

    # If user offer is >= threshold => accept. If lower but within 10% below threshold, LLM can propose counter-offer.
    accepted = False
    vendor_response = ""
    final_price = req.user_offer

    if req.user_offer >= vendor_accept_threshold:
        accepted = True
        vendor_response = f"Vendor {vendor.name} auto-accepts the proposed price of {req.user_offer:.0f}."
        explanation = "Offer meets vendor's minimum acceptable threshold."
    else:
        # Ask LLM to craft a negotiation message / propose counter
        prompt = (
            f"User offered {req.user_offer:.2f} INR for service '{svc.title}' (vendor {vendor.name}).\n"
            f"Service price_min: {svc.price_min:.2f}, price_max: {svc.price_max:.2f}.\n"
            "You are an assistant negotiating on behalf of the user. If you can persuade the vendor, suggest a counter-offer price "
            "or a compromise (like reducing scope). Provide a JSON output like { 'accepted': bool, 'counter_offer': number, 'message': str }.\n"
            "Be realistic and conservative."
        )
        llm_resp = llm_call(prompt, system="You are a negotiation assistant, pragmatic and polite.", max_tokens=200)
        # parse LLM json if present
        structured = llm_resp.structured
        if structured:
            # trust the LLM's suggested counter offer if present
            counter = structured.get("counter_offer")
            acc = structured.get("accepted")
            message = structured.get("message", llm_resp.text)
            if acc:
                accepted = True
                final_price = counter if counter else req.user_offer
                vendor_response = f"Negotiation result: vendor tentatively accepts {final_price} (via LLM)."
            else:
                accepted = False
                final_price = counter if counter else req.user_offer
                vendor_response = message
        else:
            # fallback rule-based counter-offer
            # propose midpoint between user's offer and vendor_accept_threshold
            counter = (req.user_offer + vendor_accept_threshold) / 2.0
            vendor_response = (
                f"Vendor prefers at least {vendor_accept_threshold:.0f}. Negotiation proposes counter-offer {counter:.0f}."
            )
            final_price = counter
            accepted = False
            explanation = "Auto counter-proposed due to missing LLM structured output."

    return NegotiateResponse(
        service_id=svc.id,
        original_price_min=svc.price_min,
        original_price_max=svc.price_max,
        vendor_name=vendor.name,
        offered_price=round(final_price, 2),
        vendor_response=vendor_response,
        accepted=accepted,
        explanation=locals().get("explanation", None)
    )


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": LLM_PROVIDER, "has_crewai": HAS_CREWAI}
