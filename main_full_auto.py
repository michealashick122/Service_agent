"""
Enhanced AI Event Planning Chat Application
==========================================

Run:
  python -m venv .venv
  .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/Mac)
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
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field as SQLField, create_engine, Session, select, or_, and_
from dotenv import load_dotenv
import re

load_dotenv()

# -------------------------
# Config / LLM client setup
# -------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

OPENAI_AVAILABLE = False
if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_AVAILABLE = True
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ OpenAI initialization failed: {e}")
        openai_client = None
        OPENAI_AVAILABLE = False
else:
    openai_client = None

# -------------------------
# Database Models
# -------------------------
class Vendor(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    location: str
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    rating: float = 4.5
    years_experience: Optional[int] = None
    description: Optional[str] = None
    specializations: Optional[str] = None  # JSON string of specialties

class Service(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    vendor_id: str
    category: str  # photography, catering, decor, makeup, dj, cleaning, etc.
    title: str
    description: Optional[str] = None
    price_min: float = 0.0
    price_max: float = 0.0
    duration_hours: Optional[int] = None
    capacity_min: Optional[int] = None  # min people they can serve
    capacity_max: Optional[int] = None  # max people they can serve
    tags: Optional[str] = None  # JSON string of tags
    availability_calendar: Optional[str] = None  # JSON string of available dates
    package_details: Optional[str] = None  # JSON string of what's included

class ChatSession(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str
    session_data: Optional[str] = None  # JSON string to store user preferences/context
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ChatMessage(SQLModel, table=True):
    id: Optional[str] = SQLField(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str
    message_type: str  # "user" or "assistant"
    content: str
    message_metadata: Optional[str] = None  # JSON string for additional data
    timestamp: datetime = Field(default_factory=datetime.now)

# -------------------------
# Database Setup and Seeding
# -------------------------
DATABASE_URL = "sqlite:///./event_planner_chat.db"
engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_seed():
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        # Check if data already exists
        existing = session.exec(select(Vendor)).first()
        if existing:
            return
            
        print("🌱 Seeding database with sample data...")
        
        # Create vendors
        vendors_data = [
            {
                "name": "PixelStories Photography",
                "location": "Bangalore",
                "rating": 4.7,
                "years_experience": 8,
                "contact_phone": "+91-9876543210",
                "contact_email": "info@pixelstories.com",
                "description": "Professional wedding and event photography with candid and traditional styles",
                "specializations": json.dumps(["wedding", "pre-wedding", "corporate", "family"])
            },
            {
                "name": "GrandCaterers",
                "location": "Bangalore",
                "rating": 4.4,
                "years_experience": 12,
                "contact_phone": "+91-9876543211",
                "contact_email": "orders@grandcaterers.com",
                "description": "Multi-cuisine catering for all types of events",
                "specializations": json.dumps(["wedding", "corporate", "birthday", "anniversary"])
            },
            {
                "name": "Spark Decor",
                "location": "Koramangala",
                "rating": 4.6,
                "years_experience": 6,
                "contact_phone": "+91-9876543212",
                "contact_email": "bookings@sparkdecor.com",
                "description": "Creative event decoration and stage setup",
                "specializations": json.dumps(["wedding", "birthday", "corporate", "festival"])
            },
            {
                "name": "Glamour Makeup Studio",
                "location": "Indiranagar",
                "rating": 4.8,
                "years_experience": 10,
                "contact_phone": "+91-9876543213",
                "contact_email": "book@glamourmakeup.com",
                "description": "Professional bridal and party makeup services",
                "specializations": json.dumps(["bridal", "party", "photoshoot"])
            },
            {
                "name": "Beat Masters DJ",
                "location": "Whitefield",
                "rating": 4.5,
                "years_experience": 7,
                "contact_phone": "+91-9876543214",
                "contact_email": "events@beatmasters.com",
                "description": "Professional DJ services with latest sound equipment",
                "specializations": json.dumps(["wedding", "birthday", "corporate", "club"])
            }
        ]
        
        vendors = []
        for v_data in vendors_data:
            vendor = Vendor(**v_data)
            session.add(vendor)
            vendors.append(vendor)
        
        session.commit()  # Commit to get vendor IDs
        
        # Create services
        services_data = [
            # Photography services
            {
                "vendor_id": vendors[0].id,
                "category": "photography",
                "title": "Complete Wedding Photography Package",
                "description": "Full day wedding coverage with candid and traditional shots, includes edited photos and album",
                "price_min": 30000,
                "price_max": 60000,
                "duration_hours": 12,
                "capacity_min": 50,
                "capacity_max": 500,
                "tags": json.dumps(["wedding", "candid", "traditional", "album", "drone"]),
                "package_details": json.dumps(["2 photographers", "500+ edited photos", "wedding album", "online gallery"])
            },
            {
                "vendor_id": vendors[0].id,
                "category": "photography",
                "title": "Pre-wedding & Portrait Session",
                "description": "Creative pre-wedding shoot with multiple locations and costume changes",
                "price_min": 10000,
                "price_max": 25000,
                "duration_hours": 6,
                "tags": json.dumps(["pre-wedding", "portrait", "couple", "outdoor"]),
                "package_details": json.dumps(["1 photographer", "100+ edited photos", "2 locations", "props included"])
            },
            
            # Catering services
            {
                "vendor_id": vendors[1].id,
                "category": "catering",
                "title": "Vegetarian Wedding Catering",
                "description": "Multi-cuisine vegetarian buffet with live counters and dessert station",
                "price_min": 400,
                "price_max": 800,
                "capacity_min": 50,
                "capacity_max": 1000,
                "tags": json.dumps(["vegetarian", "buffet", "live-counter", "dessert", "south-indian", "north-indian"]),
                "package_details": json.dumps(["Welcome drink", "Live chat counter", "3 main courses", "Dessert station", "Service staff"])
            },
            {
                "vendor_id": vendors[1].id,
                "category": "catering",
                "title": "Non-Vegetarian Premium Catering",
                "description": "Premium non-veg catering with BBQ and live grill stations",
                "price_min": 600,
                "price_max": 1200,
                "capacity_min": 50,
                "capacity_max": 800,
                "tags": json.dumps(["non-vegetarian", "bbq", "grill", "premium", "continental"]),
                "package_details": json.dumps(["BBQ station", "Live grill", "4 main courses", "Continental options", "Premium service"])
            },
            
            # Decoration services
            {
                "vendor_id": vendors[2].id,
                "category": "decoration",
                "title": "Wedding Stage Decoration",
                "description": "Elegant wedding stage setup with floral arrangements and lighting",
                "price_min": 25000,
                "price_max": 80000,
                "tags": json.dumps(["stage", "flowers", "lighting", "backdrop", "mandap"]),
                "package_details": json.dumps(["Stage setup", "Floral decoration", "LED lighting", "Backdrop", "Seating arrangement"])
            },
            {
                "vendor_id": vendors[2].id,
                "category": "decoration",
                "title": "Birthday Party Decoration",
                "description": "Themed birthday party decoration with balloons and banners",
                "price_min": 5000,
                "price_max": 20000,
                "capacity_min": 20,
                "capacity_max": 200,
                "tags": json.dumps(["birthday", "balloons", "banner", "theme", "kids", "adult"]),
                "package_details": json.dumps(["Balloon decoration", "Custom banners", "Table setup", "Photo booth", "Theme props"])
            },
            
            # Makeup services
            {
                "vendor_id": vendors[3].id,
                "category": "makeup",
                "title": "Bridal Makeup Package",
                "description": "Complete bridal makeup with hair styling and draping",
                "price_min": 15000,
                "price_max": 40000,
                "duration_hours": 4,
                "tags": json.dumps(["bridal", "hair-styling", "draping", "traditional", "modern"]),
                "package_details": json.dumps(["Makeup trial", "Wedding day makeup", "Hair styling", "Saree draping", "Touch-up kit"])
            },
            {
                "vendor_id": vendors[3].id,
                "category": "makeup",
                "title": "Party Makeup",
                "description": "Glamorous party makeup for special occasions",
                "price_min": 3000,
                "price_max": 8000,
                "duration_hours": 2,
                "tags": json.dumps(["party", "glamour", "evening", "special-occasion"]),
                "package_details": json.dumps(["Makeup application", "Basic hair styling", "Makeup setting", "Quick touch-up"])
            },
            
            # DJ services
            {
                "vendor_id": vendors[4].id,
                "category": "dj",
                "title": "Wedding DJ with Sound System",
                "description": "Professional DJ service with high-quality sound system and lighting",
                "price_min": 20000,
                "price_max": 50000,
                "duration_hours": 8,
                "capacity_min": 100,
                "capacity_max": 1000,
                "tags": json.dumps(["wedding", "sound-system", "lighting", "microphone", "dance-floor"]),
                "package_details": json.dumps(["Professional DJ", "Sound system", "Wireless mics", "Dance lighting", "Music mixing"])
            },
            {
                "vendor_id": vendors[4].id,
                "category": "dj",
                "title": "Birthday Party DJ Package",
                "description": "Fun DJ service perfect for birthday parties with party games music",
                "price_min": 8000,
                "price_max": 20000,
                "duration_hours": 4,
                "capacity_min": 30,
                "capacity_max": 300,
                "tags": json.dumps(["birthday", "party-games", "kids", "adult", "dance"]),
                "package_details": json.dumps(["DJ service", "Party music", "Game music", "Announcements", "Basic lighting"])
            }
        ]
        
        for s_data in services_data:
            service = Service(**s_data)
            session.add(service)
        
        session.commit()
        print("✅ Database seeded successfully!")

create_db_and_seed()

# -------------------------
# Pydantic Models for API
# -------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: str = Field(..., description="Unique identifier for the user")
    message: str = Field(..., description="User's message/query")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class ServiceRecommendation(BaseModel):
    service_id: str
    vendor_name: str
    service_title: str
    category: str
    price_range: str
    rating: float
    description: str
    contact_info: Dict[str, str]
    package_details: List[str]
    availability_info: str
    match_score: float
    recommendation_reason: str

class ChatResponse(BaseModel):
    session_id: str
    message: str
    recommendations: List[ServiceRecommendation] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    extracted_info: Optional[Dict[str, Any]] = None

# -------------------------
# AI Service Functions
# -------------------------
def extract_user_requirements(message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract user requirements from natural language using AI or rule-based parsing"""
    
    if context is None:
        context = {}
        
    # Define categories mapping for flexibility
    service_categories = {
        'photography': ['photo', 'photographer', 'camera', 'shoot', 'pictures', 'pics', 'wedding photography', 'candid'],
        'catering': ['food', 'catering', 'caterer', 'lunch', 'dinner', 'breakfast', 'meals', 'buffet', 'menu'],
        'decoration': ['decoration', 'decor', 'decorating', 'flowers', 'stage', 'backdrop', 'balloons', 'theme'],
        'makeup': ['makeup', 'beauty', 'bridal makeup', 'hair', 'styling', 'mehendi', 'henna'],
        'dj': ['dj', 'music', 'sound', 'audio', 'entertainment', 'dance', 'party music', 'sound system'],
        'cleaning': ['cleaning', 'housekeeping', 'maintenance', 'cleanup', 'sanitization']
    }
    
    event_types = {
        'wedding': ['wedding', 'marriage', 'shaadi', 'matrimony', 'bride', 'groom'],
        'birthday': ['birthday', 'bday', 'anniversary', 'celebration'],
        'corporate': ['corporate', 'office', 'business', 'company', 'professional', 'conference'],
        'festival': ['festival', 'celebration', 'cultural', 'religious', 'traditional']
    }
    
    message_lower = message.lower()
    
    # Extract services needed
    services_needed = []
    for category, keywords in service_categories.items():
        if any(keyword in message_lower for keyword in keywords):
            services_needed.append(category)
    
    # Extract event type
    event_type = None
    for etype, keywords in event_types.items():
        if any(keyword in message_lower for keyword in keywords):
            event_type = etype
            break
    
    # Extract budget using regex
    budget = None
    budget_patterns = [
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs|rupees|inr)?',
        r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|lakhs|lacs)',
        r'budget\s*(?:is|of)?\s*(?:around|about)?\s*(\d+(?:,\d{3})*)',
        r'(\d+)k\b'
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, message_lower)
        if match:
            amount = match.group(1).replace(',', '')
            if 'lakh' in match.group(0) or 'lac' in match.group(0):
                budget = float(amount) * 100000
            elif 'k' in match.group(0):
                budget = float(amount) * 1000
            else:
                budget = float(amount)
            break
    
    # Extract location
    location = None
    location_keywords = ['bangalore', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'pune', 'kolkata', 
                        'koramangala', 'indiranagar', 'whitefield', 'jp nagar', 'electronic city']
    for loc in location_keywords:
        if loc in message_lower:
            location = loc.title()
            break
    
    # Extract guest count
    guest_count = None
    guest_patterns = [
        r'(\d+)\s*(?:people|guests|persons|pax)',
        r'(?:around|about|approximately)\s*(\d+)',
        r'(\d+)\s*member'
    ]
    
    for pattern in guest_patterns:
        match = re.search(pattern, message_lower)
        if match:
            guest_count = int(match.group(1))
            break
    
    # Extract date information
    date_info = None
    date_patterns = [
        r'(\d{1,2})\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})?',
        r'(?:next|this)\s*(week|month|year)',
        r'(?:tomorrow|today|yesterday)',
        r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, message_lower)
        if match:
            date_info = match.group(0)
            break
    
    # Use AI for better extraction if available
    if OPENAI_AVAILABLE:
        try:
            ai_extraction = get_ai_extraction(message, context)
            if ai_extraction:
                # Merge AI extraction with rule-based extraction
                services_needed = ai_extraction.get('services', services_needed)
                event_type = ai_extraction.get('event_type', event_type)
                budget = ai_extraction.get('budget', budget)
                location = ai_extraction.get('location', location)
                guest_count = ai_extraction.get('guest_count', guest_count)
                date_info = ai_extraction.get('date_info', date_info)
        except Exception as e:
            print(f"AI extraction failed, using rule-based: {e}")
    
    return {
        'services_needed': services_needed,
        'event_type': event_type,
        'budget': budget,
        'location': location,
        'guest_count': guest_count,
        'date_info': date_info,
        'raw_message': message
    }

def get_ai_extraction(message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use OpenAI to extract structured information from user message"""
    
    system_prompt = """
    You are an expert event planning assistant. Extract structured information from user messages about event planning needs.
    
    Extract the following information and return as JSON:
    - services_needed: array of service categories (photography, catering, decoration, makeup, dj, cleaning)
    - event_type: string (wedding, birthday, corporate, festival, or null)
    - budget: number (in INR, convert lakhs to actual numbers, e.g., "2 lakh" = 200000)
    - location: string (city or area name)
    - guest_count: number (approximate number of attendees)
    - date_info: string (any date/time information mentioned)
    - special_requirements: array of any specific needs or preferences
    
    Return only valid JSON. If information is not available, use null for that field.
    """
    
    user_prompt = f"""
    Previous context: {json.dumps(context) if context else "None"}
    Current message: {message}
    
    Extract the event planning information from this message.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        # Try to parse JSON from response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Extract JSON from text if wrapped in other text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
    except Exception as e:
        print(f"OpenAI extraction error: {e}")
        return None

def find_matching_services(requirements: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Find services matching user requirements"""
    
    with Session(engine) as session:
        query = select(Service, Vendor).join(Vendor, Service.vendor_id == Vendor.id)
        
        # Filter by service categories if specified
        if requirements.get('services_needed'):
            query = query.where(Service.category.in_(requirements['services_needed']))
        
        # Filter by location if specified
        if requirements.get('location'):
            location = requirements['location'].lower()
            query = query.where(Vendor.location.ilike(f"%{location}%"))
        
        # Filter by capacity if guest count specified
        if requirements.get('guest_count'):
            guest_count = requirements['guest_count']
            query = query.where(
                or_(
                    Service.capacity_min.is_(None),
                    and_(Service.capacity_min <= guest_count, Service.capacity_max >= guest_count),
                    Service.capacity_max.is_(None)
                )
            )
        
        results = session.exec(query).all()
        
        # Score and rank results
        scored_results = []
        for service, vendor in results:
            score = calculate_match_score(service, vendor, requirements)
            
            # Filter by budget if specified (with some flexibility)
            if requirements.get('budget'):
                budget = requirements['budget']
                # Allow 20% flexibility in budget
                if service.price_min > budget * 1.2:
                    continue
            
            scored_results.append({
                'service': service,
                'vendor': vendor,
                'score': score
            })        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:limit]

def calculate_match_score(service: Service, vendor: Vendor, requirements: Dict[str, Any]) -> float:
    """Calculate how well a service matches user requirements"""
    
    score = 0.0
    
    # Base score from vendor rating
    score += vendor.rating * 10
    
    # Location_match
    if requirements.get('location'):
        user_location = requirements['location'].lower()
        vendor_location = vendor.location.lower()
        if user_location in vendor_location or vendor_location in user_location:
            score += 25
    
    # Budget match
    if requirements.get('budget'):
        budget = requirements['budget']
        avg_price = (service.price_min + service.price_max) / 2
        
        if service.price_min <= budget <= service.price_max:
            score += 30  # Perfect budget fit
        else:
            # Penalty for budget mismatch
            budget_diff = abs(avg_price - budget) / max(budget, avg_price)
            score -= budget_diff * 20
    
    # Capacity match
    if requirements.get('guest_count') and service.capacity_min and service.capacity_max:
        guest_count = requirements['guest_count']
        if service.capacity_min <= guest_count <= service.capacity_max:
            score += 20
    
    # Service category exact match
    if requirements.get('services_needed'):
        if service.category in requirements['services_needed']:
            score += 15
    
    # Experience bonus
    if vendor.years_experience:
        score += min(vendor.years_experience, 15)  # Cap at 15 points
    
    return round(score, 2)

def generate_ai_response(requirements: Dict[str, Any], matches: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Generate natural AI response using OpenAI or fallback to template"""
    
    if OPENAI_AVAILABLE and matches:
        return generate_openai_response(requirements, matches, context)
    else:
        return generate_template_response(requirements, matches, context)

def generate_openai_response(requirements: Dict[str, Any], matches: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Generate response using OpenAI"""
    
    # Prepare match summary for AI
    match_summaries = []
    for match in matches[:5]:  # Top 5 matches
        service = match['service']
        vendor = match['vendor']
        match_summaries.append({
            'vendor': vendor.name,
            'service': service.title,
            'category': service.category,
            'price_range': f"₹{service.price_min:,} - ₹{service.price_max:,}",
            'rating': vendor.rating,
            'location': vendor.location
        })
    
    system_prompt = """
    You are Zing AI, a friendly and helpful event planning assistant. You help users find the best service providers for their events.
    
    Guidelines:
    - Be conversational and helpful
    - Mention specific vendor names and services when recommending
    - Include price ranges when discussing options
    - Ask follow-up questions to better understand needs
    - Be enthusiastic about helping plan their event
    - If budget seems tight, suggest alternatives or negotiation possibilities
    - Always maintain a professional yet friendly tone
    """
    
    user_prompt = f"""
    User requirements: {json.dumps(requirements)}
    Available matches: {json.dumps(match_summaries)}
    Previous context: {json.dumps(context)}
    
    Generate a helpful response recommending the best services for the user's event. Be specific about vendors and prices.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"OpenAI response generation error: {e}")
        return generate_template_response(requirements, matches, context)

def generate_template_response(requirements: Dict[str, Any], matches: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    """Generate response using templates as fallback"""
    
    if not matches:
        return "I understand you're looking for event services, but I couldn't find any matches for your specific requirements. Could you please provide more details about your event, such as the type of event, location, and budget? This will help me find better options for you."
    
    event_type = requirements.get('event_type', 'event')
    services_needed = requirements.get('services_needed', [])
    budget = requirements.get('budget')
    location = requirements.get('location', 'your area')
    
    response = f"Great! I found some excellent options for your {event_type} in {location}. "
    
    # Group matches by category
    by_category = {}
    for match in matches:
        category = match['service'].category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(match)
    
    recommendations = []
    for category, category_matches in by_category.items():
        best_match = category_matches[0]  # Highest scored
        service = best_match['service']
        vendor = best_match['vendor']
        
        price_range = f"₹{service.price_min:,} - ₹{service.price_max:,}"
        recommendations.append(f"For {category}, I recommend {vendor.name} - they offer '{service.title}' with prices ranging from {price_range}. They have a {vendor.rating} star rating.")
    
    response += "\n\n" + "\n".join(recommendations)
    
    if budget:
        response += f"\n\nBased on your budget of ₹{budget:,}, these options should work well for you."
    
    response += "\n\nWould you like more details about any of these services, or do you have specific questions about pricing or availability?"
    
    return response

def get_follow_up_questions(requirements: Dict[str, Any]) -> List[str]:
    """Generate relevant follow-up questions based on missing information"""
    
    questions = []
    
    if not requirements.get('event_type'):
        questions.append("What type of event are you planning? (wedding, birthday, corporate, etc.)")
    
    if not requirements.get('budget'):
        questions.append("What's your budget range for this event?")
    
    if not requirements.get('location'):
        questions.append("Which city or area are you planning the event in?")
    
    if not requirements.get('guest_count'):
        questions.append("Approximately how many guests will be attending?")
    
    if not requirements.get('date_info'):
        questions.append("When are you planning to have the event?")
    
    if not requirements.get('services_needed'):
        questions.append("Which services do you need? (photography, catering, decoration, etc.)")
    
    return questions[:3]  # Return max 3 questions to avoid overwhelming

# -------------------------
# FastAPI Application
# -------------------------
app = FastAPI(
    title="AI Event Planning Chat Assistant",
    description="Intelligent chat assistant for event planning with dynamic service recommendations",
    version="2.0.0"
)

# Add CORS middleware for web application integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that handles natural language queries and provides intelligent responses
    """
    
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        with Session(engine) as db_session:
            # Store user message
            user_message = ChatMessage(
                session_id=session_id,
                message_type="user",
                content=request.message,
                message_metadata=json.dumps(request.context)
            )
            db_session.add(user_message)
            
            # Get or create chat session
            chat_session = db_session.exec(
                select(ChatSession).where(ChatSession.id == session_id)
            ).first()
            
            if not chat_session:
                chat_session = ChatSession(
                    id=session_id,
                    user_id=request.user_id,
                    session_data=json.dumps({})
                )
                db_session.add(chat_session)
            
            # Update session context
            session_data = json.loads(chat_session.session_data or "{}")
            session_data.update(request.context)
            
            db_session.commit()
        
        # Extract requirements from user message
        requirements = extract_user_requirements(request.message, session_data)
        
        # Update session data with extracted requirements
        session_data.update({
            'last_requirements': requirements,
            'conversation_history': session_data.get('conversation_history', []) + [request.message]
        })
        
        # Find matching services
        matches = find_matching_services(requirements)
        
        # Generate AI response
        ai_response = generate_ai_response(requirements, matches, session_data)
        
        # Create service recommendations
        recommendations = []
        for match in matches[:5]:  # Top 5 recommendations
            service = match['service']
            vendor = match['vendor']
            
            # Parse package details
            package_details = []
            if service.package_details:
                try:
                    package_details = json.loads(service.package_details)
                except:
                    package_details = [service.package_details]
            
            recommendation = ServiceRecommendation(
                service_id=service.id,
                vendor_name=vendor.name,
                service_title=service.title,
                category=service.category,
                price_range=f"₹{service.price_min:,} - ₹{service.price_max:,}",
                rating=vendor.rating,
                description=service.description or "Service details available on request",
                contact_info={
                    "phone": vendor.contact_phone or "Contact available on request",
                    "email": vendor.contact_email or "Email available on request"
                },
                package_details=package_details,
                availability_info="Available - Contact vendor for specific dates",
                match_score=match['score'],
                recommendation_reason=generate_recommendation_reason(match, requirements)
            )
            recommendations.append(recommendation)
        
        # Generate follow-up questions
        follow_up_questions = get_follow_up_questions(requirements)
        
        # Determine if clarification is needed
        requires_clarification = (
            not requirements.get('services_needed') or 
            len(matches) == 0 or
            not requirements.get('event_type')
        )
        
        # Store assistant response
        with Session(engine) as db_session:
            assistant_message = ChatMessage(
                session_id=session_id,
                message_type="assistant",
                content=ai_response,
                message_metadata=json.dumps({
                    'recommendations_count': len(recommendations),
                    'requirements': requirements
                })
            )
            db_session.add(assistant_message)
            
            # Update session data
            chat_session = db_session.get(ChatSession, session_id)
            if chat_session:
                chat_session.session_data = json.dumps(session_data)
                chat_session.updated_at = datetime.now()
            
            db_session.commit()
        
        return ChatResponse(
            session_id=session_id,
            message=ai_response,
            recommendations=recommendations,
            follow_up_questions=follow_up_questions,
            requires_clarification=requires_clarification,
            extracted_info=requirements
        )
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def generate_recommendation_reason(match: Dict[str, Any], requirements: Dict[str, Any]) -> str:
    """Generate explanation for why this service was recommended"""
    
    reasons = []
    service = match['service']
    vendor = match['vendor']
    score = match['score']
    
    # High rating reason
    if vendor.rating >= 4.5:
        reasons.append(f"Highly rated ({vendor.rating} stars)")
    
    # Location match
    if requirements.get('location'):
        user_location = requirements['location'].lower()
        vendor_location = vendor.location.lower()
        if user_location in vendor_location or vendor_location in user_location:
            reasons.append("Located in your preferred area")
    
    # Budget fit
    if requirements.get('budget'):
        budget = requirements['budget']
        if service.price_min <= budget <= service.price_max:
            reasons.append("Fits within your budget")
        elif service.price_min <= budget * 1.1:
            reasons.append("Close to your budget range")
    
    # Experience
    if vendor.years_experience and vendor.years_experience >= 5:
        reasons.append(f"{vendor.years_experience} years of experience")
    
    # Capacity match
    if requirements.get('guest_count') and service.capacity_min and service.capacity_max:
        guest_count = requirements['guest_count']
        if service.capacity_min <= guest_count <= service.capacity_max:
            reasons.append("Can handle your guest count")
    
    if not reasons:
        reasons.append("Good match for your requirements")
    
    return " • ".join(reasons)

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    
    with Session(engine) as session:
        messages = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp)
        ).all()
        
        return {
            "session_id": session_id,
            "messages": [
                {
                    "id": msg.id,
                    "type": msg.message_type,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": json.loads(msg.message_metadata or "{}")
                }
                for msg in messages
            ]
        }

@app.get("/services/categories")
async def get_service_categories():
    """Get all available service categories"""
    
    with Session(engine) as session:
        categories = session.exec(
            
            select(Service.category).distinct()
        ).all()
        
        return {"categories": categories}

@app.get("/services/search")
async def search_services(
    category: Optional[str] = None,
    location: Optional[str] = None,
    min_budget: Optional[float] = None,
    max_budget: Optional[float] = None,
    min_rating: Optional[float] = None
):
    """Search services with filters"""
    
    with Session(engine) as session:
        query = select(Service, Vendor).join(Vendor, Service.vendor_id == Vendor.id)
        
        if category:
            query = query.where(Service.category == category)
        
        if location:
            query = query.where(Vendor.location.ilike(f"%{location}%"))
        
        if min_budget:
            query = query.where(Service.price_max >= min_budget)
        
        if max_budget:
            query = query.where(Service.price_min <= max_budget)
        
        if min_rating:
            query = query.where(Vendor.rating >= min_rating)
        
        results = session.exec(query).all()
        
        services = []
        for service, vendor in results:
            services.append({
                "service_id": service.id,
                "vendor_name": vendor.name,
                "service_title": service.title,
                "category": service.category,
                "price_range": f"₹{service.price_min:,} - ₹{service.price_max:,}",
                "rating": vendor.rating,
                "location": vendor.location,
                "description": service.description,
                "contact": {
                    "phone": vendor.contact_phone,
                    "email": vendor.contact_email
                }
            })
        
        return {"services": services, "count": len(services)}

@app.get("/vendors/{vendor_id}")
async def get_vendor_details(vendor_id: str):
    """Get detailed information about a vendor"""
    
    with Session(engine) as session:
        vendor = session.get(Vendor, vendor_id)
        if not vendor:
            raise HTTPException(status_code=404, detail="Vendor not found")
        
        # Get all services by this vendor
        services = session.exec(
            select(Service).where(Service.vendor_id == vendor_id)
        ).all()
        
        vendor_info = {
            "id": vendor.id,
            "name": vendor.name,
            "location": vendor.location,
            "rating": vendor.rating,
            "years_experience": vendor.years_experience,
            "description": vendor.description,
            "contact": {
                "phone": vendor.contact_phone,
                "email": vendor.contact_email
            },
            "specializations": json.loads(vendor.specializations or "[]"),
            "services": [
                {
                    "id": svc.id,
                    "category": svc.category,
                    "title": svc.title,
                    "description": svc.description,
                    "price_range": f"₹{svc.price_min:,} - ₹{svc.price_max:,}",
                    "package_details": json.loads(svc.package_details or "[]")
                }
                for svc in services
            ]
        }
        
        return vendor_info

@app.post("/chat/feedback")
async def submit_feedback(
    session_id: str = Body(...),
    message_id: str = Body(...),
    rating: int = Body(..., ge=1, le=5),
    comment: Optional[str] = Body(None)
):
    """Submit feedback for AI responses"""
    
    # In a real application, you would store this feedback for model improvement
    return {
        "message": "Thank you for your feedback!",
        "session_id": session_id,
        "rating": rating
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": LLM_PROVIDER,
        "openai_available": OPENAI_AVAILABLE,
        "features": {
            "ai_extraction": OPENAI_AVAILABLE,
            "ai_responses": OPENAI_AVAILABLE,
            "chat_history": True,
            "service_search": True,
            "vendor_details": True
        }
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Event Planning Chat Assistant",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/chat",
            "chat_history": "/chat/history/{session_id}",
            "search_services": "/services/search",
            "service_categories": "/services/categories",
            "vendor_details": "/vendors/{vendor_id}",
            "health": "/health",
            "docs": "/docs"
        },
        "description": "Send POST requests to /chat with natural language queries to get event planning recommendations"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)