# # main_enhanced_crewai_enterprise.py
# # Python 3.11+ recommended

# import os
# import json
# import time
# import logging
# from datetime import date
# from typing import Optional, List, Dict, Any

# import httpx
# from fastapi import FastAPI, Depends, HTTPException, Header, Request, status
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field, ConfigDict
# from pydantic_settings import BaseSettings
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
# from sqlalchemy import String, Integer, Date as SQLDate, select

# # CrewAI imports with tools
# from crewai import Agent, Task, Crew, LLM as CrewLLM
# from crewai.tools import BaseTool
# # from crewai_tools import BaseTool as CrewAIBaseTool

# # === Settings ===
# class Settings(BaseSettings):
#     API_KEY: Optional[str] = None
#     DATABASE_URL: str = "sqlite+aiosqlite:///./event_planner.db"
#     OPENAI_API_KEY: Optional[str] = None  # Will be loaded from .env
#     OPENAI_MODEL: str = "gpt-4o-mini"
#     SEED_DATA: bool = True
#     CORS_ORIGINS: List[str] = ["*"]

#     model_config = ConfigDict(env_file=".env", case_sensitive=False)

# settings = Settings()

# # === Logging ===
# logging.basicConfig(
#     level=os.getenv("LOG_LEVEL", "INFO"),
#     format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
# )
# log = logging.getLogger("enhanced-crewai-planner")

# # === API Key Validation and Setup ===
# def validate_and_setup_openai():
#     """Validate and setup OpenAI API key with proper error handling"""
    
#     # Try multiple sources for the API key
#     api_key = None
    
#     # 1. Check settings (from .env file)
#     if settings.OPENAI_API_KEY:
#         api_key = settings.OPENAI_API_KEY
#         log.info("Using OpenAI API key from settings/.env file")
    
#     # 2. Check environment variable directly
#     elif os.getenv("OPENAI_API_KEY"):
#         api_key = os.getenv("OPENAI_API_KEY")
#         print("GOT OPEN API KEYYYYYYYYYYYYYYYYYYYYYY")
#         log.info("Using OpenAI API key from environment variable")
    
#     # 3. Check alternative environment variable names
#     elif os.getenv("OPENAI_KEY"):
#         api_key = os.getenv("OPENAI_KEY")
#         log.info("Using OpenAI API key from OPENAI_KEY environment variable")
    
#     if not api_key:
#         log.error("No OpenAI API key found! Please set OPENAI_API_KEY in .env file or environment")
#         return None
    
#     # Validate API key format
#     if not api_key.startswith(('sk-', 'sk-proj-')):
#         log.error(f"Invalid OpenAI API key format: {api_key[:10]}...")
#         return None
    
#     # Set the environment variable for CrewAI/LiteLLM to use
#     os.environ["OPENAI_API_KEY"] = api_key
    
#     log.info(f"OpenAI API key configured successfully (ends with: ...{api_key[-10:]})")
#     return api_key

# # Validate API key on startup
# openai_api_key = validate_and_setup_openai()

# # === DB Models ===
# class Base(DeclarativeBase):
#     pass

# class Vendor(Base):
#     __tablename__ = "vendors"
#     id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
#     name: Mapped[str] = mapped_column(String(255))
#     service_type: Mapped[str] = mapped_column(String(100))
#     city: Mapped[str] = mapped_column(String(100))
#     price_min: Mapped[int] = mapped_column(Integer)
#     price_max: Mapped[int] = mapped_column(Integer)
#     available_date: Mapped[Optional[date]] = mapped_column(SQLDate, nullable=True)
#     contact: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

# engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
# AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# # === CrewAI Tools (Enhanced) ===

# class SearchVendorsTool(BaseTool):
#     name: str = "Search Vendors"
#     description: str = "Search for event vendors in database based on location, service type, date, and budget"

#     def _run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
#         """Search for vendors - this will be called synchronously by CrewAI"""
#         import asyncio
#         return asyncio.run(self._async_run(city, service_type, date_str, max_budget))
    
#     async def _async_run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
#         """Async implementation of vendor search"""
#         slots = {}
#         if city: slots["city"] = city
#         if service_type: slots["service_type"] = service_type
#         if date_str: slots["date"] = date_str
#         if max_budget: slots["budget"] = max_budget
        
#         vendors = await fetch_vendors(slots)
        
#         vendor_data = [
#             {
#                 "id": v.id,
#                 "name": v.name,
#                 "service_type": v.service_type,
#                 "city": v.city,
#                 "price_min": v.price_min,
#                 "price_max": v.price_max,
#                 "available_date": v.available_date.isoformat() if v.available_date else None,
#                 "contact": v.contact
#             }
#             for v in vendors
#         ]
        
#         return json.dumps({
#             "vendors": vendor_data,
#             "count": len(vendor_data),
#             "search_criteria": slots
#         })
import asyncio
class SearchVendorsTool(BaseTool):
    name: str = "Search Vendors"
    description: str = "Search for event vendors in database based on location, service type, date, and budget"

    def _run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
        """Sync implementation"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new thread-safe loop for async operations
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(
                        self._async_run(city, service_type, date_str, max_budget)
                    )
                    return result
                finally:
                    new_loop.close()
            else:
                return loop.run_until_complete(
                    self._async_run(city, service_type, date_str, max_budget)
                )
        except Exception as e:
            log.error(f"Search Vendors Tool Error: {str(e)}")
            # Return empty results instead of failing
            return json.dumps({
                "error": str(e),
                "vendors": [],
                "count": 0,
                "search_criteria": {
                    "city": city, 
                    "service_type": service_type,
                    "date": date_str,
                    "max_budget": max_budget
                }
            })

    async def _async_run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
        """Async implementation"""
        try:
            slots = {}
            if city: slots["city"] = city.lower()  # Case insensitive search
            if service_type: slots["service_type"] = service_type.lower()
            if date_str: slots["date"] = date_str
            if max_budget: slots["budget"] = max_budget

            vendors = await fetch_vendors(slots)
            
            if not vendors:
                return json.dumps({
                    "vendors": [],
                    "count": 0,
                    "message": "No vendors found matching criteria",
                    "search_criteria": slots
                })

            vendor_data = [
                {
                    "id": v.id,
                    "name": v.name,
                    "service_type": v.service_type,
                    "city": v.city,
                    "price_min": v.price_min,
                    "price_max": v.price_max,
                    "available_date": v.available_date.isoformat() if v.available_date else None,
                    "contact": v.contact
                }
                for v in vendors
            ]
            
            return json.dumps({
                "vendors": vendor_data,
                "count": len(vendor_data),
                "search_criteria": slots
            })
            
        except Exception as e:
            log.error(f"Async vendor search error: {str(e)}")
            return json.dumps({
                "error": str(e),
                "vendors": [],
                "count": 0,
                "search_criteria": slots
            })
class BudgetCalculatorTool(BaseTool):
    name: str = "Calculate Budget"
    description: str = "Calculate optimal budget allocation across different event services"

    def _run(self, total_budget: int, services: str, event_type: str = "general", priority_service: str = None) -> str:
        """Calculate budget allocation"""
        try:
            # Parse services if it's a string representation of a list
            if isinstance(services, str):
                if services.startswith('[') and services.endswith(']'):
                    services = json.loads(services)
                else:
                    services = [s.strip() for s in services.split(',')]
        except:
            services = [services] if isinstance(services, str) else []
        
        # Event-specific allocations
        event_allocations = {
            "wedding": {"food": 0.40, "decoration": 0.25, "camera": 0.20, "makeup": 0.10, "cleaning": 0.05},
            "birthday": {"food": 0.35, "decoration": 0.30, "camera": 0.15, "makeup": 0.10, "cleaning": 0.10},
            "corporate": {"food": 0.45, "decoration": 0.20, "camera": 0.20, "makeup": 0.05, "cleaning": 0.10},
            "general": {"food": 0.35, "decoration": 0.25, "camera": 0.15, "makeup": 0.15, "cleaning": 0.10}
        }
        
        allocations = event_allocations.get(event_type.lower(), event_allocations["general"])
        
        # Adjust for priority service
        if priority_service and priority_service in allocations:
            boost = 0.15
            allocations[priority_service] += boost
            
            other_services = [s for s in services if s != priority_service and s in allocations]
            if other_services:
                reduction_per_service = boost / len(other_services)
                for service in other_services:
                    allocations[service] = max(0.05, allocations[service] - reduction_per_service)
        
        budget_plan = []
        total_allocated = 0
        
        for service in services:
            if service in allocations:
                amount = int(total_budget * allocations[service])
                total_allocated += amount
                budget_plan.append({
                    "service": service,
                    "allocated_budget": amount,
                    "percentage": round(allocations[service] * 100, 1),
                    "suggested_range": f"₹{amount - amount//4:,} - ₹{amount + amount//4:,}"
                })
        
        return json.dumps({
            "total_budget": total_budget,
            "allocated_budget": total_allocated,
            "remaining_budget": total_budget - total_allocated,
            "allocations": budget_plan,
            "event_type": event_type
        })

class VendorRecommendationTool(BaseTool):
    name: str = "Get Vendor Recommendations"
    description: str = "Get personalized vendor recommendations based on event requirements"

    def _run(self, requirements_json: str) -> str:
        """Sync implementation"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new thread-safe loop for async operations
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(
                        self._async_run(requirements_json)
                    )
                    return result
                finally:
                    new_loop.close()
            else:
                return loop.run_until_complete(
                    self._async_run(requirements_json)
                )
        except Exception as e:
            log.error(f"Vendor Recommendations Tool Error: {str(e)}")
            # Return empty results instead of failing
            return json.dumps({
                "error": str(e),
                "recommendations": [],
                "total_services": 0,
                "message": "Error retrieving vendor recommendations"
            })

    async def _async_run(self, requirements_json: str) -> str:
        """Async implementation"""
        try:
            # Parse requirements
            try:
                requirements = json.loads(requirements_json)
            except json.JSONDecodeError as e:
                log.error(f"Invalid requirements JSON: {str(e)}")
                return json.dumps({"error": "Invalid requirements JSON format"})
            
            city = requirements.get("city")
            services = requirements.get("services", [])
            budget = requirements.get("budget")
            date_str = requirements.get("date")
            event_type = requirements.get("event_type", "general")
            
            all_recommendations = []
            
            # Process each requested service
            for service in services:
                service_budget = None
                if budget and len(services) > 0:
                    service_budget = budget // len(services)
                
                slots = {"city": city, "service_type": service}
                if date_str: slots["date"] = date_str
                if service_budget: slots["budget"] = service_budget
                
                vendors = await fetch_vendors(slots)
                
                if vendors:
                    # Rank vendors by value (price vs features)
                    best_vendor = min(vendors, key=lambda v: v.price_min)
                    
                    recommendation = {
                        "service": service,
                        "vendor": {
                            "id": best_vendor.id,
                            "name": best_vendor.name,
                            "price_min": best_vendor.price_min,
                            "price_max": best_vendor.price_max,
                            "contact": best_vendor.contact,
                            "available_date": best_vendor.available_date.isoformat() if best_vendor.available_date else None
                        },
                        "reasoning": f"Best value option for {service} in {city}",
                        "alternatives_count": len(vendors) - 1
                    }
                    all_recommendations.append(recommendation)
            
            return json.dumps({
                "recommendations": all_recommendations,
                "event_type": event_type,
                "total_services": len(services),
                "message": f"Found recommendations for {len(all_recommendations)} out of {len(services)} requested services"
            })
            
        except Exception as e:
            log.error(f"Async vendor recommendations error: {str(e)}")
            return json.dumps({
                "error": str(e),
                "recommendations": [],
                "total_services": 0,
                "message": "Error processing vendor recommendations"
            })
# # Create tool instances
# search_vendors_tool = SearchVendorsTool()
# budget_calculator_tool = BudgetCalculatorTool()
# vendor_recommendation_tool = VendorRecommendationTool()

# # @tool("Get vendor recommendations with reasoning")
# async def get_vendor_recommendations(
#     requirements_json: str
# ) -> str:
#     """
#     Get personalized vendor recommendations based on event requirements.
    
#     Args:
#         requirements_json: JSON string with event requirements
    
#     Returns:
#         JSON string with recommended vendors and reasoning
#     """
#     try:
#         requirements = json.loads(requirements_json)
#     except:
#         return json.dumps({"error": "Invalid requirements JSON"})
    
#     city = requirements.get("city")
#     services = requirements.get("services", [])
#     budget = requirements.get("budget")
#     date_str = requirements.get("date")
#     event_type = requirements.get("event_type", "general")
    
#     all_recommendations = []
    
#     for service in services:
#         service_budget = None
#         if budget and len(services) > 0:
#             # Simple equal allocation for this tool
#             service_budget = budget // len(services)
        
#         slots = {"city": city, "service_type": service}
#         if date_str: slots["date"] = date_str
#         if service_budget: slots["budget"] = service_budget
        
#         vendors = await fetch_vendors(slots)
        
#         if vendors:
#             # Rank vendors by value (price vs features)
#             best_vendor = min(vendors, key=lambda v: v.price_min)
            
#             recommendation = {
#                 "service": service,
#                 "vendor": {
#                     "id": best_vendor.id,
#                     "name": best_vendor.name,
#                     "price_min": best_vendor.price_min,
#                     "price_max": best_vendor.price_max,
#                     "contact": best_vendor.contact,
#                     "available_date": best_vendor.available_date.isoformat() if best_vendor.available_date else None
#                 },
#                 "reasoning": f"Best value option for {service} in {city}",
#                 "alternatives_count": len(vendors) - 1
#             }
#             all_recommendations.append(recommendation)
    
#     return json.dumps({
#         "recommendations": all_recommendations,
#         "event_type": event_type,
#         "total_services": len(services),
#         "message": f"Found recommendations for {len(all_recommendations)} out of {len(services)} requested services"
#     })

# # === DB Helper Function ===
# async def fetch_vendors(slots: Dict[str, Any]) -> List[Vendor]:
#     stmt = select(Vendor)
#     filters = []
#     from sqlalchemy import func, and_
#     if slots.get("city"):
#         filters.append(func.lower(Vendor.city) == slots["city"].lower())
#     if slots.get("service_type"):
#         filters.append(func.lower(Vendor.service_type) == slots["service_type"].lower())
#     if slots.get("date"):
#         try:
#             y, m, d = map(int, slots["date"].split("-"))
#             filters.append(Vendor.available_date == date(y, m, d))
#         except Exception:
#             pass
#     if filters:
#         stmt = stmt.where(and_(*filters))
#     async with AsyncSessionLocal() as ses:
#         res = await ses.execute(stmt)
#         vendors = res.scalars().all()
    
#     if slots.get("budget"):
#         try:
#             b = int(slots["budget"])
#             vendors = [v for v in vendors if v.price_min <= b or v.price_max <= b]
#         except Exception:
#             pass
#     return vendors

# # === Enhanced CrewAI Agents ===
# crewai_llm = CrewLLM(
#     provider="openai",
#     model=settings.OPENAI_MODEL,
#     api_key=settings.OPENAI_API_KEY
# )

# # Event Planning Specialist
# event_planner = Agent(
#     role="Senior Event Planning Specialist",
#     goal="Extract detailed event requirements and coordinate the planning process",
#     backstory="""You are a senior event planning specialist with 15+ years of experience. 
#     You excel at understanding client needs, asking the right questions, and coordinating 
#     multiple vendors to create memorable events.""",
#     llm=crewai_llm,
#     tools=[search_vendors_tool],
#     verbose=True,
#     allow_delegation=True
# )

# # Budget Planning Expert
# budget_planner = Agent(
#     role="Budget Planning Expert", 
#     goal="Optimize budget allocation and ensure cost-effectiveness",
#     backstory="""You are a financial planning expert specializing in event budgets. 
#     You understand market rates, can spot good deals, and know how to maximize 
#     value within any budget constraint.""",
#     llm=crewai_llm,
#     tools=[budget_calculator_tool],
#     verbose=True
# )

# # Vendor Relations Specialist
# vendor_specialist = Agent(
#     role="Vendor Relations Specialist",
#     goal="Find the best vendors and negotiate optimal arrangements",
#     backstory="""You are a vendor relations specialist with deep knowledge of local markets. 
#     You have relationships with quality vendors and can quickly identify the best options 
#     for any event type and budget.""",
#     llm=crewai_llm,
#     tools=[search_vendors_tool, vendor_recommendation_tool],
#     verbose=True
# )

# # === Enhanced Tasks ===
# requirements_analysis_task = Task(
#     description="""Analyze the user's event planning request: "{user_message}"
    
#     Extract and organize the following information:
#     1. Event type and purpose
#     2. Location/city requirements  
#     3. Date and timing
#     4. Budget constraints
#     5. Required services
#     6. Special requirements or preferences
    
#     If any critical information is missing, identify what needs to be clarified.
#     Use the search_vendors tool to get an initial sense of availability.""",
    
#     agent=event_planner,
#     expected_output="""JSON object with:
#     - event_details: {type, city, date, budget, services[], special_requirements[]}
#     - missing_info: list of questions to ask user
#     - initial_vendor_availability: summary from search_vendors tool"""
# )

# budget_optimization_task = Task(
#     description="""Based on the requirements from the previous task, create an optimal budget plan.
    
#     Use the calculate_budget_allocation tool to:
#     1. Distribute the total budget across required services
#     2. Consider the event type for appropriate allocations
#     3. Account for any priority services mentioned
#     4. Provide budget ranges for flexibility
    
#     Requirements: {requirements_json}""",
    
#     agent=budget_planner,
#     expected_output="""JSON object with:
#     - budget_breakdown: detailed allocation per service
#     - recommendations: cost-saving tips
#     - flexibility_options: alternative budget scenarios"""
# )

# vendor_matching_task = Task(
#     description="""Find and recommend the best vendors based on requirements and budget.
    
#     Using the requirements: {requirements_json}
#     And budget plan: {budget_json}
    
#     Use get_vendor_recommendations tool to:
#     1. Find vendors for each required service
#     2. Match vendors to budget allocations
#     3. Provide alternatives and backup options
#     4. Include contact information and next steps""",
    
#     agent=vendor_specialist,
#     expected_output="""JSON object with:
#     - primary_recommendations: best vendor for each service
#     - alternatives: backup options
#     - contact_plan: suggested order for contacting vendors
#     - next_steps: actionable items for the user"""
# )

# # === FastAPI App ===
# app = FastAPI(title="Enhanced AI Event Planner", version="0.4.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.CORS_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # === Models ===
# class ChatIn(BaseModel):
#     message: str
#     session_id: Optional[str] = None

# class ChatOut(BaseModel):
#     reply: str
#     requirements: Optional[Dict[str, Any]] = None
#     budget_plan: Optional[Dict[str, Any]] = None
#     recommendations: Optional[List[Dict[str, Any]]] = None
#     next_steps: Optional[List[str]] = None

# class VendorOut(BaseModel):
#     id: int
#     name: str
#     service_type: str
#     city: str
#     price_min: int
#     price_max: int
#     available_date: Optional[date] = None
#     contact: Optional[str] = None

# # === Auth & Rate Limiting ===
# async def api_key_auth(x_api_key: Optional[str] = Header(default=None)):
#     if settings.API_KEY:
#         if not x_api_key or x_api_key != settings.API_KEY:
#             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
#     return True

# RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "3"))
# _BUCKETS: Dict[str, Dict[str, float]] = {}

# def _allow(ip: str) -> bool:
#     now = time.time()
#     b = _BUCKETS.get(ip)
#     if not b:
#         _BUCKETS[ip] = {"tokens": RATE_LIMIT_RPS, "ts": now}
#         return True
#     elapsed = now - b["ts"]
#     b["tokens"] = min(RATE_LIMIT_RPS, b["tokens"] + elapsed * RATE_LIMIT_RPS)
#     b["ts"] = now
#     if b["tokens"] >= 1.0:
#         b["tokens"] -= 1.0
#         return True
#     return False

# async def rate_guard(request: Request):
#     ip = request.client.host if request.client else "unknown"
#     if not _allow(ip):
#         raise HTTPException(429, "Rate limit exceeded")

# # === Fallback chat function ===
# async def fallback_chat(message: str) -> ChatOut:
#     """Fallback chat when CrewAI is not available"""
#     message_lower = message.lower()
#     city = None
#     service_type = None
    
#     # Extract basic info
#     for c in ["chennai", "mumbai", "delhi", "bengaluru", "bangalore"]:
#         if c in message_lower:
#             city = "Bengaluru" if c in ["bengaluru", "bangalore"] else c.capitalize()
#             break
    
#     for s in ["food", "camera", "decoration", "cleaning", "makeup"]:
#         if s in message_lower:
#             service_type = s
#             break
    
#     slots = {}
#     if city: slots["city"] = city
#     if service_type: slots["service_type"] = service_type
    
#     vendors = await fetch_vendors(slots)
    
#     if vendors:
#         vendor_list = []
#         for v in vendors[:3]:
#             vendor_list.append(
#                 f"• **{v.name}** ({v.service_type}) - ₹{v.price_min:,}-₹{v.price_max:,}\n"
#                 f"  Contact: {v.contact}"
#             )
        
#         reply = f"I found these vendors for you:\n\n" + "\n\n".join(vendor_list)
#         reply += f"\n\nWould you like help with budget planning or finding vendors for other services?"
        
#         recommendations = [
#             {
#                 "vendor_id": v.id,
#                 "name": v.name,
#                 "service_type": v.service_type,
#                 "price_range": f"₹{v.price_min:,}-₹{v.price_max:,}",
#                 "contact": v.contact
#             }
#             for v in vendors[:3]
#         ]
#     else:
#         reply = """I'd love to help plan your event! To give you the best recommendations, I need some details:

# • **City**: Where is your event?
# • **Date**: When is it happening? (YYYY-MM-DD)
# • **Event Type**: Wedding, birthday, corporate event, etc.
# • **Services Needed**: Food, photography, decoration, makeup, cleaning?
# • **Budget**: What's your approximate budget?

# Once I have these details, I can find the perfect vendors for you!"""
#         recommendations = []
    
#     return ChatOut(
#         reply=reply,
#         recommendations=recommendations,
#         next_steps=["Provide missing event details", "Review vendor recommendations", "Contact preferred vendors"]
#     )

# === Routes ===
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok", 
        "openai_configured": openai_api_key is not None,
        "crewai_available": crewai_llm is not None
    }

@app.get("/readyz")
async def readyz():
    try:
        async with engine.begin() as _:
            pass
        return {
            "status": "ready",
            "database": "connected",
            "openai_api": "configured" if openai_api_key else "missing",
            "crewai_agents": "available" if crewai_llm else "fallback_mode"
        }
    except Exception as e:
        raise HTTPException(500, f"db not ready: {e}")

@app.get("/vendors", response_model=List[VendorOut])
async def list_vendors(
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

@app.post("/chat", response_model=ChatOut, dependencies=[Depends(rate_guard), Depends(api_key_auth)])
async def chat(body: ChatIn):
    # If CrewAI is not available, use fallback
    print(f"New chat request received at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not crewai_llm or not event_planner:
        print("WARNING: CrewAI not available, using fallback mode")
        log.warning("CrewAI not available, using fallback mode")
        return await fallback_chat(body.message)
    
    try:
        # Create the enhanced crew
        print("*****calling CREW AI**********")
        planning_crew = Crew(
            agents=[event_planner, budget_planner, vendor_specialist],
            tasks=[requirements_analysis_task, budget_optimization_task, vendor_matching_task],
            llm=crewai_llm,
            verbose=True,
            process="sequential"  # Tasks run in sequence, building on each other
        )
        
        # Execute the crew
        print("*****BEFORE RESULT FROM CREW AI**********")
        result = planning_crew.kickoff(inputs={
            "user_message": body.message,
            "requirements_json": "{}",  # Will be populated by first task
            "budget_json": "{}"  # Will be populated by second task
        })
        print(result,"<<<=====result")
        # Parse the final result
        try:
            # CrewAI result is typically the output of the last task
            result_text = str(result)
            print("")
            # Try to extract JSON from the result
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                result_json = json.loads(result_text[start_idx:end_idx])
                
                # Structure the response
                recommendations = result_json.get("primary_recommendations", [])
                next_steps = result_json.get("next_steps", [])
                
                # Create a user-friendly reply
                reply_parts = []
                
                if recommendations:
                    reply_parts.append("Here are my recommendations for your event:")
                    for rec in recommendations:
                        vendor = rec.get("vendor", {})
                        service = rec.get("service", "")
                        reply_parts.append(
                            f"\n**{service.title()}**: {vendor.get('name', 'Unknown')} "
                            f"(₹{vendor.get('price_min', 0):,}-₹{vendor.get('price_max', 0):,}) "
                            f"- Contact: {vendor.get('contact', 'N/A')}"
                        )
                
                if next_steps:
                    reply_parts.append(f"\n\n**Next Steps:**")
                    for i, step in enumerate(next_steps, 1):
                        reply_parts.append(f"{i}. {step}")
                
                reply = "\n".join(reply_parts) if reply_parts else "I've analyzed your requirements. Let me know if you need more specific information!"
                
                return ChatOut(
                    reply=reply,
                    recommendations=recommendations,
                    next_steps=next_steps
                )
            
        except json.JSONDecodeError:
            pass
        
        # Fallback: return the raw result as reply
        return ChatOut(
            reply=str(result),
            recommendations=[],
            next_steps=[]
        )
        
    except Exception as e:
        log.error(f"Enhanced CrewAI error: {e}")
        
        # Fall back to simple mode
        return await fallback_chat(body.message)

# # === DB Initialization ===
# async def init_db(seed: bool = True):
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#     if seed:
#         async with AsyncSessionLocal() as ses:
#             existing = (await ses.execute(select(Vendor))).scalars().first()
#             if not existing:
#                 # Add more diverse vendor data
#                 vendors_data = [
#                     # Chennai vendors
#                     Vendor(name="Royal Caterers", service_type="food", city="Chennai",
#                            price_min=15000, price_max=60000, available_date=date(2025, 10, 24), contact="99999 11111"),
#                     Vendor(name="Elite Photography", service_type="camera", city="Chennai",
#                            price_min=12000, price_max=45000, available_date=date(2025, 10, 24), contact="99999 22222"),
#                     Vendor(name="Floral Decors", service_type="decoration", city="Chennai",
#                            price_min=8000, price_max=40000, available_date=date(2025, 10, 24), contact="99999 33333"),
#                     Vendor(name="Sparkle Cleaners", service_type="cleaning", city="Chennai",
#                            price_min=3000, price_max=12000, available_date=date(2025, 10, 24), contact="99999 44444"),
#                     Vendor(name="GlamUp Makeup", service_type="makeup", city="Chennai",
#                            price_min=5000, price_max=25000, available_date=date(2025, 10, 24), contact="99999 55555"),
                    
#                     # Mumbai vendors
#                     Vendor(name="Mumbai Feast", service_type="food", city="Mumbai",
#                            price_min=18000, price_max=75000, available_date=date(2025, 10, 24), contact="98888 11111"),
#                     Vendor(name="Bollywood Shots", service_type="camera", city="Mumbai",
#                            price_min=15000, price_max=55000, available_date=date(2025, 10, 24), contact="98888 22222"),
                    
#                     # Delhi vendors
#                     Vendor(name="Capital Catering", service_type="food", city="Delhi",
#                            price_min=20000, price_max=80000, available_date=date(2025, 10, 24), contact="97777 11111"),
#                     Vendor(name="Delhi Dreams Decor", service_type="decoration", city="Delhi",
#                            price_min=10000, price_max=50000, available_date=date(2025, 10, 24), contact="97777 33333"),
#                 ]
                
#                 ses.add_all(vendors_data)
#                 await ses.commit()

# # === Startup ===
# @app.on_event("startup")
# async def startup():
#     await init_db(seed=settings.SEED_DATA)
#     log.info("Enhanced CrewAI Event Planner service started")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# main_enhanced_crewai_enterprise.py
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
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Date as SQLDate, select

# CrewAI imports with tools
from crewai import Agent, Task, Crew, LLM as CrewLLM
from crewai.tools import BaseTool

# === Settings ===
class Settings(BaseSettings):
    API_KEY: Optional[str] = None
    DATABASE_URL: str = "sqlite+aiosqlite:///./event_planner.db"
    OPENAI_API_KEY: Optional[str] = None  # Will be loaded from .env
    OPENAI_MODEL: str = "gpt-4o-mini"
    SEED_DATA: bool = True
    CORS_ORIGINS: List[str] = ["*"]

    model_config = ConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()

# === Logging ===
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("enhanced-crewai-planner")

# === API Key Validation and Setup ===
def validate_and_setup_openai():
    """Validate and setup OpenAI API key with proper error handling"""
    
    # Try multiple sources for the API key
    api_key = None
    
    # 1. Check settings (from .env file)
    if settings.OPENAI_API_KEY:
        api_key = settings.OPENAI_API_KEY
        log.info("Using OpenAI API key from settings/.env file")
    
    # 2. Check environment variable directly
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
        print("GOT OPEN API KEYYYYYYYYYYYYYYYYYYYYYY")
        log.info("Using OpenAI API key from environment variable")
    
    # 3. Check alternative environment variable names
    elif os.getenv("OPENAI_KEY"):
        api_key = os.getenv("OPENAI_KEY")
        log.info("Using OpenAI API key from OPENAI_KEY environment variable")
    
    if not api_key:
        log.error("No OpenAI API key found! Please set OPENAI_API_KEY in .env file or environment")
        return None
    
    # Validate API key format
    if not api_key.startswith(('sk-', 'sk-proj-')):
        log.error(f"Invalid OpenAI API key format: {api_key[:10]}...")
        return None
    
    # Set the environment variable for CrewAI/LiteLLM to use
    os.environ["OPENAI_API_KEY"] = api_key
    
    log.info(f"OpenAI API key configured successfully (ends with: ...{api_key[-10:]})")
    return api_key

# # Validate API key on startup
# openai_api_key = validate_and_setup_openai()

# # === DB Models ===
# class Base(DeclarativeBase):
#     pass

# class Vendor(Base):
#     __tablename__ = "vendors"
#     id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
#     name: Mapped[str] = mapped_column(String(255))
#     service_type: Mapped[str] = mapped_column(String(100))
#     city: Mapped[str] = mapped_column(String(100))
#     price_min: Mapped[int] = mapped_column(Integer)
#     price_max: Mapped[int] = mapped_column(Integer)
#     available_date: Mapped[Optional[date]] = mapped_column(SQLDate, nullable=True)
#     contact: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

# engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
# AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# # === CrewAI Tools (Enhanced) ===

# class SearchVendorsTool(BaseTool):
#     name: str = "Search Vendors"
#     description: str = "Search for event vendors in database based on location, service type, date, and budget"

#     def _run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
#         """Search for vendors - this will be called synchronously by CrewAI"""
#         import asyncio
#         return asyncio.run(self._async_run(city, service_type, date_str, max_budget))
    
#     async def _async_run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
#         """Async implementation of vendor search"""
#         slots = {}
#         if city: slots["city"] = city
#         if service_type: slots["service_type"] = service_type
#         if date_str: slots["date"] = date_str
#         if max_budget: slots["budget"] = max_budget
        
#         vendors = await fetch_vendors(slots)
        
#         vendor_data = [
#             {
#                 "id": v.id,
#                 "name": v.name,
#                 "service_type": v.service_type,
#                 "city": v.city,
#                 "price_min": v.price_min,
#                 "price_max": v.price_max,
#                 "available_date": v.available_date.isoformat() if v.available_date else None,
#                 "contact": v.contact
#             }
#             for v in vendors
#         ]
        
#         return json.dumps({
#             "vendors": vendor_data,
#             "count": len(vendor_data),
#             "search_criteria": slots
#         })
import asyncio
class SearchVendorsTool(BaseTool):
    name: str = "Search Vendors"
    description: str = "Search for event vendors in database based on location, service type, date, and budget"

    def _run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
        """Sync implementation"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new thread-safe loop for async operations
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(
                        self._async_run(city, service_type, date_str, max_budget)
                    )
                    return result
                finally:
                    new_loop.close()
            else:
                return loop.run_until_complete(
                    self._async_run(city, service_type, date_str, max_budget)
                )
        except Exception as e:
            log.error(f"Search Vendors Tool Error: {str(e)}")
            # Return empty results instead of failing
            return json.dumps({
                "error": str(e),
                "vendors": [],
                "count": 0,
                "search_criteria": {
                    "city": city, 
                    "service_type": service_type,
                    "date": date_str,
                    "max_budget": max_budget
                }
            })

    async def _async_run(self, city: str = None, service_type: str = None, date_str: str = None, max_budget: int = None) -> str:
        """Async implementation"""
        try:
            slots = {}
            if city: slots["city"] = city.lower()  # Case insensitive search
            if service_type: slots["service_type"] = service_type.lower()
            if date_str: slots["date"] = date_str
            if max_budget: slots["budget"] = max_budget

            vendors = await fetch_vendors(slots)
            
            if not vendors:
                return json.dumps({
                    "vendors": [],
                    "count": 0,
                    "message": "No vendors found matching criteria",
                    "search_criteria": slots
                })

            vendor_data = [
                {
                    "id": v.id,
                    "name": v.name,
                    "service_type": v.service_type,
                    "city": v.city,
                    "price_min": v.price_min,
                    "price_max": v.price_max,
                    "available_date": v.available_date.isoformat() if v.available_date else None,
                    "contact": v.contact
                }
                for v in vendors
            ]
            
            return json.dumps({
                "vendors": vendor_data,
                "count": len(vendor_data),
                "search_criteria": slots
            })
            
        except Exception as e:
            log.error(f"Async vendor search error: {str(e)}")
            return json.dumps({
                "error": str(e),
                "vendors": [],
                "count": 0,
                "search_criteria": slots
            })
class BudgetCalculatorTool(BaseTool):
    name: str = "Calculate Budget"
    description: str = "Calculate optimal budget allocation across different event services"

    def _run(self, total_budget: int, services: str, event_type: str = "general", priority_service: str = None) -> str:
        """Calculate budget allocation"""
        try:
            # Parse services if it's a string representation of a list
            if isinstance(services, str):
                if services.startswith('[') and services.endswith(']'):
                    services = json.loads(services)
                else:
                    services = [s.strip() for s in services.split(',')]
        except:
            services = [services] if isinstance(services, str) else []
        
        # Event-specific allocations
        event_allocations = {
            "wedding": {"food": 0.40, "decoration": 0.25, "camera": 0.20, "makeup": 0.10, "cleaning": 0.05},
            "birthday": {"food": 0.35, "decoration": 0.30, "camera": 0.15, "makeup": 0.10, "cleaning": 0.10},
            "corporate": {"food": 0.45, "decoration": 0.20, "camera": 0.20, "makeup": 0.05, "cleaning": 0.10},
            "general": {"food": 0.35, "decoration": 0.25, "camera": 0.15, "makeup": 0.15, "cleaning": 0.10}
        }
        
        allocations = event_allocations.get(event_type.lower(), event_allocations["general"])
        
        # Adjust for priority service
        if priority_service and priority_service in allocations:
            boost = 0.15
            allocations[priority_service] += boost
            
            other_services = [s for s in services if s != priority_service and s in allocations]
            if other_services:
                reduction_per_service = boost / len(other_services)
                for service in other_services:
                    allocations[service] = max(0.05, allocations[service] - reduction_per_service)
        
        budget_plan = []
        total_allocated = 0
        
        for service in services:
            if service in allocations:
                amount = int(total_budget * allocations[service])
                total_allocated += amount
                budget_plan.append({
                    "service": service,
                    "allocated_budget": amount,
                    "percentage": round(allocations[service] * 100, 1),
                    "suggested_range": f"₹{amount - amount//4:,} - ₹{amount + amount//4:,}"
                })
        
        return json.dumps({
            "total_budget": total_budget,
            "allocated_budget": total_allocated,
            "remaining_budget": total_budget - total_allocated,
            "allocations": budget_plan,
            "event_type": event_type
        })


# # Create tool instances
# search_vendors_tool = SearchVendorsTool()
# budget_calculator_tool = BudgetCalculatorTool()
# vendor_recommendation_tool = VendorRecommendationTool()

# # @tool("Get vendor recommendations with reasoning")
# async def get_vendor_recommendations(
#     requirements_json: str
# ) -> str:
#     """
#     Get personalized vendor recommendations based on event requirements.
    
#     Args:
#         requirements_json: JSON string with event requirements
    
#     Returns:
#         JSON string with recommended vendors and reasoning
#     """
#     try:
#         requirements = json.loads(requirements_json)
#     except:
#         return json.dumps({"error": "Invalid requirements JSON"})
    
#     city = requirements.get("city")
#     services = requirements.get("services", [])
#     budget = requirements.get("budget")
#     date_str = requirements.get("date")
#     event_type = requirements.get("event_type", "general")
    
#     all_recommendations = []
    
#     for service in services:
#         service_budget = None
#         if budget and len(services) > 0:
#             # Simple equal allocation for this tool
#             service_budget = budget // len(services)
        
#         slots = {"city": city, "service_type": service}
#         if date_str: slots["date"] = date_str
#         if service_budget: slots["budget"] = service_budget
        
#         vendors = await fetch_vendors(slots)
        
#         if vendors:
#             # Rank vendors by value (price vs features)
#             best_vendor = min(vendors, key=lambda v: v.price_min)
            
#             recommendation = {
#                 "service": service,
#                 "vendor": {
#                     "id": best_vendor.id,
#                     "name": best_vendor.name,
#                     "price_min": best_vendor.price_min,
#                     "price_max": best_vendor.price_max,
#                     "contact": best_vendor.contact,
#                     "available_date": best_vendor.available_date.isoformat() if best_vendor.available_date else None
#                 },
#                 "reasoning": f"Best value option for {service} in {city}",
#                 "alternatives_count": len(vendors) - 1
#             }
#             all_recommendations.append(recommendation)
    
#     return json.dumps({
#         "recommendations": all_recommendations,
#         "event_type": event_type,
#         "total_services": len(services),
#         "message": f"Found recommendations for {len(all_recommendations)} out of {len(services)} requested services"
#     })

# # === DB Helper Function ===
# async def fetch_vendors(slots: Dict[str, Any]) -> List[Vendor]:
#     stmt = select(Vendor)
#     filters = []
#     from sqlalchemy import func, and_
#     if slots.get("city"):
#         filters.append(func.lower(Vendor.city) == slots["city"].lower())
#     if slots.get("service_type"):
#         filters.append(func.lower(Vendor.service_type) == slots["service_type"].lower())
#     if slots.get("date"):
#         try:
#             y, m, d = map(int, slots["date"].split("-"))
#             filters.append(Vendor.available_date == date(y, m, d))
#         except Exception:
#             pass
#     if filters:
#         stmt = stmt.where(and_(*filters))
#     async with AsyncSessionLocal() as ses:
#         res = await ses.execute(stmt)
#         vendors = res.scalars().all()
    
#     if slots.get("budget"):
#         try:
#             b = int(slots["budget"])
#             vendors = [v for v in vendors if v.price_min <= b or v.price_max <= b]
#         except Exception:
#             pass
#     return vendors

# # === Enhanced CrewAI Agents ===
# crewai_llm = CrewLLM(
#     provider="openai",
#     model=settings.OPENAI_MODEL,
#     api_key=settings.OPENAI_API_KEY
# )

# # Event Planning Specialist
# event_planner = Agent(
#     role="Senior Event Planning Specialist",
#     goal="Extract detailed event requirements and coordinate the planning process",
#     backstory="""You are a senior event planning specialist with 15+ years of experience. 
#     You excel at understanding client needs, asking the right questions, and coordinating 
#     multiple vendors to create memorable events.""",
#     llm=crewai_llm,
#     tools=[search_vendors_tool],
#     verbose=True,
#     allow_delegation=True
# )

# # Budget Planning Expert
# budget_planner = Agent(
#     role="Budget Planning Expert", 
#     goal="Optimize budget allocation and ensure cost-effectiveness",
#     backstory="""You are a financial planning expert specializing in event budgets. 
#     You understand market rates, can spot good deals, and know how to maximize 
#     value within any budget constraint.""",
#     llm=crewai_llm,
#     tools=[budget_calculator_tool],
#     verbose=True
# )

# # Vendor Relations Specialist
# vendor_specialist = Agent(
#     role="Vendor Relations Specialist",
#     goal="Find the best vendors and negotiate optimal arrangements",
#     backstory="""You are a vendor relations specialist with deep knowledge of local markets. 
#     You have relationships with quality vendors and can quickly identify the best options 
#     for any event type and budget.""",
#     llm=crewai_llm,
#     tools=[search_vendors_tool, vendor_recommendation_tool],
#     verbose=True
# )

# # === Enhanced Tasks ===
# requirements_analysis_task = Task(
#     description="""Analyze the user's event planning request: "{user_message}"
    
#     Extract and organize the following information:
#     1. Event type and purpose
#     2. Location/city requirements  
#     3. Date and timing
#     4. Budget constraints
#     5. Required services
#     6. Special requirements or preferences
    
#     If any critical information is missing, identify what needs to be clarified.
#     Use the search_vendors tool to get an initial sense of availability.""",
    
#     agent=event_planner,
#     expected_output="""JSON object with:
#     - event_details: {type, city, date, budget, services[], special_requirements[]}
#     - missing_info: list of questions to ask user
#     - initial_vendor_availability: summary from search_vendors tool"""
# )

# budget_optimization_task = Task(
#     description="""Based on the requirements from the previous task, create an optimal budget plan.
    
#     Use the calculate_budget_allocation tool to:
#     1. Distribute the total budget across required services
#     2. Consider the event type for appropriate allocations
#     3. Account for any priority services mentioned
#     4. Provide budget ranges for flexibility
    
#     Requirements: {requirements_json}""",
    
#     agent=budget_planner,
#     expected_output="""JSON object with:
#     - budget_breakdown: detailed allocation per service
#     - recommendations: cost-saving tips
#     - flexibility_options: alternative budget scenarios"""
# )

# vendor_matching_task = Task(
#     description="""Find and recommend the best vendors based on requirements and budget.
    
#     Using the requirements: {requirements_json}
#     And budget plan: {budget_json}
    
#     Use get_vendor_recommendations tool to:
#     1. Find vendors for each required service
#     2. Match vendors to budget allocations
#     3. Provide alternatives and backup options
#     4. Include contact information and next steps""",
    
#     agent=vendor_specialist,
#     expected_output="""JSON object with:
#     - primary_recommendations: best vendor for each service
#     - alternatives: backup options
#     - contact_plan: suggested order for contacting vendors
#     - next_steps: actionable items for the user"""
# )

# # === FastAPI App ===
# app = FastAPI(title="Enhanced AI Event Planner", version="0.4.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.CORS_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # === Models ===
# class ChatIn(BaseModel):
#     message: str
#     session_id: Optional[str] = None

# class ChatOut(BaseModel):
#     reply: str
#     requirements: Optional[Dict[str, Any]] = None
#     budget_plan: Optional[Dict[str, Any]] = None
#     recommendations: Optional[List[Dict[str, Any]]] = None
#     next_steps: Optional[List[str]] = None

# class VendorOut(BaseModel):
#     id: int
#     name: str
#     service_type: str
#     city: str
#     price_min: int
#     price_max: int
#     available_date: Optional[date] = None
#     contact: Optional[str] = None

# # === Auth & Rate Limiting ===
# async def api_key_auth(x_api_key: Optional[str] = Header(default=None)):
#     if settings.API_KEY:
#         if not x_api_key or x_api_key != settings.API_KEY:
#             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
#     return True

# RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "3"))
# _BUCKETS: Dict[str, Dict[str, float]] = {}

# def _allow(ip: str) -> bool:
#     now = time.time()
#     b = _BUCKETS.get(ip)
#     if not b:
#         _BUCKETS[ip] = {"tokens": RATE_LIMIT_RPS, "ts": now}
#         return True
#     elapsed = now - b["ts"]
#     b["tokens"] = min(RATE_LIMIT_RPS, b["tokens"] + elapsed * RATE_LIMIT_RPS)
#     b["ts"] = now
#     if b["tokens"] >= 1.0:
#         b["tokens"] -= 1.0
#         return True
#     return False

# async def rate_guard(request: Request):
#     ip = request.client.host if request.client else "unknown"
#     if not _allow(ip):
#         raise HTTPException(429, "Rate limit exceeded")

# # === Fallback chat function ===
# async def fallback_chat(message: str) -> ChatOut:
#     """Fallback chat when CrewAI is not available"""
#     message_lower = message.lower()
#     city = None
#     service_type = None
    
#     # Extract basic info
#     for c in ["chennai", "mumbai", "delhi", "bengaluru", "bangalore"]:
#         if c in message_lower:
#             city = "Bengaluru" if c in ["bengaluru", "bangalore"] else c.capitalize()
#             break
    
#     for s in ["food", "camera", "decoration", "cleaning", "makeup"]:
#         if s in message_lower:
#             service_type = s
#             break
    
#     slots = {}
#     if city: slots["city"] = city
#     if service_type: slots["service_type"] = service_type
    
#     vendors = await fetch_vendors(slots)
    
#     if vendors:
#         vendor_list = []
#         for v in vendors[:3]:
#             vendor_list.append(
#                 f"• **{v.name}** ({v.service_type}) - ₹{v.price_min:,}-₹{v.price_max:,}\n"
#                 f"  Contact: {v.contact}"
#             )
        
#         reply = f"I found these vendors for you:\n\n" + "\n\n".join(vendor_list)
#         reply += f"\n\nWould you like help with budget planning or finding vendors for other services?"
        
#         recommendations = [
#             {
#                 "vendor_id": v.id,
#                 "name": v.name,
#                 "service_type": v.service_type,
#                 "price_range": f"₹{v.price_min:,}-₹{v.price_max:,}",
#                 "contact": v.contact
#             }
#             for v in vendors[:3]
#         ]
#     else:
#         reply = """I'd love to help plan your event! To give you the best recommendations, I need some details:

# • **City**: Where is your event?
# • **Date**: When is it happening? (YYYY-MM-DD)
# • **Event Type**: Wedding, birthday, corporate event, etc.
# • **Services Needed**: Food, photography, decoration, makeup, cleaning?
# • **Budget**: What's your approximate budget?

# Once I have these details, I can find the perfect vendors for you!"""
#         recommendations = []
    
#     return ChatOut(
#         reply=reply,
#         recommendations=recommendations,
#         next_steps=["Provide missing event details", "Review vendor recommendations", "Contact preferred vendors"]
#     )

# === Routes ===
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok", 
        "openai_configured": openai_api_key is not None,
        "crewai_available": crewai_llm is not None
    }

@app.get("/readyz")
async def readyz():
    try:
        async with engine.begin() as _:
            pass
        return {
            "status": "ready",
            "database": "connected",
            "openai_api": "configured" if openai_api_key else "missing",
            "crewai_agents": "available" if crewai_llm else "fallback_mode"
        }
    except Exception as e:
        raise HTTPException(500, f"db not ready: {e}")

@app.get("/vendors", response_model=List[VendorOut])
async def list_vendors(
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

@app.post("/chat", response_model=ChatOut, dependencies=[Depends(rate_guard), Depends(api_key_auth)])
async def chat(body: ChatIn):
    # If CrewAI is not available, use fallback
    print(f"New chat request received at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not crewai_llm or not event_planner:
        print("WARNING: CrewAI not available, using fallback mode")
        log.warning("CrewAI not available, using fallback mode")
        return await fallback_chat(body.message)
    
    try:
        # Create the enhanced crew
        print("*****calling CREW AI**********")
        planning_crew = Crew(
            agents=[event_planner, budget_planner, vendor_specialist],
            tasks=[requirements_analysis_task, budget_optimization_task, vendor_matching_task],
            llm=crewai_llm,
            verbose=True,
            process="sequential"  # Tasks run in sequence, building on each other
        )
        
        # Execute the crew
        print("*****BEFORE RESULT FROM CREW AI**********")
        result = planning_crew.kickoff(inputs={
            "user_message": body.message,
            "requirements_json": "{}",  # Will be populated by first task
            "budget_json": "{}"  # Will be populated by second task
        })
        print(result,"<<<=====result")
        # Parse the final result
        try:
            # CrewAI result is typically the output of the last task
            result_text = str(result)
            print("")
            # Try to extract JSON from the result
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                result_json = json.loads(result_text[start_idx:end_idx])
                
                # Structure the response
                recommendations = result_json.get("primary_recommendations", [])
                next_steps = result_json.get("next_steps", [])
                
                # Create a user-friendly reply
                reply_parts = []
                
                if recommendations:
                    reply_parts.append("Here are my recommendations for your event:")
                    for rec in recommendations:
                        vendor = rec.get("vendor", {})
                        service = rec.get("service", "")
                        reply_parts.append(
                            f"\n**{service.title()}**: {vendor.get('name', 'Unknown')} "
                            f"(₹{vendor.get('price_min', 0):,}-₹{vendor.get('price_max', 0):,}) "
                            f"- Contact: {vendor.get('contact', 'N/A')}"
                        )
                
                if next_steps:
                    reply_parts.append(f"\n\n**Next Steps:**")
                    for i, step in enumerate(next_steps, 1):
                        reply_parts.append(f"{i}. {step}")
                
                reply = "\n".join(reply_parts) if reply_parts else "I've analyzed your requirements. Let me know if you need more specific information!"
                
                return ChatOut(
                    reply=reply,
                    recommendations=recommendations,
                    next_steps=next_steps
                )
            
        except json.JSONDecodeError:
            pass
        
        # Fallback: return the raw result as reply
        return ChatOut(
            reply=str(result),
            recommendations=[],
            next_steps=[]
        )
        
    except Exception as e:
        log.error(f"Enhanced CrewAI error: {e}")
        
        # Fall back to simple mode
        return await fallback_chat(body.message)

# # === DB Initialization ===
# async def init_db(seed: bool = True):
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#     if seed:
#         async with AsyncSessionLocal() as ses:
#             existing = (await ses.execute(select(Vendor))).scalars().first()
#             if not existing:
#                 # Add more diverse vendor data
#                 vendors_data = [
#                     # Chennai vendors
#                     Vendor(name="Royal Caterers", service_type="food", city="Chennai",
#                            price_min=15000, price_max=60000, available_date=date(2025, 10, 24), contact="99999 11111"),
#                     Vendor(name="Elite Photography", service_type="camera", city="Chennai",
#                            price_min=12000, price_max=45000, available_date=date(2025, 10, 24), contact="99999 22222"),
#                     Vendor(name="Floral Decors", service_type="decoration", city="Chennai",
#                            price_min=8000, price_max=40000, available_date=date(2025, 10, 24), contact="99999 33333"),
#                     Vendor(name="Sparkle Cleaners", service_type="cleaning", city="Chennai",
#                            price_min=3000, price_max=12000, available_date=date(2025, 10, 24), contact="99999 44444"),
#                     Vendor(name="GlamUp Makeup", service_type="makeup", city="Chennai",
#                            price_min=5000, price_max=25000, available_date=date(2025, 10, 24), contact="99999 55555"),
                    
#                     # Mumbai vendors
#                     Vendor(name="Mumbai Feast", service_type="food", city="Mumbai",
#                            price_min=18000, price_max=75000, available_date=date(2025, 10, 24), contact="98888 11111"),
#                     Vendor(name="Bollywood Shots", service_type="camera", city="Mumbai",
#                            price_min=15000, price_max=55000, available_date=date(2025, 10, 24), contact="98888 22222"),
                    
#                     # Delhi vendors
#                     Vendor(name="Capital Catering", service_type="food", city="Delhi",
#                            price_min=20000, price_max=80000, available_date=date(2025, 10, 24), contact="97777 11111"),
#                     Vendor(name="Delhi Dreams Decor", service_type="decoration", city="Delhi",
#                            price_min=10000, price_max=50000, available_date=date(2025, 10, 24), contact="97777 33333"),
#                 ]
                
#                 ses.add_all(vendors_data)
#                 await ses.commit()

# # === Startup ===
# @app.on_event("startup")
# async def startup():
#     await init_db(seed=settings.SEED_DATA)
#     log.info("Enhanced CrewAI Event Planner service started")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# main_enhanced_crewai_enterprise.py
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
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Date as SQLDate, select

# CrewAI imports with tools
from crewai import Agent, Task, Crew, LLM as CrewLLM
from crewai.tools import BaseTool

# === Settings ===
class Settings(BaseSettings):
    API_KEY: Optional[str] = None
    DATABASE_URL: str = "sqlite+aiosqlite:///./event_planner.db"
    OPENAI_API_KEY: Optional[str] = None  # Will be loaded from .env
    OPENAI_MODEL: str = "gpt-4o-mini"
    SEED_DATA: bool = True
    CORS_ORIGINS: List[str] = ["*"]

    model_config = ConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()

# === Logging ===
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("enhanced-crewai