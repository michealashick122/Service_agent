import os
import json
import time
import logging
import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
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

# CrewAI imports
from crewai import Agent, Task, Crew, LLM as CrewLLM, Process
from crewai.tools import BaseTool

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("crewai-event-planner")

# === SETTINGS ===
class Settings(BaseSettings):
    API_KEY: Optional[str] = None
    DATABASE_URL: str = "postgresql+asyncpg://postgres:Admin@localhost:5432/agentspace"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    SEED_DATA: bool = True
    CORS_ORIGINS: List[str] = ["*"]
    RATE_LIMIT_RPS: float = 5.0
    
    ENABLE_VENDOR_CACHE: bool = True
    ENABLE_LLM_CACHE: bool = True
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    LLM_TIMEOUT: int = 30
    PARALLEL_PROCESSING: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

settings = Settings()

# === DATABASE MODELS ===
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

engine = create_async_engine(
    settings.DATABASE_URL, 
    echo=False, 
    future=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=3600,
    pool_pre_ping=True,
)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# === CREWAI TOOLS ===
class VendorSearchTool(BaseTool):
    name: str = "Vendor Search Tool"
    description: str = """Search for vendors in the database based on criteria.
    Input should be a JSON string with fields: city, service_type, date (YYYY-MM-DD), budget.
    Returns list of matching vendors with their details."""
    
    async def _arun(self, query: str) -> str:
        """Async search for vendors"""
        try:
            criteria = json.loads(query)
            vendors = await self._fetch_vendors(criteria)
            
            if not vendors:
                return "No vendors found matching the criteria."
            
            result = []
            for v in vendors:
                result.append({
                    "id": v.id,
                    "name": v.name,
                    "service_type": v.service_type,
                    "city": v.city,
                    "price_range": f"₹{v.price_min:,} - ₹{v.price_max:,}",
                    "price_min": v.price_min,
                    "price_max": v.price_max,
                    "available_date": v.available_date.isoformat() if v.available_date else None,
                    "contact": v.contact
                })
            
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide valid JSON with search criteria."
        except Exception as e:
            log.error(f"Vendor search error: {e}")
            return f"Error searching vendors: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Sync wrapper for async run"""
        return asyncio.run(self._arun(query))
    
    async def _fetch_vendors(self, criteria: Dict[str, Any]) -> List[Vendor]:
        """Fetch vendors from database"""
        async with AsyncSessionLocal() as ses:
            stmt = select(Vendor)
            filters = []
            
            if criteria.get("city"):
                from sqlalchemy import func
                filters.append(func.lower(Vendor.city) == criteria["city"].lower())
            
            if criteria.get("service_type"):
                from sqlalchemy import func
                service_types = criteria["service_type"]
                if isinstance(service_types, str):
                    service_types = [service_types]
                
                from sqlalchemy import or_
                service_filters = [
                    func.lower(Vendor.service_type) == st.lower() 
                    for st in service_types
                ]
                filters.append(or_(*service_filters))
            
            if criteria.get("date"):
                try:
                    y, m, d = map(int, criteria["date"].split("-"))
                    target_date = date(y, m, d)
                    filters.append(Vendor.available_date == target_date)
                except:
                    pass
            
            if criteria.get("budget"):
                budget = int(criteria["budget"])
                filters.append(Vendor.price_min <= budget)
            
            if filters:
                from sqlalchemy import and_
                stmt = stmt.where(and_(*filters))
            
            result = await ses.execute(stmt)
            return list(result.scalars().all())


class BudgetOptimizerTool(BaseTool):
    name: str = "Budget Optimizer Tool"
    description: str = """Finds the best combination of vendors within budget.
    Input should be JSON with: vendors (list), services_needed (list), budget (int).
    Returns optimal vendor combination and cost breakdown."""
    
    def _run(self, query: str) -> str:
        """Sync run"""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """Optimize vendor selection for budget"""
        try:
            data = json.loads(query)
            vendors_data = data.get("vendors", [])
            services = data.get("services_needed", [])
            budget = data.get("budget", 0)
            
            if not vendors_data or not services or not budget:
                return "Error: Missing required fields (vendors, services_needed, budget)"
            
            # Convert vendor dicts to objects
            vendors = []
            for v_data in vendors_data:
                v = type('Vendor', (), v_data)()
                vendors.append(v)
            
            # Find best combination
            selected, total_cost, fits = await self._find_best_combination(
                vendors, services, budget
            )
            
            result = {
                "fits_budget": fits,
                "total_min_cost": total_cost,
                "budget": budget,
                "selected_vendors": [
                    {
                        "name": v.name,
                        "service_type": v.service_type,
                        "price_min": v.price_min,
                        "price_max": v.price_max,
                        "contact": v.contact
                    } for v in selected
                ],
                "budget_utilization": f"{int(total_cost/budget*100)}%" if budget > 0 else "N/A"
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            log.error(f"Budget optimization error: {e}")
            return f"Error optimizing budget: {str(e)}"
    
    async def _find_best_combination(self, vendors: List, services: List[str], 
                                     budget: int) -> Tuple[List, int, bool]:
        """Find optimal vendor combination"""
        from itertools import product
        
        vendors_by_service = {}
        for service in services:
            vendors_by_service[service] = [
                v for v in vendors if v.service_type.lower() == service.lower()
            ]
        
        missing = [s for s in services if not vendors_by_service.get(s)]
        if missing:
            available = [v for v in vendors if v.service_type in services]
            total = sum(v.price_min for v in available)
            return available, total, False
        
        service_lists = [vendors_by_service[s] for s in services]
        all_combos = list(product(*service_lists))
        
        valid_combos = []
        for combo in all_combos:
            min_cost = sum(v.price_min for v in combo)
            if min_cost <= budget:
                valid_combos.append({
                    "vendors": combo,
                    "cost": min_cost,
                    "score": budget - min_cost
                })
        
        if not valid_combos:
            cheapest = min(all_combos, key=lambda c: sum(v.price_min for v in c))
            return list(cheapest), sum(v.price_min for v in cheapest), False
        
        valid_combos.sort(key=lambda x: x["score"], reverse=True)
        best = valid_combos[0]
        
        return list(best["vendors"]), best["cost"], True


class ServiceMapperTool(BaseTool):
    name: str = "Service Type Mapper"
    description: str = """Maps user service requests to database service types.
    Input: user's service request (e.g., 'photographer', 'catering')
    Output: JSON list of matching database service types."""
    
    def _run(self, query: str) -> str:
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """Map service types using AI"""
        try:
            # Get available service types from DB
            async with AsyncSessionLocal() as ses:
                result = await ses.execute(
                    text("SELECT DISTINCT service_type FROM vendors")
                )
                db_types = [row[0] for row in result.fetchall()]
            
            # Simple mapping logic (can be enhanced with LLM)
            query_lower = query.lower().strip()
            
            mapping = {
                "photo": ["camera", "photography"],
                "photographer": ["camera", "photography"],
                "photography": ["camera", "photography"],
                "camera": ["camera", "photography"],
                "cameraman": ["camera", "photography"],
                "catering": ["food"],
                "caterer": ["food"],
                "meals": ["food"],
                "food": ["food"],
                "decor": ["decoration"],
                "decoration": ["decoration"],
                "flowers": ["decoration"],
                "cleaning": ["cleaning"],
                "cleaner": ["cleaning"],
                "makeup": ["makeup"],
                "beauty": ["makeup"]
            }
            
            matched = mapping.get(query_lower, [query_lower])
            
            # Filter to only DB types
            valid = [t for t in matched if t.lower() in [dt.lower() for dt in db_types]]
            
            if not valid:
                valid = [query_lower]
            
            return json.dumps(valid)
            
        except Exception as e:
            return json.dumps([query.lower()])


# === CREWAI AGENTS & CREW ===
class EventPlannerCrew:
    """Main CrewAI orchestration for event planning"""
    
    def __init__(self):
        self.llm = CrewLLM(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY
        ) if settings.OPENAI_API_KEY else None
        
        # Initialize tools
        self.vendor_search_tool = VendorSearchTool()
        self.budget_optimizer_tool = BudgetOptimizerTool()
        self.service_mapper_tool = ServiceMapperTool()
        
        # Create agents
        self._create_agents()
    
    def _create_agents(self):
        """Create specialized agents"""
        
        # 1. Intent Understanding Agent
        self.intent_agent = Agent(
            role="Customer Intent Analyst",
            goal="Understand customer requirements and extract structured information",
            backstory="""You are an expert at understanding customer needs for event planning.
            You analyze messages to extract: city, date, service types, budget, and event details.
            You normalize city names (bangalore->Bangalore, chennai->Chennai) and dates (YYYY-MM-DD format).""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 2. Service Mapping Agent
        self.mapping_agent = Agent(
            role="Service Type Specialist",
            goal="Map user service requests to available database service categories",
            backstory="""You understand all service types and their synonyms.
            You use the Service Mapper tool to find matching database service types.
            You know that 'photography' and 'camera' are related, 'catering' means 'food', etc.""",
            tools=[self.service_mapper_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 3. Vendor Search Agent
        self.search_agent = Agent(
            role="Vendor Search Specialist",
            goal="Find the best matching vendors from the database",
            backstory="""You are an expert at searching and filtering vendors.
            You use the Vendor Search Tool to find vendors matching customer criteria.
            You consider location, service type, availability, and budget constraints.""",
            tools=[self.vendor_search_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 4. Budget Optimization Agent
        self.budget_agent = Agent(
            role="Budget Optimization Expert",
            goal="Find the most cost-effective vendor combinations within budget",
            backstory="""You are a financial expert who optimizes vendor selection.
            You use the Budget Optimizer Tool to find the best vendor combinations.
            You maximize value while staying within budget constraints.""",
            tools=[self.budget_optimizer_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 5. Response Formatter Agent
        self.formatter_agent = Agent(
            role="Customer Communication Specialist",
            goal="Create clear, helpful responses for customers",
            backstory="""You are an expert at customer communication.
            You create friendly, informative responses that include vendor details,
            pricing, contact information, and helpful recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    async def process_query(self, user_message: str) -> Dict[str, Any]:
        """Process user query through CrewAI agents"""
        
        # Task 1: Extract Intent
        intent_task = Task(
            description=f"""Analyze this customer message and extract structured information:
            
            Message: "{user_message}"
            
            Extract:
            - intent: (QUERY_VENDORS, VENDOR_INFO, PLAN_EVENT, GENERAL_Q, CLARIFY)
            - city: normalized city name (Chennai, Mumbai, Delhi, Bangalore)
            - date: in YYYY-MM-DD format if mentioned
            - service_type: what services they need (can be multiple)
            - budget: numeric budget if mentioned
            - vendor_name: if asking about specific vendor
            - event_type: type of event if mentioned
            
            Return ONLY valid JSON with these fields.""",
            agent=self.intent_agent,
            expected_output="JSON object with extracted intent and slots"
        )
        
        # Task 2: Map Services
        service_mapping_task = Task(
            description="""Using the extracted service_type from previous task,
            use the Service Type Mapper tool to find matching database service types.
            
            Handle multiple services if needed (comma-separated).
            
            Return JSON list of mapped service types.""",
            agent=self.mapping_agent,
            expected_output="JSON array of database service types",
            context=[intent_task]
        )
        
        # Task 3: Search Vendors
        vendor_search_task = Task(
            description="""Using the extracted criteria and mapped service types,
            use the Vendor Search Tool to find matching vendors.
            
            Build a JSON search query with: city, service_type (from mapping), date, budget.
            
            Return the full vendor list from the tool.""",
            agent=self.search_agent,
            expected_output="JSON array of matching vendors",
            context=[intent_task, service_mapping_task]
        )
        
        # Task 4: Optimize Budget (conditional)
        budget_optimization_task = Task(
            description="""If multiple services are needed AND a budget is specified,
            use the Budget Optimizer Tool to find the best vendor combination.
            
            Input: vendors list, services needed, budget
            
            Return optimized selection and cost breakdown.""",
            agent=self.budget_agent,
            expected_output="JSON with optimized vendor combination and costs",
            context=[intent_task, service_mapping_task, vendor_search_task]
        )
        
        # Task 5: Format Response
        response_task = Task(
            description="""Create a friendly, informative response for the customer.
            
            Include:
            - Clear summary of what was found
            - Vendor details (name, service, price range, contact)
            - Budget analysis if applicable
            - Helpful next steps or recommendations
            
            Use proper formatting with bullet points and clear sections.
            
            Return JSON with: {
                "reply": "formatted response text",
                "recommendations": [list of vendor objects]
            }""",
            agent=self.formatter_agent,
            expected_output="JSON with formatted reply and recommendations",
            context=[intent_task, vendor_search_task, budget_optimization_task]
        )
        
        # Create and run crew
        crew = Crew(
            agents=[
                self.intent_agent,
                self.mapping_agent,
                self.search_agent,
                self.budget_agent,
                self.formatter_agent
            ],
            tasks=[
                intent_task,
                service_mapping_task,
                vendor_search_task,
                budget_optimization_task,
                response_task
            ],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute crew
        try:
            result = await asyncio.to_thread(crew.kickoff)
            
            # Parse final result
            final_output = str(result)
            
            # Try to extract JSON from result
            try:
                if "{" in final_output:
                    start = final_output.find("{")
                    end = final_output.rfind("}") + 1
                    json_str = final_output[start:end]
                    parsed = json.loads(json_str)
                    return parsed
                else:
                    return {
                        "reply": final_output,
                        "recommendations": []
                    }
            except:
                return {
                    "reply": final_output,
                    "recommendations": []
                }
                
        except Exception as e:
            log.error(f"CrewAI execution error: {e}")
            return {
                "reply": f"I encountered an error processing your request: {str(e)}",
                "recommendations": []
            }


# Initialize crew
event_planner_crew = EventPlannerCrew()

# === FASTAPI APP ===
app = FastAPI(title="CrewAI Event Planner", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === SCHEMAS ===
class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatOut(BaseModel):
    reply: str
    recommendations: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    crew_result: Optional[str] = None

# === ENDPOINTS ===
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "orchestration": "CrewAI",
        "agents": 5,
        "tools": 3
    }

@app.post("/chat", response_model=ChatOut)
async def chat_with_crew(body: ChatIn):
    """Main chat endpoint orchestrated by CrewAI"""
    start_time = time.time()
    
    try:
        # Process through CrewAI
        result = await event_planner_crew.process_query(body.message)
        
        processing_time = time.time() - start_time
        
        return ChatOut(
            reply=result.get("reply", "I couldn't process that request."),
            recommendations=result.get("recommendations", []),
            processing_time=round(processing_time, 3),
            crew_result=str(result)
        )
        
    except Exception as e:
        log.error(f"Chat error: {e}", exc_info=True)
        processing_time = time.time() - start_time
        
        return ChatOut(
            reply=f"I encountered an error: {str(e)}",
            recommendations=[],
            processing_time=round(processing_time, 3)
        )

@app.on_event("startup")
async def startup():
    """Initialize database"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    log.info("🚀 CrewAI Event Planner started with full agent orchestration")

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()
    log.info("🛑 Service shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)