from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Import your existing agent (don't modify it)
from layla_agent import run_layla

load_dotenv()

app = FastAPI(title="Layla API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key from environment
API_KEY = os.getenv("LAYLA_API_KEY", "your-secret-api-key-here")

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    state: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    state: dict

# API Key dependency
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# Health check (no auth required)
@app.get("/")
async def root():
    return {"status": "ok", "service": "Layla API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Chat endpoint (requires API key)
@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Chat with Layla agent
    
    Headers:
        X-API-Key: Your API key
    
    Body:
        {
            "message": "Hi, I'm looking for a 2 bedroom apartment",
            "state": null
        }
    """
    try:
        # Use your existing agent function - no modifications needed
        state = run_layla(request.message, request.state)
        
        # Get last message content
        last_message = state["messages"][-1]
        response_text = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        return ChatResponse(
            response=response_text,
            state=state
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


