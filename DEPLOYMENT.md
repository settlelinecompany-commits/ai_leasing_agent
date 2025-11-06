# Layla API Deployment Guide

## Files Created

✅ `api/index.py` - FastAPI wrapper with API key authentication  
✅ `requirements.txt` - API dependencies (FastAPI, uvicorn, etc.)  
✅ `vercel.json` - Vercel configuration  

## Setup Steps

### 1. Add API Key to `.env`

Add this line to your `.env` file:
```
LAYLA_API_KEY=your-secret-api-key-here-change-this
```

**Important:** Generate a strong, random API key. You can use:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Deploy to Vercel

#### Option A: Using Vercel CLI
```bash
# Install Vercel CLI if you haven't
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel deploy
```

#### Option B: Using Vercel Dashboard
1. Go to [vercel.com](https://vercel.com)
2. Import your Git repository
3. Vercel will auto-detect the Python project
4. Add environment variables:
   - `LAYLA_API_KEY` - Your API key
   - `OPENAI_API_KEY` - Your OpenAI key (already in .env)
   - `QDRANT_URL` - Your Qdrant URL (already in .env)
   - `QDRANT_API_KEY` - Your Qdrant API key (already in .env)

### 3. Test the API

Once deployed, test with:

```bash
curl -X POST https://your-app.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-here" \
  -d '{
    "message": "Hi, I'm looking for a 2 bedroom apartment",
    "state": null
  }'
```

Or with Python:
```python
import requests

response = requests.post(
    "https://your-app.vercel.app/api/chat",
    headers={"X-API-Key": "your-secret-api-key-here"},
    json={
        "message": "Hi, I'm looking for a 2 bedroom apartment",
        "state": None
    }
)

print(response.json())
```

## API Endpoints

### `GET /` or `GET /health`
Health check (no auth required)
```bash
curl https://your-app.vercel.app/health
```

### `POST /api/chat`
Chat with Layla (requires API key)

**Headers:**
- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Body:**
```json
{
  "message": "User's message here",
  "state": null  // Optional: previous conversation state
}
```

**Response:**
```json
{
  "response": "Layla's response",
  "state": {
    "messages": [...],
    "lead_info": {...},
    "tour_details": {...},
    ...
  }
}
```

## Notes

- The API wrapper uses your existing `layla_agent.py` - no modifications needed
- All your agent's dependencies will be installed automatically by Vercel
- The `requirements.txt` only includes FastAPI dependencies - your agent's deps are already working
- Make sure to set all environment variables in Vercel dashboard

## Troubleshooting

If you get import errors, make sure:
1. All your agent files (`layla_agent.py`, `layla_search.py`, etc.) are in the root directory
2. All environment variables are set in Vercel
3. Python version is compatible (Vercel supports 3.11 and 3.12)


