"""
Query Rocky Real Estate properties - Ask questions and get answers
"""
import os
import json
import re
import requests
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from typing import Optional, Dict, List

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "https://8445b063-3aeb-4395-84e5-6ae2c6a662c2.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.w4n53UJV1pGUEWGAeTsEwsgUyTfqFHQ6eUlykapMhU0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

COLLECTION_NAME = "rocky_properties"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

def parse_query_filters(query: str) -> Dict:
    """Parse query to extract structured filters"""
    filters = {
        'bedrooms': None,
        'bathrooms': None,
        'max_monthly_rent': None,
        'min_monthly_rent': None,
        'max_yearly_rent': None,
        'min_yearly_rent': None,
        'furnished': None,
        'parking': None,
        'pet_friendly': None,
        'amenities': [],
        'city': None,
        'area': None
    }
    
    query_lower = query.lower()
    
    # Extract bedrooms (e.g., "2 bedroom", "2BR", "two bedroom")
    bed_patterns = [
        r'(\d+)\s*(?:bedroom|bed|br|bedrooms)',
        r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bedroom|bed)',
        r'studio',
    ]
    for pattern in bed_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if match.group(1).lower() == 'studio':
                filters['bedrooms'] = 0
            elif match.group(1).isdigit():
                filters['bedrooms'] = int(match.group(1))
            else:
                word_to_num = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                word = match.group(1).lower()
                if word in word_to_num:
                    filters['bedrooms'] = word_to_num[word]
            break
    
    # Extract bathrooms
    bath_match = re.search(r'(\d+)\s*(?:bathroom|bath|bathrooms)', query_lower)
    if bath_match:
        filters['bathrooms'] = int(bath_match.group(1))
    
    # Extract price filters
    # "under 10k monthly", "less than 10000", "below 10k"
    price_patterns = [
        r'(?:under|below|less than|max|maximum|up to)\s*(?:aed\s*)?(\d+[km]?)\s*(?:monthly|per month|/month)',
        r'(\d+[km]?)\s*(?:monthly|per month|/month)\s*(?:or less|maximum|max)',
    ]
    for pattern in price_patterns:
        match = re.search(pattern, query_lower)
        if match:
            price_str = match.group(1).lower()
            if 'k' in price_str:
                filters['max_monthly_rent'] = float(price_str.replace('k', '')) * 1000
            else:
                filters['max_monthly_rent'] = float(price_str)
            break
    
    # "over 5k monthly", "more than 5000"
    min_price_match = re.search(r'(?:over|above|more than|min|minimum|at least)\s*(?:aed\s*)?(\d+[km]?)\s*(?:monthly|per month|/month)', query_lower)
    if min_price_match:
        price_str = min_price_match.group(1).lower()
        if 'k' in price_str:
            filters['min_monthly_rent'] = float(price_str.replace('k', '')) * 1000
        else:
            filters['min_monthly_rent'] = float(price_str)
    
    # Yearly rent
    yearly_match = re.search(r'(?:under|below|less than|max)\s*(?:aed\s*)?(\d+[km]?)\s*(?:yearly|per year|/year)', query_lower)
    if yearly_match:
        price_str = yearly_match.group(1).lower()
        if 'k' in price_str:
            filters['max_yearly_rent'] = float(price_str.replace('k', '')) * 1000
        else:
            filters['max_yearly_rent'] = float(price_str)
    
    # Furnished status
    if 'furnished' in query_lower:
        if 'unfurnished' in query_lower or 'not furnished' in query_lower:
            filters['furnished'] = False
        elif 'semi-furnished' in query_lower or 'semi furnished' in query_lower:
            filters['furnished'] = 'Semi-furnished'
        else:
            filters['furnished'] = True
    
    # Parking
    if 'parking' in query_lower:
        filters['parking'] = True
    
    # Pet friendly
    if 'pet' in query_lower and ('friendly' in query_lower or 'allowed' in query_lower):
        filters['pet_friendly'] = True
    
    # Amenities
    if 'gym' in query_lower or 'fitness' in query_lower:
        filters['amenities'].append('gym')
    if 'pool' in query_lower or 'swimming' in query_lower:
        filters['amenities'].append('pool')
    if 'security' in query_lower or '24/7' in query_lower:
        filters['amenities'].append('security')
    if 'balcony' in query_lower:
        filters['amenities'].append('balcony')
    if 'elevator' in query_lower or 'lift' in query_lower:
        filters['amenities'].append('elevator')
    
    # Location filters
    if 'dubai' in query_lower:
        filters['city'] = 'Dubai'
    
    # Common areas
    area_keywords = {
        'business bay': 'Business Bay',
        'dubai marina': 'Dubai Marina',
        'downtown': 'Downtown Dubai',
        'jvc': 'Jumeirah Village Circle (JVC)',
        'jlt': 'Jumeirah Lake Towers (JLT)',
    }
    for keyword, area in area_keywords.items():
        if keyword in query_lower:
            filters['area'] = area
            break
    
    return filters

def build_qdrant_filter(filters: Dict) -> Optional[models.Filter]:
    """
    Build Qdrant filter from parsed filters.
    
    Strategy: Only apply filters for "exact" criteria that have proper indexes:
    - Bedrooms, bathrooms (integer matches)
    - Monthly/yearly rent (range filters)
    - Furnished, parking, pet_friendly (boolean matches)
    
    Semantic criteria (amenities, area, city) are NOT filtered here because:
    - They require fuzzy/proximity matching which semantic search handles better
    - Filtering them would require exact matches and could miss relevant results
    - Semantic search can understand context (e.g., "near Business Bay" vs exact match)
    
    Args:
        filters: Dict with filter values (bedrooms, bathrooms, max_monthly_rent, etc.)
    
    Returns:
        Qdrant Filter object or None if no filters to apply
    """
    conditions = []
    
    # Bedrooms (exact integer match)
    if filters.get('bedrooms') is not None:
        conditions.append(
            models.FieldCondition(
                key="bedrooms",
                match=models.MatchValue(value=filters['bedrooms'])
            )
        )
    
    # Bathrooms (exact integer match)
    if filters.get('bathrooms') is not None:
        conditions.append(
            models.FieldCondition(
                key="bathrooms",
                match=models.MatchValue(value=filters['bathrooms'])
            )
        )
    
    # Monthly rent filters (exact range matches)
    rent_conditions = []
    if filters.get('max_monthly_rent') is not None:
        rent_conditions.append(
            models.FieldCondition(
                key="monthly_rent",
                range=models.Range(lte=filters['max_monthly_rent'])
            )
        )
    if filters.get('min_monthly_rent') is not None:
        rent_conditions.append(
            models.FieldCondition(
                key="monthly_rent",
                range=models.Range(gte=filters['min_monthly_rent'])
            )
        )
    
    # Yearly rent filters (exact range matches)
    if filters.get('max_yearly_rent') is not None:
        rent_conditions.append(
            models.FieldCondition(
                key="yearly_rent",
                range=models.Range(lte=filters['max_yearly_rent'])
            )
        )
    if filters.get('min_yearly_rent') is not None:
        rent_conditions.append(
            models.FieldCondition(
                key="yearly_rent",
                range=models.Range(gte=filters['min_yearly_rent'])
            )
        )
    
    if rent_conditions:
        conditions.extend(rent_conditions)
    
    # Furnished (exact boolean match)
    if filters.get('furnished') is not None:
        conditions.append(
            models.FieldCondition(
                key="furnished",
                match=models.MatchValue(value=filters['furnished'])
            )
        )
    
    # Parking (exact boolean match)
    if filters.get('parking') is not None:
        conditions.append(
            models.FieldCondition(
                key="parking",
                match=models.MatchValue(value=filters['parking'])
            )
        )
    
    # Pet friendly (exact boolean match)
    if filters.get('pet_friendly') is not None:
        conditions.append(
            models.FieldCondition(
                key="pet_friendly",
                match=models.MatchValue(value=filters['pet_friendly'])
            )
        )
    
    # Note: amenities, area, city are intentionally NOT filtered here
    # They are handled by semantic search for better fuzzy/proximity matching
    
    if not conditions:
        return None
    
    return models.Filter(must=conditions)

def search_properties(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.3,
    filters: Optional[Dict] = None
) -> tuple:
    """
    Search properties in Qdrant with hybrid search (semantic + filters).
    
    This function accepts explicit parameters - no hardcoded logic.
    The caller (LLM via tools) decides on score_threshold and filters.
    
    Args:
        query: Search query text
        limit: Maximum number of results
        score_threshold: Minimum similarity score (0.0 to 1.0)
            - Lower (0.25-0.3) for exploratory queries
            - Higher (0.4-0.5) for specific queries
        filters: Optional dict with filter values (bedrooms, bathrooms, max_monthly_rent, etc.)
            If None, will parse from query using parse_query_filters()
    
    Returns:
        Tuple of (results.points, filters_dict)
    """
    # Parse filters from query if not provided
    if filters is None:
        filters = parse_query_filters(query)
    
    # Build Qdrant filter (only for exact criteria)
    query_filter = build_qdrant_filter(filters)
    
    # Get embedding for query using requests (works with any OpenAI version)
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": EMBEDDING_MODEL,
        "input": query
    }
    
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        data=json.dumps(data)
    )
    
    if not response.ok:
        raise Exception(f"Error getting embedding: {response.status_code} - {response.text}")
    
    result = response.json()
    query_vector = result['data'][0]['embedding']
    
    # Search Qdrant with hybrid search
    search_params = {
        "collection_name": COLLECTION_NAME,
        "query": query_vector,
        "limit": limit,
    }
    
    # Apply score threshold (caller decides the value)
    search_params["score_threshold"] = score_threshold
    
    # Apply filter if exists
    if query_filter:
        search_params["query_filter"] = query_filter
    
    results = qdrant_client.query_points(**search_params)
    
    return results.points, filters

def format_properties_for_context(properties):
    """Format properties for LLM context"""
    context_parts = []
    for i, prop in enumerate(properties, 1):
        payload = prop.payload
        score = prop.score if hasattr(prop, 'score') else 'N/A'
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        context = f"""
Property {i} (Similarity Score: {score_str}):
- Location: {payload.get('location', 'N/A')}
- Bedrooms: {payload.get('bedrooms', 'N/A')}
- Bathrooms: {payload.get('bathrooms', 'N/A')}
- Monthly Rent: AED {payload.get('monthly_rent', 0):.0f}
- Yearly Rent: AED {payload.get('yearly_rent', 0):.0f}
- Square Feet: {payload.get('sqft', 'N/A')}
- Furnished: {payload.get('furnished', 'N/A')}
- Parking: {payload.get('parking', 'N/A')}
- Amenities: {payload.get('amenities', [])}
- URL: {payload.get('url', 'N/A')}
- Description: {payload.get('description', 'N/A')[:200]}...
"""
        context_parts.append(context)
    
    return "\n".join(context_parts)

def query_properties(question: str):
    """Answer a question about properties using RAG"""
    print(f"\n🔍 Searching for: '{question}'...")
    
    # Parse filters to determine if we have strict criteria
    filters = parse_query_filters(question)
    has_strict_filters = (
        filters.get('bedrooms') is not None or 
        filters.get('bathrooms') is not None or
        filters.get('max_monthly_rent') is not None or
        filters.get('min_monthly_rent') is not None
    )
    
    # Decide on threshold based on query type
    # This is backward-compatible logic for the standalone query_properties function
    # In layla_agent.py, the LLM decides the threshold via tools
    if has_strict_filters:
        score_threshold = 0.3  # Filters ensure relevance
    else:
        score_threshold = 0.35  # Pure semantic query needs flexibility
    
    # Search Qdrant with hybrid search
    properties, filters = search_properties(question, limit=5, score_threshold=score_threshold, filters=filters)
    
    # Display extracted filters
    active_filters = {k: v for k, v in filters.items() if v is not None and v != []}
    if active_filters:
        print(f"\n🔧 Applied Filters:")
        for key, value in active_filters.items():
            print(f"  - {key}: {value}")
    
    if not properties:
        return "I couldn't find any properties matching your query. Please try rephrasing your question."
    
    print(f"✓ Found {len(properties)} relevant properties")
    
    # Display scores
    print("\n📊 Similarity Scores:")
    for i, prop in enumerate(properties, 1):
        score = prop.score if hasattr(prop, 'score') else 'N/A'
        location = prop.payload.get('location', 'N/A')[:50]
        if isinstance(score, (int, float)):
            print(f"  {i}. Score: {score:.4f} - {location}...")
        else:
            print(f"  {i}. Score: {score} - {location}...")
    
    # Format context
    context = format_properties_for_context(properties)
    
    # Create prompt for LLM
    prompt = f"""You are Layla, a helpful property leasing assistant for Rocky Real Estate in Dubai.

Answer the user's question about properties using the information provided below. Be friendly, concise, and helpful.

User Question: {question}

Available Properties:
{context}

Instructions:
- Answer the question directly and naturally
- Mention specific property details when relevant
- Include rent prices and key features
- If multiple properties match, mention the best options
- Be conversational and helpful
- If you can't answer from the provided properties, say so

Answer:"""
    
    # Get answer from OpenAI
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        data=json.dumps(data)
    )
    
    if response.ok:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def interactive_query():
    """Interactive query interface"""
    print("="*70)
    print("🏠 ROCKY REAL ESTATE - PROPERTY QUERY SYSTEM")
    print("="*70)
    print("\nAsk me questions about properties! (Type 'quit' to exit)")
    print("\nExample questions:")
    print("  - What units do you have near Business Bay?")
    print("  - Show me 2 bedroom apartments under 10k monthly")
    print("  - Do you have properties with gym and pool?")
    print("  - What's the rent for 3 bedroom apartments?")
    print()
    
    while True:
        question = input("\n💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        answer = query_properties(question)
        print(f"\n🤖 Layla: {answer}\n")
        print("-"*70)

if __name__ == "__main__":
    # Test with a single question
    import sys
    
    if len(sys.argv) > 1:
        # Command line query
        question = " ".join(sys.argv[1:])
        answer = query_properties(question)
        print(f"\n🤖 Answer: {answer}\n")
    else:
        # Interactive mode
        interactive_query()