import os
import requests
import json

# IMPORTANT: Replace this with the exact URL from your Qdrant Cloud dashboard
# To find your URL:
# 1. Log in to https://cloud.qdrant.io
# 2. Select your cluster
# 3. Copy the URL from the "Connection" or "API" section
# EXACT URL from Qdrant Cloud dashboard (adding port :6333 for API access)
# Dashboard shows: https://d3bb8826-2c10-4f02-b416-1a875e1e41b6.eu-central-1-0.aws.cloud.qdrant.io
# But API requires port :6333
QDRANT_URL = 'https://8445b063-3aeb-4395-84e5-6ae2c6a662c2.europe-west3-0.gcp.cloud.qdrant.io'
QDRANT_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.w4n53UJV1pGUEWGAeTsEwsgUyTfqFHQ6eUlykapMhU0'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

from qdrant_client import QdrantClient, models

# Initialize client with compatibility check disabled to suppress warning
client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

# Test connection first
print("Testing Qdrant connection...")
print(f"  Attempting to connect to: {QDRANT_URL}")
try:
    collections = client.get_collections()
    print(f"✓ Connected to Qdrant! Found {len(collections.collections)} collection(s)")
except Exception as e:
    error_str = str(e)
    print(f"\n✗ Connection failed!")
    print(f"  Error: {error_str}")
    print(f"\n  Troubleshooting:")
    print(f"  1. Click on 'Test_Cluster_v1' in your Qdrant Cloud dashboard")
    print(f"  2. Go to the 'Connection' or 'API' section")
    print(f"  3. Copy the EXACT URL shown there")
    print(f"  4. Replace QDRANT_URL in this script with that exact URL")
    print(f"  5. Make sure your cluster status is 'RUNNING' (not 'UNKNOWN')")
    print(f"\n  Current URL being used: {QDRANT_URL}")
    print(f"  If this doesn't match your dashboard, update it!")
    raise

collection_name = "knowledge_base"
model_name = "BAAI/bge-small-en-v1.5"

# Try to create collection, or use existing one
print(f"\nCreating collection '{collection_name}'...")
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    print(f"✓ Collection '{collection_name}' created successfully!")
except Exception as e:
    error_msg = str(e).lower()
    if "already exists" in error_msg or "duplicate" in error_msg or "409" in error_msg:
        print(f"✓ Collection '{collection_name}' already exists!")
    else:
        print(f"✗ Error creating collection: {e}")
        raise

# Check if collection has documents
collection_info = client.get_collection(collection_name)
print(f"\nCollection '{collection_name}' has {collection_info.points_count} documents stored")

# ============================================
# TEST SEMANTIC RETRIEVAL AND SIMILARITY SCORES
# ============================================
print("\n" + "="*70)
print("TESTING SEMANTIC RETRIEVAL - Understanding Similarity Scores")
print("="*70)

def test_semantic_search(query_text, description=""):
    """Test semantic search and show detailed similarity scores"""
    print(f"\n{'─'*70}")
    print(f"Query: {query_text}")
    if description:
        print(f"Purpose: {description}")
    print(f"{'─'*70}")
    
    # Perform semantic search
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=query_text, model=model_name),
        limit=5,  # Get top 5 to see more results
    )
    
    print(f"\nFound {len(results.points)} results:")
    print(f"\n{'Rank':<6} {'Score':<8} {'Similarity':<12} {'Document Preview'}")
    print(f"{'-'*6} {'-'*8} {'-'*12} {'-'*45}")
    
    for i, point in enumerate(results.points, 1):
        # Score interpretation
        score = point.score
        if score >= 0.8:
            similarity = "Very High"
        elif score >= 0.6:
            similarity = "High"
        elif score >= 0.4:
            similarity = "Medium"
        elif score >= 0.2:
            similarity = "Low"
        else:
            similarity = "Very Low"
        
        doc_preview = point.payload['document'][:45] + "..."
        print(f"{i:<6} {score:<8.4f} {similarity:<12} {doc_preview}")
    
    # Show top match details
    if results.points:
        top_match = results.points[0]
        print(f"\n📊 Top Match Details:")
        print(f"   Score: {top_match.score:.4f}")
        print(f"   Document ID: {top_match.id}")
        print(f"   Full Document: {top_match.payload['document']}")
    
    return results

# Test different queries to understand semantic similarity
test_queries = [
    (
        "What tools should I need to use to build a web service using vector embeddings for search?",
        "Original query - should match Qdrant, FastAPI, Sentence Transformers"
    ),
    (
        "vector database",
        "Simple query - should strongly match Qdrant document"
    ),
    (
        "Python web framework",
        "Should match FastAPI document"
    ),
    (
        "machine learning framework",
        "Should match PyTorch document"
    ),
    (
        "database management system",
        "Should match MySQL document"
    ),
    (
        "text embeddings",
        "Should match Sentence Transformers document"
    ),
    (
        "Where is the nearest grocery store?",
        "Unrelated query - should have low scores or no matches"
    ),
]

# Run tests
for query, description in test_queries:
    test_semantic_search(query, description)

print("\n" + "="*70)
print("UNDERSTANDING SIMILARITY SCORES:")
print("="*70)
print("""
Similarity Score Range (Cosine Similarity):
  • 0.90 - 1.00: Very similar (almost identical meaning)
  • 0.70 - 0.89: High similarity (related concepts)
  • 0.50 - 0.69: Medium similarity (somewhat related)
  • 0.30 - 0.49: Low similarity (loosely related)
  • 0.00 - 0.29: Very low similarity (unrelated)

How it works:
  1. Your query text is converted to a 384-dimensional vector
  2. Each document in the database is also a 384-dimensional vector
  3. Cosine similarity measures the angle between vectors
  4. Higher score = smaller angle = more similar meaning
  5. Score of 1.0 = identical direction (same meaning)
  6. Score of 0.0 = perpendicular (no relationship)
""")
print("="*70)

# Original query for RAG pipeline
prompt = """
What tools should I need to use to build a web service using vector embeddings for search?
"""
print(f"\n{'='*70}")
print("RAG PIPELINE - Using semantic search results")
print(f"{'='*70}")

results = client.query_points(
    collection_name=collection_name,
    query=models.Document(text=prompt, model=model_name),
    limit=3,
)

print(f"\nTop {len(results.points)} results for RAG:")
for i, point in enumerate(results.points, 1):
    print(f"  {i}. Score: {point.score:.4f} - {point.payload['document'][:60]}...")

context = "\n".join(r.payload['document'] for r in results.points)

meta_prompt = f"""
   You are a software architect. 
   Answer the following question using the provided context. 
   If you can't find the answer, do not pretend you know it, but answer "I don't know".
   
   Question: {prompt.strip()}
   
   Context: 
   {context.strip()}
   
   Answer:
   """

API_KEY = OPENAI_API_KEY  # Remove the quotes - use the variable, not the string

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def query_openai(prompt):
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=HEADERS, data=json.dumps(data)
    )

    if response.ok:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

print(query_openai(meta_prompt))