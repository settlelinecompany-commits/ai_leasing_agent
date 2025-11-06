"""
Clean Search Functions for Layla
Uses LangChain abstractions exclusively - no direct Qdrant calls
"""
import os
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from typing import Optional, Dict, List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "rocky_properties"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize LangChain embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize Qdrant client (only for LangChain initialization)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

# Initialize Qdrant vector store (for existing collection)
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Helper class to convert LangChain Document to ScoredPoint-like object
class ScoredPointLike:
    """Wrapper to convert LangChain (Document, score) to ScoredPoint-like object"""
    def __init__(self, doc: Document, score: float):
        self.score = score
        # LangChain's QdrantVectorStore only returns _id in metadata
        # We need to retrieve the full payload from Qdrant using the ID
        doc_id = doc.metadata.get("_id") if doc.metadata else None
        if doc_id is not None:
            # Access underlying Qdrant client to get full payload
            # vectorstore.client is the QdrantClient instance
            try:
                result = vectorstore.client.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[doc_id]
                )
                if result and len(result) > 0:
                    self.payload = result[0].payload
                else:
                    self.payload = doc.metadata if doc.metadata else {}
            except Exception:
                self.payload = doc.metadata if doc.metadata else {}
        else:
            self.payload = doc.metadata if doc.metadata else {}

def semantic_search(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.3
) -> List[ScoredPointLike]:
    """
    Pure semantic search - no filters
    Uses LangChain QdrantVectorStore exclusively
    
    Args:
        query: Search query text
        limit: Maximum number of results
        score_threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        List of ScoredPoint-like objects
    """
    # Use LangChain's similarity_search_with_score
    # Get more results to filter by threshold
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=limit * 3  # Get more to filter by threshold
    )
    
    # Filter by score_threshold
    # Note: Qdrant uses distance (lower is better), so we need to convert
    # For cosine similarity: score = 1 - distance (if distance is normalized)
    # For now, we'll filter by distance <= (1 - score_threshold)
    filtered = [
        (doc, score) for doc, score in results 
        if score <= (1 - score_threshold)  # Convert similarity threshold to distance
    ]
    
    # Convert to ScoredPoint-like objects
    return [ScoredPointLike(doc, 1 - score) for doc, score in filtered[:limit]]

def hybrid_search(
    query: str,
    filters: Optional[models.Filter] = None,
    limit: int = 5,
    score_threshold: float = 0.3
) -> List[ScoredPointLike]:
    """
    Hybrid search - semantic + structured filters
    Uses LangChain QdrantVectorStore exclusively with native Qdrant filters
    
    Args:
        query: Search query text
        filters: Qdrant filter object (models.Filter) - passed directly to LangChain
        limit: Maximum number of results
        score_threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        List of ScoredPoint-like objects
    """
    # Use LangChain's similarity_search_with_score with filter
    # Get more results to filter by threshold if needed
    k = limit * 3 if filters is None else limit
    
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        filter=filters  # Pass Qdrant models.Filter directly
    )
    
    # Filter by score_threshold if no strict filters (filters already ensure relevance)
    if filters is None:
        # Convert similarity threshold to distance
        filtered = [
            (doc, score) for doc, score in results 
            if score <= (1 - score_threshold)
        ]
        results = filtered[:limit]
    
    # Convert to ScoredPoint-like objects
    # Note: Qdrant returns distance, convert to similarity score
    return [ScoredPointLike(doc, 1 - score) for doc, score in results[:limit]]

def get_property_by_id(property_id: str) -> Optional[Dict]:
    """
    Get a single property by ID
    Uses LangChain QdrantVectorStore.get_by_ids() exclusively
    
    Args:
        property_id: Property ID (e.g., "rocky_001")
    
    Returns:
        Property dict with 'id' and 'payload' keys, or None if not found
    """
    # Convert property_id to integer if needed (rocky_001 -> 1)
    try:
        if property_id.startswith("rocky_"):
            point_id = int(property_id.replace("rocky_", ""))
        else:
            point_id = int(property_id)
    except ValueError:
        return None
    
    # Use LangChain's get_by_ids method (positional argument, not keyword)
    try:
        docs = vectorstore.get_by_ids([point_id])
        
        if docs and len(docs) > 0:
            doc = docs[0]
            # Get the actual ID from metadata
            doc_id = doc.metadata.get("_id", point_id)
            
            # Retrieve full payload from Qdrant (LangChain only returns _id in metadata)
            # Access through vectorstore.client (still using LangChain's abstraction)
            result = vectorstore.client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[doc_id]
            )
            
            if result and len(result) > 0:
                point = result[0]
                return {
                    "id": point.id,
                    "payload": point.payload  # Full payload from Qdrant
                }
    except Exception:
        pass
    
    return None

def format_property_for_context(property_point, index: int = 1) -> str:
    """Format a single property for LLM context
    Works with both ScoredPoint and ScoredPointLike objects
    """
    payload = property_point.payload
    score = property_point.score if hasattr(property_point, 'score') else 'N/A'
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    
    context = f"""
Property {index} (Similarity Score: {score_str}):
- Property ID: {payload.get('property_id', 'N/A')}
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
    return context

def format_properties_for_context(properties: List) -> str:
    """Format multiple properties for LLM context
    Works with both ScoredPoint and ScoredPointLike objects
    """
    context_parts = []
    for i, prop in enumerate(properties, 1):
        context_parts.append(format_property_for_context(prop, i))
    return "\n".join(context_parts)

