"""
Upload Rocky Real Estate properties to Qdrant with OpenAI embeddings
Implements rich embedding text strategy for optimal semantic search
"""
import csv
import os
from typing import Dict, List, Optional
from qdrant_client import QdrantClient, models
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "https://8445b063-3aeb-4395-84e5-6ae2c6a662c2.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.w4n53UJV1pGUEWGAeTsEwsgUyTfqFHQ6eUlykapMhU0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

COLLECTION_NAME = "rocky_properties"
CSV_FILE = "rocky_real_estate_properties.csv"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small dimensions
BATCH_SIZE = 64  # Batch size for uploads

def create_rich_embedding_text(property: Dict) -> str:
    """
    Create rich, searchable embedding text that combines all key property information.
    This text will be embedded and used for semantic search.
    """
    parts = []
    
    # Core property info
    if property.get('bedrooms') and property.get('bedrooms') != '':
        if property['bedrooms'] == '0' or property['bedrooms'] == 0:
            parts.append("studio apartment")
        else:
            parts.append(f"{property['bedrooms']} bedroom")
    
    if property.get('bathrooms') and property.get('bathrooms') != '':
        parts.append(f"{property['bathrooms']} bathroom")
    
    if property.get('property_type') and property.get('property_type') != '':
        parts.append(property['property_type'].lower())
    elif not property.get('property_type') or property.get('property_type') == '':
        # Infer from bedrooms
        if property.get('bedrooms') and property.get('bedrooms') != '' and property.get('bedrooms') != '0':
            parts.append("apartment")
    
    # Location (most important for queries)
    if property.get('location') and property.get('location') != '':
        parts.append(f"in {property['location']}")
    
    if property.get('area') and property.get('area') != '':
        parts.append(f"area {property['area']}")
    
    if property.get('city') and property.get('city') != '':
        parts.append(f"city {property['city']}")
    
    # Size
    if property.get('sqft') and property.get('sqft') != '':
        try:
            sqft = int(float(property['sqft']))
            parts.append(f"{sqft} square feet")
        except (ValueError, TypeError):
            pass
    
    # Price context (important for affordability queries)
    if property.get('monthly_rent') and property.get('monthly_rent') != '':
        try:
            monthly = float(property['monthly_rent'])
            parts.append(f"AED {monthly:.0f} monthly rent")
        except (ValueError, TypeError):
            pass
    
    if property.get('yearly_rent') and property.get('yearly_rent') != '':
        try:
            yearly = float(property['yearly_rent'])
            parts.append(f"AED {yearly:.0f} yearly rent")
        except (ValueError, TypeError):
            pass
    
    # Furnishing status
    furnished = property.get('furnished')
    if furnished:
        if furnished == 'True' or furnished is True:
            parts.append("furnished")
        elif furnished == 'False' or furnished is False:
            parts.append("unfurnished")
        elif isinstance(furnished, str):
            parts.append(furnished.lower())
    
    # Amenities (important for feature queries)
    amenities = property.get('amenities', '')
    if amenities:
        if isinstance(amenities, str):
            # Handle comma-separated string
            amenity_list = [a.strip() for a in amenities.split(',') if a.strip()]
        elif isinstance(amenities, list):
            amenity_list = amenities
        else:
            amenity_list = []
        
        if amenity_list:
            parts.append("with " + ", ".join(amenity_list))
    
    # Key features
    if property.get('parking') and (property.get('parking') == 'True' or property.get('parking') is True):
        parking_spots = property.get('parking_spots', '1')
        if parking_spots and parking_spots != '':
            try:
                parts.append(f"{int(float(parking_spots))} parking")
            except (ValueError, TypeError):
                parts.append("parking")
        else:
            parts.append("parking")
    
    if property.get('pet_friendly') and (property.get('pet_friendly') == 'True' or property.get('pet_friendly') is True):
        parts.append("pet friendly")
    
    if property.get('security_24_7') and (property.get('security_24_7') == 'True' or property.get('security_24_7') is True):
        parts.append("24/7 security")
    
    if property.get('nearby_metro') and (property.get('nearby_metro') == 'True' or property.get('nearby_metro') is True):
        parts.append("near metro")
    
    if property.get('nearby_shops') and (property.get('nearby_shops') == 'True' or property.get('nearby_shops') is True):
        parts.append("near shops")
    
    # Combine all parts
    embedding_text = ", ".join(parts)
    
    # Fallback to existing embedding_text if we created nothing
    if not embedding_text or embedding_text.strip() == "":
        embedding_text = property.get('embedding_text', property.get('description', ''))
    
    return embedding_text

def load_properties_from_csv(csv_file: str) -> List[Dict]:
    """Load properties from CSV file"""
    properties = []
    
    print(f"📂 Loading properties from {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            properties.append(row)
    
    print(f"✓ Loaded {len(properties)} properties from CSV")
    return properties

def prepare_payload(property: Dict) -> Dict:
    """Prepare payload with all metadata for filtering"""
    payload = {}
    
    # Copy all fields from CSV
    for key, value in property.items():
        # Skip empty strings, convert to None
        if value == '':
            payload[key] = None
        else:
            # Convert numeric strings to numbers
            if key in ['monthly_rent', 'yearly_rent', 'bedrooms', 'bathrooms', 'sqft', 'parking_spots']:
                try:
                    if value:
                        payload[key] = float(value) if '.' in str(value) else int(float(value))
                    else:
                        payload[key] = None
                except (ValueError, TypeError):
                    payload[key] = None
            # Convert boolean strings
            elif key in ['furnished', 'parking', 'pet_friendly', 'security_24_7', 'nearby_metro', 'nearby_shops']:
                if value == 'True' or value is True:
                    payload[key] = True
                elif value == 'False' or value is False:
                    payload[key] = False
                else:
                    payload[key] = None
            # Handle amenities list
            elif key == 'amenities':
                if value:
                    if isinstance(value, str):
                        payload[key] = [a.strip() for a in value.split(',') if a.strip()]
                    elif isinstance(value, list):
                        payload[key] = value
                    else:
                        payload[key] = []
                else:
                    payload[key] = []
            else:
                payload[key] = value
    
    return payload

def create_payload_indexes(client: QdrantClient, collection_name: str):
    """Create payload indexes for efficient filtering"""
    indexes_to_create = [
        ("bedrooms", "integer"),
        ("bathrooms", "integer"),
        ("monthly_rent", "float"),
        ("yearly_rent", "float"),
        ("sqft", "integer"),
        ("furnished", "bool"),
        ("parking", "bool"),
        ("pet_friendly", "bool"),
        ("security_24_7", "bool"),
        ("city", "keyword"),
        ("area", "keyword"),
        ("amenities", "keyword"),
    ]
    
    for field_name, field_type in indexes_to_create:
        try:
            if field_type == "integer":
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.INTEGER
                )
            elif field_type == "float":
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.FLOAT
                )
            elif field_type == "bool":
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.BOOL
                )
            elif field_type == "keyword":
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            print(f"  ✓ Index created for '{field_name}' ({field_type})")
        except Exception as e:
            # Index might already exist, that's okay
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"  ✓ Index already exists for '{field_name}'")
            else:
                print(f"  ⚠️  Could not create index for '{field_name}': {e}")

def ensure_collection_exists(client: QdrantClient, collection_name: str, dim: int):
    """Create collection if it doesn't exist, or recreate if dimensions don't match"""
    print(f"\n🔍 Checking collection '{collection_name}'...")
    
    try:
        # Check if collection exists
        collection_info = client.get_collection(collection_name)
        current_dim = collection_info.config.params.vectors.size
        
        if current_dim != dim:
            print(f"⚠️  Collection exists but has wrong dimensions ({current_dim} vs {dim})")
            print(f"🗑️  Deleting existing collection...")
            client.delete_collection(collection_name)
            print(f"✓ Collection deleted")
            
            print(f"🆕 Creating new collection with {dim} dimensions...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )
            print(f"✓ Collection created successfully!")
        else:
            print(f"✓ Collection exists with correct dimensions ({dim})")
            print(f"📊 Current points: {collection_info.points_count}")
            
            # If collection is empty, proceed with upload
            if collection_info.points_count == 0:
                print(f"📝 Collection is empty, will upload new data")
                # Create indexes even if collection exists but is empty
                print(f"\n📊 Creating payload indexes for filtering...")
                create_payload_indexes(client, collection_name)
            else:
                # For non-empty collections, delete and recreate (auto-mode)
                print(f"🗑️  Collection has existing data, clearing and re-uploading...")
                client.delete_collection(collection_name)
                print(f"🆕 Creating new collection...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"✓ Collection recreated!")
                
                # Create payload indexes for filtering
                print(f"\n📊 Creating payload indexes for filtering...")
                create_payload_indexes(client, collection_name)
    
    except Exception as e:
        # Collection doesn't exist, create it
        if "not found" in str(e).lower() or "404" in str(e):
            print(f"🆕 Collection doesn't exist, creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )
            print(f"✓ Collection created successfully!")
            
            # Create payload indexes for filtering
            print(f"\n📊 Creating payload indexes for filtering...")
            create_payload_indexes(client, collection_name)
        else:
            raise

def upload_properties_to_qdrant(
    properties: List[Dict],
    client: QdrantClient,
    openai_client,
    collection_name: str
):
    """Upload properties to Qdrant with embeddings"""
    
    print(f"\n🚀 Starting upload process...")
    print(f"📊 Properties to upload: {len(properties)}")
    print(f"🤖 Embedding model: {EMBEDDING_MODEL}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    
    points = []
    errors = []
    
    for i, prop in enumerate(properties, 1):
        try:
            # Create rich embedding text
            embedding_text = create_rich_embedding_text(prop)
            
            # Get embedding from OpenAI
            response = openai_client.Embedding.create(
                model=EMBEDDING_MODEL,
                input=embedding_text
            )
            vector = response['data'][0]['embedding']
            
            # Prepare payload
            payload = prepare_payload(prop)
            payload['embedding_text'] = embedding_text  # Store the embedding text for reference
            
            # Convert property_id to integer (extract number from "rocky_001" -> 1)
            property_id_str = prop['property_id']
            try:
                # Extract number from "rocky_001" format
                point_id = int(property_id_str.split('_')[1])
            except (ValueError, IndexError):
                # Fallback: use hash of property_id
                point_id = hash(property_id_str) % (2**63)  # Convert to positive int
            
            # Create point
            point = models.PointStruct(
                id=point_id,  # Use integer ID
                vector=vector,
                payload=payload
            )
            points.append(point)
            
            # Progress update
            if i % 10 == 0:
                print(f"  ✓ Processed {i}/{len(properties)} properties...")
            
            # Batch upload
            if len(points) >= BATCH_SIZE:
                client.upload_points(
                    collection_name=collection_name,
                    points=points
                )
                print(f"  📤 Uploaded batch of {len(points)} points")
                points = []
        
        except Exception as e:
            error_msg = f"Error processing property {prop.get('property_id', 'unknown')}: {str(e)}"
            print(f"  ✗ {error_msg}")
            errors.append(error_msg)
            continue
    
    # Upload remaining points
    if points:
        client.upload_points(
            collection_name=collection_name,
            points=points
        )
        print(f"  📤 Uploaded final batch of {len(points)} points")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ UPLOAD COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Total properties processed: {len(properties)}")
    print(f"✅ Successfully uploaded: {len(properties) - len(errors)}")
    if errors:
        print(f"✗ Errors: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Verify upload
    collection_info = client.get_collection(collection_name)
    print(f"\n📊 Collection '{collection_name}' now has {collection_info.points_count} points")
    print(f"{'='*70}")

def test_search(client: QdrantClient, collection_name: str, openai_client):
    """Test semantic search with sample queries"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING SEMANTIC SEARCH")
    print(f"{'='*70}")
    
    test_queries = [
        "2 bedroom apartment near Business Bay",
        "affordable 1BR under 8000 monthly",
        "apartment with gym and pool",
        "3 bedroom furnished in Dubai Marina",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print(f"{'-'*70}")
        
        # Get embedding for query
        response = openai_client.Embedding.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        query_vector = response['data'][0]['embedding']
        
        # Search Qdrant using query_points (newer method)
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3,
            score_threshold=0.5  # Lower threshold to see results
        )
        
        # Convert to list format
        results = results.points if hasattr(results, 'points') else results
        
        if results:
            print(f"✓ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                payload = result.payload
                print(f"  {i}. {payload.get('location', 'N/A')[:60]}...")
                print(f"     Score: {result.score:.4f} | Beds: {payload.get('bedrooms', 'N/A')} | Baths: {payload.get('bathrooms', 'N/A')} | Rent: AED {payload.get('monthly_rent', 'N/A'):.0f}")
        else:
            print(f"⚠️  No results found (score threshold: 0.5)")

def main():
    """Main execution function"""
    print("="*70)
    print("ROCKY REAL ESTATE - QDRANT UPLOAD")
    print("="*70)
    
    # Initialize clients
    print("\n🔌 Connecting to Qdrant...")
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False
    )
    
    # Test connection
    try:
        collections = qdrant_client.get_collections()
        print(f"✓ Connected to Qdrant! Found {len(collections.collections)} collection(s)")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return
    
    print("\n🤖 Initializing OpenAI client...")
    openai.api_key = OPENAI_API_KEY
    openai_client = openai
    print(f"✓ OpenAI client initialized")
    
    # Ensure collection exists
    ensure_collection_exists(qdrant_client, COLLECTION_NAME, EMBEDDING_DIM)
    
    # Load properties from CSV
    if not os.path.exists(CSV_FILE):
        print(f"\n✗ Error: CSV file '{CSV_FILE}' not found!")
        return
    
    properties = load_properties_from_csv(CSV_FILE)
    
    if not properties:
        print(f"\n⚠️  No properties found in CSV!")
        return
    
    # Upload to Qdrant
    upload_properties_to_qdrant(
        properties=properties,
        client=qdrant_client,
        openai_client=openai_client,
        collection_name=COLLECTION_NAME
    )
    
    # Test search
    test_search(qdrant_client, COLLECTION_NAME, openai_client)
    
    print(f"\n✅ All done! Properties are now searchable in Qdrant.")
    print(f"📊 Collection: {COLLECTION_NAME}")
    print(f"🔗 Qdrant URL: {QDRANT_URL}")

if __name__ == "__main__":
    main()

