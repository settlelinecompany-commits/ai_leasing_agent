"""
Layla - The Leasing Agent
LangGraph-based intelligent property leasing agent with state management
"""
import os
import re
from datetime import datetime, timedelta
from typing import Annotated, TypedDict, Optional, Dict, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Import our modules
from layla_search import semantic_search, hybrid_search, get_property_by_id, format_properties_for_context
from layla_calendar import get_available_slots, book_slot, check_availability
from query_properties import parse_query_filters, build_qdrant_filter
from qdrant_client import models

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ============================================
# Step 1: Define the State
# ============================================
def add_messages(left, right):
    """Add messages to state"""
    return left + right

class LaylaState(TypedDict):
    """State for Layla agent - maintains conversation context"""
    messages: Annotated[List[BaseMessage], add_messages]
    selected_property: Optional[Dict]  # Currently viewed property
    search_results: Optional[List[Dict]]  # Last search results (for "first one", "second one" references)
    lead_info: Optional[Dict]  # Name, phone, email
    workflow_stage: str  # "searching" | "viewing" | "booking" | "completed"
    tour_details: Optional[Dict]  # Tour booking information

# ============================================
# Step 2: Build the Tools
# ============================================

@tool
def search_properties_tool(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.3,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    max_monthly_rent: Optional[float] = None,
    min_monthly_rent: Optional[float] = None,
    furnished: Optional[bool] = None,
    parking: Optional[bool] = None
) -> str:
    """
    Search for properties using semantic search and optional structured filters.
    
    The LLM should decide:
    - score_threshold: Lower (0.25-0.3) for exploratory queries, higher (0.4-0.5) for specific queries
    - Whether to use filters: Use filters for exact criteria (bedrooms, price), rely on semantic for fuzzy (amenities, location)
    
    Args:
        query: Natural language search query
        limit: Maximum number of results (default: 5)
        score_threshold: Minimum similarity score (0.0 to 1.0). Lower = more results, higher = more precise
        bedrooms: Exact number of bedrooms (optional)
        bathrooms: Exact number of bathrooms (optional)
        max_monthly_rent: Maximum monthly rent in AED (optional)
        min_monthly_rent: Minimum monthly rent in AED (optional)
        furnished: True/False for furnished status (optional)
        parking: True/False for parking availability (optional)
    
    Returns:
        Formatted string with property details
    """
    # Build filters dict for build_qdrant_filter
    filters_dict = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'max_monthly_rent': max_monthly_rent,
        'min_monthly_rent': min_monthly_rent,
        'furnished': furnished,
        'parking': parking,
        'amenities': [],
        'city': None,
        'area': None
    }
    
    # Build Qdrant filter (only for exact criteria)
    qdrant_filter = build_qdrant_filter(filters_dict)
    
    # Perform hybrid search
    if qdrant_filter:
        properties = hybrid_search(query, filters=qdrant_filter, limit=limit, score_threshold=score_threshold)
    else:
        properties = semantic_search(query, limit=limit, score_threshold=score_threshold)
    
    if not properties:
        return "No properties found matching your criteria. Please try adjusting your search."
    
    # Format for LLM context
    return format_properties_for_context(properties)

@tool
def get_property_details_tool(property_id: str) -> str:
    """
    Get full details for a specific property by ID.
    
    Args:
        property_id: Property ID (e.g., "rocky_001" or just "1")
    
    Returns:
        Formatted string with full property details
    """
    property_data = get_property_by_id(property_id)
    
    if not property_data:
        return f"Property {property_id} not found. Please check the property ID."
    
    payload = property_data['payload']
    
    details = f"""
Full Property Details:
- Property ID: {payload.get('property_id', 'N/A')}
- Location: {payload.get('location', 'N/A')}
- Area: {payload.get('area', 'N/A')}
- City: {payload.get('city', 'N/A')}
- Bedrooms: {payload.get('bedrooms', 'N/A')}
- Bathrooms: {payload.get('bathrooms', 'N/A')}
- Monthly Rent: AED {payload.get('monthly_rent', 0):.0f}
- Yearly Rent: AED {payload.get('yearly_rent', 0):.0f}
- Square Feet: {payload.get('sqft', 'N/A')}
- Property Type: {payload.get('property_type', 'N/A')}
- Furnished: {payload.get('furnished', 'N/A')}
- Parking: {payload.get('parking', 'N/A')}
- Parking Spots: {payload.get('parking_spots', 'N/A')}
- Amenities: {payload.get('amenities', [])}
- Pet Friendly: {payload.get('pet_friendly', 'N/A')}
- Security 24/7: {payload.get('security_24_7', 'N/A')}
- Nearby Metro: {payload.get('nearby_metro', 'N/A')}
- Nearby Shops: {payload.get('nearby_shops', 'N/A')}
- URL: {payload.get('url', 'N/A')}
- Description: {payload.get('description', 'N/A')}
"""
    return details

@tool
def check_availability_tool(property_id: str) -> str:
    """
    Check if a property is currently available for rent.
    
    Args:
        property_id: Property ID (e.g., "rocky_001")
    
    Returns:
        Availability status
    """
    property_data = get_property_by_id(property_id)
    
    if not property_data:
        return f"Property {property_id} not found."
    
    # For now, assume all properties are available
    # In production, this would check against a real availability system
    return f"Property {property_id} is currently available for rent."

@tool
def get_tour_slots_tool(property_id: str, date: Optional[str] = None) -> str:
    """
    Get available tour time slots for a property.
    
    Args:
        property_id: Property ID (e.g., "rocky_001")
        date: Optional date filter (YYYY-MM-DD format). If not provided, shows next 7 days.
    
    Returns:
        Formatted string with available time slots
    """
    slots = get_available_slots(property_id, date)
    
    if not slots:
        if date:
            return f"No available tour slots found for property {property_id} on {date}."
        else:
            return f"No available tour slots found for property {property_id} in the next 7 days."
    
    # Group by date
    slots_by_date = {}
    for slot in slots:
        date_key = slot['date']
        if date_key not in slots_by_date:
            slots_by_date[date_key] = []
        slots_by_date[date_key].append(slot['time'])
    
    result = f"Available tour slots for property {property_id}:\n\n"
    for date_key in sorted(slots_by_date.keys()):
        times = sorted(slots_by_date[date_key])
        result += f"{date_key}:\n"
        for time in times:
            result += f"  - {time}\n"
        result += "\n"
    
    return result.strip()

@tool
def book_tour_smart_tool() -> str:
    """
    Book a tour for a property. This tool checks the conversation state for all required information.
    Call this tool when the user wants to book a tour - it will automatically check what information is available
    and either book the tour or tell you what's missing.
    
    Returns:
        Booking confirmation or message indicating what information is missing
    """
    # This tool doesn't do anything - custom_tool_node will intercept it
    # and handle all the logic by checking state
    return "Checking booking information..."

@tool
def sync_to_crm_tool(
    lead_name: str,
    lead_phone: str,
    lead_email: Optional[str] = None,
    property_id: Optional[str] = None,
    notes: Optional[str] = None
) -> str:
    """
    Sync lead information to CRM system (mock implementation).
    
    Args:
        lead_name: Lead name
        lead_phone: Lead phone number
        lead_email: Lead email (optional)
        property_id: Property ID they're interested in (optional)
        notes: Additional notes (optional)
    
    Returns:
        Confirmation message
    """
    # Mock CRM sync - in production, this would call actual CRM API
    crm_data = {
        "name": lead_name,
        "phone": lead_phone,
        "email": lead_email,
        "property_id": property_id,
        "notes": notes,
        "status": "new_lead"
    }
    
    return f"Lead information synced to CRM successfully. Lead: {lead_name} ({lead_phone})"

# ============================================
# Step 3: Set up LLM and Tools
# ============================================
print("Setting up Layla agent...")

tools = [
    search_properties_tool,
    get_property_details_tool,
    check_availability_tool,
    get_tour_slots_tool,
    book_tour_smart_tool,
    sync_to_crm_tool
]

tool_node = ToolNode(tools)

# Custom tool node that handles smart booking tool
def custom_tool_node(state: LaylaState) -> Dict:
    """Custom tool node that handles smart booking tool and standard tools"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        # No tool calls, use standard tool node
        return tool_node.invoke(state)
    
    # Check if smart booking tool is being called
    smart_booking_tool_call = None
    for tc in last_message.tool_calls:
        if tc.get("name") == "book_tour_smart_tool":
            smart_booking_tool_call = tc
            break
    
    if smart_booking_tool_call:
        # Get the tool_call_id for the response
        tool_call_id = smart_booking_tool_call.get("id")
        
        # Handle smart booking tool - check state and book or return missing info
        lead_info = state.get("lead_info") or {}
        tour_details = state.get("tour_details") or {}
        selected_property = state.get("selected_property") or {}
        
        # Get property_id
        property_id = (tour_details.get("property_id") or 
                       selected_property.get("property_id"))
        
        # Check what's missing
        missing = []
        if not property_id:
            missing.append("property selection")
        if not tour_details.get("date"):
            missing.append("tour date")
        if not tour_details.get("time"):
            missing.append("tour time")
        if not lead_info.get("name"):
            missing.append("your name")
        if not lead_info.get("phone"):
            missing.append("your phone number")
        
        # Validate placeholder values
        placeholder_names = ["layla", "customer", "user", "test", "demo", "example", "placeholder"]
        placeholder_phones = ["1234567890", "0000000000", "1111111111", "123456789", "00000000"]
        
        if lead_info.get("name") and lead_info["name"].lower() in placeholder_names:
            missing.append("your actual name (not a placeholder)")
        if lead_info.get("phone") and lead_info["phone"] in placeholder_phones:
            missing.append("your actual phone number (not a placeholder)")
        
        # If missing info, return ToolMessage asking for it
        if missing:
            return {
                "messages": [ToolMessage(
                    content=f"I need the following information to book your tour: {', '.join(missing)}. Please provide this information.",
                    tool_call_id=tool_call_id
                )]
            }
        
        # All info is available - book it!
        booking_result = book_slot(
            property_id,
            tour_details["date"],
            tour_details["time"],
            lead_info["name"],
            lead_info["phone"]
        )
        
        if booking_result.get('success'):
            confirmation_id = booking_result.get('confirmation_id', 'N/A')
            return {
                "messages": [ToolMessage(
                    content=f"""
Tour booking confirmed!

Confirmation ID: {confirmation_id}
Property ID: {property_id}
Date: {tour_details['date']}
Time: {tour_details['time']}
Customer: {lead_info['name']}
Phone: {lead_info['phone']}

We'll send you a reminder 1 hour before your tour.
""",
                    tool_call_id=tool_call_id
                )]
            }
        else:
            return {
                "messages": [ToolMessage(
                    content=f"Booking failed: {booking_result.get('error', 'Unknown error')}",
                    tool_call_id=tool_call_id
                )]
            }
    
    # For all other tools, use standard tool node
    return tool_node.invoke(state)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# System prompt for Layla
def get_system_prompt(state: LaylaState) -> str:
    """Generate system prompt with current state information"""
    base_prompt = """You are Layla, a friendly and professional leasing agent for Rocky Real Estate in Dubai.

Your role:
- Help customers find properties that match their needs
- Answer questions about properties, pricing, amenities, and availability
- Book property tours when customers are ready
- Sync lead information to CRM when appropriate

Guidelines:
- Be warm, helpful, and professional
- When users say "the first one" or "that property", refer to the most recent search results or selected property
- Always confirm tour details before booking
- Use the search_properties_tool for property searches - decide on score_threshold based on query specificity:
  * Lower threshold (0.25-0.3) for exploratory queries like "properties with gym"
  * Higher threshold (0.4-0.5) for specific queries with exact criteria
- Use structured filters (bedrooms, price) when the user specifies exact criteria
- For fuzzy queries (amenities, location), rely on semantic search

Using State Information:
- Check the "Current State" section below to see what information you already have
- When the user wants to book a tour, call book_tour_smart_tool() - it will automatically check what's needed
- Only ask for information that is MISSING from state

Remember: You maintain conversation context, so you can reference previous messages and search results."""
    
    # Add state information to prompt
    state_info = []
    if state.get("lead_info"):
        lead = state["lead_info"]
        if lead.get("name"):
            state_info.append(f"- Customer name: {lead['name']}")
        if lead.get("phone"):
            state_info.append(f"- Customer phone: {lead['phone']}")
        if lead.get("email"):
            state_info.append(f"- Customer email: {lead['email']}")
    
    if state.get("tour_details"):
        tour = state["tour_details"]
        if tour.get("date"):
            state_info.append(f"- Tour date: {tour['date']}")
        if tour.get("time"):
            state_info.append(f"- Tour time: {tour['time']}")
        if tour.get("property_id"):
            state_info.append(f"- Property ID: {tour['property_id']}")
    
    if state.get("selected_property"):
        prop = state["selected_property"]
        if prop.get("property_id"):
            state_info.append(f"- Selected property: {prop.get('property_id')}")
    
    if state_info:
        base_prompt += "\n\nCurrent State (Remembered Information):\n" + "\n".join(state_info)
    
    return base_prompt

# ============================================
# Step 3.5: Information Extraction Node
# ============================================
def extract_information_node(state: LaylaState) -> Dict:
    """Extract structured information (name, phone, date, time) from messages"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # CRITICAL: Always preserve existing state - initialize state_updates with existing state
    state_updates = {}
    
    # Always copy existing state to preserve it
    current_lead_info = state.get("lead_info") or {}
    state_updates["lead_info"] = current_lead_info.copy()
    
    current_tour_details = state.get("tour_details") or {}
    state_updates["tour_details"] = current_tour_details.copy()
    
    if not last_message or not hasattr(last_message, "content"):
        # Return existing state even if no extraction
        return state_updates
    
    # Get original text (not lowercased) for name extraction
    user_text_original = str(last_message.content)
    user_text = user_text_original.lower()
    
    # Note: state_updates already initialized above with existing state
    
    # Extract name and phone together (comma-separated format like "laksh, 3122037041")
    # This should be checked FIRST before individual patterns
    # Pattern: "name, phone" or "name,phone" (e.g., "laksh, 3122037041")
    # Also handle: "yep the same property, laksh, 3122037041"
    # Handle multi-word names: "Sarah Ahmed, 0501234567"
    comma_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*(\d{8,12})(?:\s|$)',  # "Sarah Ahmed, 0501234567" or "laksh, 3122037041"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*(\d{8,12})$',  # End of string
    ]
    for comma_pattern in comma_patterns:
        comma_match = re.search(comma_pattern, user_text_original, re.IGNORECASE)
        if comma_match:
            name = comma_match.group(1).strip()
            phone = comma_match.group(2).strip()
            # Only extract if name looks reasonable (not a property ID or number)
            # Handle multi-word names
            name_clean = name.strip()
            if len(name_clean) >= 2 and not name_clean.isdigit() and name_clean.lower() not in ['yes', 'no', 'yep', 'nope']:
                if not current_lead_info.get("name") or current_lead_info.get("name") in ['yes', 'no', 'yep', 'nope']:
                    state_updates["lead_info"]["name"] = name_clean
                if not current_lead_info.get("phone") or len(str(current_lead_info.get("phone", ""))) < 8:
                    state_updates["lead_info"]["phone"] = phone
                break  # Found comma-separated format, skip individual patterns
    
    # Extract name (if not already extracted from comma-separated format)
    # Skip if we already extracted from comma-separated format
    # Handle multi-word names: "my name is Sarah Ahmed", "I'm Sarah Ahmed"
    if not state_updates["lead_info"].get("name") and not current_lead_info.get("name"):
        # Patterns: "name is John", "I'm John", "John", "my name is John", "Name is Laksh"
        # Also handle: "Name is Laksh and phone number is 3122037041"
        # Handle multi-word names: "my name is Sarah Ahmed", "I'm Sarah Ahmed"
        name_patterns = [
            r'my\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+and|\s+phone|\s+,\s*phone|$)',  # "my name is Sarah Ahmed and phone"
            r'name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+and|\s+phone|\s+,\s*phone|$)',  # "name is Sarah Ahmed and phone"
            r'i\'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+and|\s+phone|\s+,\s*phone|$)',  # "I'm Sarah Ahmed and phone"
            r'i\s+am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+and|\s+phone|\s+,\s*phone|$)',  # "I am Sarah Ahmed and phone"
            r'call\s+me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+and|\s+phone|\s+,\s*phone|$)',  # "call me Sarah Ahmed and phone"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, user_text_original, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # For standalone names, make sure it's not a phone number or date
                if pattern == r'^([A-Z][a-z]+)$':
                    # Only extract if it's a reasonable name (2+ chars, not a number)
                    if len(name) >= 2 and not name.isdigit():
                        state_updates["lead_info"]["name"] = name
                        break
                else:
                    state_updates["lead_info"]["name"] = name
                    break
    
    # Extract phone number (if not already extracted from comma-separated format)
    # Skip if we already extracted from comma-separated format
    # Also avoid matching property IDs or other numbers in the conversation
    if not state_updates["lead_info"].get("phone") and not current_lead_info.get("phone"):
        # Patterns: "phone is 0501234567", "phone number is 3122037041"
        # Avoid matching property IDs or other numbers
        phone_patterns = [
            r'(?:phone|phone number|mobile|number)(?:\s+is|\s+number\s+is)?\s*:?\s*(\d{8,12})(?:\s|$)',  # Explicit phone mention
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                phone = match.group(1).strip()
                # Only extract if it looks like a phone number (not a property ID)
                # Property IDs are usually shorter or have specific patterns
                if len(phone) >= 8 and not phone.startswith('rocky_'):
                    state_updates["lead_info"]["phone"] = phone
                    break
    
    # Extract date - improved patterns to handle "9am tomorrow", "tomorrow 9am", etc.
    # Allow overwriting if user explicitly provides a new date
    # Check if user explicitly mentions a date (not just "tomorrow" if we already have a date)
    if not current_tour_details.get("date") or "november" in user_text or "december" in user_text or "january" in user_text or "february" in user_text or "march" in user_text or "april" in user_text or "may" in user_text or "june" in user_text or "july" in user_text or "august" in user_text or "september" in user_text or "october" in user_text:
        # Patterns: "Nov 6th", "November 6", "tomorrow", "2024-11-06"
        # Also handle: "9am tomorrow", "tomorrow 9am", "tomorrow works"
        date_patterns = [
            (r'(?:tomorrow|today)(?:\s+works|\s+at|\s+\d|$)', None),  # "tomorrow", "tomorrow works", "tomorrow 9am"
            (r'(?:nov|november|dec|december|jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october)\s+(\d{1,2})(?:st|nd|rd|th)?', 1),
            (r'(\d{4}-\d{2}-\d{2})', 1),
        ]
        for pattern, group_idx in date_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                if group_idx is None:
                    # "tomorrow" or "today"
                    date_str = match.group(0).lower()
                    if "tomorrow" in date_str:
                        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                    else:  # today
                        date = datetime.now().strftime("%Y-%m-%d")
                elif re.match(r'\d{4}-\d{2}-\d{2}', match.group(group_idx)):
                    # Already in YYYY-MM-DD format
                    date = match.group(group_idx)
                else:
                    # Parse "Nov 6th" format
                    try:
                        month_map = {
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
                        parts = match.group(0).lower().split()
                        month_name = parts[0][:3]
                        day = int(match.group(group_idx))
                        month = month_map.get(month_name, datetime.now().month)
                        year = datetime.now().year
                        date = f"{year}-{month:02d}-{day:02d}"
                    except:
                        continue
                
                state_updates["tour_details"]["date"] = date
                break
    
    # Extract time - improved patterns to handle "9am tomorrow", "tomorrow 9am", etc.
    # Allow overwriting if user explicitly provides a new time
    if not current_tour_details.get("time") or re.search(r'\d{1,2}\s*(am|pm)', user_text, re.IGNORECASE) or re.search(r'\d{1,2}:\d{2}', user_text):
        # Patterns: "10am", "10:00", "10:00 AM", "10:00am", "9am tomorrow", "tomorrow 9am"
        time_patterns = [
            (r'(\d{1,2}):?(\d{2})?\s*(am|pm)(?:\s+tomorrow|\s+today|\s+works|$)', (1, 2, 3)),  # "9am tomorrow", "9am works"
            (r'(\d{1,2}):?(\d{2})?\s*(am|pm)', (1, 2, 3)),  # "9am", "10:00am"
            (r'(\d{1,2}):(\d{2})', (1, 2, None)),  # "10:00"
            (r'(\d{1,2})\s*(am|pm)', (1, None, 2)),  # "9 am", "10 pm"
        ]
        for pattern, groups in time_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                hour = int(match.group(groups[0]))
                minute = int(match.group(groups[1])) if groups[1] and match.group(groups[1]) else 0
                am_pm = match.group(groups[2]) if groups[2] and match.group(groups[2]) else None
                
                if am_pm:
                    if am_pm.lower() == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm.lower() == 'am' and hour == 12:
                        hour = 0
                
                time = f"{hour:02d}:{minute:02d}"
                state_updates["tour_details"]["time"] = time
                break
    
    # Update property_id if selected_property exists
    if state.get("selected_property") and state["selected_property"].get("property_id"):
        if not state_updates["tour_details"].get("property_id"):
            state_updates["tour_details"]["property_id"] = state["selected_property"]["property_id"]
    
    # CRITICAL: Always return state_updates, even if empty
    # This ensures existing state is preserved (LangGraph merges state_updates with existing state)
    # If we return empty dict, existing state is lost!
    # By always including lead_info and tour_details (even if unchanged), we preserve state
    return state_updates

# ============================================
# Step 4: Create Agent Node
# ============================================
def layla_agent_node(state: LaylaState) -> Dict:
    """Agent node - LLM decides which tools to use"""
    messages = state["messages"]
    
    # Get system prompt with state information
    system_prompt = get_system_prompt(state)
    
    # Add system prompt if not already present
    if not any(isinstance(msg, BaseMessage) and hasattr(msg, 'content') and msg.content and "Layla" in str(msg.content) for msg in messages):
        system_message = HumanMessage(content=system_prompt)
        messages = [system_message] + messages
    else:
        # Update system prompt with current state
        # Replace first message if it's the system prompt
        if messages and isinstance(messages[0], HumanMessage) and "Layla" in str(messages[0].content):
            messages[0] = HumanMessage(content=system_prompt)
    
    response = llm_with_tools.invoke(messages)
    
    # CRITICAL: Always preserve existing state - don't overwrite with None
    state_updates = {
        "messages": [response],
        # Preserve existing state
        "lead_info": state.get("lead_info") or {},
        "tour_details": state.get("tour_details") or {},
        "selected_property": state.get("selected_property"),
        "search_results": state.get("search_results"),
        "workflow_stage": state.get("workflow_stage", "searching")
    }
    
    # Update workflow stage based on tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc.get("name") for tc in response.tool_calls]
        
        # If searching, update workflow stage
        if "search_properties_tool" in tool_names:
            state_updates["workflow_stage"] = "searching"
        
        # If getting property details, update workflow stage
        if "get_property_details_tool" in tool_names:
            state_updates["workflow_stage"] = "viewing"
            # Store selected property
            for tc in response.tool_calls:
                if tc.get("name") == "get_property_details_tool":
                    property_id = tc.get("args", {}).get("property_id")
                    if property_id:
                        state_updates["selected_property"] = {"property_id": property_id}
                        # Also update tour_details with property_id
                        if not state_updates["tour_details"]:
                            state_updates["tour_details"] = {}
                        state_updates["tour_details"]["property_id"] = property_id
        
        # If booking tour, update workflow stage
        if "book_tour_smart_tool" in tool_names or "get_tour_slots_tool" in tool_names:
            state_updates["workflow_stage"] = "booking"
    
    return state_updates

# ============================================
# Step 5: Routing Function
# ============================================
def route_after_agent(state: LaylaState) -> str:
    """Route after agent node - check if tools need to be called"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ============================================
# Step 6: Build the Graph
# ============================================
print("Building LangGraph...")

graph_builder = StateGraph(LaylaState)

# Add nodes
graph_builder.add_node("extract", extract_information_node)
graph_builder.add_node("agent", layla_agent_node)
graph_builder.add_node("tools", custom_tool_node)

# Add edges
graph_builder.add_edge(START, "extract")
graph_builder.add_edge("extract", "agent")
graph_builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "end": END
    }
)
graph_builder.add_edge("tools", "extract")  # Extract info after tools too

# Compile graph
graph = graph_builder.compile()

print("✓ Layla agent ready!")

# ============================================
# Step 7: Run Function
# ============================================
def run_layla(user_input: str, state: Optional[LaylaState] = None) -> LaylaState:
    """
    Run Layla agent with user input
    
    Args:
        user_input: User's message
        state: Optional previous state (for multi-turn conversations)
    
    Returns:
        Updated state
    """
    if state is None:
        state = {
            "messages": [],
            "selected_property": None,
            "search_results": None,
            "lead_info": {},  # Initialize as empty dict, not None
            "workflow_stage": "searching",
            "tour_details": {}  # Initialize as empty dict, not None
        }
    
    # Add user message
    user_message = HumanMessage(content=user_input)
    state["messages"].append(user_message)
    
    # Run graph
    final_state = None
    for event in graph.stream(state):
        for node_name, node_state in event.items():
            final_state = node_state
    
    return final_state if final_state else state

# ============================================
# Main Execution (for testing)
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Layla - The Leasing Agent")
    print("="*70 + "\n")
    
    # Example conversation
    state = None
    
    queries = [
        "Do you have properties with gym?",
        "Tell me more about the first one",
        "I want to book a tour",
    ]
    
    for query in queries:
        print(f"User: {query}\n")
        state = run_layla(query, state)
        
        # Get last AI message
        if state and state.get("messages"):
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "content"):
                print(f"Layla: {last_msg.content}\n")
        
        print("-"*70 + "\n")

