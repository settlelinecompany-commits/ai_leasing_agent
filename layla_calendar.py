"""
Dummy Calendar Data for Layla - Tour Booking
Simple structure that can be easily replaced with real calendar API later
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Dummy calendar data structure
# Format: {property_id: {available_slots: [...], booked_slots: [...]}}
DUMMY_CALENDAR: Dict[str, Dict] = {}

def initialize_calendar_for_property(property_id: str):
    """Initialize calendar with default available slots for a property"""
    if property_id not in DUMMY_CALENDAR:
        # Generate available slots for next 7 days
        available_slots = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate slots: 10:00, 14:00, 16:00 for each day
        for day_offset in range(7):
            date = today + timedelta(days=day_offset)
            for hour in [10, 14, 16]:
                slot = {
                    "date": date.strftime("%Y-%m-%d"),
                    "time": f"{hour:02d}:00",
                    "available": True
                }
                available_slots.append(slot)
        
        DUMMY_CALENDAR[property_id] = {
            "available_slots": available_slots,
            "booked_slots": []
        }

def get_available_slots(property_id: str, date: Optional[str] = None) -> List[Dict]:
    """
    Get available tour slots for a property
    
    Args:
        property_id: Property ID (e.g., "rocky_001")
        date: Optional date filter (YYYY-MM-DD format)
    
    Returns:
        List of available slots: [{"date": "2024-01-15", "time": "10:00", "available": True}, ...]
    """
    # Initialize if not exists
    if property_id not in DUMMY_CALENDAR:
        initialize_calendar_for_property(property_id)
    
    calendar = DUMMY_CALENDAR[property_id]
    available = calendar["available_slots"]
    
    # Filter by date if provided
    if date:
        available = [slot for slot in available if slot["date"] == date and slot["available"]]
    else:
        available = [slot for slot in available if slot["available"]]
    
    return available

def check_availability(property_id: str, date: str, time: str) -> bool:
    """
    Check if a specific slot is available
    
    Args:
        property_id: Property ID
        date: Date (YYYY-MM-DD format)
        time: Time (HH:MM format)
    
    Returns:
        True if available, False otherwise
    """
    # Initialize if not exists
    if property_id not in DUMMY_CALENDAR:
        initialize_calendar_for_property(property_id)
    
    calendar = DUMMY_CALENDAR[property_id]
    
    # Check if slot exists and is available
    for slot in calendar["available_slots"]:
        if slot["date"] == date and slot["time"] == time:
            return slot["available"]
    
    return False

def book_slot(property_id: str, date: str, time: str, customer_name: str, customer_phone: str) -> Dict:
    """
    Book a tour slot
    
    Args:
        property_id: Property ID
        date: Date (YYYY-MM-DD format)
        time: Time (HH:MM format)
        customer_name: Customer name
        customer_phone: Customer phone number
    
    Returns:
        Booking confirmation dict or error dict
    """
    # Initialize if not exists
    if property_id not in DUMMY_CALENDAR:
        initialize_calendar_for_property(property_id)
    
    calendar = DUMMY_CALENDAR[property_id]
    
    # Find the slot
    slot_found = False
    for slot in calendar["available_slots"]:
        if slot["date"] == date and slot["time"] == time:
            if not slot["available"]:
                return {
                    "success": False,
                    "error": "Slot is already booked"
                }
            slot["available"] = False
            slot_found = True
            
            # Add to booked slots
            booking = {
                "date": date,
                "time": time,
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "booked_at": datetime.now().isoformat()
            }
            calendar["booked_slots"].append(booking)
            break
    
    if not slot_found:
        return {
            "success": False,
            "error": "Slot not found"
        }
    
    return {
        "success": True,
        "property_id": property_id,
        "date": date,
        "time": time,
        "customer_name": customer_name,
        "customer_phone": customer_phone,
        "confirmation_id": f"{property_id}_{date}_{time}".replace("-", "_").replace(":", "_")
    }

def get_booked_slots(property_id: str) -> List[Dict]:
    """Get all booked slots for a property"""
    if property_id not in DUMMY_CALENDAR:
        return []
    
    return DUMMY_CALENDAR[property_id]["booked_slots"]


