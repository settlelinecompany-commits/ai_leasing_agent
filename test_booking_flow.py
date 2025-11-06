"""
Test script for Layla booking flow
Simulates the conversation from the user's example
"""
from layla_agent import run_layla

def print_state_info(state, turn_num):
    """Print current state information"""
    print(f"\n{'='*70}")
    print(f"TURN {turn_num} - STATE INFO")
    print(f"{'='*70}")
    if state:
        lead_info = state.get("lead_info") or {}
        tour_details = state.get("tour_details") or {}
        selected_property = state.get("selected_property") or {}
        
        print(f"Lead Info: {lead_info}")
        print(f"Tour Details: {tour_details}")
        print(f"Selected Property: {selected_property}")
        print(f"Workflow Stage: {state.get('workflow_stage', 'N/A')}")
    print(f"{'='*70}\n")

def print_response(state, turn_num):
    """Print Layla's response"""
    if state and state.get("messages"):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content"):
            print(f"Layla: {last_msg.content}\n")
        elif hasattr(last_msg, "tool_calls"):
            print(f"Layla is using tools: {[tc.get('name') for tc in last_msg.tool_calls]}\n")

def test_booking_flow():
    """Test the complete booking flow"""
    print("\n" + "="*70)
    print("TEST: Complete Booking Flow")
    print("="*70 + "\n")
    
    state = None
    
    # Turn 1: Search for properties
    print("User: looking for 1 bedroom with a gym")
    state = run_layla("looking for 1 bedroom with a gym", state)
    print_response(state, 1)
    print_state_info(state, 1)
    
    # Turn 2: Select property #5
    print("User: let's do #5")
    state = run_layla("let's do #5", state)
    print_response(state, 2)
    print_state_info(state, 2)
    
    # Turn 3: Ask about availability
    print("User: sure - is it available tomorrow?")
    state = run_layla("sure - is it available tomorrow?", state)
    print_response(state, 3)
    print_state_info(state, 3)
    
    # Turn 4: Ask for next available slot
    print("User: when's the next slot available?")
    state = run_layla("when's the next slot available?", state)
    print_response(state, 4)
    print_state_info(state, 4)
    
    # Turn 5: Select date
    print("User: let's do november 7th instead")
    state = run_layla("let's do november 7th instead", state)
    print_response(state, 5)
    print_state_info(state, 5)
    
    # Turn 6: Select time
    print("User: 10am")
    state = run_layla("10am", state)
    print_response(state, 6)
    print_state_info(state, 6)
    
    # Turn 7: Provide name and phone (comma-separated)
    print("User: yep the same property, laksh, 3122037041")
    state = run_layla("yep the same property, laksh, 3122037041", state)
    print_response(state, 7)
    print_state_info(state, 7)
    
    # Turn 8: Confirm booking
    print("User: yes")
    state = run_layla("yes", state)
    print_response(state, 8)
    print_state_info(state, 8)
    
    # Verify final state
    print("\n" + "="*70)
    print("FINAL STATE VERIFICATION")
    print("="*70)
    
    if state:
        lead_info = state.get("lead_info") or {}
        tour_details = state.get("tour_details") or {}
        selected_property = state.get("selected_property") or {}
        
        print(f"\n✓ Lead Info:")
        print(f"  - Name: {lead_info.get('name', 'MISSING')}")
        print(f"  - Phone: {lead_info.get('phone', 'MISSING')}")
        
        print(f"\n✓ Tour Details:")
        print(f"  - Property ID: {tour_details.get('property_id', 'MISSING')}")
        print(f"  - Date: {tour_details.get('date', 'MISSING')}")
        print(f"  - Time: {tour_details.get('time', 'MISSING')}")
        
        print(f"\n✓ Selected Property:")
        print(f"  - Property ID: {selected_property.get('property_id', 'MISSING')}")
        
        # Check if all required info is present
        has_all_info = (
            lead_info.get("name") and
            lead_info.get("phone") and
            tour_details.get("date") and
            tour_details.get("time") and
            (tour_details.get("property_id") or selected_property.get("property_id"))
        )
        
        print(f"\n✓ All Required Info Present: {has_all_info}")
        
        if has_all_info:
            print("\n✅ SUCCESS: All information is available for booking!")
        else:
            print("\n❌ FAILURE: Missing required information for booking")
            missing = []
            if not lead_info.get("name"):
                missing.append("name")
            if not lead_info.get("phone"):
                missing.append("phone")
            if not tour_details.get("date"):
                missing.append("date")
            if not tour_details.get("time"):
                missing.append("time")
            if not tour_details.get("property_id") and not selected_property.get("property_id"):
                missing.append("property_id")
            print(f"   Missing: {', '.join(missing)}")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    test_booking_flow()


