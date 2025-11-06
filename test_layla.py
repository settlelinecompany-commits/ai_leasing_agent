"""
Test script for Layla agent
Tests all scenarios from the plan
"""
from layla_agent import run_layla

def print_response(state, query_num):
    """Print the agent's response"""
    if state and state.get('messages'):
        last_msg = state['messages'][-1]
        if hasattr(last_msg, 'content'):
            print(f"Layla: {last_msg.content}\n")
        elif hasattr(last_msg, 'tool_calls'):
            print(f"Layla is using tools: {[tc.get('name') for tc in last_msg.tool_calls]}\n")

def test_basic_search():
    """Test 1: Basic search queries"""
    print("="*70)
    print("TEST 1: Basic Search Queries")
    print("="*70 + "\n")
    
    queries = [
        "Do you have properties with gym?",
        "Show me 2 bedroom apartments under 10k monthly",
    ]
    
    state = None
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        state = run_layla(query, state)
        print_response(state, i)
        print("-"*70 + "\n")

def test_context_references():
    """Test 2: Context references (selected property remembered)"""
    print("="*70)
    print("TEST 2: Context References")
    print("="*70 + "\n")
    
    state = None
    
    # First, search for properties
    print("Query 1: Do you have properties with gym?")
    state = run_layla("Do you have properties with gym?", state)
    print_response(state, 1)
    print("-"*70 + "\n")
    
    # Then reference "the first one"
    print("Query 2: Tell me more about the first one")
    state = run_layla("Tell me more about the first one", state)
    print_response(state, 2)
    print("-"*70 + "\n")
    
    # Check if selected_property is set
    if state and state.get('selected_property'):
        print(f"✓ Selected property remembered: {state['selected_property']}\n")
    else:
        print("⚠ Selected property not set in state\n")

def test_booking_flow():
    """Test 3: Multi-step booking flow"""
    print("="*70)
    print("TEST 3: Multi-Step Booking Flow")
    print("="*70 + "\n")
    
    state = None
    
    # Step 1: Search
    print("Step 1: Search for properties")
    print("Query: Show me 2 bedroom apartments under 10k monthly")
    state = run_layla("Show me 2 bedroom apartments under 10k monthly", state)
    print_response(state, 1)
    print("-"*70 + "\n")
    
    # Step 2: View property
    print("Step 2: View property details")
    print("Query: Tell me more about the first one")
    state = run_layla("Tell me more about the first one", state)
    print_response(state, 2)
    print("-"*70 + "\n")
    
    # Step 3: Get tour slots
    print("Step 3: Get available tour slots")
    print("Query: I want to book a tour")
    state = run_layla("I want to book a tour", state)
    print_response(state, 3)
    print("-"*70 + "\n")
    
    # Step 4: Book tour (if slots were shown)
    print("Step 4: Book a specific slot")
    print("Query: Book tomorrow at 10:00")
    state = run_layla("Book tomorrow at 10:00", state)
    print_response(state, 4)
    print("-"*70 + "\n")
    
    # Check workflow stage
    if state and state.get('workflow_stage'):
        print(f"✓ Workflow stage: {state['workflow_stage']}\n")
    if state and state.get('tour_details'):
        print(f"✓ Tour details: {state['tour_details']}\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LAYLA AGENT TEST SUITE")
    print("="*70 + "\n")
    
    # Run all tests
    test_basic_search()
    test_context_references()
    test_booking_flow()
    
    print("="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")


