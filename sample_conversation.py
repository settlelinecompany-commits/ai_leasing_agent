"""
Sample Conversation - Showcasing Layla's Capabilities
A polished, realistic conversation demonstrating the agent's intelligence
"""
from layla_agent import run_layla

def print_conversation():
    """Print a polished sample conversation"""
    
    print("\n" + "="*80)
    print(" " * 20 + "LAYLA - AI LEASING AGENT")
    print(" " * 15 + "Sample Conversation Showcase")
    print("="*80 + "\n")
    
    state = None
    
    # Turn 1: Natural property search
    print("👤 User: Hi, I'm looking for a 2 bedroom apartment in Dubai Marina")
    state = run_layla("Hi, I'm looking for a 2 bedroom apartment in Dubai Marina", state)
    print(f"🤖 Layla: {state['messages'][-1].content}\n")
    print("-"*80 + "\n")
    
    # Turn 2: More specific search with filters
    print("👤 User: Actually, I need something under 80k yearly with a gym and pool")
    state = run_layla("Actually, I need something under 80k yearly with a gym and pool", state)
    print(f"🤖 Layla: {state['messages'][-1].content}\n")
    print("-"*80 + "\n")
    
    # Turn 3: Select property
    print("👤 User: Tell me more about the first one")
    state = run_layla("Tell me more about the first one", state)
    print(f"🤖 Layla: {state['messages'][-1].content}\n")
    print("-"*80 + "\n")
    
    # Turn 4: Check availability
    print("👤 User: Is it available for a tour this week?")
    state = run_layla("Is it available for a tour this week?", state)
    print(f"🤖 Layla: {state['messages'][-1].content}\n")
    print("-"*80 + "\n")
    
    # Turn 5: Select date
    print("👤 User: Let's do November 7th at 2pm")
    state = run_layla("Let's do November 7th at 2pm", state)
    print(f"🤖 Layla: {state['messages'][-1].content}\n")
    print("-"*80 + "\n")
    
    # Turn 6: Provide contact info
    print("👤 User: Sure, my name is Sarah Ahmed and my phone is 0501234567")
    state = run_layla("Sure, my name is Sarah Ahmed and my phone is 0501234567", state)
    print(f"🤖 Layla: {state['messages'][-1].content}\n")
    print("-"*80 + "\n")
    
    # Turn 7: Confirmation
    print("👤 User: Perfect, thanks!")
    state = run_layla("Perfect, thanks!", state)
    print(f"🤖 Layla: {state['messages'][-1].content}\n")
    print("="*80 + "\n")
    
    # Show final state summary
    print("📊 CONVERSATION SUMMARY")
    print("="*80)
    lead_info = state.get("lead_info") or {}
    tour_details = state.get("tour_details") or {}
    selected_property = state.get("selected_property") or {}
    
    print(f"\n✅ Lead Information Captured:")
    print(f"   • Name: {lead_info.get('name', 'N/A')}")
    print(f"   • Phone: {lead_info.get('phone', 'N/A')}")
    
    print(f"\n✅ Tour Details:")
    print(f"   • Property ID: {tour_details.get('property_id', 'N/A')}")
    print(f"   • Date: {tour_details.get('date', 'N/A')}")
    print(f"   • Time: {tour_details.get('time', 'N/A')}")
    
    print(f"\n✅ Selected Property:")
    print(f"   • Property ID: {selected_property.get('property_id', 'N/A')}")
    
    print(f"\n✅ Workflow Stage: {state.get('workflow_stage', 'N/A')}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print_conversation()


