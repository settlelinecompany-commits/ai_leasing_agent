from layla_agent import run_layla

print("="*70)
print("Layla - Interactive Test")
print("="*70)
print("\nType your questions (or 'quit' to exit)\n")

state = None
while True:
    query = input("You: ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    if not query:
        continue
    
    state = run_layla(query, state)
    
    if state and state.get("messages"):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content"):
            print(f"\nLayla: {last_msg.content}\n")
        else:
            print(f"\nLayla: [Tool calls made]\n")
    
    print("-"*70 + "\n")