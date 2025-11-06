import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore  
from langchain_core.tools import create_retriever_tool, tool
from langchain_community.tools import BraveSearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSp   litter

# Load environment variables from .env file
load_dotenv()

# Configuration - Load from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Validate that all required keys are set
required_keys = {
    "QDRANT_URL": QDRANT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "BRAVE_API_KEY": BRAVE_API_KEY,
}

missing_keys = [key for key, value in required_keys.items() if not value]
if missing_keys:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_keys)}\n"
        "Please create a .env file with these variables. See .env.example for template."
    )

# Set OpenAI API key for langchain
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ============================================
# Step 1: Define the State
# ============================================
def add_messages(left, right):
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ============================================
# Step 2: Document Processing
# ============================================
print("Loading Hugging Face documentation...")
hf_loader = WebBaseLoader("https://huggingface.co/docs")
hf_docs = hf_loader.load()

print("Loading Transformers documentation...")
transformer_loader = WebBaseLoader("https://huggingface.co/docs/transformers")
transformer_docs = transformer_loader.load()

# Split documents  
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
hf_splits = text_splitter.split_documents(hf_docs)
transformer_splits = text_splitter.split_documents(transformer_docs)

print(f"Hugging Face docs: {len(hf_splits)} chunks")
print(f"Transformers docs: {len(transformer_splits)} chunks")

# ============================================
# Step 3: Create Retrievers
# ============================================
def create_retriever(collection_name, doc_splits):
    vectorstore = QdrantVectorStore.from_documents(
        doc_splits,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
    )
    return vectorstore.as_retriever()

print("\nCreating retrievers...")
hf_retriever = create_retriever("hf_documentation", hf_splits)
transformer_retriever = create_retriever("transformer_documentation", transformer_splits)
print("✓ Retrievers created!")

# ============================================
# Step 4: Build the Tools
# ============================================
print("\nBuilding tools...")

# Hugging Face retriever tool
hf_retriever_tool = create_retriever_tool(
    hf_retriever,
    "retriever_hugging_face_documentation",
    "Search and return information about hugging face documentation, it includes the guide and Python code.",
)

# Transformers retriever tool
transformer_retriever_tool = create_retriever_tool(
    transformer_retriever,
    "retriever_transformer",
    "Search and return information specifically about transformers library",
)

# Web search tool - exactly as in tutorial
@tool("web_search_tool")
def search_tool(query):
    """Search the web for information"""
    search = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 3})
    return search.run(query)

print("✓ Tools created!")

# ============================================
# Step 5: Set up LLM and Tools
# ============================================
print("\nSetting up LLM and tools...")
tools = [hf_retriever_tool, transformer_retriever_tool, search_tool]
tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)
print("✓ LLM and tools configured!")

# ============================================
# Step 6: Create Agent Node
# ============================================
def agent(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ============================================
# Step 7: Routing and Decision Making
# ============================================
def route(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    return END

# ============================================
# Step 8: Build the Graph
# ============================================
print("\nBuilding the graph...")
graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "agent",
    route,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "agent")
graph_builder.add_edge(START, "agent")

graph = graph_builder.compile()
print("✓ Graph built and compiled!")

# ============================================
# Step 9: Run the Agent
# ============================================
def run_agent(user_input: str):
    print(f"\n{'='*70}")
    print(f"Query: {user_input}")
    print(f"{'='*70}\n")
    
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for value in event.values():
            if "messages" in value:
                last_message = value["messages"][-1]
                if hasattr(last_message, "content"):
                    print("Assistant:", last_message.content)
                elif hasattr(last_message, "tool_calls"):
                    print(f"Using tools: {[tc['name'] for tc in last_message.tool_calls]}")

# ============================================
# Main Execution
# ============================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Agentic RAG with LangGraph - Ready!")
    print("="*70)
    
    # Test query from tutorial
    query = "In the Transformers library, are there any multilingual models?"
    run_agent(query)

