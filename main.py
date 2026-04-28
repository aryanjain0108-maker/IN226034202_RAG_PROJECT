import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Initialize the EXACT same embeddings used in ingest.py
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 2. Connect to the existing database folder
# This makes 'vectorstore' available to your nodes
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
# Define State
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    review_required: bool

# Initialize LLM (Free via Groq)
llm = ChatGroq(model="llama-3.1-8b-instant")

# Node 1: Retrieval (Simulation for brevity)
def retrieve_node(state: AgentState):
    question = state["question"]
    
    # Use the retriever interface (standard LangChain practice)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 1. Multi-Query Logic (as we discussed)
    # ... your multi-query code ...
    
    # 2. Get documents
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"context": context}

def retrieve_node(state):
    question = state["question"]
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    # 1. Multi-Query Generation
    # We ask the LLM to rewrite the question in different ways
    prompt = f"Generate 3 different versions of this question to improve document retrieval: {question}"
    msg = llm.invoke(prompt)
    queries = msg.content.split("\n")
    queries.append(question) # Keep the original too
    
    # 2. Perform Retrieval for all queries
    all_docs = []
    for q in queries:
        # Assuming 'vectorstore' is initialized globally
        docs = vectorstore.similarity_search(q, k=2) 
        all_docs.extend([d.page_content for d in docs])
    
    # 3. Deduplicate (remove identical chunks)
    unique_context = "\n---\n".join(list(set(all_docs)))
    
    return {"context": unique_context}
    # In a full app, query your vectorstore here
    return {"context": "Retrieved info from PDF..."}

# Node 2: Assistant
def assistant_node(state: AgentState):

    prompt = f"""
    ROLE: You are Aryan, a live Customer Support Agent at NovaTech. 
    TASK: Respond to the user's message directly. Do NOT write an email template. 
    Do NOT act as the user. Speak AS Aryan.

    IDENTITY: My name is Aryan.
    
    CONTEXT from Knowledge Base:
    {state['context']}

    USER'S MESSAGE:
    {state['question']}

    ARYAN'S RESPONSE:"""
    
    response = llm.invoke(prompt)
    
    # Check for escalation triggers
    review = any(word in state["question"].lower() for word in ["complaint", "escalate", "manager", "unhappy"])
    
    return {"answer": response.content, "review_required": review}
# Node 3: Human Review (The HITL Node)
def human_review_node(state: AgentState):
    print(f"\n--- HUMAN REVIEW REQUIRED ---")
    print(f"AI Draft: {state['answer']}")
    return state # The graph pauses here

# Build Graph
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("assistant", assistant_node)
builder.add_node("human_review", human_review_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "assistant")

# Logic: If review_required is True, go to human_review, else End
builder.add_conditional_edges(
    "assistant",
    lambda x: "human_review" if x["review_required"] else END
)
builder.add_edge("human_review", END)

# Compile with Checkpointer and Interrupt
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human_review"])
if __name__ == "__main__":
    # thread_id is essential for HITL to keep track of the specific conversation
    config = {"configurable": {"thread_id": "customer_session_001"}}
    
    print("\n==========================================")
    print("   RAG CUSTOMER SUPPORT ACTIVE  ")
    print("   Type 'exit' or 'quit' to stop       ")
    print("==========================================\n")

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Ending session. Goodbye!")
            break

        # 1. Start or Resume the graph
        # We pass the question to the state
        initial_input = {"question": user_input}
        
        # 2. Iterate through the graph events
        # stream_mode="updates" only shows us the output of each node
        for event in graph.stream(initial_input, config, stream_mode="updates"):
            for node, values in event.items():
                # This helps you see the graph 'thinking' in your terminal
                print(f"  [Log: Node '{node}' completed]")
                
                if "answer" in values:
                    print(f"\nAssistant: {values['answer']}")

        # 3. Handle the Human-in-the-Loop (HITL) pause
        snapshot = graph.get_state(config)
        if snapshot.next:
            print(f"\n--- PAUSED FOR REVIEW (Node: {snapshot.next}) ---")
            review = input("Admin, do you approve this response? (yes/no/edit): ")
            
            if review.lower() == "yes":
                # Resume the graph with no changes
                for event in graph.stream(None, config):
                    pass 
            elif review.lower() == "edit":
                new_answer = input("Enter the corrected response: ")
                # Update the state manually before resuming
                graph.update_state(config, {"answer": new_answer})
                for event in graph.stream(None, config):
                    pass
            else:
                print("Response rejected. Please try rephrasing your question.")
