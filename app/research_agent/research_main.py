from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.4,
    max_output_tokens=8192,
    timeout=30,
    max_retries=2,)

class AgentState(TypedDict):
    objective: str
    doc_chunks: list[str]
    current_chunk_index: int
    gathered: list[str]
    refined: str
    final_output: str
    objective_definition: str
    plan: str

# State Schema
def initial_state(objective, chunks):
    return AgentState(
        objective=objective,
        doc_chunks=chunks,
        current_chunk_index=0,
        gathered=[],
        refined="",
        final_output="",
        objective_definition="",
        plan=""
    )

# DEFINE Node
def define_fn(state):
    objective = state["objective"]
    response = llm.invoke(
        f"You are an expert researcher. Define the scope of this research goal:\n\nObjective: {objective}"
    )
    return {**state, "objective_definition": response.content}

# PLAN Node
def plan_fn(state):
    response = llm.invoke(
        f"Based on this defined objective:\n\n{state['objective_definition']}\n\n"
        "Create a numbered step-by-step research plan."
    )
    return {**state, "plan": response.content}

# GATHER Node
def gather_fn(state):
    idx = state["current_chunk_index"]
    if idx >= len(state["doc_chunks"]):
        return state  # No more chunks

    chunk = state["doc_chunks"][idx]
    objective = state["objective_definition"]
    plan = state["plan"]

    # prompt = (
    #     f"Objective: {objective}\n\nPlan: {plan}\n\n"
    #     f"Given the following text from the document:\n\n{chunk}\n\n"
    #     "Extract only the insights, facts, data, or concepts relevant to the objective."
    # )

    prompt = (
        f"Objective: {objective}\n\nPlan: {plan}\n\n"
        f"Summarize this chunk into 3 concise bullet points that relate to the research objective:\n\n{chunk}"
    )

    result = llm.invoke(prompt)
    state["gathered"].append(result.content)
    state["current_chunk_index"] += 1
    return state

# Check if more gathering needed
def should_continue_gathering(state):
    return state["current_chunk_index"] < len(state["doc_chunks"])

# REFINE Node
def refine_fn(state):
    joined = "\n\n".join(state["gathered"])
    prompt = prompt = (
    f"You are a concise research assistant. Based on the following extracted notes:\n\n{joined}\n\n"
    "Please identify only the most **critical, relevant, and non-redundant** insights."
    " Summarize them in 3–5 concise bullet points under each section.\n"
    "Ensure clarity, relevance, and brevity. Ignore repeated or vague points."
)
    result = llm.invoke(prompt)
    return {**state, "refined": result.content}

# GENERATE Node
def generate_fn(state):
    refined = state["refined"]
    prompt = prompt = (
    f"Using the refined notes below, write a **short, impactful markdown report** (max 1000 words)."
    f"\nUse only headings and key bullet points. Avoid repetition. Focus on relevance to the original objective.\n\n{refined}"
)
    result = llm.invoke(prompt)
    return {**state, "final_output": result.content}

# === LangGraph Build ===
def build_research_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("DEFINE", RunnableLambda(define_fn))
    workflow.add_node("PLAN", RunnableLambda(plan_fn))
    workflow.add_node("GATHER", RunnableLambda(gather_fn))
    workflow.add_node("REFINE", RunnableLambda(refine_fn))
    workflow.add_node("GENERATE", RunnableLambda(generate_fn))

    workflow.set_entry_point("DEFINE")
    workflow.add_edge("DEFINE", "PLAN")
    workflow.add_edge("PLAN", "GATHER")

    # Loop GATHER until all chunks processed
    workflow.add_conditional_edges("GATHER", should_continue_gathering, {
        True: "GATHER",
        False: "REFINE"
    })

    workflow.add_edge("REFINE", "GENERATE")
    workflow.add_edge("GENERATE", END)

    return workflow.compile()
