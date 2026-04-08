# Graph assembly + wiring

from langgraph.graph import StateGraph, END
from state import ReviewState
from nodes import (
    ingest_pr,
    agent_tool_loop,
    classify_severity,
    block_merge_recommendation,
    format_output,
)

def route_by_severity(state: ReviewState) -> str:
    """Conditional edge function - returns the name of the next node."""
    if state.get("route") == "critical":
        return "block_merge_recommendation"
    return "format_output"

def build_graph():
    graph = StateGraph(ReviewState)

    # Register all nodes
    graph.add_node("ingest_pr", ingest_pr)
    graph.add_node("agent_tool_loop", agent_tool_loop)
    graph.add_node("classify_severity", classify_severity)
    graph.add_node("block_merge_recommendation", block_merge_recommendation)
    graph.add_node("format_output", format_output)

    # Set entry point
    graph.set_entry_point("ingest_pr")

    # Linear edges
    graph.add_edge("ingest_pr", "agent_tool_loop")
    graph.add_edge("agent_tool_loop", "classify_severity")

    # Conditional branch after classification
    graph.add_conditional_edges(
        "classify_severity",
        route_by_severity,
        {
            "block_merge_recommendation": "block_merge_recommendation",
            "format_output": "format_output",
        }
    )

    # Both paths end at END
    graph.add_edge("block_merge_recommendation", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()

if __name__ == "__main__":
    app = build_graph()
    print(app.get_graph().draw_mermaid())