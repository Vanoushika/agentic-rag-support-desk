"""LangGraph workflow for ticket resolution.

Flow:

    classify -> retrieve -> draft -> [HUMAN APPROVAL GATE] -> finalise

The graph is compiled with ``interrupt_before=["finalise"]`` and an
in-memory checkpointer, so invoking it runs classification, retrieval
and drafting, then pauses. A human decision resumes (approve) or
abandons (reject) the run — the human-in-the-loop pattern used in
production support tooling so no AI-drafted reply is sent unreviewed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .kb import KnowledgeBase
from .llm import LLM


class TicketState(TypedDict, total=False):
    ticket_id: str
    subject: str
    body: str
    category: str
    chunks: List[Dict]
    citations: List[str]
    draft_response: str
    status: str
    final_response: Optional[str]


def build_graph(llm: LLM, kb: KnowledgeBase):
    """Compile the ticket-resolution graph with an approval interrupt."""

    def classify(state: TicketState) -> Dict:
        return {"category": llm.classify(state["subject"], state["body"])}

    def retrieve(state: TicketState) -> Dict:
        chunks = kb.search(f"{state['subject']} {state['body']}", k=3)
        return {
            "chunks": chunks,
            "citations": sorted({c["source"] for c in chunks}),
        }

    def draft(state: TicketState) -> Dict:
        text = llm.draft(
            state["subject"],
            state["body"],
            state.get("category", "general"),
            state.get("chunks", []),
        )
        return {"draft_response": text, "status": "awaiting_approval"}

    def finalise(state: TicketState) -> Dict:
        return {
            "final_response": state.get("draft_response", ""),
            "status": "finalised",
        }

    graph = StateGraph(TicketState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_node("finalise", finalise)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "finalise")
    graph.add_edge("finalise", END)
    return graph.compile(
        checkpointer=MemorySaver(), interrupt_before=["finalise"]
    )
