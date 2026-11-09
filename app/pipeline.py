"""SupportDesk: orchestrates the knowledge base, the LangGraph
workflow and the in-memory ticket store used by the API layer."""

from __future__ import annotations

import os
from typing import Dict, Optional
from uuid import uuid4

from .embeddings import Embedder, get_embedder
from .graph import build_graph
from .kb import KnowledgeBase
from .llm import LLM, get_llm

_DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"
)


class SupportDesk:
    def __init__(
        self,
        llm: Optional[LLM] = None,
        embedder: Optional[Embedder] = None,
        docs_dir: Optional[str] = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.embedder = embedder or get_embedder()
        self.kb = KnowledgeBase.from_docs(docs_dir or _DEFAULT_DOCS_DIR, self.embedder)
        self.graph = build_graph(self.llm, self.kb)
        self.tickets: Dict[str, Dict] = {}

    # -- internal helpers -------------------------------------------------
    def _config(self, ticket_id: str) -> Dict:
        return {"configurable": {"thread_id": ticket_id}}

    def _record_from_state(self, ticket_id: str, state: Dict) -> Dict:
        return {
            "id": ticket_id,
            "subject": state.get("subject", ""),
            "body": state.get("body", ""),
            "category": state.get("category", "general"),
            "status": state.get("status", "awaiting_approval"),
            "draft_response": state.get("draft_response", ""),
            "final_response": state.get("final_response"),
            "citations": state.get("citations", []),
            "rejection_reason": None,
        }

    # -- public API -------------------------------------------------------
    def submit(self, subject: str, body: str) -> Dict:
        """Run a new ticket through classify/retrieve/draft, then pause
        at the human approval gate."""
        ticket_id = uuid4().hex[:12]
        state = self.graph.invoke(
            {"ticket_id": ticket_id, "subject": subject, "body": body},
            self._config(ticket_id),
        )
        record = self._record_from_state(ticket_id, state)
        self.tickets[ticket_id] = record
        return record

    def get(self, ticket_id: str) -> Optional[Dict]:
        return self.tickets.get(ticket_id)

    def approve(self, ticket_id: str) -> Optional[Dict]:
        """Resume the paused graph so the drafted reply is finalised."""
        record = self.tickets.get(ticket_id)
        if record is None:
            return None
        if record["status"] != "awaiting_approval":
            return record
        self.graph.invoke(None, self._config(ticket_id))
        state = self.graph.get_state(self._config(ticket_id)).values
        record.update(self._record_from_state(ticket_id, state))
        self.tickets[ticket_id] = record
        return record

    def reject(self, ticket_id: str, reason: Optional[str] = None) -> Optional[Dict]:
        """Abandon the paused run and mark the ticket rejected."""
        record = self.tickets.get(ticket_id)
        if record is None:
            return None
        if record["status"] == "awaiting_approval":
            record["status"] = "rejected"
            record["rejection_reason"] = reason
            self.tickets[ticket_id] = record
        return record
