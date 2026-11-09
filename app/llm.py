"""LLM providers behind a small interface.

:class:`RuleBasedLLM` is the deterministic default: it classifies tickets
with transparent keyword rules and drafts responses by grounding a short
template in the top retrieved knowledge-base chunk. It needs no API key,
which keeps tests and the eval fully offline and reproducible.

If ``OPENAI_API_KEY`` is set, :func:`get_llm` tries to return a
LangChain-backed chat model wrapper and falls back to the rule-based
implementation when that is unavailable.
"""

from __future__ import annotations

import os
from typing import Dict, List, Protocol


CATEGORIES = (
    "billing",
    "account_access",
    "subscription",
    "technical_issue",
    "order_status",
    "general",
)

# Ordered (category, keywords) rules. First category with a keyword hit wins;
# "uber eats"-style overlaps are avoided by keeping keywords specific.
_RULES: List[tuple[str, tuple[str, ...]]] = [
    (
        "account_access",
        (
            "password",
            "log in",
            "login",
            "sign in",
            "locked out",
            "two-factor",
            "two factor",
            "2fa",
            "authenticator",
            "verification code",
        ),
    ),
    (
        "billing",
        (
            "refund",
            "charged",
            "charge",
            "invoice",
            "billing",
            "payment",
            "receipt",
        ),
    ),
    (
        "subscription",
        ("subscription", "upgrade", "downgrade", "plan", "cancel my"),
    ),
    (
        "order_status",
        ("order", "delivery", "shipping", "tracking", "package", "arrive"),
    ),
    (
        "technical_issue",
        (
            "error",
            "crash",
            "bug",
            "not working",
            "broken",
            "freeze",
            "app won",
        ),
    ),
]


class LLM(Protocol):
    """Minimal chat-model surface the workflow depends on."""

    def classify(self, subject: str, body: str) -> str:
        ...

    def draft(
        self,
        subject: str,
        body: str,
        category: str,
        chunks: List[Dict],
    ) -> str:
        ...


class RuleBasedLLM:
    """Deterministic stand-in used for offline runs, tests and eval."""

    def classify(self, subject: str, body: str) -> str:
        text = f"{subject} {body}".lower()
        for category, keywords in _RULES:
            if any(keyword in text for keyword in keywords):
                return category
        return "general"

    def draft(
        self,
        subject: str,
        body: str,
        category: str,
        chunks: List[Dict],
    ) -> str:
        lines = [
            "Hi there,",
            "",
            f"Thanks for contacting support about \u201c{subject}\u201d. "
            "Based on our help centre, here is what should help:",
            "",
        ]
        if chunks:
            lines.append(chunks[0]["text"].strip())
            lines.append("")
        lines.append(
            "If that doesn't resolve it, reply to this ticket and a "
            "support agent will pick it up from here."
        )
        if chunks:
            sources = ", ".join(sorted({c["source"] for c in chunks}))
            lines.append("")
            lines.append(f"Sources: {sources}")
        return "\n".join(lines)


class ChatLLM:
    """LangChain chat-model wrapper (lazy import, optional)."""

    def __init__(self, model: str | None = None) -> None:
        from langchain_openai import ChatOpenAI

        self._impl = ChatOpenAI(
            model=model
            or os.environ.get("SUPPORT_DESK_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    def classify(self, subject: str, body: str) -> str:
        prompt = (
            "Classify this support ticket into exactly one of: "
            + ", ".join(CATEGORIES)
            + f".\nSubject: {subject}\nBody: {body}\nReply with the label only."
        )
        answer = self._impl.invoke(prompt).content.strip().lower()
        return answer if answer in CATEGORIES else "general"

    def draft(
        self,
        subject: str,
        body: str,
        category: str,
        chunks: List[Dict],
    ) -> str:
        context = "\n\n".join(c["text"] for c in chunks)
        prompt = (
            "You are a support agent. Draft a helpful, concise reply to "
            "the ticket using only the help-centre context below. Do not "
            "invent policies that are not in the context.\n\n"
            f"Context:\n{context}\n\nTicket subject: {subject}\n"
            f"Ticket body: {body}\nCategory: {category}"
        )
        return self._impl.invoke(prompt).content.strip()


def get_llm() -> LLM:
    """Return the best available LLM for this environment."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return ChatLLM()
        except Exception:
            pass
    return RuleBasedLLM()
