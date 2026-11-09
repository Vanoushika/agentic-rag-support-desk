import os

from app.embeddings import HashingEmbedder
from app.kb import KnowledgeBase

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"
)


def _kb() -> KnowledgeBase:
    return KnowledgeBase.from_docs(DOCS_DIR, HashingEmbedder())


def test_password_query_retrieves_password_doc_first():
    results = _kb().search("I forgot my password and can't log in", k=3)
    assert results, "expected at least one result"
    assert results[0]["source"] == "password-reset.md"


def test_refund_query_retrieves_billing_doc_first():
    results = _kb().search("I was charged twice and want a refund", k=3)
    assert results[0]["source"] == "billing-refunds.md"


def test_search_respects_k_and_returns_scores():
    results = _kb().search("how do I track my order delivery", k=2)
    assert len(results) == 2
    assert all("score" in r for r in results)
    assert results[0]["source"] == "order-tracking.md"
