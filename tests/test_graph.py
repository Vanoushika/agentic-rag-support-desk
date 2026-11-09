def test_workflow_stops_at_approval_gate(desk):
    record = desk.submit(
        "Charged twice this month",
        "I was charged twice for my subscription and I want a refund.",
    )
    assert record["status"] == "awaiting_approval"
    assert record["category"] == "billing"
    assert record["draft_response"]
    assert "billing-refunds.md" in record["citations"]
    assert record["final_response"] is None


def test_approve_finalises_the_draft(desk):
    record = desk.submit(
        "Forgot my password",
        "I forgot my password and can't log in.",
    )
    approved = desk.approve(record["id"])
    assert approved["status"] == "finalised"
    assert approved["final_response"] == approved["draft_response"]


def test_reject_marks_ticket_rejected(desk):
    record = desk.submit(
        "Where is my order?",
        "My order hasn't arrived and tracking hasn't updated.",
    )
    rejected = desk.reject(record["id"], reason="needs a human agent")
    assert rejected["status"] == "rejected"
    assert rejected["rejection_reason"] == "needs a human agent"
    assert rejected["final_response"] is None


def test_unknown_ticket_returns_none(desk):
    assert desk.get("does-not-exist") is None
    assert desk.approve("does-not-exist") is None
    assert desk.reject("does-not-exist") is None
