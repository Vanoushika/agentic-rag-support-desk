"""FastAPI application for the Agentic RAG Support Desk.

Endpoints:
    GET  /health                          service health check
    POST /tickets                         submit a ticket (runs the
                                          workflow up to the approval gate)
    GET  /tickets/{ticket_id}             fetch one ticket
    POST /tickets/{ticket_id}/approve     approve the drafted reply
    POST /tickets/{ticket_id}/reject      reject the drafted reply
"""

from __future__ import annotations

from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import SupportDesk

load_dotenv()


class TicketCreate(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    customer_email: Optional[str] = None


class RejectRequest(BaseModel):
    reason: Optional[str] = None


class TicketResponse(BaseModel):
    id: str
    subject: str
    body: str
    category: str
    status: str
    draft_response: str
    final_response: Optional[str] = None
    citations: List[str] = []
    rejection_reason: Optional[str] = None


def create_app(desk: Optional[SupportDesk] = None) -> FastAPI:
    app = FastAPI(title="Agentic RAG Support Desk", version="0.1.0")
    desk = desk or SupportDesk()
    app.state.desk = desk

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/tickets", response_model=TicketResponse, status_code=201)
    def create_ticket(payload: TicketCreate) -> dict:
        return desk.submit(payload.subject, payload.body)

    @app.get("/tickets/{ticket_id}", response_model=TicketResponse)
    def get_ticket(ticket_id: str) -> dict:
        record = desk.get(ticket_id)
        if record is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return record

    @app.post("/tickets/{ticket_id}/approve", response_model=TicketResponse)
    def approve_ticket(ticket_id: str) -> dict:
        record = desk.approve(ticket_id)
        if record is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return record

    @app.post("/tickets/{ticket_id}/reject", response_model=TicketResponse)
    def reject_ticket(ticket_id: str, payload: RejectRequest | None = None) -> dict:
        record = desk.reject(ticket_id, payload.reason if payload else None)
        if record is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return record

    return app


app = create_app()
