import time
import uuid
from typing import List, Dict, Any, Optional

class ReviewStatus:
    PENDING = "pending"
    FEEDBACK_REQUIRED = "feedback_required"
    BUG_RAISED = "bug_raised"
    RESOLVED = "resolved"
    APPROVED = "approved"
    REJECTED = "rejected"

class ReviewTicket:
    def __init__(self, requester: str, changes: Dict[str, Any], tier: str):
        self.id = str(uuid.uuid4())[:8]
        self.timestamp = time.time()
        self.requester = requester
        self.changes = changes
        self.tier = tier
        self.status = ReviewStatus.PENDING
        self.feedback: List[Dict[str, Any]] = []  # List of {author, text, type: feedback|bug}
        self.updated_at = self.timestamp

    def add_feedback(self, author: str, text: str, feedback_type: str = "feedback"):
        self.feedback.append({
            "timestamp": time.time(),
            "author": author,
            "text": text,
            "type": feedback_type
        })
        # Prevent manual feedback from downgrading APPROVED status
        if self.status == ReviewStatus.APPROVED:
            self.updated_at = time.time()
            return

        if feedback_type == "bug":
            self.status = ReviewStatus.BUG_RAISED
        else:
            self.status = ReviewStatus.FEEDBACK_REQUIRED
        self.updated_at = time.time()

    def resolve_bugs(self):
        # Only resolve if currently in bug state
        if self.status == ReviewStatus.BUG_RAISED:
            self.status = ReviewStatus.RESOLVED
            self.updated_at = time.time()

    def approve(self, approver: str):
        # Approved is a terminal state for the review workflow
        self.status = ReviewStatus.APPROVED
        self.updated_at = time.time()
        self.feedback.append({
            "timestamp": time.time(),
            "author": approver,
            "text": "Approved for final deployment",
            "type": "approval"
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "requester": self.requester,
            "changes": self.changes,
            "tier": self.tier,
            "status": self.status,
            "feedback": self.feedback,
            "updated_at": self.updated_at
        }

class ReviewManager:
    def __init__(self):
        self.tickets: Dict[str, ReviewTicket] = {}

    def create_ticket(self, requester: str, changes: Dict[str, Any], tier: str) -> str:
        ticket = ReviewTicket(requester, changes, tier)
        self.tickets[ticket.id] = ticket
        return ticket.id

    def get_ticket(self, ticket_id: str) -> Optional[ReviewTicket]:
        return self.tickets.get(ticket_id)

    def get_all_tickets(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tickets.values()]

    def submit_feedback(self, ticket_id: str, author: str, text: str, is_bug: bool = False):
        ticket = self.get_ticket(ticket_id)
        if ticket:
            ticket.add_feedback(author, text, "bug" if is_bug else "feedback")
            return True
        return False

    def resolve_ticket(self, ticket_id: str):
        ticket = self.get_ticket(ticket_id)
        if ticket:
            ticket.resolve_bugs()
            return True
        return False

    def approve_ticket(self, ticket_id: str, approver: str):
        ticket = self.get_ticket(ticket_id)
        if ticket:
            ticket.approve(approver)
            return True
        return False
