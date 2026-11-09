import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.embeddings import HashingEmbedder
from app.llm import RuleBasedLLM
from app.pipeline import SupportDesk

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"
)


@pytest.fixture()
def desk() -> SupportDesk:
    """A SupportDesk wired to the deterministic offline providers."""
    return SupportDesk(
        llm=RuleBasedLLM(), embedder=HashingEmbedder(), docs_dir=DOCS_DIR
    )
