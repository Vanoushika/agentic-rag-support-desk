from app.kb import chunk_text


def test_short_text_stays_one_chunk():
    text = "# Title\n\nA short help article with one paragraph of content."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "short help article" in chunks[0]


def test_long_paragraph_is_split_with_no_content_lost():
    text = " ".join(f"word{i}" for i in range(400))  # ~2,300 chars
    chunks = chunk_text(text, max_chars=300, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # First and last tokens survive the split.
    assert "word0" in chunks[0]
    assert "word399" in chunks[-1]


def test_paragraphs_are_packed_not_mixed():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, max_chars=40)
    assert len(chunks) >= 2
    assert chunks[0].startswith("First paragraph")


def test_blank_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("\n\n  \n") == []
