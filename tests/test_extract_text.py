from src.llm.gemini import extract_text


def test_plain_string_passthrough():
    assert extract_text("hello world") == "hello world"


def test_list_of_text_blocks():
    content = [{"type": "text", "text": "hello", "extras": {"signature": "abc123"}}]
    assert extract_text(content) == "hello"


def test_list_of_multiple_text_blocks_joined():
    content = [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
    assert extract_text(content) == "part one\npart two"


def test_thinking_blocks_are_skipped():
    content = [
        {"type": "thinking", "text": "internal reasoning, not for the user"},
        {"type": "text", "text": "final answer"},
    ]
    assert extract_text(content) == "final answer"


def test_list_of_plain_strings():
    assert extract_text(["hello", "world"]) == "hello\nworld"


def test_unrecognized_block_falls_back_to_str():
    content = [{"type": "tool_use", "id": "1", "input": {}}]
    result = extract_text(content)
    assert "tool_use" in result


def test_non_string_non_list_falls_back_to_str():
    assert extract_text(123) == "123"


def test_empty_list_returns_empty_string():
    assert extract_text([]) == ""
