"""Unit tests for the Anthropic API wrapper.

Every test drives LLMService._call_api through a stubbed httpx client, so
nothing here makes a network call or needs a real API key. Tests are
driven with asyncio.run() rather than pytest-asyncio to keep the suite
free of extra plugins.
"""

import asyncio

import httpx
import pytest

from services import llm_service
from services.llm_service import LLMService, LLMServiceError


def run(coro):
    return asyncio.run(coro)


def response(status_code, json_body=None, text=""):
    return httpx.Response(
        status_code,
        json=json_body if json_body is not None else None,
        text=None if json_body is not None else text,
        request=httpx.Request("POST", llm_service.API_URL),
    )


class StubClient:
    """Stands in for httpx.AsyncClient, replaying a scripted list of outcomes.

    Each entry is either an httpx.Response to return or an exception to
    raise. Calls beyond the script reuse the final entry.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, *args, **kwargs):
        outcome = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def stub(monkeypatch):
    """Install a StubClient and remove retry sleeps."""
    monkeypatch.setattr(llm_service, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm_service, "BACKOFF_SECONDS", 0)

    def install(script):
        client = StubClient(script)
        monkeypatch.setattr(llm_service.httpx, "AsyncClient", client)
        return client

    return install


OK_BODY = {"content": [{"type": "text", "text": "Sinus tachycardia."}]}


class TestSuccess:
    def test_returns_the_text_block(self, stub):
        stub([response(200, OK_BODY)])
        assert run(LLMService()._call_api([])) == "Sinus tachycardia."

    def test_joins_multiple_text_blocks(self, stub):
        stub([response(200, {"content": [
            {"type": "text", "text": "One. "},
            {"type": "text", "text": "Two."},
        ]})])
        assert run(LLMService()._call_api([])) == "One. Two."

    def test_ignores_non_text_blocks(self, stub):
        stub([response(200, {"content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Answer."},
        ]})])
        assert run(LLMService()._call_api([])) == "Answer."


class TestMissingKey:
    def test_absent_key_fails_before_any_request(self, monkeypatch):
        monkeypatch.setattr(llm_service, "ANTHROPIC_API_KEY", "")
        client = StubClient([response(200, OK_BODY)])
        monkeypatch.setattr(llm_service.httpx, "AsyncClient", client)

        with pytest.raises(LLMServiceError, match="ANTHROPIC_API_KEY"):
            run(LLMService()._call_api([]))
        assert client.calls == 0


class TestRetries:
    def test_rate_limit_is_retried_then_succeeds(self, stub):
        client = stub([response(429, {"error": {"message": "slow down"}}),
                       response(200, OK_BODY)])
        assert run(LLMService()._call_api([])) == "Sinus tachycardia."
        assert client.calls == 2

    def test_timeout_is_retried(self, stub):
        client = stub([httpx.ReadTimeout("too slow"), response(200, OK_BODY)])
        assert run(LLMService()._call_api([])) == "Sinus tachycardia."
        assert client.calls == 2

    def test_retry_budget_is_bounded(self, stub):
        client = stub([response(503, {"error": {"message": "overloaded"}})])
        with pytest.raises(LLMServiceError, match="after 3 attempts"):
            run(LLMService()._call_api([]))
        assert client.calls == llm_service.MAX_ATTEMPTS

    def test_client_errors_are_not_retried(self, stub):
        client = stub([response(401, {"error": {"message": "invalid x-api-key"}})])
        with pytest.raises(LLMServiceError, match="401"):
            run(LLMService()._call_api([]))
        assert client.calls == 1


class TestMalformedResponses:
    def test_missing_content_key_is_reported_clearly(self, stub):
        stub([response(200, {"unexpected": True})])
        with pytest.raises(LLMServiceError, match="unexpected response shape"):
            run(LLMService()._call_api([]))

    def test_non_json_body_is_reported_clearly(self, stub):
        stub([response(200, text="<html>502 Bad Gateway</html>")])
        with pytest.raises(LLMServiceError, match="unexpected response shape"):
            run(LLMService()._call_api([]))

    def test_empty_answer_is_rejected(self, stub):
        stub([response(200, {"content": [{"type": "text", "text": "   "}]})])
        with pytest.raises(LLMServiceError, match="empty answer"):
            run(LLMService()._call_api([]))

    def test_error_detail_falls_back_to_body_text(self, stub):
        stub([response(400, text="Bad Request: messages must not be empty")])
        with pytest.raises(LLMServiceError, match="messages must not be empty"):
            run(LLMService()._call_api([]))


class TestFormatContext:
    def test_renders_source_and_category_for_each_chunk(self):
        chunks = [
            {"source": "BNF", "category": "drugs", "content": "Warfarin...", "score": 0.9},
            {"source": "NICE", "category": "guidelines", "content": "CKD..."},
        ]
        rendered = LLMService()._format_context(chunks)
        assert "Source: BNF" in rendered
        assert "Category: guidelines" in rendered
        assert "Relevance: N/A" in rendered  # score missing on the second chunk

    def test_empty_chunk_list_renders_empty(self):
        assert LLMService()._format_context([]) == ""
