import httpx
import pytest
import respx
from httpx import Response

from local_model_studio.comfy_client import ComfyClient


@respx.mock
def test_is_available_returns_true_on_system_stats_200() -> None:
    respx.get("http://127.0.0.1:8188/system_stats").mock(return_value=Response(200))
    client = ComfyClient("http://127.0.0.1:8188")

    assert client.is_available() is True


@respx.mock
def test_is_available_returns_false_on_http_error() -> None:
    respx.get("http://127.0.0.1:8188/system_stats").mock(return_value=Response(503))
    client = ComfyClient("http://127.0.0.1:8188")

    assert client.is_available() is False


@respx.mock
def test_is_available_returns_false_on_connection_error() -> None:
    respx.get("http://127.0.0.1:8188/system_stats").mock(side_effect=httpx.ConnectError("no comfy"))
    client = ComfyClient("http://127.0.0.1:8188")

    assert client.is_available() is False


@respx.mock
def test_queue_prompt_returns_prompt_id() -> None:
    respx.post("http://127.0.0.1:8188/prompt").mock(return_value=Response(200, json={"prompt_id": "abc123"}))
    client = ComfyClient("http://127.0.0.1:8188")

    prompt_id = client.queue_prompt({"1": {"inputs": {}}})

    assert prompt_id == "abc123"


@respx.mock
def test_queue_prompt_posts_workflow_under_prompt_key() -> None:
    route = respx.post("http://127.0.0.1:8188/prompt").mock(
        return_value=Response(200, json={"prompt_id": "abc123"})
    )
    client = ComfyClient("http://127.0.0.1:8188")

    client.queue_prompt({"1": {"inputs": {"seed": 42}}})

    assert route.calls.last.request.read() == b'{"prompt":{"1":{"inputs":{"seed":42}}}}'


@respx.mock
def test_queue_prompt_raises_when_prompt_id_missing() -> None:
    respx.post("http://127.0.0.1:8188/prompt").mock(return_value=Response(200, json={}))
    client = ComfyClient("http://127.0.0.1:8188")

    with pytest.raises(RuntimeError, match="prompt_id"):
        client.queue_prompt({"1": {"inputs": {}}})


@respx.mock
def test_queue_prompt_raises_when_prompt_id_invalid() -> None:
    respx.post("http://127.0.0.1:8188/prompt").mock(return_value=Response(200, json={"prompt_id": 123}))
    client = ComfyClient("http://127.0.0.1:8188")

    with pytest.raises(RuntimeError, match="prompt_id"):
        client.queue_prompt({"1": {"inputs": {}}})
