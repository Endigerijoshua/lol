import asyncio

import pytest
from httpx import Request, Response, TimeoutException

from app.services import osm


class ScriptedClient:
    """Fake httpx client whose per-URL responses are supplied by the test."""

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.requests: list[str] = []

    async def get(self, url, params=None, headers=None, timeout=None):
        self.requests.append(url)
        outcome = self.responses.get(url)
        if isinstance(outcome, Exception):
            raise outcome
        response = outcome
        response.raise_for_status()
        return response

    async def aclose(self):
        pass


def _json_response(content: dict) -> Response:
    import json

    response = Response(200, request=Request("GET", "http://mirror"))
    response._content = json.dumps(content).encode()
    return response


def _status_response(status: int) -> Response:
    response = Response(status, request=Request("GET", "http://mirror"))
    return response


def _bad_json_response_garbage() -> Response:
    response = Response(200, request=Request("GET", "http://mirror"))
    response._content = b"<html>not json at all"
    return response


def test_fetch_region_falls_back_from_timeout_to_next_mirror():
    first, second = osm.OVERPASS_URLS[0], osm.OVERPASS_URLS[1]
    client = ScriptedClient(
        {
            first: TimeoutException("timed out"),
            second: _json_response({"elements": [{"type": "way", "id": 1}]}),
        }
    )
    result = asyncio.run(osm.fetch_region(72.0, 15.0, 81.0, 22.5, "industrial", client, 90))
    assert result == [{"type": "way", "id": 1}]
    assert client.requests == [first, second]


def test_fetch_region_falls_back_from_http_500_to_next_mirror():
    first, second = osm.OVERPASS_URLS[0], osm.OVERPASS_URLS[1]
    client = ScriptedClient(
        {
            first: _status_response(500),
            second: _json_response({"elements": [{"type": "way", "id": 2}]}),
        }
    )
    result = asyncio.run(osm.fetch_region(72.0, 15.0, 81.0, 22.5, "industrial", client, 90))
    assert result == [{"type": "way", "id": 2}]
    assert client.requests == [first, second]


def test_fetch_region_falls_back_from_garbage_body_to_next_mirror():
    """A 200 response with a non-JSON body must not kill the whole tile fetch."""
    first, second = osm.OVERPASS_URLS[0], osm.OVERPASS_URLS[1]
    client = ScriptedClient(
        {
            first: _bad_json_response_garbage(),
            second: _json_response({"elements": [{"type": "way", "id": 3}]}),
        }
    )
    result = asyncio.run(osm.fetch_region(72.0, 15.0, 81.0, 22.5, "vegetation", client, 90))
    assert result == [{"type": "way", "id": 3}]
    assert client.requests == [first, second]


def test_fetch_region_raises_runtime_error_when_all_mirrors_fail():
    client = ScriptedClient({url: _status_response(503) for url in osm.OVERPASS_URLS})
    with pytest.raises(RuntimeError, match="Overpass unavailable"):
        asyncio.run(osm.fetch_region(72.0, 15.0, 81.0, 22.5, "industrial", client, 90))
    assert len(client.requests) == len(osm.OVERPASS_URLS)


def test_fetch_region_uses_per_request_timeout():
    first = osm.OVERPASS_URLS[0]

    class TimeoutCapturingClient(ScriptedClient):
        def __init__(self):
            super().__init__({first: _json_response({"elements": []})})
            self.timeouts = []

        async def get(self, url, params=None, headers=None, timeout=None):
            self.timeouts.append(timeout)
            return await super().get(url, params=params, headers=headers, timeout=timeout)

    client = TimeoutCapturingClient()
    asyncio.run(osm.fetch_region(72.0, 15.0, 81.0, 22.5, "industrial", client, 90))
    assert client.timeouts == [osm.OVERPASS_REQUEST_TIMEOUT_SECONDS]


def test_server_startup_does_not_import_overpass_network():
    """Importing the OSM service (and the app) must not perform any network IO.

    A hanging Overpass mirror can never slow down `uvicorn` boot because every
    fetch happens lazily on the first zone request, never at import time.
    """
    import time

    start = time.perf_counter()
    import app.main  # noqa: F401

    elapsed = time.perf_counter() - start
    assert elapsed < osm.OVERPASS_REQUEST_TIMEOUT_SECONDS
