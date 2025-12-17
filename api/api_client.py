"""Simple REST API client for http://127.0.0.1:8080

Provides a small, dependency-light wrapper around `requests` for
common HTTP verbs with JSON handling and basic error messages.

Example:
    from services.api_client import APIClient

    client = APIClient()  # defaults to http://127.0.0.1:8080
    resp = client.get('/')
    print(resp)

Install dependency:
    python -m pip install requests
"""
from typing import Any, Dict, Optional

import requests


class APIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        if not path:
            return self.base_url
        path = path.lstrip("/")
        return f"{self.base_url}/{path}"

    def request(
        self,
        method: str,
        path: str = "",
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        url = self._url(path)
        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {e}") from e

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Attempt to include response body for easier debugging
            body = None
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise RuntimeError(f"HTTP {resp.status_code}: {body}") from e

        # Try to return JSON, fall back to text
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def get(self, path: str = "", params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        return self.request("GET", path=path, params=params, headers=headers)

    def post(self, path: str = "", json: Optional[Any] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        return self.request("POST", path=path, json=json, headers=headers)

    def put(self, path: str = "", json: Optional[Any] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        return self.request("PUT", path=path, json=json, headers=headers)

    def delete(self, path: str = "", json: Optional[Any] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        return self.request("DELETE", path=path, json=json, headers=headers)

    def close(self) -> None:
        self.session.close()
