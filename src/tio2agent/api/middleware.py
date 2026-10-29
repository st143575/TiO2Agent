"""Reject oversized uploads before the body is read.

FastAPI parses the whole multipart body before calling the route handler, so the
streaming size check in FileStore only runs after the upload has been received.
This middleware rejects requests whose Content-Length is already too large.
(Chunked uploads without Content-Length are still caught by FileStore; in production,
also set a body limit on the reverse proxy, e.g. nginx ``client_max_body_size``.)
"""

from starlette.types import ASGIApp, Receive, Scope, Send

from tio2agent.api.errors import error_response


class UploadSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, path: str, max_body_bytes: int) -> None:
        self.app = app
        self.path = path.rstrip("/")
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"].rstrip("/") == self.path
        ):
            length = dict(scope["headers"]).get(b"content-length")
            if length is not None and length.isdigit() and int(length) > self.max_body_bytes:
                response = error_response(
                    413,
                    "FILE_TOO_LARGE",
                    "文件超过大小上限",
                    {"max_bytes": self.max_body_bytes},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
