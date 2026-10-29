from fastapi import APIRouter

from tio2agent import __version__

router = APIRouter(tags=["health"])


@router.get("/health", summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
