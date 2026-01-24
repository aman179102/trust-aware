from __future__ import annotations

from fastapi import FastAPI

from app.api import router as api_router


def create_app() -> FastAPI:
    """Application factory used by both uvicorn and tests."""
    app = FastAPI(
        title="Trust-Aware AI Decision System",
        description=(
            "Local, explainable text classification with confidence-aware "
            "decisions and human-review fallbacks."
        ),
        version="0.1.0",
    )

    # All analysis endpoints live under the root path for simplicity.
    app.include_router(api_router, tags=["analysis"])

    return app


# FastAPI app instance for `uvicorn main:app`.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Convenience for local development: `python main.py`.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
