from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from work_assistant.main import create_app
from work_assistant.settings import Settings


@pytest.fixture
async def app_client(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Any]]:
    database_path = tmp_path / "product.db"
    settings = Settings(
        app_env="test",
        database_url=f"sqlite+aiosqlite:///{database_path}",
        checkpoint_database_url="postgresql://unused:unused@localhost/unused",
        model_mode="fake",
        fake_step_delay_seconds=0.08,
        sse_poll_interval_seconds=0.005,
        sse_keepalive_seconds=0.02,
        run_timeout_seconds=5,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await app.state.database.create_schema_for_tests()
        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app
