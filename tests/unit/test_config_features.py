"""GET/PUT /config/features 功能开关接口测试。"""

from __future__ import annotations

import httpx
import pytest
import yaml
from fastapi import Depends, FastAPI

from aistudio_api.api.dependencies import require_web_auth
from aistudio_api.api.routes_system import (
    protected_router as system_protected_router,
)
from aistudio_api.infrastructure.gateway import model_defaults


@pytest.fixture()
def config_env(tmp_path, monkeypatch):
    """独立 config.yaml 环境,避免污染仓库配置缓存。"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "features": {
                    "atg": {"mode": "off", "empty_retries": 1},
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        model_defaults, "_resolve_config_path", lambda *_a, **_k: cfg
    )
    model_defaults.invalidate_config_cache()
    yield cfg
    model_defaults.invalidate_config_cache()


def _build_client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(
        system_protected_router, dependencies=[Depends(require_web_auth)]
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        trust_env=False,
    )


@pytest.mark.asyncio
async def test_get_features_returns_merged_defaults(config_env):
    async with _build_client() as client:
        r = await client.get("/config/features")
    assert r.status_code == 200
    features = r.json()["features"]
    # yaml 覆盖生效
    assert features["atg"]["mode"] == "off"
    assert features["atg"]["empty_retries"] == 1
    # 默认值补齐
    assert features["formats"]["openai"] is True
    assert features["atg"]["tool_prefix"] == "atg_emit_"


@pytest.mark.asyncio
async def test_put_features_deep_merges_and_persists(config_env):
    async with _build_client() as client:
        r = await client.put(
            "/config/features",
            json={
                "updates": {
                    "atg": {"mode": "prefer"},
                    "formats": {"openai": False},
                }
            },
        )
    assert r.status_code == 200
    features = r.json()["features"]
    assert features["atg"]["mode"] == "prefer"
    assert features["formats"]["openai"] is False
    # 未提供的键保留
    assert features["atg"]["empty_retries"] == 1
    # 持久化到 yaml 且缓存失效后可读
    persisted = yaml.safe_load(config_env.read_text(encoding="utf-8"))
    assert persisted["features"]["atg"]["mode"] == "prefer"
    assert (
        model_defaults.get_features_config(config_env)["formats"]["openai"] is False
    )


@pytest.mark.asyncio
async def test_put_features_rejects_empty_updates(config_env):
    async with _build_client() as client:
        r = await client.put("/config/features", json={"updates": {}})
    assert r.status_code == 400
