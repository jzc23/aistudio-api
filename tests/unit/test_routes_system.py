"""Tests for system and configuration routes."""

from __future__ import annotations

import httpx
import pytest

from aistudio_api.api.app import app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.asyncio
async def test_routing_state_endpoint(monkeypatch):
    class FakePipeline:
        def snapshot(self):
            return {
                "circuit": {"open": False, "failures": 0},
                "instances": {
                    "live_instances": 2,
                    "realms": [{"realm_id": "realm-a"}],
                    "limits": {"global_max": 2},
                    "planned_instances": 1,
                },
            }

    monkeypatch.setattr(
        "aistudio_api.application.routing_pipeline.get_routing_pipeline",
        lambda: FakePipeline(),
    )
    async with _client() as client:
        resp = await client.get("/routing/state")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["circuit"]["open"] is False
        assert payload["instances"]["planned_instances"] == 1


def test_surface_instance_snapshot_uses_unified_pool() -> None:
    from typing import Any, cast

    from aistudio_api.infrastructure.gateway.client import AIStudioClient

    class FakePool:
        def snapshot(self):
            return [
                {
                    "slot_id": "slot-1",
                    "account_id": "a",
                    "state": "busy",
                    "chat_ready": True,
                    "build_ready": False,
                }
            ]

        def page_counts(self):
            return {"chat": 1, "build": 0}

        @property
        def live_contexts(self):
            return 1

        @property
        def max_contexts(self):
            return 2

        def budget_snapshot(self):
            return {"global_max_pages": 2, "max_contexts": 2}

    client: Any = AIStudioClient.__new__(AIStudioClient)
    client._session = None
    client._instance_pool = FakePool()
    client._features_config = lambda: {
        "slots": [{"account_id": "a", "channel": "pg", "enabled": True}]
    }

    snapshot = cast("dict[str, object]", client.surface_instance_snapshot())
    contexts = cast("list[dict[str, object]]", snapshot["contexts"])
    assert contexts[0]["slot_id"] == "slot-1"
    assert snapshot["pages"] == {"chat": 1, "build": 0}
    assert snapshot["context_count"] == 1
    assert snapshot["planned_instances"] == 1
    assert "realms" not in snapshot
    assert snapshot["control_session"] == {"alive": False, "url": None}


@pytest.mark.asyncio
async def test_health_endpoint():
    async with _client() as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_auth_check_endpoint():
    async with _client() as client:
        resp = await client.get("/auth/check")
        assert resp.status_code == 200
        data = resp.json()
        assert "auth_enabled" in data


@pytest.mark.asyncio
async def test_spa_endpoints():
    import asyncio

    async with _client() as client:
        responses = await asyncio.gather(
            *[
                client.get(path, headers={"Accept": "text/html"})
                for path in ("/", "/login", "/accounts", "/settings")
            ]
        )
        for resp in responses:
            assert resp.status_code == 200
            assert "<!DOCTYPE html>" in resp.text
            assert "Accept" in resp.headers.get("Vary", "")
            assert "no-store" in resp.headers.get("Cache-Control", "")


@pytest.mark.asyncio
async def test_get_config_endpoint():
    async with _client() as client:
        resp = await client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "port" in data
        assert "browser_port" in data
        assert "yaml_content" in data


@pytest.mark.asyncio
async def test_update_config_yaml_validation():
    async with _client() as client:
        resp = await client.put(
            "/config/yaml", json={"yaml_content": "invalid: [yaml: broken"}
        )
        assert resp.status_code == 400

        resp_non_dict = await client.put(
            "/config/yaml", json={"yaml_content": "- item1\n- item2"}
        )
        assert resp_non_dict.status_code == 400


@pytest.mark.asyncio
async def test_rotation_status_and_clear_cooldown():
    from unittest.mock import MagicMock

    from aistudio_api.api.state import runtime_state
    from aistudio_api.application.account_rotator import AccountRotator

    mock_rotator = MagicMock(spec=AccountRotator)
    mock_rotator.get_all_stats.return_value = {"acc_1": {"requests": 1}}

    runtime_state.rotator = mock_rotator
    async with _client() as client:
        resp = await client.get("/rotation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "sticky"
        assert "accounts" in data

        resp_clear = await client.post(
            "/rotation/clear-cooldown", json={"account_id": "acc_1"}
        )
        assert resp_clear.status_code == 200
        mock_rotator.clear_cooldown.assert_called_once_with("acc_1", model=None)


@pytest.mark.asyncio
async def test_system_resources_endpoint(monkeypatch):
    from aistudio_api.api.state import runtime_state

    previous = runtime_state.instance_manager
    runtime_state.instance_manager = None
    try:
        async with _client() as client:
            resp = await client.get("/system/resources")
    finally:
        runtime_state.instance_manager = previous
    assert resp.status_code == 200
    payload = resp.json()
    assert "cpu_percent" in payload
    assert "memory_mb" in payload
    assert "instances" in payload


@pytest.mark.asyncio
async def test_config_yaml_save_preserves_concurrent_api_keys(monkeypatch, tmp_path):
    """编辑器旧快照保存不得覆盖期间新增的 API Key（回归测试）。"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """# 测试配置
logging:
  level: INFO
model_defaults:
  models: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AISTUDIO_CONFIG_FILE", str(config_file))
    from aistudio_api.infrastructure.gateway.model_defaults import (
        invalidate_config_cache,
    )

    invalidate_config_cache()

    async with _client() as client:
        created = await client.post(
            "/api-keys", json={"name": "persist-key", "key": "sk-test-fixed-001"}
        )
        assert created.status_code == 200

        # 模拟编辑器的旧快照：内容里完全没有 api_keys 段
        stale_snapshot = config_file.read_text(encoding="utf-8").split("api_keys:")[0]
        saved = await client.put("/config/yaml", json={"yaml_content": stale_snapshot})
        assert saved.status_code == 200
        assert "api_keys" in saved.json().get("preserved_keys", [])

        keys = await client.get("/api-keys")
        assert keys.status_code == 200
        items = keys.json()
        assert any(item["key"] == "sk-test-fixed-001" for item in items)

    assert "sk-test-fixed-001" in config_file.read_text(encoding="utf-8")
