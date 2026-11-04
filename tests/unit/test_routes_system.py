"""Tests for system and configuration routes."""

from __future__ import annotations

import httpx
import pytest
import yaml

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


@pytest.mark.asyncio
async def test_config_yaml_save_three_way_logging_merge(monkeypatch, tmp_path):
    """日志段三方合并：未动快照保留外部修改；主动编辑尊重提交值；无基线不保护。"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "logging:\n  level: INFO\nmodel_defaults:\n  models: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AISTUDIO_CONFIG_FILE", str(config_file))
    from aistudio_api.infrastructure.gateway.model_defaults import (
        invalidate_config_cache,
    )

    invalidate_config_cache()
    baseline = config_file.read_text(encoding="utf-8")

    async with _client() as client:
        # 外部入口修改日志（日志设置卡片走的就是这个接口）
        resp = await client.put("/config/logging", json={"level": "DEBUG"})
        assert resp.status_code == 200

        # 场景1: 编辑器快照未动 logging（提交值 == 基线）→ 保留磁盘上的 DEBUG
        saved = await client.put(
            "/config/yaml",
            json={"yaml_content": baseline, "base_yaml_content": baseline},
        )
        assert saved.status_code == 200
        assert "logging" in saved.json()["preserved_keys"]
        assert "DEBUG" in config_file.read_text(encoding="utf-8")

        # 场景2: 用户在编辑器中主动把 logging 改为 WARNING → 尊重提交值
        edited = baseline.replace("level: INFO", "level: WARNING")
        saved2 = await client.put(
            "/config/yaml",
            json={"yaml_content": edited, "base_yaml_content": baseline},
        )
        assert saved2.status_code == 200
        assert "logging" not in saved2.json().get("preserved_keys", [])
        assert "WARNING" in config_file.read_text(encoding="utf-8")

        # 场景3: 外部再次修改后，不带基线（旧客户端）→ logging 保持旧行为不保护
        resp3 = await client.put("/config/logging", json={"level": "ERROR"})
        assert resp3.status_code == 200
        saved3 = await client.put("/config/yaml", json={"yaml_content": edited})
        assert saved3.status_code == 200
        assert "logging" not in saved3.json().get("preserved_keys", [])
        # 旧快照 WARNING 覆盖了磁盘上的 ERROR（未提供基线时的既有语义）
        assert "WARNING" in config_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_config_yaml_save_replaces_stale_api_keys_block_cleanly(
    monkeypatch, tmp_path
):
    """替换路径回归：快照携带旧 api_keys 块时，落盘文件必须合法且保留磁盘新值。

    该场景曾因正则块边界识别缺陷产出粘行的非法 YAML 导致容器启动崩溃。
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "logging:\n  level: INFO\napi_keys:\n- name: old\n  key: sk-old\n"
        "  created_at: '2026-01-01 00:00:00'\nmodel_defaults:\n  models: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AISTUDIO_CONFIG_FILE", str(config_file))
    from aistudio_api.infrastructure.gateway.model_defaults import (
        invalidate_config_cache,
    )

    invalidate_config_cache()
    stale_snapshot = config_file.read_text(encoding="utf-8")

    async with _client() as client:
        # 外部新增一个 Key（磁盘值变为 old + new）
        created = await client.post(
            "/api-keys", json={"name": "new", "key": "sk-new-002"}
        )
        assert created.status_code == 200

        # 编辑器旧快照（只含 old）保存 → 触发块替换路径
        saved = await client.put("/config/yaml", json={"yaml_content": stale_snapshot})
        assert saved.status_code == 200
        assert "api_keys" in saved.json()["preserved_keys"]

    final_text = config_file.read_text(encoding="utf-8")
    # 落盘文件必须可完整解析（曾在此处产出非法 YAML）
    final_doc = yaml.safe_load(final_text)
    assert isinstance(final_doc, dict)
    keys = {item["key"] for item in final_doc["api_keys"]}
    assert keys == {"sk-old", "sk-new-002"}
    # 其他段落不受影响
    assert final_doc["logging"]["level"] == "INFO"
    assert isinstance(final_doc["model_defaults"], dict)
    # 无粘行：每个键独占一行
    assert "created_at: '2026'- name" not in final_text


@pytest.mark.asyncio
async def test_config_yaml_save_logging_preserve_keeps_file_parseable(
    monkeypatch, tmp_path
):
    """三方合并保留 logging 后，落盘文件必须可解析且等级为磁盘值。"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "logging:\n  level: INFO\nmodel_defaults:\n  models: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AISTUDIO_CONFIG_FILE", str(config_file))
    from aistudio_api.infrastructure.gateway.model_defaults import (
        invalidate_config_cache,
    )

    invalidate_config_cache()
    baseline = config_file.read_text(encoding="utf-8")

    async with _client() as client:
        await client.put("/config/logging", json={"level": "DEBUG"})
        saved = await client.put(
            "/config/yaml",
            json={"yaml_content": baseline, "base_yaml_content": baseline},
        )
        assert saved.status_code == 200

    final_doc = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert isinstance(final_doc, dict)
    assert final_doc["logging"]["level"] == "DEBUG"
    assert isinstance(final_doc["model_defaults"], dict)
    assert "DEBUGmodel_defaults" not in config_file.read_text(encoding="utf-8")
