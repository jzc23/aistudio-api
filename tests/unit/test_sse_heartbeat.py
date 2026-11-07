"""SSE 心跳保活包装器测试。"""

from __future__ import annotations

import asyncio

import pytest

from aistudio_api.api.sse_heartbeat import heartbeat_sse


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


async def _slow_source(delays: list[float], items: list[str]):
    for delay, item in zip(delays, items, strict=True):
        if delay:
            await asyncio.sleep(delay)
        yield item


@pytest.mark.asyncio
async def test_heartbeat_inserted_on_silence():
    # 两个数据块之间静默 0.12s,interval=0.04 → 应插入 2 条心跳
    out = await _collect(
        heartbeat_sse(
            _slow_source([0.0, 0.12, 0.0], ["data: 1\n\n", "data: 2\n\n", "data: 3\n\n"]),
            interval=0.04,
        )
    )
    assert out[0] == "data: 1\n\n"
    assert out[-1] == "data: 3\n\n"
    heartbeats = [c for c in out if c.startswith(":")]
    assert len(heartbeats) == 2
    assert all(c == ": keep-alive\n\n" for c in heartbeats)
    # 顺序保持:1 → 心跳 → 心跳 → 2 → 3
    assert out.index("data: 2\n\n") > out.index("data: 1\n\n")


@pytest.mark.asyncio
async def test_no_heartbeat_when_flowing():
    out = await _collect(
        heartbeat_sse(
            _slow_source([0, 0], ["data: a\n\n", "data: b\n\n"]),
            interval=0.5,
        )
    )
    assert out == ["data: a\n\n", "data: b\n\n"]


@pytest.mark.asyncio
async def test_exception_propagates():
    async def boom():
        yield "data: 1\n\n"
        raise RuntimeError("upstream died")

    with pytest.raises(RuntimeError, match="upstream died"):
        await _collect(heartbeat_sse(boom(), interval=0.05))


@pytest.mark.asyncio
async def test_zero_interval_passthrough():
    out = await _collect(
        heartbeat_sse(_slow_source([0.05, 0], ["a", "b"]), interval=0)
    )
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_empty_source():
    async def empty():
        return
        yield  # pragma: no cover

    assert await _collect(heartbeat_sse(empty(), interval=0.05)) == []
