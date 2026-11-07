"""SSE 心跳保活:长时间无数据时输出注释行,防止网关空闲超时断连(504)。

ATG 抗截断开启时,回答经合成函数通道传输,首段可见内容出现前可能有
数十秒静默;流中途的深度思考阶段同样静默。反向代理(雷池/Cloudflare/
Nginx,典型空闲超时 60s)会在静默期间掐断连接。

SSE 规范允许以 ``:`` 开头的注释行——所有事件源解析器(含 OpenAI/Google
SDK)都会忽略它,但字节照样流动,足以让网关保持连接。本包装器在底层
异步生成器静默超过 ``interval`` 秒时插入一条注释行,底层产生数据则立即
透传,不打乱原有事件顺序。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

DEFAULT_HEARTBEAT_INTERVAL = 15.0
HEARTBEAT_COMMENT = "keep-alive"


async def heartbeat_sse(
    source: AsyncGenerator[str, None],
    *,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    comment: str = HEARTBEAT_COMMENT,
) -> AsyncGenerator[str, None]:
    """包装 SSE 字符串生成器,静默超过 interval 秒时插入 ``: comment`` 注释行。

    实现要点:用独立 Task 承载底层的 ``__anext__()``,``asyncio.wait`` 只做
    超时观察而**不取消**底层读取——直接 ``wait_for`` 会在超时时把
    CancelledError 注入底层生成器(正在读 httpx 流),破坏上游连接。
    """
    if interval <= 0:
        async for chunk in source:
            yield chunk
        return

    ait = source.__aiter__()
    pending: asyncio.Task[str] | None = None
    try:
        pending = asyncio.ensure_future(ait.__anext__())
        while True:
            _done, still_pending = await asyncio.wait({pending}, timeout=interval)
            if still_pending:
                yield f": {comment}\n\n"
                continue
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                return
            yield chunk
            pending = asyncio.ensure_future(ait.__anext__())
    finally:
        if pending is not None and not pending.done():
            pending.cancel()


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL",
    "HEARTBEAT_COMMENT",
    "heartbeat_sse",
]
