"""ASGI 生命周期绑定的流式响应。

上游 fcdc53e 语义:客户端断开(或首响应头即断)时,Starlette 不会执行尚未
启动的 body iterator 的 ``finally``;此 subclass 在真实 ASGI 调用结束时
显式 ``aclose()`` 迭代器并取消挂起的非流式生成任务,让 aclosing 清理链
(路由 -> 心跳包装 -> 渲染器 -> 传输层)同步释放上游连接。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TypeVar

from anyio import CancelScope
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

T = TypeVar("T")


async def close_generation_task(task: asyncio.Task[T]) -> None:
    """响应结束后等待后台任务释放资源;清理错误不得覆盖原响应错误。"""
    with CancelScope(shield=True):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class RequestStreamingResponse(StreamingResponse):
    """在真实 ASGI 生命周期结束时关闭迭代器与非流式生成任务。"""

    def __init__(
        self,
        content: AsyncGenerator[str | bytes, None],
        *,
        media_type: str,
        generation_task: asyncio.Task[T] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        merged = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        if headers:
            merged.update(headers)
        super().__init__(content, media_type=media_type, headers=merged)
        self._iterator = content
        self._generation_task = generation_task

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # 首响应头就断开时,尚未启动的 body iterator 不会执行自身 finally。
            with CancelScope(shield=True):
                try:
                    await self._iterator.aclose()
                finally:
                    if self._generation_task is not None:
                        await close_generation_task(self._generation_task)


__all__ = ["RequestStreamingResponse", "close_generation_task"]
