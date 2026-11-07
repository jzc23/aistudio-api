"""Gemini-compatible application service handlers (protocol rendering).

Request orchestration lives in :mod:`aistudio_api.application.generation_core`;
this module renders protocol-neutral outcomes into Gemini wire shapes.

The retry / error-classification helpers are re-exported here for backwards
compatibility with existing callers and tests.
"""

from __future__ import annotations

import json

from fastapi.responses import StreamingResponse

from aistudio_api.api.response_models import (
    GeminiCandidateResponse,
    GeminiContentResponse,
    GeminiGenerateContentResponse,
)
from aistudio_api.api.responses import to_gemini_parts, to_gemini_usage_metadata
from aistudio_api.api.schemas import GeminiGenerateContentRequest
from aistudio_api.application.generation_core import (
    AUTH_403_MAX_RETRIES,
    WIRE_TO_GEMINI_FINISH_REASON,
    Auth403RetryManager,
    classify_gemini_error_payload,
    clean_upstream_error_message,
    generate,
    handle_attempt_exception,
    stream_generation,
    to_gemini_finish_reason,
)
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.utils.logger import get_logger

logger = get_logger("api.gemini")

__all__ = [
    "AUTH_403_MAX_RETRIES",
    "WIRE_TO_GEMINI_FINISH_REASON",
    "Auth403RetryManager",
    "classify_gemini_error_payload",
    "clean_upstream_error_message",
    "format_sse_error",
    "format_sse_event",
    "format_sse_usage",
    "handle_attempt_exception",
    "handle_gemini_generate_content",
    "to_gemini_finish_reason",
]


async def handle_gemini_generate_content(
    model_path: str,
    req: GeminiGenerateContentRequest,
    client: AIStudioClient,
    *,
    stream: bool,
):
    if stream:
        return _build_gemini_streaming_response(
            client=client, req=req, model_path=model_path
        )

    outcome = await generate(
        model_path=model_path, req=req, client=client, api_format="gemini"
    )
    output = outcome.output
    return GeminiGenerateContentResponse(
        candidates=[
            GeminiCandidateResponse(
                content=GeminiContentResponse(
                    parts=to_gemini_parts(
                        output.text,
                        function_calls=output.function_calls,
                        function_responses=output.function_responses,
                        thinking=output.thinking,
                        images=output.images,
                        reasoning_images=output.reasoning_images,
                    ),
                ),
                finishReason=to_gemini_finish_reason(
                    output.candidates[0].finish_reason
                    if output.candidates
                    else None
                ),
                index=0,
            )
        ],
        usageMetadata=to_gemini_usage_metadata(output.usage),
        modelVersion=outcome.model,
        responseId=output.response_id or None,
    )


def format_sse_event(
    event_type: str,
    text: object,
    *,
    model_version: str | None = None,
    response_id: str | None = None,
    thought_signature: str | None = None,
) -> str | None:
    """Format a single Gemini streaming event into SSE chunk string."""
    if not text:
        return None

    parts: list[dict[str, object]] = []
    if event_type == "body":
        p: dict[str, object] = {"text": str(text)}
        if thought_signature:
            p["thoughtSignature"] = thought_signature
        parts.append(p)
    elif event_type == "thinking":
        p = {"text": str(text), "thought": True}
        if thought_signature:
            p["thoughtSignature"] = thought_signature
        parts.append(p)
    elif event_type == "tool_calls":
        fc_list = text if isinstance(text, list) else []
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in to_gemini_parts("", function_calls=fc_list)
        ]
        if thought_signature and parts:
            parts[0]["thoughtSignature"] = thought_signature
    elif event_type == "images":
        img_list = text if isinstance(text, list) else []
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in to_gemini_parts("", images=img_list)
        ]
        if thought_signature and parts:
            parts[0]["thoughtSignature"] = thought_signature
    elif event_type == "reasoning_images":
        r_img_list = text if isinstance(text, list) else []
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in to_gemini_parts("", reasoning_images=r_img_list)
        ]
        if thought_signature and parts:
            parts[0]["thoughtSignature"] = thought_signature
    else:
        return None

    payload: dict[str, object] = {
        "candidates": [{"content": {"role": "model", "parts": parts}, "index": 0}]
    }
    if model_version:
        payload["modelVersion"] = model_version
    if response_id:
        payload["responseId"] = response_id
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def format_sse_usage(
    final_usage: dict[str, object] | None,
    *,
    finish_reason: str = "STOP",
    model_version: str | None = None,
    response_id: str | None = None,
) -> str:
    """Format final completion usage metadata into SSE chunk string."""
    effective_usage = final_usage or {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    usage_dict = to_gemini_usage_metadata(effective_usage).model_dump(
        mode="json", exclude_none=True
    )
    payload: dict[str, object] = {
        "candidates": [
            {
                "content": {"role": "model", "parts": []},
                "finishReason": finish_reason,
                "index": 0,
            }
        ],
        "usageMetadata": usage_dict,
    }
    if model_version:
        payload["modelVersion"] = model_version
    if response_id:
        payload["responseId"] = response_id
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def format_sse_error(exc: Exception) -> str:
    """Format exception into standard Gemini SSE error chunk."""
    code, msg, status_str = classify_gemini_error_payload(exc)
    return (
        "data: "
        + json.dumps(
            {
                "error": {
                    "code": code,
                    "message": msg,
                    "status": status_str,
                }
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )


def _build_gemini_streaming_response(
    *,
    client: AIStudioClient,
    req: GeminiGenerateContentRequest,
    model_path: str,
) -> StreamingResponse:
    async def sse_chunks():
        try:
            async for event in stream_generation(
                model_path=model_path,
                req=req,
                client=client,
                api_format="gemini",
            ):
                if event.kind == "usage":
                    yield format_sse_usage(
                        event.payload if isinstance(event.payload, dict) else None,
                        finish_reason=event.finish_reason or "STOP",
                        model_version=event.model,
                        response_id=event.response_id,
                    )
                    continue
                chunk = format_sse_event(
                    event.kind,
                    event.payload,
                    model_version=event.model,
                    response_id=event.response_id,
                    thought_signature=event.thought_signature,
                )
                if chunk:
                    yield chunk
        except Exception as exc:
            logger.warning("流式生成异常: model=%s, error=%s", model_path, exc)
            yield format_sse_error(exc)

    # 心跳保活:静默超过 15s 输出 SSE 注释行,防止网关空闲超时 504
    # (ATG 合成通道首段内容出现前常有长静默)
    async def stream_response():
        from aistudio_api.api.sse_heartbeat import heartbeat_sse

        async for chunk in heartbeat_sse(sse_chunks()):
            yield chunk

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
