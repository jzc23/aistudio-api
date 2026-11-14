"""Gemini-compatible application service handlers (protocol rendering).

Request orchestration lives in :mod:`aistudio_api.application.generation_core`;
this module renders protocol-neutral outcomes into Gemini wire shapes.

The retry / error-classification helpers are re-exported here for backwards
compatibility with existing callers and tests.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from aistudio_api.api.response_models import (
    GeminiCandidateResponse,
    GeminiContentResponse,
    GeminiCountTokensResponse,
    GeminiCreateDecisionResponse,
    GeminiDecisionModalityUsage,
    GeminiDecisionUsageResponse,
    GeminiGenerateContentResponse,
)
from aistudio_api.api.responses import to_gemini_parts, to_gemini_usage_metadata
from aistudio_api.api.schemas import (
    GeminiCountTokensRequest,
    GeminiCreateDecisionRequest,
    GeminiGenerateContentRequest,
)
from aistudio_api.api.state import runtime_state
from aistudio_api.application.account_orchestrator import (
    MAX_RETRIES,
    ensure_active_account,
    record_rotator_event,
    try_switch_account,
)
from aistudio_api.application.chat_service import normalize_gemini_request
from aistudio_api.application.generation_core import (
    AUTH_403_MAX_RETRIES,
    SHORT_RATE_LIMIT_COOLDOWN_SECONDS,
    WIRE_TO_GEMINI_FINISH_REASON,
    Auth403RetryManager,
    active_account_cooldown_remaining,
    classify_gemini_error_payload,
    clean_upstream_error_message,
    generate,
    handle_attempt_exception,
    is_daily_quota_message,
    stream_generation,
    to_gemini_finish_reason,
)
from aistudio_api.domain.errors import AuthError, SessionExpiredError
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.session import SessionContext
from aistudio_api.infrastructure.utils.logger import get_logger

logger = get_logger("api.gemini")

__all__ = [
    "AUTH_403_MAX_RETRIES",
    "SHORT_RATE_LIMIT_COOLDOWN_SECONDS",
    "WIRE_TO_GEMINI_FINISH_REASON",
    "Auth403RetryManager",
    "active_account_cooldown_remaining",
    "classify_gemini_error_payload",
    "clean_upstream_error_message",
    "ensure_active_account",
    "format_sse_error",
    "format_sse_event",
    "format_sse_usage",
    "handle_attempt_exception",
    "handle_gemini_count_tokens",
    "handle_gemini_decisions",
    "handle_gemini_generate_content",
    "is_daily_quota_message",
    "record_rotator_event",
    "runtime_state",
    "to_gemini_finish_reason",
    "try_switch_account",
]


async def _do_generate_content(
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

    # 启动非流式生成任务并提供防 524 代理超时的保活机制 (NONSTREAM_KEEPALIVE)
    gen_task = asyncio.create_task(
        _do_generate_content(model_path, req, client, stream=False)
    )
    try:
        # 前 10 秒直接等待结果；绝大多数请求在此期间返回标准对象
        return await asyncio.wait_for(asyncio.shield(gen_task), timeout=10.0)
    except TimeoutError:
        logger.info(
            "非流式请求耗时超过 10s，激活 HTTP 换行保活流: model=%s", model_path
        )

        async def keepalive_stream():
            yield b"\n"
            try:
                while not gen_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(gen_task), timeout=10.0)
                        break
                    except TimeoutError:
                        yield b"\n"
                res = await gen_task
                yield res.model_dump_json(by_alias=True, exclude_none=True).encode(
                    "utf-8"
                )
            except Exception as exc:
                code, msg, status_str = classify_gemini_error_payload(exc)
                err_payload = {
                    "error": {
                        "code": code,
                        "message": msg,
                        "status": status_str,
                    }
                }
                yield json.dumps(err_payload, ensure_ascii=False).encode("utf-8")
            finally:
                if not gen_task.done():
                    gen_task.cancel()
                elif not gen_task.cancelled():
                    _ = gen_task.exception()

        return StreamingResponse(
            keepalive_stream(),
            media_type="application/json",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
        if thought_signature and parts and not parts[0].get("thoughtSignature"):
            parts[0]["thoughtSignature"] = thought_signature
    elif event_type == "images":
        img_list = text if isinstance(text, list) else []
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in to_gemini_parts("", images=img_list)
        ]
        if thought_signature and parts and not parts[0].get("thoughtSignature"):
            parts[0]["thoughtSignature"] = thought_signature
    elif event_type == "reasoning_images":
        r_img_list = text if isinstance(text, list) else []
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in to_gemini_parts("", reasoning_images=r_img_list)
        ]
        if thought_signature and parts and not parts[0].get("thoughtSignature"):
            parts[0]["thoughtSignature"] = thought_signature
    elif event_type == "thought_signature":
        parts = [{"thoughtSignature": str(text)}]
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


async def handle_gemini_decisions(
    req: GeminiCreateDecisionRequest,
    client: AIStudioClient,
) -> GeminiCreateDecisionResponse:
    last_error: Exception | None = None
    auth_manager = Auth403RetryManager()
    total_attempts = MAX_RETRIES + AUTH_403_MAX_RETRIES
    raw_model = req.model or "gemini-decision-1"
    norm_model = raw_model if raw_model.startswith("models/") else f"models/{raw_model}"
    questions_dict: dict[str, dict[str, object]] = {
        key: val.model_dump(mode="json", exclude_none=True)
        for key, val in req.questions.items()
    }

    for attempt in range(total_attempts):
        await ensure_active_account(attempt, model=norm_model)
        request_ctx: SessionContext | None = None
        try:
            logger.info(
                "Decisions 请求: model=%s, questions=%d, 尝试=%d/%d",
                norm_model,
                len(questions_dict),
                attempt + 1,
                total_attempts,
            )
            force_recreate = auth_manager.force_recreate_signer
            auth_manager.force_recreate_signer = False
            prepared = auth_manager.current_prepared
            if prepared is None:
                prepared = await client.prepare_decision_request(
                    model=norm_model,
                    raw_input=req.input,
                    questions=questions_dict,
                    temperature=req.temperature,
                    force_recreate_signer=force_recreate,
                )
            request_ctx = prepared.ctx
            auth_manager.current_prepared = prepared
            result = await client.create_decision(
                model=norm_model,
                prepared=prepared,
            )
            auth_manager.current_prepared = None

            raw_usage = result.get("usage")
            usage_map = raw_usage if isinstance(raw_usage, dict) else {}
            prompt_tokens = int(str(usage_map.get("total_input_tokens") or 0))
            total_tokens = int(str(usage_map.get("total_tokens") or prompt_tokens))
            rotator_usage: dict[str, object] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": total_tokens,
            }
            record_rotator_event("success", model=norm_model, usage=rotator_usage)
            runtime_state.record(
                norm_model,
                "success",
                usage=rotator_usage,
                finish_reason="STOP",
            )
            raw_answers = result.get("answers")
            answers = raw_answers if isinstance(raw_answers, dict) else {}
            raw_mods = usage_map.get("input_tokens_by_modality")
            modalities = [
                GeminiDecisionModalityUsage(
                    modality=str(item.get("modality") or "text"),
                    tokens=int(str(item.get("tokens") or 0)),
                )
                for item in (raw_mods if isinstance(raw_mods, list) else [])
                if isinstance(item, dict)
            ]
            return GeminiCreateDecisionResponse(
                answers=answers,
                usage=GeminiDecisionUsageResponse(
                    total_input_tokens=prompt_tokens,
                    input_tokens_by_modality=modalities,
                    total_output_tokens=0,
                    total_tokens=total_tokens,
                ),
            )
        except Exception as exc:
            last_error = exc
            if isinstance(exc, AuthError) and not isinstance(exc, SessionExpiredError):
                if await auth_manager.handle_403(exc, model=norm_model, client=client):
                    continue
                raise
            if await handle_attempt_exception(
                exc,
                attempt=attempt,
                model_path=norm_model,
                normalized_model=norm_model,
                request_ctx=request_ctx,
            ):
                auth_manager.current_prepared = None
                continue
            raise

    raise HTTPException(
        429,
        detail={
            "message": f"All accounts reached daily quota for model '{norm_model}'. Quota resets at 00:00 PST.",
            "type": "rate_limit_exceeded",
        },
    ) from last_error


async def handle_gemini_count_tokens(
    model_path: str,
    req: GeminiCountTokensRequest,
    client: AIStudioClient,
) -> GeminiCountTokensResponse:
    if req.generateContentRequest is not None:
        effective_req = req.generateContentRequest
    elif req.contents is not None:
        effective_req = GeminiGenerateContentRequest(
            contents=req.contents,
            systemInstruction=req.systemInstruction,
            tools=req.tools,
        )
    else:
        raise HTTPException(
            400,
            detail={
                "message": "contents or generateContentRequest is required",
                "type": "bad_request",
            },
        )

    normalized = normalize_gemini_request(effective_req, model_path)
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        await ensure_active_account(attempt, model=normalized.model)
        try:
            res = await client.count_tokens(
                model=normalized.model,
                contents=normalized.contents,
            )
            raw_cached = res.get("cachedContentTokenCount")
            cached_count = (
                int(str(raw_cached))
                if raw_cached is not None and not isinstance(raw_cached, bool)
                else None
            )
            raw_details = res.get("promptTokensDetails")
            details = (
                [d for d in raw_details if isinstance(d, dict)]
                if isinstance(raw_details, list)
                else None
            )
            return GeminiCountTokensResponse(
                totalTokens=int(str(res.get("totalTokens") or 0)),
                cachedContentTokenCount=cached_count,
                promptTokensDetails=details,
            )
        except Exception as exc:
            last_error = exc
            request_ctx = getattr(
                getattr(client, "session", None), "active_context", None
            )
            if await handle_attempt_exception(
                exc,
                attempt=attempt,
                model_path=model_path,
                normalized_model=normalized.model,
                request_ctx=request_ctx,
            ):
                continue
            raise

    raise HTTPException(
        429,
        detail={
            "message": f"All accounts reached daily quota for model '{normalized.model}'.",
            "type": "rate_limit_exceeded",
        },
    ) from last_error
