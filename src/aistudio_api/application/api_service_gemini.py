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
    _effective_max_retries,
    _is_prepared_binding_current,
    _record_request_rotator_event,
    _request_context_snapshot,
    active_account_cooldown_remaining,
    classify_gemini_error_payload,
    clean_upstream_error_message,
    generate,
    handle_attempt_exception,
    is_daily_quota_message,
    stream_generation,
    to_gemini_finish_reason,
)
from aistudio_api.domain.errors import (
    AccountRestrictedError,
    AccountVerificationRequiredError,
    AuthError,
    SessionExpiredError,
    TransportError,
)
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.session import SessionContext
from aistudio_api.infrastructure.utils.logger import get_logger

logger = get_logger("api.gemini")

__all__ = [
    "AUTH_403_MAX_RETRIES",
    "MAX_RETRIES",
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
    from aistudio_api.api.streaming_response import (
        RequestStreamingResponse,
        close_generation_task,
    )

    gen_task = asyncio.create_task(
        _do_generate_content(model_path, req, client, stream=False)
    )
    try:
        # 前 10 秒直接等待结果；绝大多数请求在此期间返回标准对象
        return await asyncio.wait_for(asyncio.shield(gen_task), timeout=10.0)
    except asyncio.CancelledError:
        # shield 不能让已断开的客户端留下后台生成任务
        await close_generation_task(gen_task)
        raise
    except TimeoutError:
        logger.info(
            "非流式请求耗时超过 10s，激活 HTTP 换行保活流: model=%s", model_path
        )

        async def keepalive_stream():
            try:
                yield b"\n"
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
                await close_generation_task(gen_task)

        return RequestStreamingResponse(
            keepalive_stream(),
            media_type="application/json",
            generation_task=gen_task,
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
    elif event_type == "images":
        img_list = text if isinstance(text, list) else []
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in to_gemini_parts("", images=img_list)
        ]
    elif event_type == "reasoning_images":
        r_img_list = text if isinstance(text, list) else []
        parts = [
            part.model_dump(mode="json", exclude_none=True)
            for part in to_gemini_parts("", reasoning_images=r_img_list)
        ]
    elif event_type == "thought_signature":
        parts = [{"thoughtSignature": str(text)}]
    else:
        return None

    if (
        thought_signature
        and parts
        and event_type in ("tool_calls", "images", "reasoning_images")
    ):
        existing_signature = parts[0].get("thoughtSignature")
        if not existing_signature:
            parts[0]["thoughtSignature"] = thought_signature
        elif existing_signature != thought_signature:
            # 独立签名不能覆盖工具或图片自己的签名,按原顺序保留独立 Part
            # (上游 fcdc53e / tool roundtrip thoughtSignatures 语义)
            parts.insert(0, {"thoughtSignature": thought_signature})

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
        # aclosing:外层(SSE 响应/心跳包装)关闭时确定性关闭 stream_generation,
        # 让 aclosing 清理链直达传输层(上游 fcdc53e 流资源生命周期语义)
        from contextlib import aclosing

        try:
            async with aclosing(
                stream_generation(
                    model_path=model_path,
                    req=req,
                    client=client,
                    api_format="gemini",
                )
            ) as stream:
                async for event in stream:
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
    # (ATG 合成通道首段内容出现前常有长静默);aclosing 保证关闭链确定性传播
    async def stream_response():
        from contextlib import aclosing

        from aistudio_api.api.sse_heartbeat import heartbeat_sse

        async with aclosing(heartbeat_sse(sse_chunks())) as wrapped:
            async for chunk in wrapped:
                yield chunk

    # ASGI 生命周期绑定:客户端断开时同步关闭迭代器(aclosing 链直达传输层)
    from aistudio_api.api.streaming_response import RequestStreamingResponse

    return RequestStreamingResponse(
        stream_response(),
        media_type="text/event-stream",
    )


async def handle_gemini_decisions(
    req: GeminiCreateDecisionRequest,
    client: AIStudioClient,
) -> GeminiCreateDecisionResponse:
    last_error: Exception | None = None
    auth_manager = Auth403RetryManager()
    max_retries = _effective_max_retries()
    total_attempts = max_retries + AUTH_403_MAX_RETRIES
    ordinary_attempt = 0
    raw_model = req.model or "gemini-decision-1"
    norm_model = raw_model if raw_model.startswith("models/") else f"models/{raw_model}"
    questions_dict: dict[str, dict[str, object]] = {
        key: val.model_dump(mode="json", exclude_none=True)
        for key, val in req.questions.items()
    }

    for attempt in range(total_attempts):
        # 403 重试单独计数,普通失败共用普通预算(上游 fcdc53e/PR #21)
        if ordinary_attempt >= max_retries:
            break
        ensured_ctx = await ensure_active_account(attempt, model=norm_model)
        effective_ctx = auth_manager.pinned_ctx or (
            ensured_ctx
            if isinstance(ensured_ctx, SessionContext)
            else _request_context_snapshot(client)
        )
        request_ctx: SessionContext | None = effective_ctx
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
            if (
                prepared is not None
                and not force_recreate
                and not _is_prepared_binding_current(prepared)
            ):
                # 与生成接口一致,切号、禁用或凭据更新后不能重放缓存签名
                prepared = None
            if prepared is None or force_recreate:
                prepared = await client.prepare_decision_request(
                    model=norm_model,
                    raw_input=req.input,
                    questions=questions_dict,
                    temperature=req.temperature,
                    force_recreate_signer=force_recreate,
                    ctx=effective_ctx,
                )
            request_ctx = prepared.ctx or effective_ctx
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
            _record_request_rotator_event(
                "success", model=norm_model, request_ctx=request_ctx,
                usage=rotator_usage,
            )
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
            if isinstance(exc, AuthError) and not isinstance(
                exc,
                (
                    SessionExpiredError,
                    AccountVerificationRequiredError,
                    AccountRestrictedError,
                ),
            ):
                if await auth_manager.handle_403(
                    exc, model=norm_model, client=client, request_ctx=request_ctx
                ):
                    continue
                raise
            if await handle_attempt_exception(
                exc,
                attempt=ordinary_attempt,
                model_path=norm_model,
                normalized_model=norm_model,
                request_ctx=request_ctx,
            ):
                ordinary_attempt += 1
                auth_manager.current_prepared = None
                auth_manager.pinned_ctx = None
                continue
            raise

    # 普通预算耗尽后保留实际故障,不能将会话失效或传输失败改报日配额
    error = last_error or TransportError("Request retry budget exhausted.")
    status_code, message, _ = classify_gemini_error_payload(error)
    raise HTTPException(status_code, detail={"message": message}) from error


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

    if (
        effective_req.systemInstruction is not None
        and effective_req.systemInstruction.parts
    ) or effective_req.tools:
        # 当前计数 RPC 只承载 contents,拒绝静默丢弃会影响计数的字段
        raise HTTPException(
            400,
            detail={
                "message": (
                    "countTokens currently supports contents only; "
                    "systemInstruction and tools are not supported."
                ),
                "type": "bad_request",
            },
        )

    normalized = normalize_gemini_request(effective_req, model_path)
    last_error: Exception | None = None
    for attempt in range(_effective_max_retries()):
        ensured_ctx = await ensure_active_account(attempt, model=normalized.model)
        # 在请求进入客户端前绑定实际账号,失败时不读取已切换的全局会话
        request_ctx = (
            ensured_ctx
            if isinstance(ensured_ctx, SessionContext)
            else _request_context_snapshot(client)
        )
        try:
            if isinstance(ensured_ctx, SessionContext):
                res = await client.count_tokens(
                    model=normalized.model,
                    contents=normalized.contents,
                    ctx=ensured_ctx,
                )
            else:
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
            if await handle_attempt_exception(
                exc,
                attempt=attempt,
                model_path=model_path,
                normalized_model=normalized.model,
                request_ctx=request_ctx,
            ):
                continue
            raise

    error = last_error or TransportError("Request retry budget exhausted.")
    status_code, message, _ = classify_gemini_error_payload(error)
    raise HTTPException(status_code, detail={"message": message}) from error
