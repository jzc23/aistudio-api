"""Protocol-neutral generation pipeline shared by the Gemini and OpenAI faces.

Owns ATG (anti-truncation gateway) wrapping/unwrapping, retries, account
switching, 403 replay escalation, empty-response retries and usage accounting.
Rendering to a specific wire protocol (Gemini SSE / OpenAI SSE / JSON) happens
outside, in the API layer.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from aistudio_api.api.schemas import GeminiGenerateContentRequest
from aistudio_api.api.state import runtime_state
from aistudio_api.application.account_orchestrator import (
    MAX_RETRIES,
    ensure_active_account,
    record_rotator_event,
    try_switch_account,
)
from aistudio_api.application.atg import AtgProcessor, create_atg_processor
from aistudio_api.application.atg.extractor import SyntheticRequiredError
from aistudio_api.application.chat_service import (
    NormalizedGeminiRequest,
    normalize_gemini_request,
)
from aistudio_api.domain.errors import (
    AistudioError,
    AuthError,
    RequestError,
    SessionExpiredError,
    TransportError,
    UsageLimitExceeded,
    is_transient_rpc_error,
    is_transport_error,
    parse_upstream_grpc_error,
)
from aistudio_api.domain.models import ModelOutput
from aistudio_api.infrastructure.gateway.client import (
    AIStudioClient,
    PreparedRequest,
)
from aistudio_api.infrastructure.utils.logger import get_logger

logger = get_logger("generation")

# ATG event types forwarded through the streaming unwrapper. usage /
# finish_reason / response_id / thought_signature keep their dedicated branches.
_ATG_STREAM_EVENT_TYPES = frozenset(
    {"body", "thinking", "tool_calls", "images", "reasoning_images"}
)


@dataclass(frozen=True)
class StreamEvent:
    """One protocol-neutral streaming event.

    ``kind`` uses the existing gateway event vocabulary:
    body / thinking / tool_calls / images / reasoning_images / usage.
    The terminal ``usage`` event carries the finish reason and response id.
    """

    kind: str
    payload: object
    model: str | None = None
    response_id: str | None = None
    thought_signature: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class GenerationOutcome:
    """Protocol-neutral non-streaming generation result."""

    output: ModelOutput
    model: str
    finish_reason: str
    response_id: str | None
    surface: str = "chat"


AUTH_403_MAX_RETRIES = (
    5  # 403 阶梯重试机制：前 2 次重放，后 2 次重新签名，最后 1 次切号
)


class Auth403RetryManager:
    """403 阶梯重试管理器：前 2 次重放，后 2 次重新签名，最后 1 次切号。"""

    def __init__(self, max_retries: int = AUTH_403_MAX_RETRIES) -> None:
        self.count: int = 0
        self.max_retries: int = max_retries
        self.current_prepared: PreparedRequest | None = None

    async def handle_403(
        self,
        exc: Exception,
        *,
        model: str,
        client: AIStudioClient,
        has_yielded_data: bool = False,
    ) -> bool:
        if has_yielded_data:
            raise HTTPException(
                403, detail={"message": str(exc), "type": "auth_error"}
            ) from exc

        self.count += 1
        record_rotator_event("auth_error", model=model)

        # 1. 前 2 次 403: 重放原有已签名报文 (Replay 1/2, 2/2)
        if self.count <= 2:
            self.current_prepared = client.last_prepared_request
            logger.warning(
                "遭遇 403 权限异常，执行第 %d/2 次请求重放 (Replay): model=%s",
                self.count,
                model,
            )
            return True

        # 2. 后 2 次 403: 重新向签名服务生成新快照 (Re-sign 1/2, 2/2)
        if self.count <= 4:
            self.current_prepared = None
            # 第 3 次 403: 先尝试轮换当前账号 Cookie 再重签(上游 722b84c 语义)
            if self.count == 3:
                account_svc = runtime_state.account_service
                active_acc = account_svc.get_active_account() if account_svc else None
                if (
                    account_svc is not None
                    and active_acc is not None
                    and hasattr(account_svc, "rotate_account_cookies")
                ):
                    try:
                        session = getattr(client, "session", None)
                        await account_svc.rotate_account_cookies(
                            active_acc.id, session
                        )
                    except Exception:
                        pass
            logger.warning(
                "遭遇 403 权限异常，重放未恢复，执行第 %d/2 次重新签名 (Re-sign): model=%s",
                self.count - 2,
                model,
            )
            return True

        # 3. 最后 1 次 403: 切号故障转移 (Failover switch account)
        if self.count == 5:
            self.current_prepared = None
            account_svc = runtime_state.account_service
            active_acc = account_svc.get_active_account() if account_svc else None
            failed_id = active_acc.id if active_acc else None
            if await try_switch_account(model=model, failed_account_id=failed_id):
                logger.warning(
                    "遭遇 403 权限异常，重放与重签均未恢复，已切换至备用账号: model=%s",
                    model,
                )
                return True
            logger.error("遭遇 403 权限异常且无可用备用账号，终止重试")
            raise HTTPException(
                403, detail={"message": str(exc), "type": "auth_error"}
            ) from exc

        raise HTTPException(
            403, detail={"message": str(exc), "type": "auth_error"}
        ) from exc


WIRE_TO_GEMINI_FINISH_REASON: dict[int, str] = {
    0: "FINISH_REASON_UNSPECIFIED",
    1: "STOP",
    2: "MAX_TOKENS",
    3: "SAFETY",
    4: "RECITATION",
    5: "OTHER",
    6: "LANGUAGE",
    7: "BLOCKLIST",
    8: "PROHIBITED_CONTENT",
    9: "SPII",
    10: "MALFORMED_FUNCTION_CALL",
    11: "IMAGE_SAFETY",
    12: "UNEXPECTED_TOOL_CALL",
    13: "TOO_MANY_TOOL_CALLS",
    14: "IMAGE_PROHIBITED_CONTENT",
    15: "IMAGE_OTHER",
    16: "NO_IMAGE",
    17: "IMAGE_RECITATION",
    18: "MISSING_THOUGHT_SIGNATURE",
}


def to_gemini_finish_reason(wire_code: int | None) -> str:
    if wire_code is None:
        return "STOP"
    return WIRE_TO_GEMINI_FINISH_REASON.get(wire_code, "FINISH_REASON_UNSPECIFIED")


def clean_upstream_error_message(raw_msg: str) -> str:
    """Extract a clean, readable error message from Google/JSPB upstream error strings."""
    if not raw_msg:
        return ""
    text = raw_msg.strip()
    # 若不包含 JSPB/batchexecute 特征结构，直接保留原始错误文本
    if "[,[" not in text and "[null,[" not in text and not text.startswith("[,"):
        return text
    inner = text
    while True:
        m = re.match(r"^HTTP\s+\d+:\s*(.*)$", inner)
        if m:
            inner = m.group(1).strip()
        else:
            break
    _, msg = parse_upstream_grpc_error(inner)
    return msg or inner


def classify_gemini_error_payload(exc: Exception) -> tuple[int, str, str]:
    """Extract standard Gemini HTTP status code, message, and status string from an exception."""
    if isinstance(exc, SessionExpiredError):
        return (
            401,
            "All accounts have expired sessions. Please import fresh cookies.",
            "UNAUTHENTICATED",
        )
    if isinstance(exc, AuthError):
        return 403, str(exc), "PERMISSION_DENIED"
    if isinstance(exc, UsageLimitExceeded):
        return 429, str(exc), "RESOURCE_EXHAUSTED"
    if isinstance(exc, (ValueError, RequestValidationError)):
        return 400, str(exc), "INVALID_ARGUMENT"
    if isinstance(exc, HTTPException):
        status = exc.status_code
        detail = exc.detail
        raw_msg = detail.get("message") if isinstance(detail, dict) else detail
        msg = str(raw_msg) if raw_msg is not None else ""
        clean_msg = clean_upstream_error_message(msg)
        status_map = {
            400: "INVALID_ARGUMENT",
            401: "UNAUTHENTICATED",
            403: "PERMISSION_DENIED",
            404: "NOT_FOUND",
            429: "RESOURCE_EXHAUSTED",
            500: "INTERNAL",
            502: "UNAVAILABLE",
            503: "UNAVAILABLE",
        }
        return (
            status,
            clean_msg,
            status_map.get(status, "INTERNAL" if status >= 500 else "UNKNOWN"),
        )
    if isinstance(exc, RequestError):
        status = exc.status if exc.status > 0 else 500
        clean_msg = clean_upstream_error_message(str(exc))
        status_map = {
            400: "INVALID_ARGUMENT",
            401: "UNAUTHENTICATED",
            403: "PERMISSION_DENIED",
            404: "NOT_FOUND",
            429: "RESOURCE_EXHAUSTED",
            500: "INTERNAL",
            502: "UNAVAILABLE",
            503: "UNAVAILABLE",
        }
        return (
            status,
            clean_msg,
            status_map.get(status, "INTERNAL" if status >= 500 else "UNKNOWN"),
        )
    if isinstance(exc, TransportError):
        return 502, str(exc), "UNAVAILABLE"
    if isinstance(exc, AistudioError):
        return 500, str(exc), "INTERNAL"
    return 500, str(exc), "INTERNAL"


async def handle_attempt_exception(
    exc: Exception,
    *,
    attempt: int,
    model_path: str,
    normalized_model: str | None,
    has_yielded_data: bool = False,
) -> bool:
    """Handle per-attempt error classification and account switching.

    Returns True if the operation should be retried, or raises HTTPException / re-raises.
    """
    target_model = normalized_model or model_path
    account_svc = runtime_state.account_service
    active_acc = account_svc.get_active_account() if account_svc else None
    failed_id = active_acc.id if active_acc else None

    if isinstance(exc, ValueError):
        raise HTTPException(
            400, detail={"message": str(exc), "type": "bad_request"}
        ) from exc

    if isinstance(exc, SessionExpiredError):
        logger.warning("账号登录已失效，被重定向到登录页: %s", exc)
        if not has_yielded_data and await try_switch_account(
            model=target_model,
            failed_account_id=failed_id,
            is_session_expired=True,
        ):
            logger.info("已切换账号重试: 尝试=%d/%d", attempt + 1, MAX_RETRIES)
            return True
        raise HTTPException(
            401,
            detail={
                "message": "All accounts encountered login redirect or expired sessions. Please import fresh cookies.",
                "type": "auth_error",
            },
        ) from exc

    if isinstance(exc, AuthError):
        logger.warning("账号权限被拒绝 (HTTP 403): %s", exc)
        record_rotator_event("auth_error", model=target_model)

        raise HTTPException(
            403, detail={"message": str(exc), "type": "auth_error"}
        ) from exc

    if isinstance(exc, UsageLimitExceeded):
        runtime_state.record(target_model, "rate_limited")
        record_rotator_event("rate_limited", model=target_model)
        if not has_yielded_data and await try_switch_account(
            model=target_model, failed_account_id=failed_id
        ):
            logger.warning(
                "触发 429 配额限制，切换账号重试: 尝试=%d/%d, model=%s",
                attempt + 1,
                MAX_RETRIES,
                target_model,
            )
            return True
        logger.warning("所有账号在模型 %s 上均处于冷却状态", target_model)
        raise HTTPException(
            429,
            detail={
                "message": f"All accounts reached daily quota for model '{target_model}'. Quota resets at 00:00 PST.",
                "type": "rate_limit_exceeded",
            },
        ) from exc

    if (
        isinstance(exc, RequestError)
        and exc.status == 204
        and attempt == 0
        and not has_yielded_data
    ):
        logger.warning("收到上游 204 空响应，立即重试")
        return True

    # 针对 Google 上游偶发且完全随机的 RPC 路由 404 (Ambiguous request for service '' and method '/GenerativeService.*')：
    # 此报错为上游端点偶发抖动，在未产生输出时当场立即重试（严禁切号以防浪费配额，其他 404 严格按原样报错抛出）
    if (
        is_transient_rpc_error(exc)
        and not has_yielded_data
        and attempt < MAX_RETRIES - 1
    ):
        logger.warning(
            "上游偶发 404 方法未找到，立即重试: 尝试=%d/%d, 错误=%s",
            attempt + 1,
            MAX_RETRIES,
            clean_upstream_error_message(str(exc)),
        )
        return True
    if is_transport_error(exc):
        logger.warning(
            "网关传输或签名异常，切换账号重试: 尝试=%d/%d, 错误=%s",
            attempt + 1,
            MAX_RETRIES,
            exc,
        )
        if not has_yielded_data and await try_switch_account(
            model=target_model, failed_account_id=failed_id
        ):
            logger.info("已切换账号重试: 尝试=%d/%d", attempt + 1, MAX_RETRIES)
            return True
        if isinstance(exc, TransportError):
            runtime_state.record(target_model, "errors")
            record_rotator_event("error", model=target_model)
            logger.warning("上游服务传输中断 (502): %s", exc)
            raise HTTPException(
                502, detail={"message": str(exc), "type": "upstream_error"}
            ) from exc
    if isinstance(exc, RequestError):
        clean_msg = clean_upstream_error_message(str(exc))
        status_code = exc.status if exc.status > 0 else 500
        error_type = (
            "bad_request"
            if status_code == 400
            else "not_found"
            if status_code == 404
            else "upstream_error"
        )
        logger.warning(
            "上游返回请求错误: status=%d, message=%s", status_code, clean_msg
        )
        raise HTTPException(
            status_code, detail={"message": clean_msg, "type": error_type}
        ) from exc

    if isinstance(exc, AistudioError):
        runtime_state.record(target_model, "errors")
        record_rotator_event("error", model=target_model)
        logger.warning("服务端处理异常: %s", exc)
        raise HTTPException(
            500, detail={"message": str(exc), "type": "server_error"}
        ) from exc

    runtime_state.record(target_model, "errors")
    record_rotator_event("error", model=target_model)
    logger.error("请求处理发生未捕获异常: %s", exc)
    logger.debug("异常堆栈详情:", exc_info=True)
    raise HTTPException(
        500, detail={"message": str(exc), "type": "server_error"}
    ) from exc


def _output_is_empty(output: ModelOutput) -> bool:
    """ATG emptiness test: no text, no thinking and no tool calls at all."""
    if not output.candidates:
        return True
    candidate = output.candidates[0]
    return not (candidate.text or candidate.thinking or candidate.function_calls)


def _create_atg(
    api_format: str,
    atg_factory: Callable[[], AtgProcessor] | None,
) -> AtgProcessor:
    if atg_factory is not None:
        return atg_factory()
    from aistudio_api.application.atg.config import resolve_atg_mode

    return create_atg_processor({"mode": resolve_atg_mode(api_format)})


def _wrap_atg(
    atg: AtgProcessor,
    req: GeminiGenerateContentRequest,
    model_path: str,
) -> tuple[GeminiGenerateContentRequest, str]:
    """Inject the synthetic transport once, before normalization.

    Returns the (possibly replaced) request and the synthetic tool name
    (empty when not wrapped).
    """
    if not atg.enabled:
        return req, ""
    atg_body = req.model_dump(mode="json", exclude_none=True)
    wrap = atg.wrap(atg_body, model_path)
    if wrap.client_synthetic is not None:
        logger.info(
            "[ATG] 客户端已自带合成通道，跳过本地包装: tool=%s, reason=%s",
            wrap.client_synthetic.name or "(instruction-only)",
            wrap.client_synthetic.reason,
        )
        return req, ""
    if wrap.synthetic_hit:
        logger.info(
            "[ATG] 已注入合成通道: tool=%s, model=%s, contents=%d, tools=%d",
            wrap.tool_name,
            model_path,
            len(atg_body.get("contents") or []),
            len(atg_body.get("tools") or []),
        )
        return GeminiGenerateContentRequest.model_validate(wrap.body), wrap.tool_name
    logger.info("[ATG] 跳过包装: model=%s (非文本模型)", model_path)
    return req, ""


async def generate(
    *,
    model_path: str,
    req: GeminiGenerateContentRequest,
    client: AIStudioClient,
    api_format: str = "gemini",
    atg_factory: Callable[[], AtgProcessor] | None = None,
) -> GenerationOutcome:
    """Run a non-streaming generation, returning a protocol-neutral outcome."""
    atg = _create_atg(api_format, atg_factory)
    req, tool_name = _wrap_atg(atg, req, model_path)

    last_error: Exception | None = None
    auth_manager = Auth403RetryManager()
    total_attempts = MAX_RETRIES + AUTH_403_MAX_RETRIES

    for attempt in range(total_attempts):
        await ensure_active_account(attempt, model=model_path)
        normalized: NormalizedGeminiRequest | None = None
        try:
            normalized = normalize_gemini_request(req, model_path)
            logger.info(
                "处理非流式生成请求: format=%s, model=%s, 对话轮数=%d, 尝试=%d/%d",
                api_format,
                normalized.model,
                len(req.contents),
                attempt + 1,
                total_attempts,
            )

            # ATG empty-response retry: a fully empty (collapsed) upstream result
            # is re-sent up to cfg.empty_retries times. This layer never switches
            # accounts; it only re-sends within the current attempt.
            max_empty = atg.cfg.empty_retries if (atg.enabled and tool_name) else 0
            output: ModelOutput | None = None
            for empty_attempt in range(max_empty + 1):
                output = await client.generate_content(
                    model=normalized.model,
                    contents=normalized.contents,
                    system_instruction_content=normalized.system_instruction,
                    tools=normalized.tools,
                    tool_config=normalized.tool_config,
                    safety_settings=normalized.safety_settings,
                    temperature=normalized.temperature,
                    top_p=normalized.top_p,
                    top_k=normalized.top_k,
                    max_tokens=normalized.max_tokens,
                    generation_config_overrides=normalized.generation_config_overrides,
                    prepared=auth_manager.current_prepared,
                )
                auth_manager.current_prepared = None
                if not (atg.enabled and tool_name):
                    break
                if not _output_is_empty(output):
                    break
                if empty_attempt < max_empty:
                    logger.warning(
                        "[ATG] 空响应，重发 (%d/%d): model=%s",
                        empty_attempt + 1,
                        max_empty,
                        model_path,
                    )
                    continue
                logger.warning(
                    "[ATG] empty response retries exhausted: model=%s", model_path
                )
                if atg.cfg.mode == "required":
                    raise SyntheticRequiredError(
                        "synthetic tool call required but not present in upstream response"
                    )
            assert output is not None

            if atg.enabled and tool_name:
                unwrap = atg.unwrap_response(output, tool_name)
                output = unwrap.output

            final_fr = to_gemini_finish_reason(
                output.candidates[0].finish_reason if output.candidates else None
            )
            record_rotator_event("success", model=normalized.model, usage=output.usage)
            runtime_state.record(
                normalized.model,
                "success",
                usage=output.usage,
                finish_reason=final_fr,
            )
            pt = output.usage.get("prompt_tokens") if output.usage else 0
            ct = output.usage.get("completion_tokens") if output.usage else 0
            logger.info(
                "非流式请求完成: model=%s, finish_reason=%s, 输入tokens=%s, 输出tokens=%s",
                normalized.model,
                final_fr,
                pt,
                ct,
            )
            return GenerationOutcome(
                output=output,
                model=normalized.model,
                finish_reason=final_fr,
                response_id=output.response_id or None,
            )
        except Exception as exc:
            last_error = exc
            if isinstance(exc, AuthError):
                target_m = normalized.model if normalized else model_path
                if await auth_manager.handle_403(exc, model=target_m, client=client):
                    continue
                raise
            if await handle_attempt_exception(
                exc,
                attempt=attempt,
                model_path=model_path,
                normalized_model=normalized.model if normalized else None,
            ):
                continue
            raise

    raise HTTPException(
        429,
        detail={
            "message": f"All accounts reached daily quota for model '{model_path}'. Quota resets at 00:00 PST.",
            "type": "rate_limit_exceeded",
        },
    ) from last_error


async def stream_generation(
    *,
    model_path: str,
    req: GeminiGenerateContentRequest,
    client: AIStudioClient,
    api_format: str = "gemini",
    atg_factory: Callable[[], AtgProcessor] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run a streaming generation, yielding protocol-neutral events.

    The terminal event has ``kind == "usage"`` and carries the final usage
    payload plus finish reason / response id. Errors raise; the protocol
    renderer decides how to surface them.
    """
    atg = _create_atg(api_format, atg_factory)
    effective_req, tool_name = _wrap_atg(atg, req, model_path)

    stream_start_time = time.perf_counter()
    logged_first_chunk = False
    logger.info(
        "处理流式生成请求: format=%s, model=%s, 对话轮数=%d",
        api_format,
        model_path,
        len(effective_req.contents),
    )
    final_usage: dict[str, object] | None = None
    final_finish_reason: str = "STOP"
    latest_response_id: str | None = None
    buffered_thought_sig: str | None = None
    max_empty = atg.cfg.empty_retries if (atg.enabled and tool_name) else 0
    auth_manager = Auth403RetryManager()
    total_attempts = MAX_RETRIES + AUTH_403_MAX_RETRIES
    normalized: NormalizedGeminiRequest | None = None
    stream_completed = False
    last_stream_error: Exception | None = None

    for stream_attempt in range(total_attempts):
        has_yielded_data = False
        last_stream_error = None
        final_usage = None
        final_finish_reason = "STOP"
        latest_response_id = None
        buffered_thought_sig = None
        await ensure_active_account(stream_attempt, model=model_path)
        normalized = normalize_gemini_request(effective_req, model_path)
        # Fresh per-attempt unwrapper: a retry must not carry stale
        # buffered/consumed state from the previous attempt.
        unwrapper = (
            atg.create_stream_unwrapper(tool_name)
            if (atg.enabled and tool_name)
            else None
        )
        empty_attempt = 0
        retry_outer = False
        while True:
            produced_content = False
            try:
                async for event_type, text in client.stream_generate_content(
                    model=normalized.model,
                    contents=normalized.contents,
                    system_instruction_content=normalized.system_instruction,
                    tools=normalized.tools,
                    tool_config=normalized.tool_config,
                    safety_settings=normalized.safety_settings,
                    temperature=normalized.temperature,
                    top_p=normalized.top_p,
                    top_k=normalized.top_k,
                    max_tokens=normalized.max_tokens,
                    generation_config_overrides=normalized.generation_config_overrides,
                    prepared=auth_manager.current_prepared,
                ):
                    if event_type == "usage":
                        final_usage = text if isinstance(text, dict) else None
                    elif event_type == "finish_reason":
                        if text not in (None, 0, 22):
                            final_finish_reason = (
                                to_gemini_finish_reason(text)
                                if isinstance(text, int)
                                else str(text)
                            )
                        if (
                            unwrapper is not None
                            and final_finish_reason == "TOOL_CALLS"
                            and unwrapper.should_remap_finish_reason()
                        ):
                            final_finish_reason = "STOP"
                    elif event_type == "response_id":
                        latest_response_id = str(text)
                    elif event_type == "thought_signature":
                        buffered_thought_sig = str(text)
                    elif (
                        unwrapper is not None and event_type in _ATG_STREAM_EVENT_TYPES
                    ):
                        if text:
                            produced_content = True
                        for index, (out_type, out_text) in enumerate(
                            unwrapper.process_event(event_type, text)
                        ):
                            if out_text:
                                has_yielded_data = True
                                if not logged_first_chunk:
                                    logged_first_chunk = True
                                    ttft_ms = (
                                        time.perf_counter() - stream_start_time
                                    ) * 1000
                                    logger.info(
                                        "流式响应开始推送: model=%s, 首包耗时=%.1fms",
                                        normalized.model,
                                        ttft_ms,
                                    )
                            yield StreamEvent(
                                kind=out_type,
                                payload=out_text,
                                model=normalized.model,
                                response_id=latest_response_id,
                                thought_signature=(
                                    buffered_thought_sig if index == 0 else None
                                ),
                            )
                        buffered_thought_sig = None
                    else:
                        if text:
                            has_yielded_data = True
                            if not logged_first_chunk:
                                logged_first_chunk = True
                                ttft_ms = (
                                    time.perf_counter() - stream_start_time
                                ) * 1000
                                logger.info(
                                    "流式响应开始推送: model=%s, 首包耗时=%.1fms",
                                    normalized.model,
                                    ttft_ms,
                                )
                        yield StreamEvent(
                            kind=event_type,
                            payload=text,
                            model=normalized.model,
                            response_id=latest_response_id,
                            thought_signature=buffered_thought_sig,
                        )
                        buffered_thought_sig = None
                auth_manager.current_prepared = None
            except Exception as exc:
                last_stream_error = exc
                if isinstance(exc, AuthError) and not has_yielded_data:
                    target_m = normalized.model if normalized else model_path
                    if await auth_manager.handle_403(
                        exc,
                        model=target_m,
                        client=client,
                        has_yielded_data=has_yielded_data,
                    ):
                        retry_outer = True
                        break
                    raise
                if await handle_attempt_exception(
                    exc,
                    attempt=stream_attempt,
                    model_path=model_path,
                    normalized_model=normalized.model if normalized else None,
                    has_yielded_data=has_yielded_data,
                ):
                    retry_outer = True
                    break
                raise

            # ATG empty-response retry: an attempt that produced no content
            # event is re-sent up to cfg.empty_retries times (no account switch).
            if max_empty > 0 and not produced_content:
                if empty_attempt < max_empty:
                    empty_attempt += 1
                    logger.warning(
                        "[ATG] 流式空响应，重发 (%d/%d): model=%s",
                        empty_attempt,
                        max_empty,
                        model_path,
                    )
                    continue
                logger.warning(
                    "[ATG] empty response retries exhausted: model=%s",
                    model_path,
                )
                if atg.cfg.mode == "required":
                    raise SyntheticRequiredError(
                        "synthetic tool call required but not present in "
                        "upstream response"
                    )
            break
        if retry_outer:
            continue

        # Flush buffered output and log unwrap statistics.
        if unwrapper is not None:
            for out_type, out_text in unwrapper.finalize():
                if out_text:
                    has_yielded_data = True
                yield StreamEvent(
                    kind=out_type,
                    payload=out_text,
                    model=normalized.model,
                    response_id=latest_response_id,
                    thought_signature=buffered_thought_sig,
                )
                buffered_thought_sig = None
        stream_completed = True
        break

    if not stream_completed and last_stream_error is not None:
        raise last_stream_error
    target_model = normalized.model if normalized else model_path
    record_rotator_event("success", model=target_model, usage=final_usage)
    if normalized is not None:
        runtime_state.record(
            normalized.model,
            "success",
            usage=final_usage,
            finish_reason=final_finish_reason,
        )
    elapsed_total_ms = (time.perf_counter() - stream_start_time) * 1000
    pt = final_usage.get("prompt_tokens") if final_usage else 0
    ct = final_usage.get("completion_tokens") if final_usage else 0
    logger.info(
        "流式请求完成: model=%s, 耗时=%.1fms, finish_reason=%s, 输入tokens=%s, 输出tokens=%s",
        target_model,
        elapsed_total_ms,
        final_finish_reason,
        pt,
        ct,
    )
    yield StreamEvent(
        kind="usage",
        payload=final_usage,
        model=target_model,
        response_id=latest_response_id,
        finish_reason=final_finish_reason,
    )
