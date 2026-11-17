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
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    AccountRestrictedError,
    AccountVerificationRequiredError,
    AistudioError,
    AuthError,
    RequestError,
    SessionExpiredError,
    StaleSessionError,
    TransportError,
    UsageLimitExceeded,
    is_transient_rpc_error,
    is_transport_error,
    parse_upstream_grpc_error,
)
from aistudio_api.domain.models import ModelOutput
from aistudio_api.infrastructure.account.account_store import AccountMeta
from aistudio_api.infrastructure.gateway.client import (
    AIStudioClient,
    PreparedRequest,
)
from aistudio_api.infrastructure.gateway.session import SessionContext
from aistudio_api.infrastructure.utils.logger import get_logger

logger = get_logger("generation")

# ATG event types forwarded through the streaming unwrapper. usage /
# finish_reason / response_id / thought_signature keep their dedicated branches.
_ATG_STREAM_EVENT_TYPES = frozenset(
    {"body", "thinking", "tool_calls", "images", "reasoning_images"}
)

# Renderer-consumable event kinds: only these may consume the buffered
# standalone thoughtSignature (unknown kinds render to nothing and must not
# swallow a pending signature -- upstream fcdc53e tool roundtrip semantics).
_RENDERABLE_STREAM_EVENT_TYPES = frozenset(
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


# 分钟级短冷却阈值:冷却时长不超过该值视为瞬时 RPM 限流,而非今日配额耗尽
# (上游 19fe769 / Issue #16)
SHORT_RATE_LIMIT_COOLDOWN_SECONDS = 600


def is_daily_quota_message(message: str) -> bool:
    """判断上游清理后的错误消息是否明确指向"日配额耗尽"。"""
    lowered = (message or "").casefold()
    return any(
        marker in lowered for marker in ("daily", "per day", "per_day", "perday")
    )


def active_account_cooldown_remaining(account_id: str | None, model: str) -> float:
    """查询账号在目标模型上的剩余冷却秒数,用于区分瞬时 RPM 限流与日配额耗尽。"""
    if not account_id:
        return 0.0
    rotator = runtime_state.rotator
    if rotator is None:
        return 0.0
    try:
        stats = rotator.get_account_stats(account_id)
        if stats is None:
            return 0.0
        return float(stats.get_cooldown_remaining(model))
    except Exception:
        return 0.0


def _request_context_snapshot(client: AIStudioClient) -> SessionContext:
    """在签名前绑定本次账号,未知来源用空上下文阻止回退到新活跃账号。"""
    session = getattr(client, "session", None)
    context = getattr(session, "active_context", None) if session is not None else None
    return context if isinstance(context, SessionContext) else SessionContext()


def _request_account(
    request_ctx: SessionContext | None,
) -> tuple[AccountMeta | None, str | None]:
    """按本次请求凭据定位账号,显式未知来源不能落到新的全局账号。"""
    account_svc = runtime_state.account_service
    if account_svc is None:
        return None, None
    if request_ctx is not None:
        if not request_ctx.auth_file:
            return None, None
        account_id = Path(request_ctx.auth_file).parent.name
        return account_svc.get_account(account_id), account_id
    account = account_svc.get_active_account()
    return account, account.id if account else None


def _is_request_generation_current(request_ctx: SessionContext | None) -> bool:
    """认证结果只能影响生成该请求时使用的 Cookie 代际。"""
    account_svc = runtime_state.account_service
    if request_ctx is None or not request_ctx.auth_file or account_svc is None:
        return True
    return bool(
        account_svc.is_cookie_generation_current(
            Path(request_ctx.auth_file).parent.name, request_ctx.cookie_id
        )
    )


def _record_request_rotator_event(
    event: Literal["auth_error", "rate_limited", "error", "success"],
    *,
    model: str,
    request_ctx: SessionContext | None,
    usage: dict[str, object] | None = None,
) -> None:
    """只记录本次请求的当前代际账号,未知或已删除来源不影响新活跃账号。"""
    if request_ctx is None:
        record_rotator_event(event, model=model, usage=usage)
        return
    if not request_ctx.auth_file:
        return
    account, account_id = _request_account(request_ctx)
    rotator = runtime_state.rotator
    if (
        account is None
        or account_id is None
        or rotator is None
        or not _is_request_generation_current(request_ctx)
    ):
        return
    record_rotator_event(event, model=model, usage=usage, account_id=account_id)


def _is_prepared_binding_current(prepared: PreparedRequest) -> bool:
    """重放缓存签名时核对账户启用状态、代际、登录索引与订阅权益。"""
    account_svc = runtime_state.account_service
    if account_svc is None:
        return True
    ctx = prepared.ctx
    account = account_svc.get_active_account()
    if (
        ctx is None
        or not ctx.auth_file
        or account is None
        or not account.enabled
        or Path(ctx.auth_file).parent.name != account.id
        or not _is_request_generation_current(ctx)
        or ctx.auth_user != account.auth_user
    ):
        return False
    from aistudio_api.infrastructure.account.cookie_parser import normalize_benefit_tier

    return normalize_benefit_tier(ctx.tier) == normalize_benefit_tier(account.tier)


class Auth403RetryManager:
    """403 阶梯重试管理器：前 2 次重放，后 2 次重新签名，最后 1 次切号。

    force_recreate_signer 用于通知本地/远程签名服务销毁僵死热 Worker 冷重建
    (上游 19fe769 语义):第 3 次(重签)与第 5 次(切号)置位,由调用方在
    下一次请求中消费后立即归零,避免重复触发昂贵的冷启动。
    pinned_ctx (上游 85e22c3/fcdc53e 语义):重签期间把请求钉在原账号会话上,
    不随并发切号漂移;切号成功后清空。
    """

    def __init__(self, max_retries: int = AUTH_403_MAX_RETRIES) -> None:
        self.count: int = 0
        self.max_retries: int = max_retries
        self.current_prepared: PreparedRequest | None = None
        self.pinned_ctx: SessionContext | None = None
        self.force_recreate_signer: bool = False

    @staticmethod
    def _diagnostic_context(
        client: AIStudioClient,
        prepared: PreparedRequest | None,
        request_ctx: SessionContext | None,
    ) -> str:
        """403 诊断上下文:账号/签名服务/worker 池/UA,定位 403 根因。"""
        signer_id = "unknown"
        outer_ua = ""
        pool_state = ""
        try:
            from aistudio_api.infrastructure.gateway.model_defaults import (
                get_active_waa_server,
            )

            waa_cfg = get_active_waa_server()
            signer_id = str(waa_cfg.get("id") or "unknown")
            outer_ua = str(waa_cfg.get("user_agent") or "")
        except Exception:
            pass
        if prepared is not None and isinstance(prepared.headers, dict):
            outer_ua = str(prepared.headers.get("User-Agent") or outer_ua)
        session = getattr(client, "session", None)
        signer = getattr(session, "_signer", None) if session is not None else None
        if signer is not None:
            pool_state = str(getattr(signer, "last_pool_state", "") or "")
        acc, acc_id = _request_account(request_ctx)
        auth_user = (
            acc.auth_user
            if acc is not None
            else (request_ctx.auth_user if request_ctx is not None else "0")
        )
        return (
            f"account={acc_id or 'unknown'}(u/{auth_user}), "
            f"signer={signer_id}, pool={pool_state or 'unknown'}, "
            f"ua={outer_ua[:48] or 'default'}"
        )

    async def handle_403(
        self,
        exc: Exception,
        *,
        model: str,
        client: AIStudioClient,
        request_ctx: SessionContext | None = None,
        has_yielded_data: bool = False,
    ) -> bool:
        if request_ctx is None and self.current_prepared is not None:
            request_ctx = self.current_prepared.ctx
        if not _is_request_generation_current(request_ctx):
            self.current_prepared = None
            self.pinned_ctx = None
            if not has_yielded_data:
                return True
            raise HTTPException(
                503, detail="Request credentials were replaced."
            ) from exc
        if has_yielded_data:
            raise HTTPException(
                403, detail={"message": str(exc), "type": "auth_error"}
            ) from exc

        diag = self._diagnostic_context(client, self.current_prepared, request_ctx)
        self.count += 1
        _record_request_rotator_event(
            "auth_error", model=model, request_ctx=request_ctx
        )

        # 1. 前 2 次 403: 重放已签名报文 (Replay 1/2, 2/2)
        #    本次快照由请求处理器绑定,不读取客户端共享的最近请求(并发隔离)。
        if self.count <= 2:
            self.force_recreate_signer = False
            logger.warning(
                "遭遇 403 权限异常，执行第 %d/2 次请求重放 (Replay): model=%s, %s",
                self.count,
                model,
                diag,
            )
            return True

        # 2. 后 2 次 403: 重新向签名服务生成新快照 (Re-sign 1/2, 2/2)
        if self.count <= 4:
            self.current_prepared = None
            self.pinned_ctx = request_ctx
            # 第 3 次 403: 先尝试轮换当前账号 Cookie 再重签(上游 722b84c 语义)
            if self.count == 3:
                account_svc = runtime_state.account_service
                active_acc, _ = _request_account(request_ctx)
                if (
                    account_svc is not None
                    and active_acc is not None
                    and hasattr(account_svc, "rotate_account_cookies")
                ):
                    try:
                        session = getattr(client, "session", None)
                        result = await account_svc.rotate_account_cookies(
                            active_acc.id, session
                        )
                        if result.get("stale") is True or not (
                            _is_request_generation_current(request_ctx)
                        ):
                            self.pinned_ctx = None
                            return True
                    except Exception:
                        pass
            self.force_recreate_signer = self.count == 3
            logger.warning(
                "遭遇 403 权限异常，执行第 %d/2 次重新签名 (Re-sign): model=%s, "
                "force_recreate=%s, %s",
                self.count - 2,
                model,
                self.force_recreate_signer,
                diag,
            )
            return True

        # 3. 最后 1 次 403: 切号故障转移 (Failover switch account)
        if self.count == 5:
            self.current_prepared = None
            self.pinned_ctx = None
            self.force_recreate_signer = True
            _, failed_id = _request_account(request_ctx)
            if await try_switch_account(
                model=model,
                failed_account_id=failed_id,
                expected_cookie_id=request_ctx.cookie_id if request_ctx else None,
            ):
                logger.warning(
                    "遭遇 403 权限异常，重放与重签均未恢复，已切换至备用账号: "
                    "model=%s, %s",
                    model,
                    diag,
                )
                return True
            logger.error(
                "遭遇 403 权限异常且无可用备用账号，终止重试: model=%s, %s",
                model,
                diag,
            )
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
            504: "DEADLINE_EXCEEDED",
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
            504: "DEADLINE_EXCEEDED",
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
    request_ctx: SessionContext | None = None,
) -> bool:
    """Handle per-attempt error classification and account switching.

    Returns True if the operation should be retried, or raises HTTPException / re-raises.
    ``request_ctx`` pins the failed account to the request's own session context
    (upstream d4ddd09 semantics) so account switches never mis-target after a
    previous rotation has already changed the active account.
    """
    target_model = normalized_model or model_path
    account_svc = runtime_state.account_service
    active_acc, failed_id = _request_account(request_ctx)

    # Cookie 代际更新(重新导入/轮转)导致的陈旧签名:未出数据直接重备请求,
    # 已出数据只能中断(上游 62cb23b 语义)
    if isinstance(exc, StaleSessionError):
        if not has_yielded_data:
            logger.info("请求凭据代际已更新，重新准备请求: model=%s", target_model)
            return True
        raise HTTPException(
            503, detail="Request credentials were replaced."
        ) from exc

    if isinstance(exc, ValueError):
        raise HTTPException(
            400, detail={"message": str(exc), "type": "bad_request"}
        ) from exc

    # 认证类错误先核对代际:旧代际的结果不得处罚/切走当前新账号
    # (上游 fcdc53e 语义)
    if isinstance(exc, AuthError) and not _is_request_generation_current(request_ctx):
        if not has_yielded_data:
            logger.info("忽略旧 Cookie 代际的认证结果: account=%s", failed_id)
            return True
        raise HTTPException(
            503, detail="Request credentials were replaced."
        ) from exc

    if isinstance(exc, SessionExpiredError):
        if not has_yielded_data and account_svc is not None and active_acc is not None:
            # 遇到 401 时先探活会话并尝试轮换滚动 Cookie,若拿到新 PSIDTS 则原地重试
            try:
                from aistudio_api.infrastructure.account.cookie_parser import (
                    load_cookies_from_auth_file,
                    probe_aistudio_session,
                )

                auth_path = account_svc.get_account_auth_path(active_acc.id)
                cdict = load_cookies_from_auth_file(auth_path) if auth_path else {}
                before_ts = cdict.get("__Secure-1PSIDTS", "")
                if cdict:
                    alive, _ = await probe_aistudio_session(cdict)
                    if not _is_request_generation_current(request_ctx):
                        return True
                    if alive and hasattr(account_svc, "rotate_account_cookies"):
                        session = (
                            runtime_state.client.session
                            if runtime_state.client
                            else None
                        )
                        rot_res = await account_svc.rotate_account_cookies(
                            active_acc.id, session
                        )
                        if rot_res.get("stale") is True or not (
                            _is_request_generation_current(request_ctx)
                        ):
                            return True
                        new_cookies = rot_res.get("cookies")
                        after_ts = (
                            new_cookies.get("__Secure-1PSIDTS", "")
                            if isinstance(new_cookies, dict)
                            else ""
                        )
                        if rot_res.get("ok") and after_ts and after_ts != before_ts:
                            logger.info(
                                "账号 %s 触发 401 后完成 Cookie 轮换，原地重试",
                                active_acc.id,
                            )
                            return True
            except Exception as rot_err:
                logger.debug("401 Cookie 轮换探测异常: %s", rot_err)

        if not _is_request_generation_current(request_ctx):
            return True
        logger.warning("账号登录已失效: account=%s, %s", failed_id, exc)
        if not has_yielded_data and await try_switch_account(
            model=target_model,
            failed_account_id=failed_id,
            is_session_expired=True,
            expected_cookie_id=request_ctx.cookie_id if request_ctx else None,
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

    if isinstance(exc, (AccountVerificationRequiredError, AccountRestrictedError)):
        # 账号需要身份验证或被服务级限制:立即自动禁用并切号,
        # 不再浪费 403 重放/重签预算(上游 fcdc53e 语义)
        if not _is_request_generation_current(request_ctx):
            if not has_yielded_data:
                logger.info(
                    "忽略旧 Cookie 代际的验证/受限结果: account=%s", failed_id
                )
                return True
            raise HTTPException(
                503, detail="Request credentials were replaced."
            ) from exc
        _record_request_rotator_event(
            "auth_error", model=target_model, request_ctx=request_ctx
        )
        if account_svc is not None and failed_id:
            disabled = account_svc.toggle_account_enabled(failed_id, False)
            auth_u = (
                active_acc.auth_user
                if active_acc is not None
                else (request_ctx.auth_user if request_ctx is not None else "0")
            )
            reason = (
                "需要身份验证"
                if isinstance(exc, AccountVerificationRequiredError)
                else "服务受限"
            )
            logger.error(
                "账号 %s (u/%s) %s，已自动禁用 (disabled=%s): %s",
                failed_id,
                auth_u,
                reason,
                disabled is not None and not disabled.enabled,
                exc,
            )
        if not has_yielded_data and await try_switch_account(
            model=target_model,
            failed_account_id=failed_id,
            expected_cookie_id=request_ctx.cookie_id if request_ctx else None,
        ):
            logger.info("已切换账号重试 (%d/%d)", attempt + 1, MAX_RETRIES)
            return True
        raise HTTPException(
            403, detail={"message": str(exc), "type": "auth_error"}
        ) from exc

    if isinstance(exc, AuthError):
        logger.warning("账号权限被拒绝 (HTTP 403): %s", exc)
        _record_request_rotator_event(
            "auth_error", model=target_model, request_ctx=request_ctx
        )

        raise HTTPException(
            403, detail={"message": str(exc), "type": "auth_error"}
        ) from exc

    if isinstance(exc, UsageLimitExceeded):
        runtime_state.record(target_model, "rate_limited")
        _record_request_rotator_event(
            "rate_limited", model=target_model, request_ctx=request_ctx
        )
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
        # 区分瞬时 RPM 限流与今日配额耗尽(上游 19fe769 / Issue #16):
        # 冷却剩余较短或上游消息明示 per-day 才报"日配额",否则报短时限流
        upstream_detail = clean_upstream_error_message(str(exc))
        cooldown_remaining = active_account_cooldown_remaining(
            failed_id, target_model
        )
        is_daily_quota = (
            is_daily_quota_message(upstream_detail)
            or cooldown_remaining > SHORT_RATE_LIMIT_COOLDOWN_SECONDS
        )
        if is_daily_quota:
            message = (
                f"All accounts reached daily quota for model '{target_model}'. "
                "Quota resets at 00:00 PST."
            )
        else:
            retry_hint = (
                f"retry after ~{int(cooldown_remaining)}s"
                if cooldown_remaining > 0
                else "retry shortly"
            )
            message = (
                f"Rate limit exceeded (HTTP 429, {retry_hint}): "
                f"{upstream_detail or 'upstream quota exhausted'}"
            )
        logger.warning(
            "模型 %s 所有账号均在冷却中 (cooldown_remaining=%ds, daily_quota=%s)",
            target_model,
            int(cooldown_remaining),
            is_daily_quota,
        )
        raise HTTPException(
            429,
            detail={"message": message, "type": "rate_limit_exceeded"},
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
        # 传输失败只能使用普通预算,最后一次失败或已发出内容时不再切号
        # (上游 fcdc53e/PR #21 语义:attempt 为普通预算计数,不含 403 重试)
        if not has_yielded_data and attempt < MAX_RETRIES - 1:
            logger.warning(
                "网关传输或签名异常，切换账号重试: 尝试=%d/%d, 错误=%s",
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
            if await try_switch_account(
                model=target_model, failed_account_id=failed_id
            ):
                logger.info("已切换账号重试: 尝试=%d/%d", attempt + 1, MAX_RETRIES)
                return True
        runtime_state.record(target_model, "errors")
        _record_request_rotator_event(
            "error", model=target_model, request_ctx=request_ctx
        )
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
        _record_request_rotator_event(
            "error", model=target_model, request_ctx=request_ctx
        )
        logger.warning("服务端处理异常: %s", exc)
        raise HTTPException(
            500, detail={"message": str(exc), "type": "server_error"}
        ) from exc

    runtime_state.record(target_model, "errors")
    _record_request_rotator_event(
        "error", model=target_model, request_ctx=request_ctx
    )
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
    ordinary_attempt = 0
    request_ctx: SessionContext | None = None

    for attempt in range(total_attempts):
        # 403 重试单独计数,普通失败(429/传输/登录失效)共用 MAX_RETRIES 预算,
        # 两类预算互不挤占(上游 fcdc53e/PR #21 语义)
        if ordinary_attempt >= MAX_RETRIES:
            break
        ensured_ctx = await ensure_active_account(attempt, model=model_path)
        effective_ctx = auth_manager.pinned_ctx or (
            ensured_ctx
            if isinstance(ensured_ctx, SessionContext)
            else _request_context_snapshot(client)
        )
        request_ctx = effective_ctx
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

            force_recreate = auth_manager.force_recreate_signer
            auth_manager.force_recreate_signer = False
            prepared = auth_manager.current_prepared
            if (
                prepared is not None
                and not force_recreate
                and not _is_prepared_binding_current(prepared)
            ):
                # 切号、禁用或凭据更新后不能重放缓存签名
                logger.info("缓存签名的账户绑定已变化，重新准备请求")
                prepared = None
            if prepared is None or force_recreate:
                prepared = await client.prepare_request(
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
                    force_recreate_signer=force_recreate,
                    ctx=effective_ctx,
                )
            request_ctx = prepared.ctx or effective_ctx
            auth_manager.current_prepared = prepared

            # ATG empty-response retry: a fully empty (collapsed) upstream result
            # is re-sent up to cfg.empty_retries times. This layer never switches
            # accounts; it only re-sends within the current attempt.
            max_empty = atg.cfg.empty_retries if (atg.enabled and tool_name) else 0
            output: ModelOutput | None = None
            for empty_attempt in range(max_empty + 1):
                output = await client.generate_content(
                    model=normalized.model,
                    prepared=prepared,
                )
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
            _record_request_rotator_event(
                "success",
                model=normalized.model,
                request_ctx=request_ctx,
                usage=output.usage,
            )
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
            if isinstance(exc, AuthError) and not isinstance(
                exc,
                (
                    SessionExpiredError,
                    AccountVerificationRequiredError,
                    AccountRestrictedError,
                ),
            ):
                target_m = normalized.model if normalized else model_path
                if await auth_manager.handle_403(
                    exc, model=target_m, client=client, request_ctx=request_ctx
                ):
                    continue
                raise
            if await handle_attempt_exception(
                exc,
                attempt=ordinary_attempt,
                model_path=model_path,
                normalized_model=normalized.model if normalized else None,
                request_ctx=request_ctx,
            ):
                # 普通预算只在这里消耗,403 重试不占用
                ordinary_attempt += 1
                auth_manager.current_prepared = None
                auth_manager.pinned_ctx = None
                continue
            raise

    # 预算耗尽时保留最后一次故障类型,不能统一伪装成配额耗尽
    error = last_error or TransportError("Request retry budget exhausted.")
    status_code, message, _ = classify_gemini_error_payload(error)
    raise HTTPException(status_code, detail={"message": message}) from error


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
    ordinary_attempt = 0
    normalized: NormalizedGeminiRequest | None = None
    request_ctx: SessionContext | None = None
    stream_completed = False
    last_stream_error: Exception | None = None

    for stream_attempt in range(total_attempts):
        # 403 重试单独计数,普通失败共用 MAX_RETRIES 预算(上游 fcdc53e/PR #21)
        if ordinary_attempt >= MAX_RETRIES:
            break
        has_yielded_data = False
        last_stream_error = None
        final_usage = None
        final_finish_reason = "STOP"
        latest_response_id = None
        buffered_thought_sig = None
        ensured_ctx = await ensure_active_account(stream_attempt, model=model_path)
        effective_ctx = auth_manager.pinned_ctx or (
            ensured_ctx
            if isinstance(ensured_ctx, SessionContext)
            else _request_context_snapshot(client)
        )
        request_ctx = effective_ctx
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
                force_recreate = auth_manager.force_recreate_signer
                auth_manager.force_recreate_signer = False
                prepared = auth_manager.current_prepared
                if (
                    prepared is not None
                    and not force_recreate
                    and not _is_prepared_binding_current(prepared)
                ):
                    logger.info("缓存签名的账户绑定已变化，重新准备请求")
                    prepared = None
                if prepared is None or force_recreate:
                    prepared = await client.prepare_request(
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
                        generation_config_overrides=(
                            normalized.generation_config_overrides
                        ),
                        force_recreate_signer=force_recreate,
                        ctx=effective_ctx,
                    )
                request_ctx = prepared.ctx or effective_ctx
                auth_manager.current_prepared = prepared
                async with aclosing(
                    client.stream_generate_content(
                        model=normalized.model,
                        prepared=prepared,
                    )
                ) as events:
                    async for event_type, text in events:
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
                            if buffered_thought_sig:
                                # 连续的独立签名按到达顺序输出,不能互相覆盖
                                # (上游 fcdc53e / tool roundtrip 语义)
                                yield StreamEvent(
                                    kind="thought_signature",
                                    payload=buffered_thought_sig,
                                    model=normalized.model,
                                    response_id=latest_response_id,
                                )
                                has_yielded_data = True
                            buffered_thought_sig = str(text)
                        elif (
                            unwrapper is not None
                            and event_type in _ATG_STREAM_EVENT_TYPES
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
                            renderable = (
                                event_type in _RENDERABLE_STREAM_EVENT_TYPES
                            )
                            if text and renderable:
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
                            # 仅有实际载荷且可渲染的事件才消费签名缓冲;
                            # "done"/None 载荷与未知类型事件不能清空,否则
                            # 尾随独立签名丢失(上游 Issue #15 / fcdc53e)
                            if text and renderable:
                                buffered_thought_sig = None
                auth_manager.current_prepared = None
            except Exception as exc:
                last_stream_error = exc
                if (
                    isinstance(exc, AuthError)
                    and not isinstance(
                        exc,
                        (
                            SessionExpiredError,
                            AccountVerificationRequiredError,
                            AccountRestrictedError,
                        ),
                    )
                    and not has_yielded_data
                ):
                    target_m = normalized.model if normalized else model_path
                    if await auth_manager.handle_403(
                        exc,
                        model=target_m,
                        client=client,
                        request_ctx=request_ctx,
                        has_yielded_data=has_yielded_data,
                    ):
                        retry_outer = True
                        break
                    raise
                if await handle_attempt_exception(
                    exc,
                    attempt=ordinary_attempt,
                    model_path=model_path,
                    normalized_model=normalized.model if normalized else None,
                    has_yielded_data=has_yielded_data,
                    request_ctx=request_ctx,
                ):
                    ordinary_attempt += 1
                    auth_manager.current_prepared = None
                    auth_manager.pinned_ctx = None
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
    # 尾随独立 thoughtSignature:流结束时仍有未被后续内容消费的签名,单独补发,
    # 避免下一轮对话因缺失签名被上游拒绝(上游 19fe769 / Issue #15)
    if buffered_thought_sig:
        yield StreamEvent(
            kind="thought_signature",
            payload=buffered_thought_sig,
            model=normalized.model if normalized else model_path,
            response_id=latest_response_id,
        )
        buffered_thought_sig = None
    target_model = normalized.model if normalized else model_path
    _record_request_rotator_event(
        "success", model=target_model, request_ctx=request_ctx, usage=final_usage
    )
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
