"""诊断:RotateCookies 令牌轮转可行性(本地运行,Cookie 不外发)。

用法(服务器项目目录下,先重启到含 /v1/cookies/rotate 的新版 waa-signer):
    uv run python scripts/diag_rotatecookies.py data/accounts/<账号ID>/auth.json \
        [--proxy http://host:port] [--signer http://127.0.0.1:8081]

实验矩阵(每条路径都只打印状态码与 Cookie 名,绝不打印值):
  A1/A2  直连 httpx POST accounts.google.com/RotateCookies
         (两种 Content-Type 变体,SAPISIDHASH + Cookie 头)
  B      同 A,但叠加 waa-signer /v1/proof/mint 返回的动态协议头
  C      直连 MakerSuite 域的 $rpc/.../RotateCookies(探测该端点是否存在)
  D      经 waa-signer 新端点 /v1/cookies/rotate(Firefox 152 uTLS 指纹栈)
         —— 需要 signer 重编译,不可用时自动跳过

任一路径拿到新令牌后,追加三项判定探测(走 GetAiStudioBenefitTier RPC):
  T1  旧 Cookie 全套       → 旧令牌是否被 Google 即时作废?
  T2  核心 Cookie + 新令牌 → 新令牌是否可用?
  T3  仅核心 Cookie        → 剥离 PSIDTS 路线是否依然可行?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from http.cookies import SimpleCookie
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from aistudio_api.infrastructure.account.cookie_parser import (
    calculate_sapisid_hash,
)

ROTATE_URL_ACCOUNTS = "https://accounts.google.com/RotateCookies"
ROTATE_URL_MAKERSUITE = (
    "https://alkalimakersuite-pa.clients6.google.com/$rpc/"
    "google.internal.alkali.applications.makersuite.v1.MakerSuiteService/RotateCookies"
)
TIER_RPC_URL = (
    "https://alkalimakersuite-pa.clients6.google.com/$rpc/"
    "google.internal.alkali.applications.makersuite.v1.MakerSuiteService/"
    "GetAiStudioBenefitTier"
)
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0"
)


def load_cookies(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    jar: dict[str, str] = {}
    for item in data.get("cookies", []):
        if isinstance(item, dict) and item.get("name"):
            jar[str(item["name"])] = str(item.get("value") or "")
    return jar


def split(jar: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    core = {k: v for k, v in jar.items() if "PSIDTS" not in k and k != "SIDCC"}
    tokens = {k: v for k, v in jar.items() if "PSIDTS" in k or k == "SIDCC"}
    return core, tokens


def header(jar: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in jar.items() if v)


def parse_set_cookie_names(lines: list[str]) -> list[str]:
    names: list[str] = []
    for line in lines:
        c = SimpleCookie()
        try:
            c.load(line.strip())
        except Exception:
            continue
        names.extend(k for k in c if k)
    return names


def parse_rotations(lines: list[str]) -> dict[str, str]:
    rotated: dict[str, str] = {}
    for line in lines:
        c = SimpleCookie()
        try:
            c.load(line.strip())
        except Exception:
            continue
        for k in c:
            if c[k].value:
                rotated[k] = c[k].value
    return rotated


async def tier_call(
    client: httpx.AsyncClient, jar: dict[str, str], auth_user: str
) -> int:
    headers = {
        "User-Agent": UA,
        "Content-Type": "application/json+protobuf",
        "x-goog-api-key": "AIzaSyDdP816MREB3SkjZO04QXbjsigfcI0GWOs",
        "x-goog-authuser": auth_user,
        "Authorization": calculate_sapisid_hash(jar) or "",
        "Cookie": header(jar),
        "Origin": "https://aistudio.google.com",
        "Referer": "https://aistudio.google.com/",
    }
    try:
        resp = await client.post(TIER_RPC_URL, headers=headers, content="[]")
        return resp.status_code
    except httpx.HTTPError:
        return -1


async def try_direct(
    client: httpx.AsyncClient,
    jar: dict[str, str],
    auth_user: str,
    url: str,
    content_type: str,
    label: str,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, int | None, dict[str, str]]:
    headers = {
        "User-Agent": UA,
        "Content-Type": content_type,
        "x-goog-authuser": auth_user,
        "Authorization": calculate_sapisid_hash(jar) or "",
        "Cookie": header(jar),
        "Origin": "https://accounts.google.com",
        "Referer": "https://accounts.google.com/",
    }
    if extra_headers:
        headers.update(extra_headers)
    try:
        resp = await client.post(url, headers=headers, content=b"")
        lines = resp.headers.get_list("set-cookie")
        names = parse_set_cookie_names(lines)
        print(f"{label:34s} → HTTP {resp.status_code}  Set-Cookie: {names or '无'}")
        return label, resp.status_code, parse_rotations(lines)
    except httpx.HTTPError as exc:
        print(f"{label:34s} → 传输失败: {str(exc)[:60]}")
        return label, None, {}


async def mint_headers(
    signer_client: httpx.AsyncClient, signer_url: str, jar: dict[str, str], auth_user: str
) -> dict[str, str]:
    """从 /v1/proof/mint 取动态协议头(失败返回空)。"""
    try:
        resp = await signer_client.post(
            f"{signer_url}/v1/proof/mint",
            json={
                "cookies": header(jar),
                "authUser": int(auth_user),
                "g1Tier": "TIER1",
                "model": "models/gemini-3.7-flash",
                "prompt": "",
                "wireContents": [],
            },
            timeout=30.0,
        )
        data = resp.json()
        raw = data.get("headers") or {}
        return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}
    except Exception:
        return {}


async def try_via_signer(
    signer_client: httpx.AsyncClient, signer_url: str, jar: dict[str, str], auth_user: str
) -> dict[str, str]:
    try:
        resp = await signer_client.post(
            f"{signer_url}/v1/cookies/rotate",
            json={
                "cookies": header(jar),
                "authUser": int(auth_user),
                "body": "",
            },
            timeout=60.0,
        )
        data = resp.json()
        status = data.get("status")
        lines = data.get("set_cookies") or []
        names = parse_set_cookie_names(lines)
        print(f"{'D signer指纹栈rotate':34s} → HTTP {status}  Set-Cookie: {names or '无'}")
        return parse_rotations(lines)
    except Exception as exc:
        print(f"{'D signer指纹栈rotate':34s} → 不可用({str(exc)[:50]})")
        return {}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("auth_json")
    parser.add_argument("--auth-user", default="0")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--signer", default="http://127.0.0.1:8081")
    args = parser.parse_args()

    auth_path = Path(args.auth_json)
    if not auth_path.is_file():
        raise SystemExit(f"auth.json 不存在: {auth_path}")

    jar = load_cookies(auth_path)
    core, tokens = split(jar)
    print(f"已加载 {len(jar)} 个 Cookie(核心 {len(core)}, 令牌 {len(tokens)}: {sorted(tokens)})")
    print()

    best_rotation: dict[str, str] = {}
    async with httpx.AsyncClient(
        timeout=20.0, proxy=args.proxy, follow_redirects=False, trust_env=False
    ) as client, httpx.AsyncClient(
        timeout=60.0, trust_env=False
    ) as signer_client:
        # A: 直连两变体
        for ct, label in (
            ("application/json+protobuf", "A1 直连(json+protobuf)"),
            ("application/x-www-form-urlencoded", "A2 直连(form)"),
        ):
            _lbl, _st, rotated = await try_direct(
                client, jar, args.auth_user, ROTATE_URL_ACCOUNTS, ct, label
            )
            if rotated and not best_rotation:
                best_rotation = rotated

        # B: 叠加 proof mint 动态头
        dyn = await mint_headers(signer_client, args.signer, jar, args.auth_user)
        if dyn:
            keep = {
                k: v
                for k, v in dyn.items()
                if k.lower()
                in ("x-aistudio-visit-id", "x-goog-api-key", "x-goog-ext-519733851-bin")
            }
            _lbl, _st, rotated = await try_direct(
                client,
                jar,
                args.auth_user,
                ROTATE_URL_ACCOUNTS,
                "application/json+protobuf",
                "B 直连+proof动态头",
                extra_headers=keep,
            )
            if rotated and not best_rotation:
                best_rotation = rotated
        else:
            print(f"{'B 直连+proof动态头':34s} → mint 不可用,跳过")

        # C: MakerSuite 域变体
        _lbl, _st, rotated = await try_direct(
            client,
            jar,
            args.auth_user,
            ROTATE_URL_MAKERSUITE,
            "application/json+protobuf",
            "C MakerSuite域rotate",
        )
        if rotated and not best_rotation:
            best_rotation = rotated

        # D: signer 指纹栈
        rotated = await try_via_signer(
            signer_client, args.signer, jar, args.auth_user
        )
        if rotated and not best_rotation:
            best_rotation = rotated

        # 判定探测
        print()
        t1 = await tier_call(client, jar, args.auth_user)
        print(f"{'T1 旧Cookie全套':34s} → HTTP {t1}")
        if best_rotation:
            t2 = await tier_call(client, {**core, **best_rotation}, args.auth_user)
            print(f"{'T2 核心+新令牌':34s} → HTTP {t2}")
        t3 = await tier_call(client, core, args.auth_user)
        print(f"{'T3 仅核心Cookie':34s} → HTTP {t3}")

    print()
    print("==== 判定 ====")
    if best_rotation:
        print(f"轮转成功路径返回令牌: {sorted(best_rotation)}")
        print(f"- T1={t1}: {'旧令牌仍有效(风险低)' if t1 == 200 else '旧令牌被即时作废(必须原子切换)'}")
        print(f"- T3={t3}: {'剥离 PSIDTS 路线可行(兜底)' if t3 == 200 else '核心凭据疑似失效,请重新导入'}")
    else:
        print("所有路径均未拿到新令牌 —— 阶段0判定:需依赖 signer 路径(D)或方案终止")
        print(f"- T1={t1}, T3={t3}(当前凭据状态参考)")


if __name__ == "__main__":
    asyncio.run(main())
