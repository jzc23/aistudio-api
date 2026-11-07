"""账号凭据健康探测(RotateCookies 实验的收尾产物)。

实验结论(2026-11):令牌无可轮转来源亦无需轮转——SIDCC 家族随流量轮转且
有效期 2 年,PSIDTS 为静态导入且持续有效。因此保活退化为"每日体检":
对每个账号用其完整 Cookie 调 GetAiStudioBenefitTier RPC(等效 T1 探测),
全部 200 → 凭据健康;非 200 → 该账号需要重新导入 Cookie。

用法:
    uv run python scripts/credential_health_check.py [--proxy http://host:port]
    uv run python scripts/credential_health_check.py data/accounts/<id>/auth.json

退出码: 0=全部健康  1=存在失效账号  2=无账号。适合 systemd timer / cron:
    健康时静默,异常时非零退出 + ERROR 日志行(接 journalctl 告警即可)。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from aistudio_api.infrastructure.account.cookie_parser import (
    calculate_sapisid_hash,
)

TIER_RPC_URL = (
    "https://alkalimakersuite-pa.clients6.google.com/$rpc/"
    "google.internal.alkali.applications.makersuite.v1.MakerSuiteService/"
    "GetAiStudioBenefitTier"
)
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0"
)


def _accounts(accounts_dir: Path) -> list[tuple[str, str, Path]]:
    """[(account_id, auth_user, auth.json 路径)],按 registry 顺序。"""
    registry_path = accounts_dir / "registry.json"
    ordered: list[tuple[str, str]] = []
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            accounts = registry.get("accounts") or {}
            active = registry.get("active_account_id")
            ids: list[str] = []
            if active and active in accounts:
                ids.append(str(active))
            ids.extend(str(k) for k in accounts if k != active)
            ordered = [
                (k, str((accounts.get(k) or {}).get("auth_user") or "0"))
                for k in ids
            ]
        except (OSError, json.JSONDecodeError):
            pass
    if not ordered:
        ordered = [(p.parent.name, "0") for p in sorted(accounts_dir.glob("*/auth.json"))]
    return [
        (aid, user, accounts_dir / aid / "auth.json")
        for aid, user in ordered
        if (accounts_dir / aid / "auth.json").is_file()
    ]


def _load_cookies(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    jar: dict[str, str] = {}
    for item in data.get("cookies", []):
        if isinstance(item, dict) and item.get("name"):
            jar[str(item["name"])] = str(item.get("value") or "")
    return jar


async def probe(
    client: httpx.AsyncClient, jar: dict[str, str], auth_user: str
) -> int:
    headers = {
        "User-Agent": UA,
        "Content-Type": "application/json+protobuf",
        "x-goog-api-key": "AIzaSyDdP816MREB3SkjZO04QXbjsigfcI0GWOs",
        "x-goog-authuser": auth_user,
        "Authorization": calculate_sapisid_hash(jar) or "",
        "Cookie": "; ".join(f"{k}={v}" for k, v in jar.items() if v),
        "Origin": "https://aistudio.google.com",
        "Referer": "https://aistudio.google.com/",
    }
    try:
        resp = await client.post(TIER_RPC_URL, headers=headers, content="[]")
        return resp.status_code
    except httpx.HTTPError:
        return -1


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("auth_json", nargs="?", default=None)
    parser.add_argument("--proxy", default=None)
    args = parser.parse_args()

    if args.auth_json:
        targets = [("-", "0", Path(args.auth_json))]
        if not targets[0][2].is_file():
            print(f"ERROR auth.json 不存在: {targets[0][2]}")
            return 2
    else:
        accounts_dir = Path(
            os.environ.get("AISTUDIO_ACCOUNTS_DIR")
            or (Path(__file__).resolve().parents[1] / "data" / "accounts")
        )
        targets = _accounts(accounts_dir)
        if not targets:
            print(f"ERROR 账号目录下没有账号: {accounts_dir}")
            return 2

    failed = 0
    async with httpx.AsyncClient(
        timeout=15.0, proxy=args.proxy, trust_env=False
    ) as client:
        for account_id, auth_user, auth_path in targets:
            jar = _load_cookies(auth_path)
            status = await probe(client, jar, auth_user)
            ok = status == 200
            if not ok:
                failed += 1
            line = f"{'OK  ' if ok else 'FAIL'} {account_id} (u/{auth_user}) → HTTP {status}"
            print(line if ok else f"ERROR {line} —— 建议重新导入该账号 Cookie")

    if failed:
        print(f"RESULT: {failed}/{len(targets)} 个账号凭据异常")
        return 1
    print(f"RESULT: 全部 {len(targets)} 个账号凭据健康")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
