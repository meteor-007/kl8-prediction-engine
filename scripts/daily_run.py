# -*- coding: utf-8 -*-
"""每日全自动执行器：读取点位 -> 运行预测 -> 自动提交 -> 推送当前分支。

【设计目标】
1. 点位默认从 `每日预测指令.txt` 的 `[点位]` 行读取，每天只改这一行；
2. 严格串行：文件锁防止手动运行与定时任务并发；
3. 流水线失败时不提交、不推送，保留现场供排查；
4. 无变更时跳过提交；本地有未推送提交时补推；
5. 日志写入 output/logs/（已被 .gitignore 忽略），不污染仓库。

@author czw
@since 2026-11-11
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from kl8_engine.pipeline import PipelineResult, run_pipeline
from kl8_engine.points import PointsError, parse_points

POINT_LINE_RE = re.compile(r"^\s*\[点位\]\s*(.+?)\s*$", re.MULTILINE)

GITHUB_FALLBACK_IPS = (
    "20.87.245.0",
    "4.228.31.150",
    "20.29.134.23",
    "140.82.112.3",
    "140.82.113.3",
    "140.82.114.3",
    "140.82.116.3",
    "20.27.177.113",
)
"""直连 github.com 失败时尝试的 GitHub HTTPS 入口 IP（可用 KL8_GITHUB_IP 覆盖）。"""


class Tee:
    """把输出同时写到控制台与日志文件。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"


def read_points_from_instruction(path: Path) -> str:
    """从指令文件读取当日点位。"""
    if not path.exists():
        raise FileNotFoundError(f"指令文件不存在: {path}")
    match = POINT_LINE_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"未在 {path.name} 中找到 '[点位]' 行")
    return match.group(1).strip()


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """执行 git 命令。"""
    result = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} 失败（退出码 {result.returncode}）:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def ensure_git_repo() -> None:
    """确认当前目录是 git 仓库。"""
    result = git(["rev-parse", "--is-inside-work-tree"], check=False)
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise RuntimeError("当前目录不是 git 仓库，无法自动提交/推送")


def current_branch() -> str:
    """当前分支名。"""
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def remote_exists() -> bool:
    """是否存在 origin 远程。"""
    return git(["remote", "get-url", "origin"], check=False).returncode == 0


def _tcp_reachable(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    """快速判断目标 IP 的 443 端口是否可连。"""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def push_with_fallback(branch: str) -> None:
    """先直连推送；失败时自动尝试 GitHub 备用 IP。

    某些网络会把 github.com 解析到不可达的 IP，但 api.github.com 与
    GitHub 的其他入口 IP 仍可用。这里用 git 的 http.curloptResolve
    在不修改系统 DNS/hosts 的前提下切换连接目标。
    """
    direct = git(["push", "origin", branch], check=False)
    if direct.returncode == 0:
        return

    print("⚠ 直连 github.com 推送失败，尝试备用入口 IP ...")
    detail = (direct.stderr or direct.stdout).strip()
    if detail:
        print(f"  {detail.splitlines()[0][:200]}")

    candidates: list[str] = []
    env_ip = os.environ.get("KL8_GITHUB_IP", "").strip()
    if env_ip:
        candidates.append(env_ip)
    candidates.extend(ip for ip in GITHUB_FALLBACK_IPS if ip not in candidates)

    for ip in candidates:
        if not _tcp_reachable(ip):
            print(f"  - {ip}: 不可达")
            continue
        result = git(
            ["-c", f"http.curloptResolve=github.com:443:{ip}", "push", "origin", branch],
            check=False,
        )
        if result.returncode == 0:
            print(f"✔ 已通过备用入口 {ip} 推送")
            return
        first_line = (result.stderr or result.stdout).strip().splitlines()
        print(f"  - {ip}: 推送失败 {first_line[0][:160] if first_line else ''}")

    raise RuntimeError("直连与所有备用入口均推送失败")


def auto_commit_and_push(
    message: str,
    body: str,
    push: bool,
) -> bool:
    """暂存全部变更并提交；有需要时推送当前分支。

    @return 是否创建了新提交
    """
    ensure_git_repo()
    branch = current_branch()
    git(["add", "-A"])

    staged = git(["diff", "--cached", "--quiet"], check=False)
    committed = staged.returncode != 0
    if committed:
        git(["commit", "-m", message, "-m", body])
        print(f"✔ 已提交到分支 {branch}: {message}")
    else:
        print("ℹ 没有文件变更，跳过提交")

    if not push:
        print("ℹ 已按 --no-push 跳过推送")
        return committed

    if not remote_exists():
        print("⚠ 未配置 origin 远程，跳过推送")
        return committed

    # 无论本次是否有新提交，只要本地领先远程就补推（覆盖上次推送失败的情况）
    ahead = git(["rev-list", "--count", f"origin/{branch}..HEAD"], check=False)
    if ahead.returncode == 0 and int(ahead.stdout.strip() or "0") > 0:
        push_with_fallback(branch)
        print(f"✔ 已推送 {branch} -> origin/{branch}")
    else:
        print("ℹ 本地没有领先远程的提交，无需推送")
    return committed


def run_prediction(args: argparse.Namespace, points: tuple[int, ...]) -> PipelineResult:
    """执行完整预测流水线。"""
    def progress(msg: str) -> None:
        print(f"   ⏳ {msg}", flush=True)

    history_path = Path(args.history)
    if not history_path.is_absolute():
        history_path = PROJECT_ROOT / history_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    print(f"📌 本次点位 {len(points)} 码: {' '.join(f'{n:02d}' for n in points)}")
    result = run_pipeline(
        history_path=str(history_path),
        output_dir=str(output_dir),
        no_sync=args.no_sync,
        dl_weight=args.dl_weight,
        seq_len=args.seq_len,
        epochs=args.epochs,
        budget_shrinkage=args.budget,
        points=points,
        points_mode=args.points_mode,
        fusion_mode=args.fusion,
        comparison=not args.no_compare,
        progress=progress,
    )

    official = "Borda" if result.fusion_mode == "borda" else "weighted"
    print("\n" + "=" * 72)
    print(f"✔ 目标期号: 第 {result.target_period} 期 | 正式版融合: {official}")
    print(f"🔥 金胆: {' '.join(f'{n:02d}' for n in result.ensemble.gold_pool)}")
    print(f"⭐ 主选: {' '.join(f'{n:02d}' for n in result.ensemble.main_pool)}")
    if result.comparison_ensemble is not None:
        comparison = "weighted" if official == "Borda" else "Borda"
        print(
            f"🧪 实验版({comparison}) 金胆: "
            f"{' '.join(f'{n:02d}' for n in result.comparison_ensemble.gold_pool)}"
        )
    print(f"⏱ 耗时: {result.elapsed_seconds:.1f}s")
    print("=" * 72)
    return result


def acquire_lock(lock_path: Path) -> bool:
    """获取进程锁；已存在则返回 False。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"pid={os.getpid()} at={datetime.now().isoformat()}".encode("utf-8"))
    os.close(fd)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="快乐8 每日全自动执行：预测 + 自动提交 + 推送当前分支"
    )
    parser.add_argument("--points", default=None, help="点位字符串；缺省从指令文件 [点位] 行读取")
    parser.add_argument(
        "--instruction",
        default="每日预测指令.txt",
        help="指令文件路径（默认 每日预测指令.txt）",
    )
    parser.add_argument("--history", default="kl8_history.txt")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-sync", action="store_true", help="跳过联网同步")
    parser.add_argument("--fusion", choices=("borda", "weighted"), default="borda")
    parser.add_argument("--no-compare", action="store_true", help="关闭双融合并排对比")
    parser.add_argument("--points-mode", choices=("compare", "boost"), default="compare")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--dl-weight", type=float, default=0.55)
    parser.add_argument("--no-commit", action="store_true", help="只运行预测，不提交")
    parser.add_argument("--no-push", action="store_true", help="提交但不推送")
    parser.add_argument("--skip-run", action="store_true", help="跳过预测，只执行提交/推送")
    parser.add_argument("--message", default=None, help="自定义提交标题")
    parser.add_argument("--dry-run", action="store_true", help="只打印解析结果，不运行")
    parser.add_argument("--log-dir", default="output/logs")
    args = parser.parse_args()

    instruction = Path(args.instruction)
    if not instruction.is_absolute():
        instruction = PROJECT_ROOT / instruction
    try:
        points_raw = args.points or read_points_from_instruction(instruction)
        points = parse_points(points_raw)
    except (OSError, ValueError, PointsError) as exc:
        print(f"❌ 点位读取失败: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"点位({len(points)}): {' '.join(f'{n:02d}' for n in points)}")
        print(f"正式版融合: {args.fusion} | 并排对比: {'关闭' if args.no_compare else '开启'}")
        print(f"同步: {'跳过' if args.no_sync else '开启'} | epochs={args.epochs}")
        return 0

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"daily_{datetime.now():%Y%m%d_%H%M%S}.log"

    lock_path = PROJECT_ROOT / "output" / ".daily_run.lock"
    if not acquire_lock(lock_path):
        print(f"❌ 已有另一个每日任务在运行（锁: {lock_path}）", file=sys.stderr)
        return 3

    original_stdout, original_stderr = sys.stdout, sys.stderr
    exit_code = 0
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)
            print(f"=== 每日自动执行 {datetime.now():%Y-%m-%d %H:%M:%S} ===")
            print(f"项目: {PROJECT_ROOT}")
            print(f"日志: {log_path}")

            period_text = "?"
            if not args.skip_run:
                try:
                    result = run_prediction(args, points)
                    period_text = str(result.target_period)
                except Exception as exc:
                    print(f"\n❌ 预测流水线失败: {exc}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    exit_code = 1
                    return exit_code
            else:
                print("ℹ 已按 --skip-run 跳过预测")

            if args.no_commit:
                print("ℹ 已按 --no-commit 跳过提交/推送")
                return exit_code

            title = args.message or f"chore(daily): {date.today().isoformat()} 第{period_text}期预测 [auto]"
            body = (
                f"target_period: {period_text}\n"
                f"points: {' '.join(f'{n:02d}' for n in points)}\n"
                f"official_fusion: {args.fusion}\n"
                f"generated_by: scripts/daily_run.py"
            )
            try:
                auto_commit_and_push(title, body, push=not args.no_push)
            except Exception as exc:
                print(f"\n❌ 自动提交/推送失败: {exc}", file=sys.stderr)
                exit_code = 1
            return exit_code
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        try:
            lock_path.unlink()
        except OSError:
            pass
        print(f"日志已写入: {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())