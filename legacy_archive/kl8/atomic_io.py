# -*- coding: utf-8 -*-
"""L0 基础设施：原子写与幂等落盘工具（契约 R48-e、R49-f，P8-M13 修复）。

【为什么需要单独一层】
P8 体检卡 M13 指出：TXT/微信/Excel 与状态文件此前都是 ``open(path,"w")`` 直写，
进程中断会留下**半截文件**；Excel 写失败还会先 ``print`` 再照常返回路径，让上层误判
成功。本模块把「同目录临时文件 → ``os.replace`` → 备份旧件」收敛为一处实现，
所有落盘方（reporting / pipeline / adaptive）复用，避免各写各的。

【为什么必须同目录】``os.replace`` 仅在同一文件系统内保证原子性；跨盘会退化为复制+删除。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

#: 备份后缀（契约 R49-f：覆盖前保留旧件，便于事故回滚）。
BACKUP_SUFFIX: Final[str] = ".bak"


def _ensure_parent(path: Path) -> None:
    """确保目标文件的父目录存在（不存在则递归创建）。

    @param path 目标文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _backup_previous(path: Path) -> None:
    """把已存在的旧文件复制为 ``.bak``（不移动，避免读者短期读不到）。

    @param path 目标文件路径
    @raises OSError 备份失败（由调用方决定是否降级）
    """
    if not path.exists():
        return
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    with open(path, "rb") as source, open(backup, "wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    backup: bool = True,
) -> Path:
    """原子写二进制文件（先写同目录临时文件，再 ``os.replace``）。

    @param path 目标路径
    @param payload 二进制内容
    @param backup 是否在覆盖前把旧文件备份为 ``.bak``
    @return 目标路径
    @raises OSError 目录不可创建、临时写失败或替换失败
    """
    target = Path(path)
    _ensure_parent(target)
    if backup:
        _backup_previous(target)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=target.parent, delete=False, suffix=".tmp"
    )
    tmp_name = handle.name
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    backup: bool = True,
) -> Path:
    """原子写文本文件（UTF-8 by default，``\\n`` 原样写入）。

    @param path 目标路径
    @param text 文本内容
    @param encoding 编码（默认 UTF-8）
    @param backup 是否在覆盖前备份旧件
    @return 目标路径
    @raises OSError 目录不可创建、临时写失败或替换失败
    """
    return atomic_write_bytes(path, text.encode(encoding), backup=backup)


def atomic_copy(src: str | Path, dst: str | Path, *, backup: bool = True) -> Path:
    """把已有文件（如 openpyxl 写好的临时 xlsx）原子搬移到目标路径。

    @param src 源文件（通常位于目标同目录的临时文件）
    @param dst 目标路径
    @param backup 是否在覆盖前备份旧件
    @return 目标路径
    @raises OSError 读源或替换失败
    """
    source = Path(src)
    target = Path(dst)
    _ensure_parent(target)
    if backup:
        _backup_previous(target)
    os.replace(source, target)
    return target