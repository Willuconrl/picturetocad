# src/dxf_repro.py
"""DXF 写出的**比特级可复现**支持（SPEC §3.3：「同一输入两次运行产出比特级一致」）。

问题（ezdxf 1.4.4 实测，四个独立来源，缺一不可）
------------------------------------------------
1. `$TDCREATE/$TDUPDATE/$VERSIONGUID/$FINGERPRINTGUID`
   → 默认写当前时间与新随机 GUID。ezdxf 提供开关
   `options.write_fixed_meta_data_for_testing = True` 可固定。
2. `CREATED_BY_EZDXF / WRITTEN_BY_EZDXF` 两个 DICTIONARYVAR 的值来自
   `ezdxf.document.ezdxf_marker_string()`（`"<版本> @ <ISO 时间>"`），**不受**上面的开关
   控制 → 只能临时替换该函数。
3. CLASSES 段顺序：`sections/classes.py:REQUIRED_CLASSES` 是 **set**，
   `add_required_classes()` 直接迭代它，且类定义是在**写出时**才补齐的 →
   顺序随 Python 字符串哈希随机化（同进程稳定、跨进程不稳定）。实测 5 次生成
   得到 2 种字节。对策：写出前补齐并按 key 排序重建 OrderedDict。
4. 平台换行：`Drawing.saveas()` 用文本模式写，Windows 上把 LF 变成 CRLF →
   同一份 DXF 在 Windows/Linux 上字节不同。对策：自己写字节（固定 LF + UTF-8）。

契约
----
- 本模块是**唯一**允许操纵 ezdxf 写出时序的地方；`dxf_writer.py` 与测试夹具都走它。
- 任何一步失效必须**抛错**，不得静默降级（否则回归基准会悄悄漂移）。
- 代价：DXF 头里不再有 ezdxf 版本与保存时间。版本信息改由 `manifest.json` 记录
  （见 `src/artifacts.py`），并写在 `$PROJECTNAME` / `$LASTSAVEDBY` 的业务元数据里。
"""

from __future__ import annotations

import contextlib
import io
from collections import OrderedDict
from typing import Iterator

import ezdxf
from ezdxf import document as ezdxf_document
from ezdxf.document import CREATED_BY_EZDXF, WRITTEN_BY_EZDXF, Drawing

#: 替换 ezdxf 默认标记（含时间戳）的固定串；不含时间/版本，故可复现
FIXED_EZDXF_MARKER = "r2v-tool deterministic output"


@contextlib.contextmanager
def deterministic_ezdxf() -> Iterator[None]:
    """临时固定 ezdxf 的写出元数据（时间戳、GUID、标记串）。可安全嵌套。"""
    original_marker = ezdxf_document.ezdxf_marker_string
    previous_option = ezdxf.options.write_fixed_meta_data_for_testing
    ezdxf_document.ezdxf_marker_string = lambda: FIXED_EZDXF_MARKER
    ezdxf.options.write_fixed_meta_data_for_testing = True
    try:
        yield
    finally:
        ezdxf_document.ezdxf_marker_string = original_marker
        ezdxf.options.write_fixed_meta_data_for_testing = previous_option


def pin_class_order(doc: Drawing) -> None:
    """补齐必需 CLASS 并按名字字典序重建表，固定 CLASSES 段的写出顺序。"""
    doc.classes.add_required_classes(doc.dxfversion)
    items = sorted(doc.classes.classes.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    doc.classes.classes = OrderedDict(items)


def assert_class_order(doc: Drawing) -> None:
    """写出后复检：ezdxf 不得再追加未排序的类定义。"""
    keys = list(doc.classes.classes.keys())
    if keys != sorted(keys):
        raise RuntimeError(
            "CLASSES 段顺序不稳定：ezdxf 在写出时追加了未排序的类定义，"
            "请复查 src/dxf_repro.py:pin_class_order"
        )


def dxf_bytes(doc: Drawing) -> bytes:
    """把文档导出成确定性字节（LF、UTF-8、固定元数据、排序 CLASSES）。

    注意：`CREATED_BY_EZDXF` 是在 `ezdxf.new()` 那一刻写进去的（含创建时间），
    开关管不到它，必须在写出前显式覆盖，否则同一场景在不同进程里创建就会漂移。
    """
    pin_class_order(doc)
    with deterministic_ezdxf():
        meta = doc.ezdxf_metadata()
        meta[CREATED_BY_EZDXF] = FIXED_EZDXF_MARKER
        meta[WRITTEN_BY_EZDXF] = FIXED_EZDXF_MARKER  # 写出时会被固定函数再写一次
        stream = io.StringIO()
        doc.write(stream)
        assert_class_order(doc)
        data = doc.encode(stream.getvalue())
    if FIXED_EZDXF_MARKER.encode() not in data:
        raise RuntimeError(
            "DXF 内的 ezdxf 标记未被替换：ezdxf 实现变了，请复查 src/dxf_repro.py"
        )
    if b"\r\n" in data:
        raise RuntimeError("DXF 里出现 CRLF：写出路径没有走字节写入，请复查 src/dxf_repro.py")
    return data


def save_deterministic(doc: Drawing, path: str | "os.PathLike[str]") -> int:
    """确定性写盘，返回字节数。调用方负责目录已存在。"""
    from pathlib import Path

    data = dxf_bytes(doc)
    Path(path).write_bytes(data)
    return len(data)


__all__ = [
    "FIXED_EZDXF_MARKER",
    "assert_class_order",
    "deterministic_ezdxf",
    "dxf_bytes",
    "pin_class_order",
    "save_deterministic",
]
