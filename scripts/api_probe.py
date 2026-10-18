#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ezdxf API 契约核对 + dxf_writer 冒烟测试。

为什么需要这个脚本
------------------
`src/dxf_writer.py` 曾经对 ezdxf 的 11 处调用在 ezdxf 1.4.4 上不成立，而且失败是
**静默的**（被 `try/except` 吞成一条 warning），结果是"写出成功"但图元是错的。
本脚本把这些事实固化成断言：

- `--facts`  逐条核对当前 ezdxf 的 API 是否仍符合下列事实。**升级 ezdxf 后先跑这个**，
  任何一条从 PASS 变 FAIL 都意味着 `dxf_writer.py` 里的写法需要重新评估。
- `--smoke`  对真实的 `src/dxf_writer.py` 跑端到端写出并回读验收（含几何断言）。
- `--guards` 验证输入守卫与容差语义（缺失高度、退化矩阵、NaN、非法半径、自检容差边界）。

已核对的 API 事实（基线 ezdxf 1.4.4）
------------------------------------
1. `LWPolyline.set_bulge()` / `get_bulge()` **不存在** -> 写 bulge 只能用
   `add_lwpolyline(..., format="xyseb")` + 每顶点 5 元组 `(x, y, sw, ew, bulge)`
2. `add_lwpolyline()` 的关键字是 `close`（不是 `closed`）
3. `format="xyb"` 不是合法格式串，bulge 会被静默丢弃
4. `add_text()` 没有 `insert` 参数 -> 位置必须走 `dxfattribs={"insert": (x, y)}`
5. `MText.height` / `MText.width` 不是 DXF 属性（赋值静默无效）->
   必须用 `mt.dxf.char_height` / `mt.dxf.width`
6. `add_mtext()` 的合法属性名是 `insert`，`insertion_point` 抛 DXFAttributeError
7. `$COMMENTS` 不是合法头变量（DXFKeyError）。可落盘并回读的替代：
   `$LASTSAVEDBY` / `$PROJECTNAME` / `header.custom_vars`；
   `Drawing.comments` 是幽灵属性（可赋值，但类上没有该属性）
8. `f"{rms_mm:.4f}"` 在 `rms_mm=None` 时抛 TypeError
9. `LinetypeTable` 没有 `keys()`；可迭代（产出 Linetype 对象）且支持按名字 `in`
10. 图层可以引用不存在的线型而不报错，`doc.audit()` 也通过 -> 必须自己校验
11. DXF 的 `lineweight` 只接受有限档位（1/100mm）；`LWPOLYLINE.const_width` 直写毫米
12. 内置线型名是 `Continuous`（大小写与图层表里的 `CONTINUOUS` 不同）

用法
----
    python scripts/api_probe.py --facts
    python scripts/api_probe.py --smoke
    python scripts/api_probe.py --guards
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "src" / "dxf_writer.py"


def _require_deps():
    try:
        import ezdxf  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        print(f"需要 ezdxf 与 numpy：{exc}\n  python -m pip install -r requirements.txt")
        raise SystemExit(2)


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, got, want) -> None:
        ok = got == want
        self.failures += 0 if ok else 1
        line = f"  {'PASS' if ok else 'FAIL'} {label}: {got!r}"
        if not ok:
            line += f"  (期望 {want!r})"
        print(line)

    def summary(self, title: str) -> int:
        print(f"\n{title} failures={self.failures}")
        return 1 if self.failures else 0


# --------------------------------------------------------------------------- #
# --facts：核对 ezdxf 的 API 事实
# --------------------------------------------------------------------------- #

def facts() -> int:
    import inspect

    import ezdxf
    from ezdxf.entities import LWPolyline, MText

    rep = Report()
    print(f"ezdxf {ezdxf.__version__}")
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()

    print("\n[1-3] LWPolyline 的 bulge 写法")
    rep.check("LWPolyline 无 set_bulge", hasattr(LWPolyline, "set_bulge"), False)
    rep.check("LWPolyline 无 get_bulge", hasattr(LWPolyline, "get_bulge"), False)
    params = list(inspect.signature(msp.add_lwpolyline).parameters)
    rep.check("add_lwpolyline 关键字含 close", "close" in params, True)
    rep.check("add_lwpolyline 关键字不含 closed", "closed" in params, False)
    pl = msp.add_lwpolyline([(0, 0, 0, 0, 0.5), (10, 0, 0, 0, -0.5)],
                            format="xyseb", close=False)
    rep.check("format='xyseb' 保留 bulge",
              [round(v[4], 6) for v in pl.get_points(format="xyseb")], [0.5, -0.5])

    print("\n[4] add_text 的参数")
    tparams = list(inspect.signature(msp.add_text).parameters)
    rep.check("add_text 无 insert 参数", "insert" in tparams, False)
    t = msp.add_text("x", height=2.5, dxfattribs={"layer": "0", "insert": (1.0, 2.0)})
    rep.check("insert 走 dxfattribs 生效", (t.dxf.insert.x, t.dxf.insert.y), (1.0, 2.0))

    print("\n[5-6] MText 的属性名")
    rep.check("MText 无 DXF 属性 height", "height" in MText.DXFATTRIBS, False)
    rep.check("MText 有 DXF 属性 char_height", "char_height" in MText.DXFATTRIBS, True)
    mt = msp.add_mtext("x", dxfattribs={"layer": "0", "insert": (3.0, 4.0)})
    mt.dxf.char_height = 2.5
    rep.check("dxf.char_height 生效", mt.dxf.char_height, 2.5)
    rep.check("幽灵赋值 mt.height 不入属性表",
              "height" in mt.dxf.all_existing_dxf_attribs(), False)

    print("\n[7] 元数据落点")
    header_ok = {}
    for key in ("$COMMENTS", "$LASTSAVEDBY", "$PROJECTNAME"):
        try:
            doc.header[key] = "probe"
            header_ok[key] = True
        except Exception:  # noqa: BLE001
            header_ok[key] = False
    rep.check("$COMMENTS 非法（基线行为）", header_ok["$COMMENTS"], False)
    rep.check("$LASTSAVEDBY 合法", header_ok["$LASTSAVEDBY"], True)
    rep.check("$PROJECTNAME 合法", header_ok["$PROJECTNAME"], True)
    rep.check("Drawing 无 comments 属性", hasattr(doc, "comments"), False)

    print("\n[8] rms_mm=None 的格式化")
    rms = None
    try:
        f"{rms:.4f}"
        formatted = "ok"
    except TypeError:
        formatted = "TypeError"
    rep.check("None 用 :.4f 会抛 TypeError", formatted, "TypeError")

    print("\n[9-10] 线型表与图层校验")
    rep.check("LinetypeTable 无 keys()", hasattr(doc.linetypes, "keys"), False)
    names = [e.dxf.name for e in doc.linetypes]
    rep.check("内置线型名是 Continuous", "Continuous" in names, True)
    rep.check("内置线型名不是 CONTINUOUS", "CONTINUOUS" in names, False)
    rep.check("linetypes 支持按名字 in", "Continuous" in doc.linetypes, True)
    doc.layers.add("PROBE_BAD", color=7, linetype="NO_SUCH_LT")
    rep.check("图层可引用不存在的线型而不报错",
              doc.layers.get("PROBE_BAD").dxf.linetype, "NO_SUCH_LT")

    print("\n[11] 线宽属性")
    ln = msp.add_line((0, 0), (1, 1), dxfattribs={"layer": "0", "lineweight": 18})
    rep.check("LINE.lineweight 可写", ln.dxf.lineweight, 18)
    check_pl = msp.add_lwpolyline([(0, 0), (5, 0)], format="xyseb",
                                  dxfattribs={"layer": "0", "const_width": 0.35})
    rep.check("LWPOLYLINE.const_width 可写", check_pl.dxf.const_width, 0.35)

    return rep.summary("FACTS")


# --------------------------------------------------------------------------- #
# --smoke：对真实模块做端到端验收
# --------------------------------------------------------------------------- #

def smoke() -> int:
    import ezdxf

    rep = Report()
    out = Path(tempfile.gettempdir()) / "api_probe_smoke.dxf"
    metrics = out.with_name("metrics_s8.json")

    print("运行 python -m src.dxf_writer selftest（模块内置冒烟自检）")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "src.dxf_writer", "selftest", "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print("  exit code:", proc.returncode)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
    rep.check("selftest 退出码 0", proc.returncode, 0)
    if not out.exists():
        print("  未生成 DXF，无法继续")
        return rep.summary("SMOKE")

    doc = ezdxf.readfile(str(out))
    ents = list(doc.modelspace())
    kinds = sorted(e.dxftype() for e in ents)
    print("  回读实体:", kinds)
    rep.check("实体齐全", kinds, ["ARC", "CIRCLE", "LINE", "LWPOLYLINE", "TEXT"])

    arc = next(e for e in ents if e.dxftype() == "ARC")
    rep.check("ARC 0->90 经 Y 翻转写成 270->0",
              (round(arc.dxf.start_angle, 6), round(arc.dxf.end_angle, 6)), (270.0, 0.0))
    pl = next(e for e in ents if e.dxftype() == "LWPOLYLINE")
    rep.check("bulge 保留且随反射取反",
              [round(v[4], 6) for v in pl.get_points(format="xyseb")], [-0.5, 0.0, 0.0])
    rep.check("LWPOLYLINE const_width 写出毫米线宽",
              round(pl.dxf.const_width, 6), 0.5)
    txt = next(e for e in ents if e.dxftype() == "TEXT")
    rep.check("TEXT 高度落盘", round(txt.dxf.height, 6), 2.5)
    rep.check("TEXT 中文未损坏", "中文" in txt.dxf.text, True)
    rep.check("$INSUNITS = 4 (mm)", doc.header.get("$INSUNITS"), 4)
    rep.check("$MEASUREMENT = 1 (公制)", doc.header.get("$MEASUREMENT"), 1)
    rep.check("$PROJECTNAME 带 RMS",
              "RMS=" in (doc.header.get("$PROJECTNAME") or ""), True)
    layers = sorted(layer.dxf.name for layer in doc.layers)
    rep.check("CENTER 图层线型为 DASHED",
              doc.layers.get("CENTER").dxf.linetype, "DASHED")

    if metrics.exists():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        for key in ("ok", "entity_counts", "timing_ms", "error_estimate",
                    "warnings", "mismatches", "checked_entities"):
            rep.check(f"metrics_s8.json 含 {key}", key in data, True)
        rep.check("自检几何偏差为 0", data.get("max_geometry_deviation_mm"), 0.0)
    else:
        rep.check("生成 metrics_s8.json", False, True)

    for f in (out, metrics, out.with_name("dxf_report.txt")):
        if f.exists():
            os.remove(str(f))
    return rep.summary("SMOKE")


# --------------------------------------------------------------------------- #
# --guards：输入守卫与容差语义
# --------------------------------------------------------------------------- #

def guards() -> int:
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from src.dxf_writer import DXFSelfCheckError, DXFWriter, LayerConfig  # noqa: PLC0415

    rep = Report()

    def expect_raise(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"  PASS {label}: ValueError -> {str(exc)[:56]}...")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {label}: 抛了 {type(exc).__name__} 而非 ValueError")
        else:
            print(f"  FAIL {label}: 未报错")
        rep.failures += 1

    def writer(**kw):
        base = dict(image_height_mm=100.0,
                    layer_config=LayerConfig(bins=(0.25, 0.42),
                                             names=("LINES_W0.18", "LINES_W0.35",
                                                    "LINES_W0.50")))
        base.update(kw)
        return DXFWriter(**base)

    print("== 构造期守卫 ==")
    expect_raise("image_height_mm 缺失", lambda: DXFWriter())
    expect_raise("世界矩阵非 3x3",
                 lambda: DXFWriter(image_height_mm=100.0,
                                   transform_mm_img_to_world=np.eye(4)))
    expect_raise("世界矩阵退化 det=0",
                 lambda: DXFWriter(image_height_mm=100.0,
                                   transform_mm_img_to_world=np.array(
                                       [[1.0, 0, 0], [0, 0.0, 0], [0, 0, 1.0]])))
    expect_raise("LayerConfig 名字数不匹配",
                 lambda: LayerConfig(bins=(0.25,), names=("A",)))
    expect_raise("LayerConfig bins 未升序",
                 lambda: LayerConfig(bins=(0.5, 0.25), names=("A", "B", "C")))

    print("\n== 写出期守卫 ==")
    expect_raise("坐标含 NaN",
                 lambda: writer().add_line((0, float("nan")), (10, 10), "LINES_W0.35"))
    expect_raise("未知图层名",
                 lambda: writer().add_line((0, 0), (10, 10), "LINES_W9.99"))

    w = writer(transform_mm_img_to_world=np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]))
    ok = any("非单位缩放" in s for s in w.warnings)
    print(f"  {'PASS' if ok else 'FAIL'} 非单位缩放告警: {w.warnings}")
    rep.failures += 0 if ok else 1

    w2 = writer()
    w2.add_circle((0, 0), -1.0)
    w2.add_circle((0, 0), float("inf"))
    w2.add_arc((0, 0), 5.0, 30.0, 30.0)
    w2.add_arc((0, 0), 0.0, 0.0, 90.0)
    ok2 = w2._entities_written == 0 and len(w2.warnings) == 4
    print(f"  {'PASS' if ok2 else 'FAIL'} 非法半径/零扫掠角被丢弃: "
          f"写入 {w2._entities_written} 个，{len(w2.warnings)} 条告警")
    rep.failures += 0 if ok2 else 1

    print("\n== 自检容差语义（篡改登记几何后回读比对）==")
    out_dir = Path(tempfile.gettempdir()) / "api_probe_guards"
    out_dir.mkdir(exist_ok=True)

    w3 = writer()
    w3.add_line((0, 90), (10, 90), "LINES_W0.35")
    s = w3._written_geom[0]["start"]
    w3._written_geom[0]["start"] = (s[0], s[1] + 1e-4)      # 容差内
    try:
        r3 = w3.save(out_dir / "tol_ok.dxf", rms_mm=0.1)
        rep.check("偏差 1e-4 < 容差 1e-3 通过", r3["ok"], True)
        rep.check("最大偏差被记录", r3["max_geometry_deviation_mm"] > 0, True)
    except DXFSelfCheckError as exc:
        rep.check("偏差 1e-4 应通过", f"却失败: {exc}", True)

    w4 = writer()
    w4.add_line((0, 90), (10, 90), "LINES_W0.35")
    s = w4._written_geom[0]["start"]
    w4._written_geom[0]["start"] = (s[0], s[1] + 2e-3)      # 超容差
    try:
        r4 = w4.save(out_dir / "tol_bad.dxf", rms_mm=0.1)
        rep.check("偏差 2e-3 > 容差 1e-3 必须失败", r4["ok"], False)
    except DXFSelfCheckError:
        print("  PASS 偏差 2e-3 > 容差 1e-3 被自检抓住")
    rep.check("自检失败异常带退出码 3", DXFSelfCheckError.exit_code, 3)

    for f in out_dir.glob("*"):
        f.unlink()
    out_dir.rmdir()
    return rep.summary("GUARDS")


def main() -> int:
    parser = argparse.ArgumentParser(description="ezdxf API 契约核对与 dxf_writer 冒烟测试")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--facts", action="store_true", help="核对 ezdxf 的 API 事实")
    group.add_argument("--smoke", action="store_true", help="对真实模块做端到端验收")
    group.add_argument("--guards", action="store_true", help="验证输入守卫与容差语义")
    args = parser.parse_args()
    _require_deps()
    if args.facts:
        return facts()
    if args.smoke:
        return smoke()
    return guards()


if __name__ == "__main__":
    sys.exit(main())
