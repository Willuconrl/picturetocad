# src/s3_skeleton.py
"""S3 骨架化与拓扑重建（SPEC §4 S3：最容易崩的环节，单独验收）。

流程
----
1. **双路径骨架**：形态学细化（`skimage.morphology.thin`）与距离变换中轴
   （`medial_axis`）各出一张骨架，用「沿骨架的笔画宽度变异系数」选更稳的那张
   （CV 小 = 粗细均匀 = 交叉处没有多余分支）。
2. **图构建**：8 邻域度数 ≠2 的像素聚类成节点，度数 =2 的链成边。
3. **毛刺修剪**：一端是端点、长度 < `spur_removal.length_threshold_linewidths × 本地线宽`
   的边剪掉；`endpoint_curvature_check` 时，与母边夹角很小的短尾视为线段的自然结尾，
   保留（不误删真端点）。
4. **节点合并**：`node.redundant_collapse` 时，度 2 且偏离两侧连线 ≤ `collinear_tol_px`
   的节点合并；`node.snap_distance_linewidths` 内相邻节点吸附成一个。
5. **节点分类**：端点 / 角点 / T / Y / 十字；判定用入射角与 `node.angle_tol_deg`。
6. **断线桥接**：两个端点间隙 ≤ `bridge.max_gap_linewidths × 本地线宽`、方向延续性
   ≤ `bridge.max_angle_deg` 时连一条边，并标 `bridged: true`。

坐标契约（见 `docs/coordinate_system.md`）
------------------------------------------
沿用 S2 的**连续像素坐标**：像素 (col,row) 覆盖 [col,col+1)×[row,row+1)，中心在
(col+0.5, row+0.5)。骨架像素天然在格点上，因此节点/边坐标一律 +0.5 后再写出，
下游（S4/S7）不需要再做半像素补偿。

确定性
------
节点与边按坐标字典序编号；所有集合类中间结果都 `sorted()` 后使用；
同一输入两次运行产出的 graph.json 必须**逐字节相同**（由测试保证）。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from skimage.morphology import medial_axis, skeletonize, thin

from src.artifacts import StageMetrics, write_json, write_metrics
from src.config import R2VConfig

NodeType = Literal["endpoint", "corner", "tee", "wye", "cross", "isolated"]

#: 8 邻域偏移（dx, dy），顺序固定 → 遍历确定性
_NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)

#: 折线简化容差（px）：只用于压掉骨架的锯齿，不改变拓扑
_RDP_TOLERANCE_PX = 0.5

#: 拐角判定：简化后折线在某个顶点的转向角 ≥ 此值就切成两个边并建"角点"节点。
#: 这是**分类器内部常数**（骨架锯齿通常是 45° 以内的小转向），不是精度判据。
_CORNER_TURN_DEG = 20.0

#: 把一段短尾判定为"母边的自然结尾"的夹角阈值（度）
_COLLINEAR_TAIL_DEG = 20.0


class SkeletonError(ValueError):
    """S3 输入或参数非法。"""


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphNode(_Model):
    id: int
    x: float
    y: float
    type: NodeType
    degree: int
    edges: tuple[int, ...]
    incident_angles_deg: tuple[float, ...]
    merged_from: int = Field(default=1, description="合并了多少个原始骨架节点像素簇")


class GraphEdge(_Model):
    id: int
    node_a: int
    node_b: int
    points: tuple[tuple[float, float], ...]
    length_px: float
    linewidth_px: float
    bridged: bool = False
    closed: bool = False
    closed_loop: bool = Field(
        default=False,
        description="整条边是一个闭合环（图框/窗洞轮廓）。S8 读它写单条 closed=True 的 LWPOLYLINE，"
        "而不是拆成若干独立 LINE——闭合语义在 CAD/CAM 里是有意义的拓扑。",
    )


class TopologyGraph(_Model):
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    skeleton_path: Literal["thin", "medial_axis"]
    width_cv: dict[str, float]

    def node(self, node_id: int) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise SkeletonError(f"没有 id={node_id} 的节点")

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.type] = counts.get(node.type, 0) + 1
        return dict(sorted(counts.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "skeleton_path": self.skeleton_path,
            "width_cv": dict(sorted(self.width_cv.items())),
            "nodes": [n.model_dump(mode="json") for n in self.nodes],
            "edges": [e.model_dump(mode="json") for e in self.edges],
        }


# ---------------------------------------------------------------------------
# 骨架与图构建
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _RawEdge:
    a: int          # 节点像素索引
    b: int
    points: list[tuple[float, float]]   # 连续坐标（已 +0.5）
    pixels: list[tuple[int, int]]       # 骨架像素 (x, y) 索引


def _neighbours(x: int, y: int) -> Iterable[tuple[int, int]]:
    for dx, dy in _NEIGHBOURS:
        yield x + dx, y + dy


def _skeleton_arrays(binary: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """两路骨架 + 各自的距离变换（用于线宽与选择）。"""
    mask = binary > 0
    if not mask.any():
        raise SkeletonError("S3 输入二值图为空：没有可骨架化的前景")
    thin_skel = thin(mask).astype(bool)
    thin_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    # rng 必须固定：medial_axis 默认用随机数做**距离相等时的平局裁决**，
    # 不固定会出现"同一输入两次骨架像素数不同（792 vs 793）"，直接破坏确定性契约。
    medial_skel, medial_distance = medial_axis(mask, return_distance=True, rng=0)
    return {
        "thin": (thin_skel, thin_distance.astype(np.float32)),
        "medial_axis": (medial_skel.astype(bool), medial_distance.astype(np.float32)),
    }


def _width_cv(skeleton: np.ndarray, distance: np.ndarray) -> float:
    values = 2.0 * distance[skeleton]
    if values.size < 2:
        return 0.0
    mean = float(values.mean())
    return float(values.std() / mean) if mean > 1e-9 else 0.0


def _degree_map(skeleton: np.ndarray) -> dict[tuple[int, int], int]:
    ys, xs = np.nonzero(skeleton)
    degree: dict[tuple[int, int], int] = {}
    for x, y in zip(xs.tolist(), ys.tolist()):
        count = 0
        for nx, ny in _neighbours(x, y):
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx]:
                count += 1
        degree[(x, y)] = count
    return degree


def _cluster_nodes(node_pixels: set[tuple[int, int]]) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    """把相邻的节点像素聚成节点簇；返回簇列表与 像素→簇号 映射（按坐标排序保证确定性）。"""
    clusters: list[list[tuple[int, int]]] = []
    mapping: dict[tuple[int, int], int] = {}
    for start in sorted(node_pixels, key=lambda p: (p[1], p[0])):
        if start in mapping:
            continue
        index = len(clusters)
        members: list[tuple[int, int]] = []
        queue = deque([start])
        mapping[start] = index
        while queue:
            x, y = queue.popleft()
            members.append((x, y))
            for nx, ny in _neighbours(x, y):
                if (nx, ny) in node_pixels and (nx, ny) not in mapping:
                    mapping[(nx, ny)] = index
                    queue.append((nx, ny))
        clusters.append(sorted(members, key=lambda p: (p[1], p[0])))
    return clusters, mapping


def _trace_edges(
    skeleton: np.ndarray,
    node_pixels: set[tuple[int, int]],
    mapping: dict[tuple[int, int], int],
) -> list[_RawEdge]:
    """沿着度 2 像素链追边。"""
    edges: list[_RawEdge] = []
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def walk(start: tuple[int, int], first: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
        path = [start]
        previous = start
        current = first
        while current not in node_pixels:
            path.append(current)
            nxt = None
            for candidate in sorted(_neighbours(*current), key=lambda p: (p[1], p[0])):
                if candidate == previous:
                    continue
                cy, cx = candidate[1], candidate[0]
                if 0 <= cy < skeleton.shape[0] and 0 <= cx < skeleton.shape[1] and skeleton[cy, cx]:
                    nxt = candidate
                    break
            if nxt is None:
                break
            previous, current = current, nxt
        return path, current

    for node in sorted(node_pixels, key=lambda p: (p[1], p[0])):
        for neighbour in sorted(_neighbours(*node), key=lambda p: (p[1], p[0])):
            x, y = neighbour
            if not (0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1] and skeleton[y, x]):
                continue
            if (node, neighbour) in visited:
                continue
            path, endpoint = walk(node, neighbour)
            visited.add((node, neighbour))
            # 反向也要标记，否则从另一端会再追一条完全相同的边
            if len(path) >= 2:
                visited.add((endpoint, path[-1]))
            else:
                visited.add((endpoint, node))
            if mapping[node] == mapping.get(endpoint, -1):
                # 同一个节点簇内部相邻像素之间会追出长度为 1px 的自环，全是垃圾
                continue
            pixels = path + [endpoint]
            points = [((px + 0.5), (py + 0.5)) for px, py in pixels]
            edges.append(_RawEdge(a=mapping[node], b=mapping[endpoint], points=points, pixels=pixels))
    return edges


def _rdp(points: Sequence[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Douglas–Peucker 简化（迭代实现，避免深递归）。"""
    if len(points) <= 2:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        ax, ay = points[start]
        bx, by = points[end]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        best_index, best_distance = -1, -1.0
        for index in range(start + 1, end):
            px, py = points[index]
            if norm < 1e-12:
                distance = math.hypot(px - ax, py - ay)
            else:
                distance = abs(dy * px - dx * py + bx * ay - by * ax) / norm
            if distance > best_distance:
                best_index, best_distance = index, distance
        if best_distance > tolerance:
            keep[best_index] = True
            stack.append((start, best_index))
            stack.append((best_index, end))
    return [point for point, kept in zip(points, keep) if kept]


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return float(
        sum(
            math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            for i in range(len(points) - 1)
        )
    )


# ---------------------------------------------------------------------------
# 拓扑操作
# ---------------------------------------------------------------------------
class _MutableGraph:
    """中间表示：节点坐标 + 边（邻接表），便于反复剪枝/合并。"""

    def __init__(self) -> None:
        self.nodes: dict[int, list[float]] = {}
        self.edges: dict[int, dict[str, Any]] = {}
        self._next_node = 0
        self._next_edge = 0

    def add_node(self, x: float, y: float, merged: int = 1) -> int:
        node_id = self._next_node
        self._next_node += 1
        self.nodes[node_id] = [float(x), float(y), float(merged)]
        return node_id

    def add_edge(
        self,
        a: int,
        b: int,
        points: Sequence[tuple[float, float]],
        *,
        linewidth_px: float,
        bridged: bool = False,
        closed: bool = False,
    ) -> int:
        edge_id = self._next_edge
        self._next_edge += 1
        pts = list(points)
        self.edges[edge_id] = {
            "a": a,
            "b": b,
            "points": pts,
            "length": _polyline_length(pts),
            "linewidth": float(linewidth_px),
            "bridged": bool(bridged),
            "closed": bool(closed),
        }
        return edge_id

    def incident(self, node_id: int) -> list[int]:
        return sorted(eid for eid, e in self.edges.items() if e["a"] == node_id or e["b"] == node_id)

    def other(self, edge_id: int, node_id: int) -> int:
        edge = self.edges[edge_id]
        return edge["b"] if edge["a"] == node_id else edge["a"]

    def degree(self, node_id: int) -> int:
        return len(self.incident(node_id))

    def remove_edge(self, edge_id: int) -> None:
        self.edges.pop(edge_id, None)

    def remove_node(self, node_id: int) -> None:
        self.nodes.pop(node_id, None)

    def prune_isolated(self) -> None:
        for node_id in sorted(self.nodes):
            if not self.incident(node_id):
                self.remove_node(node_id)


def _edge_direction_at(edge: dict[str, Any], node_id: int) -> tuple[float, float]:
    """从 node_id 出发沿该边前 5 个点的方向（单位向量）。"""
    points = edge["points"]
    if edge["b"] == node_id:
        points = list(reversed(points))
    if len(points) < 2:
        return (0.0, 0.0)
    look = min(5, len(points) - 1)
    dx = points[look][0] - points[0][0]
    dy = points[look][1] - points[0][1]
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 1e-12 else (0.0, 0.0)


def prune_spurs(
    graph: _MutableGraph,
    *,
    linewidth_px: float,
    length_threshold_linewidths: float,
    endpoint_curvature_check: bool,
) -> int:
    """删掉短毛刺，返回删除条数。

    规则：一端度为 1、长度 < 阈值×线宽 → 候选毛刺；
    若开启 `endpoint_curvature_check` 且它与母边几乎共线（夹角 ≤20°），
    视为线段自然结尾（缩细/像素化造成），**保留**——这就是「不误删真端点」。
    """
    threshold = max(1.0, length_threshold_linewidths * linewidth_px)
    removed = 0
    changed = True
    while changed:
        changed = False
        for edge_id in sorted(graph.edges):
            edge = graph.edges[edge_id]
            a, b = edge["a"], edge["b"]
            if a == b:
                continue
            if graph.degree(a) != 1 and graph.degree(b) != 1:
                continue
            if edge["length"] >= threshold:
                continue
            endpoint = a if graph.degree(a) == 1 else b
            parent_node = b if endpoint == a else a
            if endpoint_curvature_check:
                parents = [eid for eid in graph.incident(parent_node) if eid != edge_id]
                if parents:
                    spur_dir = _edge_direction_at(edge, endpoint)
                    parent_dir = _edge_direction_at(graph.edges[parents[0]], parent_node)
                    cosine = max(-1.0, min(1.0, -(
                        spur_dir[0] * parent_dir[0] + spur_dir[1] * parent_dir[1]
                    )))
                    angle = math.degrees(math.acos(cosine))
                    if angle <= _COLLINEAR_TAIL_DEG:
                        continue  # 与母边共线：是线段的自然结尾，不是毛刺
            graph.remove_edge(edge_id)
            graph.remove_node(endpoint)
            removed += 1
            changed = True
    graph.prune_isolated()
    return removed


def merge_collinear_nodes(graph: _MutableGraph, *, tolerance_px: float) -> int:
    """合并度 2 且几乎在一条直线上的节点（共线连续节点）。"""
    merged = 0
    changed = True
    while changed:
        changed = False
        for node_id in sorted(graph.nodes):
            incident = graph.incident(node_id)
            if len(incident) != 2:
                continue
            first, second = incident
            e1, e2 = graph.edges[first], graph.edges[second]
            other_a = graph.other(first, node_id)
            other_b = graph.other(second, node_id)
            if other_a == other_b:  # 自环
                continue
            x, y = graph.nodes[other_a][:2]
            x2, y2 = graph.nodes[other_b][:2]
            nx, ny = graph.nodes[node_id][:2]
            dx, dy = x2 - x, y2 - y
            norm = math.hypot(dx, dy)
            if norm < 1e-9:
                continue
            deviation = abs(dy * nx - dx * ny + x2 * y - y2 * x) / norm
            if deviation > tolerance_px:
                continue
            points = list(e1["points"]) if e1["b"] == node_id else list(reversed(e1["points"]))
            points2 = list(e2["points"]) if e2["a"] == node_id else list(reversed(e2["points"]))
            combined = points + points2[1:]
            linewidth = (e1["linewidth"] + e2["linewidth"]) / 2.0
            graph.remove_edge(first)
            graph.remove_edge(second)
            graph.remove_node(node_id)
            graph.add_edge(other_a, other_b, combined, linewidth_px=linewidth,
                           bridged=e1["bridged"] or e2["bridged"])
            merged += 1
            changed = True
            break
    return merged


def snap_close_nodes(graph: _MutableGraph, *, distance_px: float) -> int:
    """把距离小于 `distance_px` 的节点吸成一个（端点抖动造成的伪双节点）。"""
    snapped = 0
    changed = True
    while changed:
        changed = False
        ids = sorted(graph.nodes)
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                if first not in graph.nodes or second not in graph.nodes:
                    continue
                ax, ay = graph.nodes[first][:2]
                bx, by = graph.nodes[second][:2]
                if math.hypot(ax - bx, ay - by) > distance_px:
                    continue
                for edge_id in graph.incident(second):
                    edge = graph.edges[edge_id]
                    if edge["a"] == second:
                        edge["a"] = first
                    if edge["b"] == second:
                        edge["b"] = first
                graph.remove_node(second)
                snapped += 1
                changed = True
                break
            if changed:
                break
    return snapped


def bridge_gaps(
    graph: _MutableGraph,
    *,
    linewidth_px: float,
    max_gap_linewidths: float,
    max_angle_deg: float,
) -> int:
    """把间隙不大且方向延续的两个端点连起来，返回桥接条数。"""
    max_gap = max(1.0, max_gap_linewidths * linewidth_px)
    endpoints = [node for node in sorted(graph.nodes) if graph.degree(node) == 1]
    connected = {
        (min(edge["a"], edge["b"]), max(edge["a"], edge["b"])) for edge in graph.edges.values()
    }
    candidates: list[tuple[float, int, int]] = []
    for i, first in enumerate(endpoints):
        ax, ay = graph.nodes[first][:2]
        for second in endpoints[i + 1:]:
            if (min(first, second), max(first, second)) in connected:
                continue  # 已经直连，桥接会造出平行边
            bx, by = graph.nodes[second][:2]
            distance = math.hypot(ax - bx, ay - by)
            if distance > max_gap:
                continue
            candidates.append((distance, first, second))
    candidates.sort()
    bridged = 0
    used: set[int] = set()
    for distance, first, second in candidates:
        if first in used or second in used:
            continue
        ax, ay = graph.nodes[first][:2]
        bx, by = graph.nodes[second][:2]
        to_other = (bx - ax, by - ay)
        norm = math.hypot(*to_other)
        if norm < 1e-9:
            continue
        d = (to_other[0] / norm, to_other[1] / norm)
        # 方向延续性：边的**外指方向**（= -沿边方向）与"指向对方"的夹角 ≤ max_angle_deg
        ok = True
        for node, direction in ((first, d), (second, (-d[0], -d[1]))):
            edge_ids = graph.incident(node)
            if not edge_ids:
                ok = False
                break
            edge_dir = _edge_direction_at(graph.edges[edge_ids[0]], node)
            outward = (-edge_dir[0], -edge_dir[1])
            cosine = max(-1.0, min(1.0, outward[0] * direction[0] + outward[1] * direction[1]))
            angle = math.degrees(math.acos(cosine))
            if angle > max_angle_deg:
                ok = False
                break
        if not ok:
            continue
        graph.add_edge(
            first, second, [(ax, ay), (bx, by)], linewidth_px=linewidth_px, bridged=True
        )
        used.update({first, second})
        bridged += 1
    return bridged


def _resample_polyline(points: list[tuple[float, float]], *, step: float) -> list[tuple[float, float]]:
    """按弧长 step 线性重采样折线（用于窗口转角）。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return [tuple(p) for p in pts]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    total = float(seg.sum())
    if total < 1e-9:
        return [tuple(p) for p in pts]
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    count = max(2, int(total / max(step, 1e-6)) + 1)
    targets = np.linspace(0.0, total, count)
    xs = np.interp(targets, cumulative, pts[:, 0])
    ys = np.interp(targets, cumulative, pts[:, 1])
    return list(zip(xs.tolist(), ys.tolist()))


def _window_turns(dense: list[tuple[float, float]], w: int) -> list[float]:
    """每点的**窗口转角**（度）：用前后相隔 w 个采样（≈w px）的弦夹角。

    返回长度与 dense 相同的列表，两端（不足 w）填 0。
    """
    count = len(dense)
    turns = [0.0] * count
    pts = np.asarray(dense, dtype=np.float64)
    for index in range(w, count - w):
        v1 = pts[index] - pts[index - w]
        v2 = pts[index + w] - pts[index]
        n1 = float(np.hypot(*v1))
        n2 = float(np.hypot(*v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosine = max(-1.0, min(1.0, float(v1 @ v2) / (n1 * n2)))
        turns[index] = math.degrees(math.acos(cosine))
    return turns


def _neighbour_turn_baseline(turns: list[float], index: int, w: int) -> float:
    """拐点邻域之外的"基线转角"中位数：圆弧上它与拐点转角同量级，真拐点远小于它。"""
    count = len(turns)
    span = max(2 * w, w + 2)
    samples = [
        turns[i]
        for i in range(0, count, max(1, w // 2))
        if abs(i - index) > span and turns[i] > 0.0
    ]
    # 少于 3 个基线样本（短边/短弧）视为证据不足，返回 -1 → 保守拒绝切分
    return float(np.median(samples)) if len(samples) >= 3 else -1.0


def _project_back(
    raw_points: object,
    rdp_points: list[tuple[float, float]],
    dense: list[tuple[float, float]],
    dense_index: int,
) -> tuple[list[tuple[float, float]], int]:
    """把重采样坐标映射回**原始折线**上最近的点索引（切点必须落在真实采样点上）。"""
    raw = [(float(p[0]), float(p[1])) for p in raw_points]  # type: ignore[union-attr]
    if not raw:
        return rdp_points, -1
    target = np.asarray(dense[dense_index], dtype=np.float64)
    array = np.asarray(raw, dtype=np.float64)
    index = int(np.argmin(np.hypot(array[:, 0] - target[0], array[:, 1] - target[1])))
    return raw, index


def split_corners(
    graph: _MutableGraph,
    *,
    linewidth_px: float,
    min_arm_ratio: float = 2.0,
    turn_deg: float = _CORNER_TURN_DEG,
    simplify_tol: float = 1.0,
    window_linewidths: float = 2.0,
    contrast_ratio: float = 2.0,
) -> int:
    """把折线里明显的拐点切成独立节点（类型 corner），返回切分次数。

    要求拐点两侧的臂都 ≥ `min_arm_ratio × 线宽`：T 形交叉处细化会留下 3–4 px 的小钩子，
    也满足"转向 > 30°"，若不设这条限制，每个 T 都会被切成两个假角点。
    """
    split = 0
    min_arm = max(3.0, min_arm_ratio * linewidth_px)
    w_samples = max(4, int(round(window_linewidths * linewidth_px)))
    changed = True
    while changed:
        changed = False
        for edge_id in sorted(graph.edges):
            edge = graph.edges[edge_id]
            if edge["a"] == edge["b"]:
                # 闭合环暂不做拐点切分：环的追踪在 8 连通下可能退化成螺旋，
                # 强切会把一个圆切成 4 段（实测 S4 因此测不出 CIRCLE）。
                # 图框等闭合多边形的拐点切分留待 M7 与 S8 一起处理，见 README 已知限制。
                continue
            points = _rdp(edge["points"], simplify_tol)
            if len(points) < 3:
                continue
            # M9-4.1：拐角在骨架上是**圆化**的（笔宽 5px 时 120° 内角摊到十几个像素上），
            # 单点转角永远到不了 30°，因此改用**窗口转角**：在 1px 重采样后取 ±w 的弦夹角，
            # 并用"拐点转角 / 邻域基线转角"的对比度排除圆弧（圆弧的转角沿弧长均匀分布）。
            dense = _resample_polyline([(p[0], p[1]) for p in points], step=1.0)
            if len(dense) < 2 * w_samples + 1:
                continue
            turns = _window_turns(dense, w_samples)
            best_index = int(np.argmax(turns)) if turns else -1
            best_turn = float(turns[best_index]) if best_index >= 0 else 0.0
            if best_index < w_samples or best_turn < turn_deg:
                continue
            # 拐点 vs 圆弧的稳定判别：**转角饱和**。拐点在窗口加倍后转角几乎不变
            # （t2 ≈ t1），圆弧的转角与窗口成正比（t2 ≈ 2·t1）。用 t1/t2 做判据，
            # 不依赖邻域基线，故对短边/短弧与像素量化都稳。
            wide = _window_turns(dense, 2 * w_samples)
            if best_index >= 2 * w_samples:
                wide_turn = float(wide[best_index])
            else:
                wide_turn = float(wide[min(len(wide) - 1, 2 * w_samples)])
            if wide_turn <= 1e-6 or (best_turn / wide_turn) < contrast_ratio:
                continue
            left_arm = _polyline_length(dense[: best_index + 1])
            right_arm = _polyline_length(dense[best_index:])
            if left_arm < min_arm or right_arm < min_arm:
                continue
            # 切回原始（未重采样）点列：按弧长比例取最近点
            points, best_index = _project_back(edge["points"], points, dense, best_index)
            if len(points) < 3 or not 0 < best_index < len(points) - 1:
                continue
            a, b = edge["a"], edge["b"]
            corner = graph.add_node(points[best_index][0], points[best_index][1])
            left = points[: best_index + 1]
            right = points[best_index:]
            width = float(edge["linewidth"])
            bridged = bool(edge["bridged"])
            graph.remove_edge(edge_id)
            graph.add_edge(a, corner, left, linewidth_px=width, bridged=bridged)
            graph.add_edge(corner, b, right, linewidth_px=width, bridged=bridged)
            split += 1
            changed = True
            break
    return split


def _classify(
    angles: Sequence[float], degree: int, *, angle_tol_deg: float
) -> NodeType:
    """按入射角与度数分类。角度是**从节点指向边**的方向角（度，0..360）。"""
    if degree == 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree == 2:
        # 完全共线的度 2 节点应在合并阶段消掉；剩下的就是拐点
        return "corner"
    if degree == 3:
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = abs((angles[i] - angles[j]) % 360.0)
                straight = min(diff, 360.0 - diff)
                if abs(straight - 180.0) <= angle_tol_deg:
                    return "tee"
        return "wye"
    return "cross"


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """8 邻域连通分量（返回每块的布尔 mask，顺序按最小像素坐标排序，保证确定性）。"""
    remaining = mask.copy()
    components: list[np.ndarray] = []
    ys, xs = np.nonzero(remaining)
    for y, x in sorted(zip(ys.tolist(), xs.tolist()), key=lambda p: (p[0], p[1])):
        if not remaining[y, x]:
            continue
        stack = [(x, y)]
        remaining[y, x] = False
        members: list[tuple[int, int]] = []
        while stack:
            cx, cy = stack.pop()
            members.append((cx, cy))
            for nx, ny in _neighbours(cx, cy):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    stack.append((nx, ny))
        block = np.zeros_like(mask, dtype=bool)
        for cx, cy in members:
            block[cy, cx] = True
        components.append(block)
    return components


def _loop_parts(
    skeleton: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, list[tuple[float, float]], bool]]]:
    """没有节点像素的纯环（例如理想圆环）：返回 (节点坐标表, [(a,b,points,closed)])。"""
    ys, xs = np.nonzero(skeleton)
    if len(xs) == 0:
        return [], []
    order = sorted(range(len(xs)), key=lambda i: (int(ys[i]), int(xs[i])))
    start = (int(xs[order[0]]), int(ys[order[0]]))
    path = [start]
    previous: tuple[int, int] | None = None
    current = start
    closed = False
    limit = int(skeleton.sum()) + 4
    while len(path) <= limit:
        nxt = None
        for candidate in sorted(_neighbours(*current), key=lambda p: (p[1], p[0])):
            cx, cy = candidate
            if not (0 <= cy < skeleton.shape[0] and 0 <= cx < skeleton.shape[1] and skeleton[cy, cx]):
                continue
            if candidate == previous:
                continue
            if candidate == start and len(path) > 3:
                path.append(start)
                closed = True
                nxt = None
                break
            nxt = candidate
            break
        if closed:
            break
        if nxt is None:
            break
        previous, current = current, nxt
        path.append(current)
    points = [(px + 0.5, py + 0.5) for px, py in path]
    if closed:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        return [(cx, cy)], [(0, 0, points, True)]
    return [points[0], points[-1]], [(0, 1, points, False)]


def build_graph(
    skeleton: np.ndarray,
    distance: np.ndarray,
    *,
    cfg: R2VConfig,
    linewidth_px: float,
) -> tuple[_MutableGraph, int, int, int]:
    """骨架 → 可变的节点/边图；返回 (graph, 原始节点数, 原始边数, 修剪条数)。"""
    degree = _degree_map(skeleton)
    node_pixels = {pixel for pixel, value in degree.items() if value != 2}
    clusters, mapping = _cluster_nodes(node_pixels)
    raw_edges = _trace_edges(skeleton, node_pixels, mapping)

    graph = _MutableGraph()
    node_ids: dict[int, int] = {}
    for index, members in enumerate(clusters):
        cx = sum(p[0] + 0.5 for p in members) / len(members)
        cy = sum(p[1] + 0.5 for p in members) / len(members)
        node_ids[index] = graph.add_node(cx, cy, merged=len(members))

    for edge in raw_edges:
        points = _rdp(edge.points, _RDP_TOLERANCE_PX)
        widths = [2.0 * float(distance[y, x]) for x, y in edge.pixels]
        local_width = float(np.median(widths)) if widths else linewidth_px
        graph.add_edge(
            node_ids[edge.a], node_ids[edge.b], points, linewidth_px=local_width,
            closed=edge.a == edge.b,
        )

    covered = np.zeros_like(skeleton, dtype=bool)
    for edge in raw_edges:
        for x, y in edge.pixels:
            covered[y, x] = True
    remaining = skeleton & ~covered
    if remaining.any():
        # 没有节点的纯环（矩形框、圆环）：之前只要图里有任何节点，这些环就会被整条丢掉
        # （实测：图框矩形的骨架凭空消失，S4 只拟出内部那根线）
        for component in _components(remaining):
            node_coords, loop_edges = _loop_parts(component)
            local_ids = [graph.add_node(float(cx), float(cy)) for cx, cy in node_coords]
            sample_edge = raw_edges[0] if raw_edges else None
            width = sample_edge.pixels if sample_edge else []
            local_width = linewidth_px
            if width:
                values = [2.0 * float(distance[y, x]) for x, y in width]
                local_width = float(np.median(values)) if values else linewidth_px
            for a, b, points, closed in loop_edges:
                graph.add_edge(
                    local_ids[a], local_ids[b], _rdp(points, _RDP_TOLERANCE_PX),
                    linewidth_px=local_width, closed=closed,
                )

    raw_nodes = len(graph.nodes)
    raw_edge_count = len(graph.edges)
    pruned = prune_spurs(
        graph,
        linewidth_px=linewidth_px,
        length_threshold_linewidths=cfg.skeleton.spur_removal.length_threshold_linewidths,
        endpoint_curvature_check=cfg.skeleton.spur_removal.endpoint_curvature_check,
    )
    return graph, raw_nodes, raw_edge_count, pruned


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class S3Result:
    graph: TopologyGraph
    warnings: tuple[str, ...]
    pruned_spurs: int
    merged_nodes: int
    snapped_nodes: int
    bridged_edges: int
    split_corners: int
    raw_nodes: int
    raw_edges: int


def _line_fit_near_node(points: np.ndarray, node_xy: tuple[float, float], window_px: float) -> tuple[np.ndarray, np.ndarray] | None:
    """在靠近节点的一段（弧长 window_px）上做 TLS 直线拟合，返回 (点, 单位方向)。"""
    distances = np.hypot(points[:, 0] - node_xy[0], points[:, 1] - node_xy[1])
    mask = (distances >= 0.5 * window_px) & (distances <= window_px)
    sample = points[mask]
    if len(sample) < 3:
        return None
    centre = sample.mean(axis=0)
    _, _, vt = np.linalg.svd(sample - centre, full_matrices=False)
    direction = vt[0] / (np.linalg.norm(vt[0]) or 1.0)
    return centre, direction


def _intersect_lines(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
    (p1, d1), (p2, d2) = a, b
    matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(matrix)) < 1e-9:
        return None
    t = np.linalg.solve(matrix, p2 - p1)
    return p1 + t[0] * d1


def _fit_circle_kasa(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Kåsa 代数圆拟合（闭式解），返回 (圆心, 半径)；点数不足或退化时返回 None。"""
    if len(points) < 5:
        return None
    x = points[:, 0]
    y = points[:, 1]
    matrix = np.column_stack([x, y, np.ones_like(x)])
    rhs = x * x + y * y
    try:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = solution[0] / 2.0
    cy = solution[1] / 2.0
    value = solution[2] + cx * cx + cy * cy
    if not np.isfinite(value) or value <= 0:
        return None
    return np.array([cx, cy]), float(math.sqrt(value))


def _line_circle_intersections(
    point: np.ndarray, direction: np.ndarray, centre: np.ndarray, radius: float
) -> list[np.ndarray]:
    """直线与圆的闭式交点（解一元二次方程，无迭代）。"""
    offset = point - centre
    b = 2.0 * float(offset @ direction)
    c = float(offset @ offset) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return []
    root = math.sqrt(disc)
    return [point + t * direction for t in ((-b - root) / 2.0, (-b + root) / 2.0)]


def sharpen_corners(graph: TopologyGraph, *, min_angle_deg: float, window_px: float) -> tuple[TopologyGraph, int]:
    """把 line↔line 的 `corner` 节点替换为**两侧直线段的解析交点**（M9-1）。

    形态学骨架在折点处必然"切角"：30° 内角内缩 0.86mm、60° 0.30mm、90° 0.18mm
    （见 `scripts/diagnose_corner.py`）。这里对每个 `corner` 节点，用两侧近端一段点
    各拟合一条直线，求交后把节点与相邻边的端点都移到交点上；内角小于门限（急弯）
    时保持原样，避免破坏端帽。

    只处理**度=2 且两条边都是直线型**（点数足够且切线连续）的拐点；拓扑与段数不变。
    """
    nodes = [node.model_copy() for node in graph.nodes]
    edges = [edge.model_copy() for edge in graph.edges]
    by_id = {node.id: index for index, node in enumerate(nodes)}
    sharpened = 0
    for node in nodes:
        if node.type != "corner" or node.degree != 2 or len(node.edges) != 2:
            continue
        incident = [edges[edge_id] for edge_id in node.edges if edge_id < len(edges)]
        if len(incident) != 2 or any(edge.closed or edge.closed_loop for edge in incident):
            continue
        node_xy = (node.x, node.y)
        fits = []
        for edge in incident:
            pts = np.asarray(edge.points, dtype=np.float64)
            if len(pts) < 4:
                fits = []
                break
            fits.append(_line_fit_near_node(pts, node_xy, window_px))
        if len(fits) != 2 or fits[0] is None or fits[1] is None:
            continue
        (c1, d1), (c2, d2) = fits
        # 方向统一成"远离节点"，内角 = 两方向夹角
        v1 = c1 - np.asarray(node_xy)
        v2 = c2 - np.asarray(node_xy)
        if d1 @ v1 < 0:
            d1 = -d1
        if d2 @ v2 < 0:
            d2 = -d2
        cosine = float(np.clip(d1 @ d2, -1.0, 1.0))
        interior_deg = math.degrees(math.acos(cosine))
        if interior_deg < min_angle_deg:
            continue
        intersection = _intersect_lines((c1, d1), (c2, d2))
        if intersection is None:
            # line↔arc 接缝：一侧拟合直线，另一侧拟合圆，求闭式交点（M9-1 Phase 3）
            circles = []
            for edge in incident:
                pts = np.asarray(edge.points, dtype=np.float64)
                circles.append(_fit_circle_kasa(pts))
            pair = None
            for index in (0, 1):
                circle = circles[index]
                line_fit = fits[1 - index]
                if circle is None or line_fit is None:
                    continue
                centre_c, radius_c = circle
                line_point, line_dir = line_fit
                # 圆拟合残差要明显小于直线拟合，才认定为"弧侧"
                pts_arc = np.asarray(incident[index].points, dtype=np.float64)
                circle_rms = float(np.sqrt(np.mean((np.hypot(pts_arc[:,0]-centre_c[0], pts_arc[:,1]-centre_c[1]) - radius_c) ** 2)))
                line_rms = float(np.sqrt(np.mean(((pts_arc - line_point) @ np.array([-line_dir[1], line_dir[0]])) ** 2)))
                if circle_rms >= line_rms:
                    continue
                candidates = _line_circle_intersections(line_point, line_dir, centre_c, radius_c)
                if not candidates:
                    continue
                best = min(candidates, key=lambda pt: math.hypot(pt[0] - node.x, pt[1] - node.y))
                # 切向角：直线方向 vs 圆在该点的切线（都指向远离节点一侧）
                normal = (best - centre_c) / max(float(np.linalg.norm(best - centre_c)), 1e-9)
                tangent = np.array([-normal[1], normal[0]])
                if tangent @ (line_point - best) < 0:
                    tangent = -tangent
                cosine = float(np.clip(abs(tangent @ line_dir), -1.0, 1.0))
                angle_deg = math.degrees(math.acos(cosine))
                if angle_deg < min_angle_deg:
                    continue
                pair = best
                break
            intersection = pair
            if intersection is None:
                continue
        offset = float(np.hypot(intersection[0] - node.x, intersection[1] - node.y))
        if not np.isfinite(offset) or offset > 3.0 * window_px:
            continue
        new_xy = (round(float(intersection[0]), 4), round(float(intersection[1]), 4))
        nodes[by_id[node.id]] = node.model_copy(update={"x": new_xy[0], "y": new_xy[1]})
        for edge in incident:
            pts = [list(p) for p in edge.points]
            if edge.node_a == node.id:
                pts[0] = [new_xy[0], new_xy[1]]
            if edge.node_b == node.id:
                pts[-1] = [new_xy[0], new_xy[1]]
            trimmed = np.asarray(pts, dtype=np.float64)
            length = float(np.hypot(*(np.diff(trimmed, axis=0).T)).sum()) if len(trimmed) > 1 else edge.length_px
            updated = edge.model_copy(
                update={
                    "points": tuple((round(float(px), 4), round(float(py), 4)) for px, py in trimmed),
                    "length_px": round(length, 4),
                }
            )
            edges[edge.id] = updated
            incident = [updated if item.id == edge.id else item for item in incident]
        sharpened += 1
    if sharpened == 0:
        return graph, 0
    return graph.model_copy(update={"nodes": tuple(nodes), "edges": tuple(edges)}), sharpened


def run_s3_skeleton(
    binary: np.ndarray,
    cfg: R2VConfig,
    *,
    dpi: float,
    linewidth_px: float | None = None,
    out_dir: str | Path | None = None,
) -> tuple[S3Result, StageMetrics]:
    """S3 主流程。`binary` 是 S2 的输出（uint8，非零=墨迹）。"""
    t0 = time.perf_counter()
    mask = np.asarray(binary)
    if mask.ndim != 2:
        raise SkeletonError(f"S3 需要二维二值图，收到形状 {mask.shape}")
    if not (mask > 0).any():
        raise SkeletonError("S3 输入二值图为空")
    if linewidth_px is None or linewidth_px <= 0:
        foreground = mask > 0
        skeleton_pixels = int(skeletonize(foreground).sum())
        linewidth_px = float(foreground.sum()) / max(1, skeleton_pixels)

    warnings: list[str] = []
    variants = _skeleton_arrays(mask)
    scores = {name: _width_cv(skel, dist) for name, (skel, dist) in variants.items()}
    chosen = min(scores, key=lambda name: (scores[name], name))
    if cfg.skeleton.dual_path is False:
        chosen = "thin"
        warnings.append("skeleton.dual_path=false：只用形态学细化，不比较中轴路径")
    skeleton, distance = variants[chosen]

    graph, raw_nodes, raw_edges, pruned = build_graph(
        skeleton, distance, cfg=cfg, linewidth_px=linewidth_px
    )

    corners = 0
    if cfg.skeleton.node.classify:
        corners = split_corners(
            graph,
            linewidth_px=linewidth_px,
            turn_deg=float(cfg.skeleton.node.corner_turn_deg),
            contrast_ratio=float(cfg.skeleton.node.corner_contrast_ratio),
            window_linewidths=float(cfg.skeleton.node.corner_window_linewidths),
        )

    merged = 0
    if cfg.skeleton.node.redundant_collapse:
        merged = merge_collinear_nodes(graph, tolerance_px=cfg.skeleton.node.collinear_tol_px)
    snapped = 0
    snap_px = cfg.skeleton.node.snap_distance_linewidths * linewidth_px
    if snap_px > 0:
        snapped = snap_close_nodes(graph, distance_px=snap_px)
    bridged = 0
    if cfg.skeleton.bridge.enable:
        bridged = bridge_gaps(
            graph,
            linewidth_px=linewidth_px,
            max_gap_linewidths=cfg.skeleton.bridge.max_gap_linewidths,
            max_angle_deg=cfg.skeleton.bridge.max_angle_deg,
        )
    # 合并/吸附可能产生新的度 1 短尾
    pruned += prune_spurs(
        graph,
        linewidth_px=linewidth_px,
        length_threshold_linewidths=cfg.skeleton.spur_removal.length_threshold_linewidths,
        endpoint_curvature_check=cfg.skeleton.spur_removal.endpoint_curvature_check,
    )
    graph.prune_isolated()

    # 节点/边重编号：按坐标字典序，保证同输入两次运行完全一致
    node_order = sorted(graph.nodes, key=lambda nid: (round(graph.nodes[nid][1], 6),
                                                      round(graph.nodes[nid][0], 6), nid))
    remap = {old: new for new, old in enumerate(node_order)}
    edge_order = sorted(
        graph.edges,
        key=lambda eid: (
            min(remap[graph.edges[eid]["a"]], remap[graph.edges[eid]["b"]]),
            max(remap[graph.edges[eid]["a"]], remap[graph.edges[eid]["b"]]),
            -round(graph.edges[eid]["length"], 6),
            eid,
        ),
    )
    edge_remap = {old: new for new, old in enumerate(edge_order)}

    incident: dict[int, list[int]] = defaultdict(list)
    for edge_id, edge in graph.edges.items():
        incident[edge["a"]].append(edge_id)
        if edge["b"] != edge["a"]:
            incident[edge["b"]].append(edge_id)

    nodes: list[GraphNode] = []
    for old_id in node_order:
        x, y, merged_count = graph.nodes[old_id]
        edge_ids = sorted(edge_remap[eid] for eid in incident.get(old_id, []))
        angles: list[float] = []
        for eid in incident.get(old_id, []):
            edge = graph.edges[eid]
            dir_x, dir_y = _edge_direction_at(edge, old_id)
            angles.append(round(math.degrees(math.atan2(dir_y, dir_x)) % 360.0, 4))
        angles.sort()
        node_type = _classify(angles, len(edge_ids), angle_tol_deg=cfg.skeleton.node.angle_tol_deg)
        if any(graph.edges[eid]["a"] == graph.edges[eid]["b"] for eid in incident.get(old_id, [])):
            node_type = "isolated"  # 纯环（例如一个圆环）：没有真正的分支
        if cfg.skeleton.node.classify is False:
            node_type = "endpoint" if len(edge_ids) == 1 else "cross"
        nodes.append(
            GraphNode(
                id=remap[old_id],
                x=round(float(x), 4),
                y=round(float(y), 4),
                type=node_type,
                degree=len(edge_ids),
                edges=tuple(edge_ids),
                incident_angles_deg=tuple(angles),
                merged_from=int(merged_count),
            )
        )

    edges: list[GraphEdge] = []
    for old_id in edge_order:
        edge = graph.edges[old_id]
        edges.append(
            GraphEdge(
                id=edge_remap[old_id],
                node_a=remap[edge["a"]],
                node_b=remap[edge["b"]],
                points=tuple((round(px, 4), round(py, 4)) for px, py in edge["points"]),
                length_px=round(edge["length"], 4),
                linewidth_px=round(edge["linewidth"], 4),
                bridged=bool(edge["bridged"]),
                closed=bool(edge["closed"]),
                closed_loop=bool(edge["closed"]),
            )
        )

    topology = TopologyGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        skeleton_path=chosen,  # type: ignore[arg-type]
        width_cv={name: round(value, 6) for name, value in scores.items()},
    )

    # M9-1：拐点锐化（解析交点替换骨架顶点）。只动 corner 节点坐标与相邻边端点。
    topology, sharpened = sharpen_corners(
        topology,
        min_angle_deg=float(cfg.skeleton.corner_sharpen_min_angle_deg),
        window_px=max(6.0, 3.0 * float(linewidth_px)),
    )
    if sharpened:
        warnings.append(f"corner_sharpening: {sharpened} 个拐点用解析交点替换")

    timing_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    metrics = StageMetrics(
        stage="s3",
        counts={
            "nodes": len(nodes),
            "edges": len(edges),
            "raw_nodes": raw_nodes,
            "raw_edges": raw_edges,
            "pruned_spurs": pruned,
            "merged_nodes": merged,
            "snapped_nodes": snapped,
            "bridged_edges": bridged,
            "split_corners": corners,
            "skeleton_px": int(skeleton.sum()),
            "warnings": len(warnings),
        },
        timing_ms=timing_ms,
        error_estimate={
            "linewidth_px": round(float(linewidth_px), 4),
            "width_cv_thin": scores["thin"],
            "width_cv_medial_axis": scores["medial_axis"],
        },
        warnings=tuple(warnings),
        details={
            "skeleton_path": chosen,
            "node_types": topology.counts_by_type(),
            "bridged_edge_ids": [e.id for e in edges if e.bridged],
            "dpi": dpi,
        },
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "graph.json", topology.to_json())
        write_metrics(out, metrics)

    return (
        S3Result(
            graph=topology,
            warnings=tuple(warnings),
            pruned_spurs=pruned,
            merged_nodes=merged,
            snapped_nodes=snapped,
            bridged_edges=bridged,
            split_corners=corners,
            raw_nodes=raw_nodes,
            raw_edges=raw_edges,
        ),
        metrics,
    )


__all__ = [
    "GraphEdge",
    "GraphNode",
    "S3Result",
    "SkeletonError",
    "TopologyGraph",
    "bridge_gaps",
    "build_graph",
    "merge_collinear_nodes",
    "prune_spurs",
    "run_s3_skeleton",
    "snap_close_nodes",
]
