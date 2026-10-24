# Changelog

## M9-3
- fix(S4): LINE↔ARC 接缝对齐（角度判定 + 2° 跨度门禁 + G1 豁免）
- 根因：直线/弧写回槽位碰撞导致 LINE 被覆盖为第二条 ARC
- 效果：bulge P95 185.9 → 0.242 px；golden 四场景全绿；355 passed / 0 xfailed

## M9-2
- feat(geom): 闭式直线↔圆求交工具 `geom_utils` + 5 单测
- diag: 接缝偏差探针（切点偏离曲率极小 0.59 mm、圆心漂移 0、交点越界 0.625°）

## M9-1
- fix(S3): 拐点锐化（line↔line + line↔arc 解析交点替换骨架顶点，门控 60°）
- 说明：`sphere` 类相切过渡不产生 corner 节点，接缝问题移交 S4（M9-3 已解决）

## M8-4
- feat(S4): 切分预言机（单模型拦截 + 双模型 BIC + 积分图 SSE），单边切分 4.89 ms
- fix(golden_compare): 折线直线段按 spacing 重采样、bulge 多段线炸成原子基元
- 效果：bulge 的虚假 P95 94.1 px 纠正为真实 0.244 px

## M8-3 / M8-2 / M8-1
- fix(S2): 线宽改 `min(2×p90(EDT), 面积/骨架长)`，高估 1.20× → 1.01×
- ci: GitHub Actions（Linux 3.10/3.12 + Windows 3.10）、`scripts/ci_local.sh`
- feat(S3/S4): 骨架转角切角诊断与轨配对平滑

## M7
- feat(S8): DXF 写出（LINE/ARC/CIRCLE/LWPOLYLINE）+ 回读自检 + 未配准水印
- feat(golden): 五维 golden 比对（Hausdorff P95 / 原子计数 / 关键尺寸 / 图层 / 闭合）

## S6（文字候选区三轮）
- feat(S6): 候选区提取 → 行合并 → OCR 引擎适配层（缺失不崩、降级字段齐全）
- feat(S8): TEXT 实体写出（XDATA 溯源 line_id/engine/confidence），低置信度只留掩膜
