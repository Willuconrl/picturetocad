# 第三方依赖许可证清单

> 本文件由 `scripts/gen_licenses.py` 从**已安装包的 metadata** 生成，不要手改；改依赖后重新运行脚本。
>
> SPDX 判定只采信 `License-Expression` 或语义唯一的分类器；其余标为 `UNKNOWN` 并列入文末待人工确认清单——**不猜**。

| 包 | 版本 | SPDX | 来源 | 判定依据 |
|---|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org | metadata: License-Expression；requirements 分支锁定 2.2.6/2.5.4，本表按实装 2.5.4 记录 |
| scipy | 1.18.1 | BSD-3-Clause | https://scipy.org/ | metadata: License（含唯一 SPDX 标识）；requirements 分支锁定 1.15.3/1.18.1，本表按实装 1.18.1 记录 |
| scikit-image | 0.26.0 | BSD-3-Clause | https://scikit-image.org | source=LICENSE_file_inference (LICENSE.txt)；requirements 分支锁定 0.25.2/0.26.0，本表按实装 0.26.0 记录 |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | https://github.com/opencv/opencv-python | classifier: License :: OSI Approved :: Apache Software License |
| ezdxf | 1.4.4 | MIT | https://github.com/mozman/ezdxf | classifier: License :: OSI Approved :: MIT License |
| fonttools | 4.65.0 | MIT | http://github.com/fonttools/fonttools | metadata: License |
| pyparsing | 3.3.3 | MIT | https://github.com/pyparsing/pyparsing/ | metadata: License-Expression |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions | metadata: License-Expression |
| pydantic | 2.14.1 | MIT | https://github.com/pydantic/pydantic | metadata: License-Expression |
| pydantic-settings | 2.15.0 | MIT | https://github.com/pydantic/pydantic-settings | metadata: License-Expression |
| PyYAML | 6.0.3 | MIT | https://pyyaml.org/wiki/PyYAMLDocumentation | classifier: License :: OSI Approved :: MIT License |
| Pillow | 12.3.0 | MIT-CMU | https://python-pillow.github.io | metadata: License-Expression |
| pytest | 9.1.1 | MIT | https://docs.pytest.org/en/latest/ | metadata: License-Expression |
| hypothesis | 6.168.5 | MPL-2.0 | https://hypothesis.works | metadata: License-Expression |

## 可选依赖（OCR，SPEC §2）

> 这些包**不在** `requirements.txt` 里，只有装了 `requirements-ocr.txt` 才进入运行时。
> 不装也能跑：S6 走降级路径（`text=""` / `confidence=0` / `engine="none"` + WARN）。

| 包 | 版本区间 | SPDX | 用途 | 判定依据 |
|---|---|---|---|---|
| paddleocr | >=3.7.0,<3.8.0 | Apache-2.0 | 主 OCR 引擎 | PyPI metadata `license: Apache License 2.0`（2026-10-21 核查） |
| paddlepaddle | >=3.0.0 | Apache-2.0 | paddleocr 的推理基座 | PyPI metadata（PaddlePaddle 官方项目） |
| pytesseract | >=0.3.13,<0.4 | Apache-2.0 | 降级 OCR 引擎（需系统 tesseract 二进制） | PyPI metadata |

## 红线（SPEC §2）

- 禁止 GPL 依赖进入主分发：Potrace 一律不用。
- PySide6 为 LGPL-3：必须以独立包/动态链接方式隔离，并在本文件写明用户替换义务。

## 待人工确认

- 无（全部依赖都有可判定的 SPDX 标识）。
