# PDF / Word 内容与复杂表格结构化解析流水线

这是一个准确率优先的实验性离线流水线，支持 PDF、DOC 和 DOCX。代码、模型权重、Python 环境、输入数据、缓存、日志和结果都保存在宿主机 `/data/wy/tools/parser`；Docker 镜像只提供 Python、LibreOffice 等可复现运行环境，迁移时挂载整个目录即可。

当前流程把 PyMuPDF、pdfplumber、RapidOCR、Table Transformer 和 MinerU2.5-Pro 放在一条有明确职责与补救条件的逐页流水线中。不解析图内容。

## 当前流程

```text
逐页处理
     │
     ├─ 1. 判断文字层
     │     ├─ 可靠文字层：PyMuPDF提取文字及坐标
     │     └─ 无可靠文字层：渲染页面 → RapidOCR
     │                         优先GPU，失败转CPU
     │
     ├─ 2. 检测表格候选
     │     ├─ pdfplumber lines：仅用于有文字层页面
     │     └─ TATR Detection：所有页面，仅提供区域建议
     │
     ├─ 3. 视觉线框硬门控
     │     ├─ 合格pdfplumber lines：直接保留
     │     ├─ 不合格pdfplumber lines：强网格才保留为补救候选
     │     ├─ TATR Detection：强网格才保留为候选
     │     └─ 无强网格：作为正文/误检丢弃
     │
     ├─ 4. 合并、去重候选框
     │
     └─ 5. 逐个处理候选
           ├─ pdfplumber结果质量合格
           │      └─ 直接采用
           │
           └─ TATR Detection候选
                  ↓
              TATR-v1.1-All恢复结构
                  ↓
              分配文字
                ├─ 文字层页：PyMuPDF原生文字框
                └─ 扫描页：RapidOCR文字框
                  ↓
              结构与文字质量检查
                ├─ 合格：采用
                ├─ 有充分表格证据但结构失败：MinerU表格区域补救
                └─ 缺少表格证据：作为图形/误检丢弃
```

“充分表格证据”不是模型投票。目前门控要求证据分达到阈值，并且至少有一种结构性来源：pdfplumber 的二维网格、TATR 的部分二维结构或图像中的强横纵线网格。单独一个高置信 TATR Detection 框不足以触发 MinerU，目的是避免流程图、电路图等被生成式模型幻觉成表格。

候选融合只接受质量合格的 pdfplumber lines 表格，以及通过强视觉网格门控的失败 lines 结果和 TATR Detection 框。pdfplumber 不运行文字对齐的无边框 `text` 策略；TATR Detection 分数本身也不能绕过线框门控。无线框候选不会运行 TATR 或 MinerU，区域文字继续作为正文。该策略有意不解析无边框表格。

每页只被分为“可靠原生文字层”或“扫描/不可靠文字层”。原生文字层可靠性检查记录字符数、可打印字符比例、语义字符比例和替换字符比例，判为不可靠时整页使用 RapidOCR。

## Word 流程

```text
DOCX ────────────────┐
                     ├─ OpenXML 原生解析 ─ 结构检查 ─ JSON + 表格 HTML + 核验 HTML
DOC ─ LibreOffice ─ DOCX┘
                              │
                              └─ 原生字符少于 20 且存在图片
                                      └─ LibreOffice 转 PDF → 复用既有 PDF 流水线
```

Word 正文、标题、列表和表格按 OpenXML 中的文档顺序处理。表格直接根据 `tblGrid`、`gridSpan` 和 `vMerge` 恢复行列及合并跨度；不使用 OCR 或视觉模型重复猜测原生 Word 表格。图片只计数并跳过，不提取文件、不解析内容。DOCX 是流式文档，页码和页面坐标依赖字体及渲染环境，因此原生结果使用文档顺序和 XML 路径定位，`bbox` 保持 `null`。

DOC 先由同一容器内的 LibreOffice 转换为 DOCX。图像型 Word 才转为 PDF 并调用已经冻结的 PDF 流水线；该回退仍遵守 PDF 的 GPU 和精度要求，不做 CPU 降级。

## 模型与精度策略

- TATR Detection 和 `microsoft/table-transformer-structure-recognition-v1.1-all` 必须使用 CUDA，结构模型固定 FP32，不做 CPU 降级。
- MinerU2.5-Pro 只识别通过证据门控但 TATR 质量检查失败的表格裁剪区域，不解析整页。`PARSER_MINERU_DTYPE=auto` 时，`torch.cuda.is_bf16_supported()` 为真就用 BF16，否则用 FP32；可显式指定 `bf16`、`fp16` 或 `fp32`。
- RapidOCR 使用统一版 `rapidocr` 包，优先使用 Torch CUDA 引擎；初始化或推理失败时静默切换 ONNX Runtime CPU。切换事件仍会写入页面 JSON 和 manifest，避免“静默”变成不可审计。
- 模型运行时只读取 `/workspace/models`，不依赖容器镜像中的权重。TATR 和 MinerU 禁止 CPU fallback。
- 启动时默认要求所映射 GPU 至少有 6 GiB 空闲显存；不足则在处理任何页面前停止。可用 `PARSER_MIN_FREE_GIB` 调整门槛，但不会触发模型或精度降级。

## 目录

```text
config/            路径、阈值和运行策略
pdfpipe/           主流水线、模型适配、表格归一化、指标
wordpipe/          DOC/DOCX 转换、OpenXML 原生解析和人工核验报告
content_export/    面向知识库的原文 JSON、HTML、Markdown 导出
scripts/           构建、安装、下载、运行和评估脚本
models/            TATR、MinerU、RapidOCR 权重（宿主机）
vendor/sources/    固定提交的官方源码快照
envs/pipeline/     唯一 Python 环境（宿主机，挂载进容器）
data/              输入和测试数据
runs/              逐页 JSON、表格 HTML、原图、叠加图、原始结果
logs/              安装、下载和运行日志
manifests/          Docker、依赖、源码、模型版本与 SHA-256
cache/              pip、Hugging Face、ModelScope 等缓存
```

## 打包并离线部署

目标机无需访问网络，但必须预先具备 Linux amd64、Docker、NVIDIA 驱动和 NVIDIA Container Toolkit。当前 Python 环境使用 CUDA 12.8 版 PyTorch，目标机驱动必须兼容；H100 可使用 BF16。

1. 在当前已完成搭建的源机器制作离线包

离线包必须同时包含 Docker 镜像，以及宿主机上的代码、Python 依赖、模型和固定源码。打包目录应位于项目目录之外：

```bash
PROJECT_DIR=/data/wy/tools/parser
PACKAGE_DIR=/data/wy/tools/parser-offline
mkdir -p "$PACKAGE_DIR"

docker save pdf-structure-parser:gpu \
  | gzip -1 > "$PACKAGE_DIR/docker-image.tar.gz"

tar \
  --exclude='./cache' \
  --exclude='./runs' \
  --exclude='./data' \
  --exclude='./logs' \
  --exclude='./.pytest_cache' \
  --exclude='*/__pycache__' \
  -czf "$PACKAGE_DIR/parser-runtime.tar.gz" \
  -C "$PROJECT_DIR" .

cd "$PACKAGE_DIR"
sha256sum docker-image.tar.gz parser-runtime.tar.gz > SHA256SUMS
```

`parser-runtime.tar.gz` 中会保留 `envs/pipeline`、`models`、`vendor`、`pdfpipe`、`wordpipe`、`content_export`、`scripts`、`config` 和 `manifests`。LibreOffice 和中文字体位于 Docker 镜像内。默认排除可重建缓存、历史结果、输入数据和日志；若需要迁移这些内容，应单独归档。

2. 将整个离线包复制到目标机

需要复制以下三个文件：

```text
parser-offline/
├── docker-image.tar.gz
├── parser-runtime.tar.gz
└── SHA256SUMS
```

3. 在目标机校验并加载

```bash
PACKAGE_DIR=/目标机离线包路径/parser-offline
TARGET_DIR=/data/wy/tools/parser

cd "$PACKAGE_DIR"
sha256sum -c SHA256SUMS

gzip -dc docker-image.tar.gz | docker load
mkdir -p "$TARGET_DIR"
tar -xzf parser-runtime.tar.gz -C "$TARGET_DIR"
```

项目可以解压到其他宿主机路径，因为运行时始终将项目根目录挂载为容器内的 `/workspace`。归档必须保留符号链接和执行权限；使用上述 `tar` 命令即可。

4. 检查离线运行环境

```bash
cd /data/wy/tools/parser
docker image inspect pdf-structure-parser:gpu
nvidia-smi

PARSER_GPUS=device=0 bash scripts/container.sh \
  envs/pipeline/bin/python -c \
  'import torch, fitz, pdfplumber, rapidocr, transformers; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

输出应显示 CUDA 可用并识别出目标 GPU。该检查只验证容器、Python 环境和 CUDA，不加载模型权重。

5. 开始离线解析

```bash
mkdir -p data/input
# 将待解析 PDF 放入 data/input 后执行
PARSER_GPUS=device=0 ./scripts/run.sh data/input/document.pdf --run-id document-v1

# 多 GPU 目录批量
./scripts/run.sh data/input --run-id batch-v1 --parallel-gpus 0,1,2,3

# Word 原生解析不需要 GPU
PARSER_GPUS=all ./scripts/run.sh data/input/document.docx --run-id word-v1
```

目标机不要执行 `scripts/setup.sh`、`scripts/bootstrap.sh` 或 `scripts/download-models.sh`，这些脚本面向联网搭建。正常解析只读取本地 `envs/pipeline`、`models` 和 `vendor`。

## 运行

统一入口为 `./scripts/run.sh FILE_OR_DIRECTORY [参数]`。单文件根据 `.pdf`、`.doc`、`.docx` 自动选择解析器。混合格式目录需要显式指定 `--format pdf` 或 `--format word`。TATR 和 MinerU 必须使用 GPU；Word 原生解析不加载模型，只有图像型 Word 回退需要 GPU。

### 运行方式

1. 单个 PDF 完整运行

```bash
mkdir -p data/input
PARSER_GPUS=device=1 ./scripts/run.sh data/input/document.pdf --run-id document-v1
```

2. 目录批量运行

单 GPU：

```bash
PARSER_GPUS=device=1 ./scripts/run.sh data/input --run-id batch-v1
```

多 GPU：

```bash
./scripts/run.sh data/input --run-id batch-v1 --parallel-gpus 0,1,2,3
```

多 GPU 模式为每张宿主机 GPU 启动一个独立容器进程，按文件名轮询分配 PDF；worker 数不会超过 PDF 数，每个 worker 内逐文件、逐页处理。目录只读取顶层 `*.pdf`，不递归子目录。

3. 中断后断点续跑

```bash
PARSER_GPUS=device=1 ./scripts/run.sh data/input/document.pdf --run-id document-v1 --resume
```

`--resume` 直接复用已有 `page.json`。改变 DPI、模型或阈值后必须换新 run id。

4. 临时调参或限页运行

```bash
PARSER_GPUS=device=1 ./scripts/run.sh data/input/document.pdf \
  --run-id document-pages-v1 \
  --tatr-detection-threshold 0.5 \
  --pages 1,3-5
```

5. 禁用 MinerU 做消融测试

```bash
PARSER_GPUS=device=1 ./scripts/run.sh data/input/document.pdf \
  --run-id document-no-mineru \
  --disable-mineru
```

6. 使用完整的自定义配置

```bash
cp config/default.json config/scan-sensitive.json
PARSER_GPUS=device=1 ./scripts/run.sh test_scan.pdf \
  --run-id test-scan-config-v1 \
  --config config/scan-sensitive.json
```

7. 解析 Word 文件或目录

```bash
./scripts/run.sh data/input/document.docx --run-id word-v1
./scripts/run.sh data/input/legacy.doc --run-id legacy-word-v1

# 目录中同时有 PDF 和 Word 时显式选择 Word
./scripts/run.sh data/test_files --format word --run-id word-batch-v1
```

Word 目录模式读取顶层 `.doc` 和 `.docx`，不递归。`--resume` 复用已完成的 `document.json`。`--config`、`--dpi` 和 `--disable-mineru` 只在图像型 Word 转 PDF 回退时生效；Word 原生解析没有模型阈值。

8. 为已有结果补生成知识库原文文件

```bash
# 单个文档结果目录、document.json 或整个批次目录均可
./scripts/export-content.sh runs/word-v1
./scripts/export-content.sh runs/document-v1/example-12345678/document.json
```

最新运行会自动生成原文文件，不需要再执行该命令。批量补导出会递归查找 `document.json`，并对多 GPU 批次中的符号链接结果去重。

### 参数配置

#### 1. 可修改参数

`FILE_OR_DIRECTORY` 必填（PDF、DOC、DOCX 文件或目录） 输入文件；目录模式不递归。  
`--format` 自动（`pdf/word`） 混合格式目录必须指定要处理的格式；单文件通常不需要。  
`--run-id` UTC 时间（字符串） 设置输出目录 `runs/<run-id>`；已有目录必须配合 `--resume`。  
`--resume` 关闭（开关） 复用已完成页面；参数或模型变化后不要使用。  
`--config` `config/default.json`（完整 JSON 文件路径） 使用另一套完整配置。  
`--parallel-gpus` 不启用（逗号分隔的宿主机 GPU 编号，如 `0,1`） 仅用于目录批量运行；每张 GPU 启动一个 worker，不能重复指定 GPU。

`--pages` 不指定（页码或闭区间列表，如 `1,3-5`） 只处理指定页面；页码从 1 开始，重复页自动去重，超过文档总页数时报错，不能与 `--max-pages` 同时使用。  
`--max-pages` 不限制（正整数） 只处理前 N 页，仅用于调试。

`--dpi` 144（整数，`>=96`；建议 `144～300`） 设置页面渲染分辨率；越高越利于小字扫描件，但耗时、存储和显存开销越大。  
`--tatr-detection-threshold` 0.6（`0～1`；建议 `0.5～0.7`） 设置表格候选最低置信度；降低可减少漏表，但会增加误检。  
`tatr_structure_threshold` 0.5（`0～1`；建议 `0.4～0.6`） 控制 TATR 结构对象门槛，在 JSON 配置中修改。  
`candidate_match_iou` 0.25（`0～1`；建议 `0.2～0.5`） 控制 pdfplumber 候选与 TATR 锚点匹配以及原生候选合并，在 JSON 配置中修改。  
`mineru_table_evidence_threshold` 4（整数 `0～11`） 控制是否调用 MinerU；越低召回更高但图形误判风险更大，在 JSON 配置中修改。  
`--disable-mineru` 关闭（开关） 禁用 MinerU 最终补救，用于消融测试。

`PARSER_GPUS` all（Docker GPU 选择器，如 `device=1`） 设置普通单进程运行映射的宿主机 GPU；多 GPU 批量由 `--parallel-gpus` 自动设置。  
`PARSER_DEVICE` `cuda:0`（`cuda` 或 `cuda:N`） 设置容器内 TATR/MinerU 设备，不支持 CPU。  
`PARSER_MIN_FREE_GIB` 6（非负浮点数） 设置启动所需的最低空闲显存；不足时直接停止。  
`PARSER_MINERU_DTYPE` auto（`auto/bf16/fp16/fp32` 及全称） 设置 MinerU 精度；`auto` 在支持时使用 BF16，否则使用 FP32。TATR 固定使用 FP32。  
`RAPIDOCR_DEVICE` auto（`auto/cuda/cpu`） 设置 RapidOCR 设备；`auto` 和 `cuda` 失败后允许静默回退 CPU。  
`PARSER_IMAGE` `pdf-structure-parser:gpu`（本机 Docker 镜像名） 设置运行镜像。  
`PARSER_DTYPE` auto（与 MinerU dtype 相同） 兼容后备参数；正常运行应使用 `PARSER_MINERU_DTYPE`。

`defaults.max_pages` null（保留字段） 当前不生效，限页请使用 `--max-pages`。

命令行的 `--dpi` 和 `--tatr-detection-threshold` 优先于 JSON 配置。最终生效参数写入 `manifest.json`。

#### 2. 固定参数

`原生文字层判定` 可见字符至少 20、词框至少 3、可打印字符比例至少 0.95、字母/数字比例至少 0.25、替换字符比例不超过 0.01；全部满足才使用 PyMuPDF，否则整页使用 RapidOCR。

`pdfplumber 线框检测` 仅使用 lines 策略，snap/join tolerance 均为 3 PDF 点；质量不合格但强视觉网格成立时可作为 TATR 补救候选。

`TATR 输入` 模型尺寸参数 800，候选裁剪外扩 10 像素，结构对象预筛阈值 0.30。

`候选合并` 较小框覆盖率至少 0.72；pdfplumber 同形候选去重 IoU 大于 0.90；TATR 框之间只有 IoU 至少 0.80 才视为重复检测。

`表格质量检查` 至少 2 行 2 列，非空单元格比例至少 0.28；来源文字召回至少 0.35，MinerU 至少 0.45；TATR 最终结构置信度至少 0.25。

`强视觉网格` 横线和纵线各至少 3 条、交点像素至少 8，长线至少占对应边长 35%；是失败 pdfplumber-lines 和全部 TATR Detection 候选进入结构解析的硬门槛，也继续作为 MinerU 门控证据。

`MinerU 证据计分` TATR Detection 至少 0.85 加 2 分；pdfplumber 二维网格、TATR 部分二维结构、强视觉网格各加 3 分，并且至少存在一种结构性证据。

`OCR 行聚类` 垂直中心容差 4 PDF 点。

`跨页候选` 前表底边从页顶计超过 78%、后表顶边小于 25% 且列数相同；只标记，不自动合并。

## 输出

每个文档位于 `runs/<run-id>/<文件名>-<sha256前8位>/`：

- `document.json`：全文页面、文字块、文字框、采用表格、未解决跨页候选和错误。
- `content.json`：面向知识库的结构化原文，只保留页/分节、正文、标题、列表和表格结构及文字。
- `content.html`：与 `content.json` 对应的语义化 HTML 原文。
- `content.md`：与 `content.json` 对应的 Markdown 原文；表格使用 Markdown 兼容的原生 HTML，以保留合并单元格。
- `manifest.json`：输入哈希、运行参数、GPU/dtype、RapidOCR 实际 provider、模型和依赖版本。
- 批次根目录的 `run.log`：自动记录启动检查、模型输出和逐页进度。
- `index.html`：原页叠加框、表格 HTML、正文和丢弃/未解决原因的人工核验页。
- `pages/page-XXXX/page.png` 与 `overlay.png`：逐页原图和采用结果叠加图。
- `pages/page-XXXX/page.json`：可恢复的逐页完整结果。
- `pages/page-XXXX/tables/*.html`：每张采用表格的独立 HTML。
- `pages/page-XXXX/raw/tatr-detection.json`：所有页面的 TATR 检测原始结果。
- `pages/page-XXXX/raw/<candidate>/`：候选裁剪、TATR 结构、MinerU 原始输出和最终决策链。

多 GPU 批次额外生成 `parallel.json`，记录 GPU、worker、退出码及日志路径。worker 原始批次保存在 `.parallel/worker-*/`，批次根目录通过同名符号链接提供与单 GPU 模式一致的文档访问路径；断点续跑时必须保持原 worker 数量和 PDF 文件清单。

统一单元格字段包括 `row`、`col`、`rowspan`、`colspan`、`text`、`bbox` 和 `source`。MinerU 的 HTML 若不能可靠恢复单元格几何，`bbox` 保持 `null`，并通过 `source.region_bbox` 保留表格区域位置；不会伪造单元格坐标。质量检查覆盖空结构、行列不足、非法/重叠跨度、非矩形网格、过多空单元格、来源文字召回和 TATR 结构置信度。所有失败尝试和未解决分歧均保留。

Word 文档同样输出到 `runs/<run-id>/<文件名>-<sha256前8位>/`：

- `document.json`：按顺序保存正文块、表格、分节、页眉页脚、未解决项和错误。
- `manifest.json`：输入哈希、解析模式、依赖版本、LibreOffice 版本及转换记录。
- `index.html`：按原始阅读顺序展示正文和表格，供远程人工核验；不生成页面预览。
- `tables/*.html`：每张表格的独立 HTML。
- `raw/ooxml.json` 与 `raw/tables/*.xml`：包统计、跳过项和表格原始 OpenXML。
- `converted/source.docx`：仅旧版 DOC 输入生成。
- `converted/source.pdf` 与 `pages/`：仅图像型 Word 回退生成。

Word 表格沿用相同的单元格字段；由于流式 Word 文档没有可靠原生页面坐标，其 `page` 和 `bbox` 为 `null`，并通过 `source.part`、`source.xml_path`、行列及文档顺序定位。

三个 `content.*` 文件由 PDF 和 Word 共用的独立导出层生成。`content.json` 使用字段白名单，不包含输入路径、坐标、解析器、模型来源、置信度、候选框、质量检查、运行日志、错误或未解决分歧；表格只保留 `rows`、`columns` 以及单元格的 `row`、`col`、`rowspan`、`colspan`、`header`、`text`。Word 的粗体、斜体、下划线、上下标和超链接仅在实际存在时保留，嵌套表格递归放入所属单元格；页眉、页脚、脚注和尾注保存在 `supplementary` 中。PDF 内部仍利用坐标确定正文与表格的阅读顺序，但坐标不会写入原文文件。

## 有标注评估

只有提供可靠标注时才计算指标：

```bash
./scripts/evaluate.sh ground-truth.json runs/RUN/DOC/document.json metrics.json
```

输出 TEDS、TEDS-S、GriTS_Top、GriTS_Con；两侧都有可靠单元格框时才输出 GriTS_Loc。无标注运行在 `document.json` 和 `manifest.json` 中明确标记 `metrics.status=not_computed`，模型一致性绝不作为准确率。

## 已知限制

- 当前按不对同一页内部的“部分原生、部分扫描”做分区 OCR。
- 跨页表格只生成保守候选，不自动合并，避免错误拼接；需要人工核验或后续增加文档级规则。
- MinerU 区域补救返回的 HTML 通常没有可靠单元格像素框；结构和文字通过质量检查后可采用，但几何缺失会明确记录。
- 无边框、只有外框、内部横线或竖线不足的表格会被有意作为正文保留；当前只解析具有强横纵线网格的表格。
- 正文阅读顺序使用 PyMuPDF 几何顺序或 RapidOCR 行聚类，对极端排版仍需人工复核。
- Word 图片只计数并跳过；只有整体接近图像型文档时才转 PDF 回退，不处理原生正文中的局部截图表格。
- Word 浮动文本框随锚定段落保留，复杂环绕排版的视觉阅读顺序可能与 Word 渲染结果不同，并会记录为未解决项。
- 原生 Word 不生成页面预览，也不伪造页码或坐标；人工核验 HTML 展示的是结构化阅读顺序。

历史实验目录保留用于追溯，但当前入口只产生上述单流水线格式。
