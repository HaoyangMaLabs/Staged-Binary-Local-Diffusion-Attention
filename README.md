# Staged Binary-Local Diffusion Attention

论文代码与实验记录：以少量逐层扩大的因果传播路径实现时间轴上的信息扩散。

- [英文论文 PDF](BinaryDiffusionAttention/paper_en.pdf)、[英文源码](BinaryDiffusionAttention/paper_en.md)
- [实验协议及分析](BinaryDiffusionAttention/experiment.md)
- [实验入口说明](BinaryDiffusionAttention/README.md)
- [模型实现说明](BiAttn/README.md)

目录结构：

```text
BiAttn/                    模型、共享注意力组件、拓扑与 Triton 内核
BinaryDiffusionAttention/   共享数据/比较配置、实验入口、论文与图表脚本
  data/                    WikiText-2、PTB、Tiny Shakespeare 数据
  results/                 原始 JSON、CSV 和训练日志
  figures/                 论文图表
tests/                    模型与因果性回归测试
```

安装原实验环境（Python 3.14、PyTorch 2.14，CUDA 13.0 轮子）：

```bash
uv sync --locked
```

在仓库根目录运行小规模 CPU 检查，结果写入单独目录：

```bash
uv run python -m unittest discover -s tests
uv run python BinaryDiffusionAttention/run_graph_coverage.py --lengths 256 --layers 8 --routes dense,H0,H8
uv run python BinaryDiffusionAttention/run_mini_experiment.py --device cpu --steps 2 --seq-len 32 --batch-size 1 --dim 16 --layers 3 --heads 2 --ff-hidden 32 --modes full,h8 --output-dir /tmp/sbda-smoke
```

长上下文复现命令见论文附录 B。默认数据位置由脚本自身位置确定；自定义输出路径相对于执行时的工作目录。

重建图表或论文 PDF 时按需安装排版依赖：

```bash
uv run --with matplotlib python BinaryDiffusionAttention/plot_paper_figures.py
uv run --with pypandoc-binary --with typst --with pymupdf python BinaryDiffusionAttention/export_paper_pdf.py
```

论文、原始数据和实验记录已保留。书籍训练、蒸馏、TPU、双阶段词表及其他独立实验的辅助脚本已移除；视频节目资料已排除。核心实现保留论文附录引用的路径及实验所用参数默认值。[文件清单](IMPORT_MANIFEST.json) 记录当前校验值及导入时的校验值；未重新运行论文的大规模训练。

结构扫描可独立使用系统 Python；训练及模型测试需要 PyTorch。训练与图扫描共用拓扑规则，长上下文训练和吞吐测试共用比较配置。长上下文脚本支持 `--device cpu` 执行小窗口 FP32 检查；论文 CUDA 路径仍使用 BF16。
