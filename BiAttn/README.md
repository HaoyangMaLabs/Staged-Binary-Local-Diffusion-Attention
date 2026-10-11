# 模型实现

本目录提供论文实验使用的模型骨干与注意力实现。

| 文件 | 职责 |
|---|---|
| `model.py` | `BiAttnConfig`、`BiAttnLM`，RMSNorm、SwiGLU、残差块与绑定词表 |
| `topology.py` | 纯 Python 源偏移规则及 H4 贪心调度，供训练和图扫描共用 |
| `left_v2.py` | 路径覆盖统计、精确 token gather 和 SBDA 注意力 |
| `attention_common.py` | 共享 RoPE、head normalization 和 GQA 辅助函数 |
| `three_track.py` | 稠密、局部及历史双向基线 |
| `sbda_triton.py` | CUDA 融合前向和反向内核 |

SBDA 的源偏移为：前两层 `{0, 2**(layer-1)}`，第三层开始 `{0, 1, 2, 2**(layer-1)}`。越过左边界的源被屏蔽；`h8` 是历史实验中同一路径的别名。

```python
from BiAttn import BiAttnConfig, BiAttnLM

model = BiAttnLM(BiAttnConfig(
    vocab_size=128, dim=64, layers=16, heads=4, ff_hidden=128,
    attention_backend="sbda",
))
```

`attention_backend="full"` 选择稠密因果基线，`"left_v2"` 配合 `path_topology` 选择拓扑消融。原实现中的局部、双向和 MoE 配置保留用于兼容；论文使用固定宽度骨干与 dense SwiGLU。独立项目中的稀疏 FFN 未包含在本仓库，相关配置已移除。

模型默认值保留原实验行为；实验脚本会显式选择注意力后端。CPU 使用 PyTorch 路径，支持的 CUDA 设备可使用融合 Triton 路径。测试位于仓库根目录 `tests/`。

`left_v2.py` 继续导出论文引用的 `topology_offsets` 和 `sbda_stage`，兼容原复现路径。`BiAttn` 根入口按需加载模型，结构扫描不需要安装 PyTorch。
