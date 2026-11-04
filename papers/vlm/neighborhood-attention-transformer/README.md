# NAT：邻域注意力 Transformer

- **原题**：Neighborhood Attention Transformer
- **作者**：Ali Hassani、Steven Walton、Jiachen Li、Shen Li、Humphrey Shi
- **会议**：CVPR 2023
- **译本依据**：用户提供的 17 页 PDF；对应 arXiv:2204.07143v5，2023-05-16，含附录
- **原文**：[PDF](https://arxiv.org/pdf/2204.07143v5) · [arXiv 条目](https://arxiv.org/abs/2204.07143v5) · [CVF 会议页面](https://openaccess.thecvf.com/content/CVPR2023/html/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.html) · [同版本源文件](https://arxiv.org/src/2204.07143v5)
- **项目**：[作者代码仓库](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
- **主题**：二维滑动窗口注意力、视觉 Transformer、视觉编码器、CUDA 实现
- **收录日期**：2026-11-04

## 文件

| 文件 | 内容 |
| --- | --- |
| [translation.zh.md](translation.zh.md) | 摘要、正文、致谢及附录 A–C 的中文译文；41 条 References 保留英文 |
| [translation-notes.zh.md](translation-notes.zh.md) | 术语、校订、原文歧义及核对范围 |
| [assets/](assets/) | 13 幅原图与 4 个编号公式的高清裁图 |

## 阅读说明

本文是二维局部注意力与视觉骨干网络论文，未直接研究图像 VLM。若主要想理解二维滑动窗口如何工作，可先读图 1、图 2、第 3.1 节与附录图 VIII；附录 C 解释边缘位置如何处理以及平移等变性的限制。第 3.4 节比较复杂度；需要了解高效实现时，再读附录 A。

译文使用 Unicode 数学符号和 HTML 上下标显示行内公式；4 个编号公式同时保留原文图像与可复制的 LaTeX。9 张表保留原文数值，表 4 分为两个部分显示。References 位于正文与附录之间，保持原文顺序。

这是未经作者审定的非官方译本。原论文版权归原权利人所有，本仓库不为该论文及译文赋予额外开放许可，也不保存源 PDF 副本。性能数据与实现描述均属于论文当时的版本；未复现代码或实验。
