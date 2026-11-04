# Swin Transformer：使用移位窗口的层次化视觉 Transformer

- **原题**：Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- **作者**：Ze Liu、Yutong Lin、Yue Cao、Han Hu、Yixuan Wei、Zheng Zhang、Stephen Lin、Baining Guo
- **机构**：微软亚洲研究院、中国科学技术大学、西安交通大学、清华大学
- **版本**：用户提供的 ICCV 2021 会议版，11 页，会议录页码 10012–10022，2021 年 10 月
- **原文**：[会议页面](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) · [会议版 PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) · [arXiv:2103.14030v2](https://arxiv.org/abs/2103.14030v2)
- **项目**：[作者代码仓库](https://github.com/microsoft/Swin-Transformer)
- **主题**：视觉 Transformer、移位窗口注意力、层次化骨干网络、视觉编码器
- **收录日期**：2026-11-04

## 文件

| 文件 | 内容 |
| --- | --- |
| [translation.zh.md](translation.zh.md) | 摘要、正文第 1–5 节、致谢与 4 条原文脚注；76 条 References 保留英文 |
| [translation-notes.zh.md](translation-notes.zh.md) | 会议版与 arXiv 版差异、校订和公式／表格核对记录 |
| [assets/](assets/) | 4 幅原图与 4 个编号公式的高清裁图 |

## 阅读说明

理解二维局部注意力时，建议先看图 2 和第 3.2 节，再看图 4 中循环移位与掩码如何实现批量计算。第 3.1 节说明图像块合并和层次化结构。需要比较 NAT 时，应区分 Swin 的 **shifted window（移位窗口）**与 NAT 的 **sliding window（滑动窗口）**：两者的窗口划分方式不同。

译文以用户提供的会议版为准，不把 arXiv v2 中额外的附录、表格行或段落混入正文。会议版虽有“详见附录”的表述，但这份 11 页 PDF 不含附录。6 张表拆成 9 个子表以便阅读，数值及缺失值保持原文。行内公式采用可直接显示的数学符号和上下标，编号公式同时保留图像与 LaTeX。

本译文未经作者审定。原论文版权归原权利人所有；辅助核对所用的 arXiv 版本标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。整理改动包括中文翻译、Markdown 排版、原图提取、公式裁图和中文标签对照。未复现论文实验；性能比较均属于论文发表时的版本与设置。本仓库不保存源 PDF 副本。
