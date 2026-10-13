# RLT：利用游程词元化加速视频 Transformer

- **原题**：Don’t Look Twice: Faster Video Transformers with Run-Length Tokenization
- **作者**：Rohan Choudhury、Guanglei Zhu、Sihan Liu、Koichiro Niinuma、Kris M. Kitani、László A. Jeni
- **发表**：NeurIPS 2024；arXiv v1 提交于 2024-11-07
- **原文**：[arXiv:2411.05222v1](https://arxiv.org/abs/2411.05222v1)
- **项目页**：[RLT](https://rccchoudhury.github.io/projects/rlt/)
- **主题**：VLM、视频理解、词元压缩、训练与推理效率
- **收录日期**：2026-10-13

## 文件

| 文件 | 内容 |
| --- | --- |
| [translation.zh.md](translation.zh.md) | 完整中文译文，公式用高清图显示并保留可展开的 LaTeX 源码，图片使用相对路径 |
| [translation-notes.zh.md](translation-notes.zh.md) | 正文校订依据及原文待核实问题，集中保存于独立文件 |
| [assets/](assets/) | 论文的 6 幅原始图像和 4 幅原公式高清裁图 |

## 阅读说明

译文涵盖正文、致谢、56 条参考文献与附录 A、B。图像保留原图，并附中文图注和标签对照；Markdown 使用原公式的高清裁图，并保留可展开的 LaTeX 源码，表格数值与原文逐项核对。原文请通过上方 arXiv 链接阅读；本目录不再保存 PDF 副本。

本译文由 Codex 翻译、校订和重新排版，是未经原作者审定的非官方整理译本。明显笔误、重复段落及与方法或表格不符的文字概括已直接修正；正文不再穿插逐段译注，修改依据和待核实问题集中保存在核对记录中。仓库尚未复现其代码或实验。

## Markdown 校订（2026-10-13）

修正下标、序列省略号和标点；统一 tubelet 时间大小符号，修正静态块剪枝方向、阈值趋势及训练时间笔误；合并重复段落，并根据表格整理速度、耗时和长度编码的结论。原图、原公式裁图及 LaTeX 源码、表内实验数值保留。

原论文采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可；作者署名、原文链接和版本信息如上。中文翻译与校订记录同样按 CC BY 4.0 提供。
