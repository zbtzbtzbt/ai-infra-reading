# AI Infra Reading

个人的 LLM、VLM 与 AI Infra 阅读仓库，用于保存论文、中文译文、博客资料和技术解析。重点记录方法如何工作、结论依赖什么条件，以及哪些工程问题仍需验证。

## 目录

```text
papers/
  llm/          # 大语言模型论文、原文、译文及单篇解析
  vlm/          # 视觉语言模型与多模态论文、原文、译文及单篇解析
  ai-infra/     # 训练、推理、分布式系统与计算优化论文
blogs/
  llm/          # 博客资料与阅读笔记
  vlm/
  ai-infra/
notes/
  llm/          # 跨论文的专题分析与方法比较
  vlm/
  ai-infra/
templates/
  paper-review.md
  blog-review.md
```

每篇文献使用一个简短的英文目录名，例如 `papers/vlm/run-length-tokenization/`。有多个相关主题时，选择主分类，并在条目中记录其余标签。

## 已收录

| 文献 | 来源与版本 | 主题 | 文件 |
| --- | --- | --- | --- |
| RLT：Don't Look Twice | [arXiv:2411.05222v1](https://arxiv.org/abs/2411.05222v1)，NeurIPS 2024 | VLM；视频词元压缩 | [条目与阅读说明](papers/vlm/run-length-tokenization/README.md) · [中文 Markdown](papers/vlm/run-length-tokenization/translation.zh.md) |
| Ariadne Attention：注意力头语义专门化与混合 ViT 注意力 | [arXiv:2608.28383v1](https://arxiv.org/abs/2608.28383v1)，2026-08-28 | VLM；混合注意力；视觉编码器效率 | [条目与阅读说明](papers/vlm/ariadne-attention/README.md) · [中文 Markdown](papers/vlm/ariadne-attention/translation.zh.md) · [核对记录](papers/vlm/ariadne-attention/translation-notes.zh.md) |
| NAT：邻域注意力 Transformer | [arXiv:2204.07143v5](https://arxiv.org/abs/2204.07143v5)，CVPR 2023 | VLM 视觉编码器基础；二维滑动窗口注意力 | [条目与阅读说明](papers/vlm/neighborhood-attention-transformer/README.md) · [中文 Markdown](papers/vlm/neighborhood-attention-transformer/translation.zh.md) · [核对记录](papers/vlm/neighborhood-attention-transformer/translation-notes.zh.md) |
| Swin Transformer：使用移位窗口的层次化视觉 Transformer | [ICCV 2021 会议版](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) | VLM 视觉编码器基础；二维移位窗口注意力 | [条目与阅读说明](papers/vlm/swin-transformer/README.md) · [中文 Markdown](papers/vlm/swin-transformer/translation.zh.md) · [核对记录](papers/vlm/swin-transformer/translation-notes.zh.md) |

上述中文翻译均为非官方整理译本，具体翻译范围、校订方式与待核实问题见各条目的阅读说明和核对记录。论文报告的实验结果属于原作者主张，本仓库尚未复现其代码或实验。

## 收录与阅读

1. 选定分类，建立文献目录，在 `README.md` 中记录标题、作者、原始链接、发布日期及具体版本。
2. 译文只保存为 `translation.zh.md`，不生成或收录 PDF 副本；原文使用 arXiv、DOI 或作者页面的链接。配图放在同级 `assets/`，保持相对链接可用。
3. 复制 [论文解析模板](templates/paper-review.md) 或 [博客阅读模板](templates/blog-review.md) 到文献目录，写下问题、方法、证据、局限和待验证项，并补充本页索引。
4. 对照原文核查术语、公式、图表编号与数值。正文可修正明显笔误、合并重复内容；重要校订依据与不确定问题集中放在独立核对记录，保持正文连贯。需要汇总多篇文献时，将专题分析放入 `notes/`，链接回各原始条目。

技术结论应能追溯到具体章节、公式、图表或外部证据。复核记录应包含代码版本、配置与运行环境；尚未执行的验证保留为待验证项。
