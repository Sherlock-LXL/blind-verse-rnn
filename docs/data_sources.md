# 数据来源与使用说明 / Data Sources and Usage Notes

## 中文

Blind Verse 的最终预训练语料以现代诗为主体，并混合现代汉语译文、宋词和古诗。仓库保留数据准备代码，但不重新分发下载或处理后的原始文本。

### 上游来源

1. [Iess/chinese_modern_poetry](https://huggingface.co/datasets/Iess/chinese_modern_poetry)  
   `src/prepare_modern.py` 从 `response` 字段提取现代诗行，清理非汉字字符、限制长度并去重。该数据集页面标示为 MIT License。

2. [xmj2002/Chinese_modern_classical](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical)  
   `src/prepare_translation.py` 从 `modern` 字段提取现代汉语片段，按标点切分、限制长度并去重。数据集页面标示为 Apache License 2.0，并说明其数据来自 [NiuTrans/Classical-Modern](https://github.com/NiuTrans/Classical-Modern)。

3. [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)  
   `src/prepare_song.py` 与 `src/prepare_tang.py` 从本地下载副本中提取宋词和唐诗片段，按标点切分、限制长度、去重并抽样。该上游仓库标示为 MIT License。

### 历史 benchmark

为了让后续两次语料扩容仍使用相同标准比较，`vocab_benchmark.json` 与 `pretrain_benchmark.txt` 在实验期间由作者人工冻结并保存。它们不是 `prepare_pretrain.py` 自动生成的文件。历史 `.pth` 文件也由作者按优化阶段人工命名，以对应 `optimization_log.md` 中的实验记录。

### 分发边界

- 本仓库不提交原始或处理后的外部语料；
- 上游页面的许可证标签应与各项目的最新许可证和数据说明共同核对；
- 数据集许可证不当然解决具体文本、作者权利或模型权重分发的全部问题；
- 最终模型计划通过 GitHub Release 单独分发，代码许可证由作者另行添加；
- 使用者应自行确认其用途、地区和发布方式所需的授权与合规要求。

## English

The final Blind Verse pretraining collection is mainly modern poetry, mixed with modern Chinese translations, Song ci poetry, and classical poetry. This repository keeps the preparation code but does not redistribute downloaded or processed raw text.

### Upstream sources

1. [Iess/chinese_modern_poetry](https://huggingface.co/datasets/Iess/chinese_modern_poetry)  
   `src/prepare_modern.py` extracts modern-poetry lines from the `response` field, removes non-Chinese characters, applies length filters, and deduplicates the results. The dataset page labels the dataset under the MIT License.

2. [xmj2002/Chinese_modern_classical](https://huggingface.co/datasets/xmj2002/Chinese_modern_classical)  
   `src/prepare_translation.py` extracts modern Chinese segments from the `modern` field, splits them by punctuation, applies length filters, and deduplicates the results. The dataset page labels the dataset under Apache License 2.0 and states that it is based on [NiuTrans/Classical-Modern](https://github.com/NiuTrans/Classical-Modern).

3. [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)  
   `src/prepare_song.py` and `src/prepare_tang.py` extract Song ci poetry and Tang-poetry segments from a local downloaded copy, split by punctuation, apply length filters, deduplicate, and sample the results. The upstream repository labels itself under the MIT License.

### Historical benchmarks

To compare the two later corpus expansions under the same standard, `vocab_benchmark.json` and `pretrain_benchmark.txt` were manually frozen and saved by the author during the experiments. They are not generated automatically by `prepare_pretrain.py`. Historical `.pth` files were also named manually by optimization stage to match the experiment records in `optimization_log.md`.

### Distribution boundary

- Raw or processed external corpora are not committed to this repository;
- Upstream license labels should be reviewed together with each project's current license and data documentation;
- A dataset license does not automatically settle every question concerning individual texts, author rights, or model-weight distribution;
- The final model is intended for separate distribution through a GitHub Release, while the author will add the code license separately;
- Users are responsible for confirming the permissions and compliance requirements applicable to their use, jurisdiction, and publication method.
