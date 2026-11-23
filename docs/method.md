# 模型与复现 · Model and Reproduction

## 数据

目标 Blind Verse 语料划分：

| Split | 句数 |
| --- | ---: |
| Train | 1,292 |
| Validation | 159 |
| Test | 160 |

最终大规模预训练数据以现代诗为主体，并混合宋词、古诗和现代翻译类文本：

| 项目 | 数值 |
| --- | ---: |
| Pretraining train | 612,349 句 |
| Pretraining validation | 32,228 句 |
| 合计 | 644,577 句（约 645k） |
| Training characters | 5,134,711 |
| Frozen vocabulary | 5,160 |

为了在两次语料扩容前后使用同一标准比较，`vocab_benchmark.json` 与 `pretrain_benchmark.txt` 是实验期间人工冻结并保存的 benchmark 文件，不是 `prepare_pretrain.py` 自动生成的产物。若要严格复现历史结果，需要保留这两个冻结文件及其原始映射。

外部语料不会随本仓库重新分发。来源、处理用途和许可注意事项见 [docs/data_sources.md](data_sources.md)。


## 训练流程

1. **Run 0：从零训练**  
   仅使用目标语料，vocab size 1555，不使用外部预训练。
2. **预训练与消融**  
   比较模型尺寸、batch size、Adam/AdamW、dropout、weight tying 与数据规模。数据扩容是最主要的提升来源；2048 未显著优于 1536，weight tying 明显变差。
3. **最终预训练**  
   E1536/H1536/L1/dropout 0.1，batch 64，AdamW，lr `4e-4`，weight decay `1e-4`，gradient clipping `1.0`；使用 `ReduceLROnPlateau`，`mode="min"`、`factor=0.5`、`patience=1`、`min_lr=1e-5`。
4. **微调**  
   从 `best_pretrained_model_645kdata.pth` 加载，使用目标 train/val，batch 32，AdamW，lr `4e-5`，weight decay `1e-4`，gradient clipping `1.0`，最多 30 epochs，early stopping patience 3。
5. **最终测试**  
   使用所有 non-PAD token 的 loss 总和除以 non-PAD token 数量，得到 token-average CrossEntropy。

实验期间的 `.pth` 文件名由人工指定，用于对应 `docs/optimization_log.md` 中的不同优化阶段；这些命名属于历史实验记录，不是自动版本管理系统。


## 结果与指标口径

训练期日志使用 batch-average：先计算每个 batch 的平均 loss，再对 batch 求平均。

| 结果 | Loss | 口径 |
| --- | ---: | --- |
| Run 0 best validation | 5.3966 | 旧 batch-average |
| Before fine-tuning validation | 3.9668 | 旧 batch-average |
| Best fine-tuned validation | 3.7379 | 旧 batch-average |

最终评估使用更规范的 token-average：

| 结果 | 数值 |
| --- | ---: |
| Final validation loss | 3.7171 |
| Final test loss | 3.3620 |
| Test perplexity | 28.85 |

两套 loss 口径不能直接混为同一组结果。Run 0 使用 vocab size 1555，最终模型使用 vocab size 5160，因此两者也不是严格受控的 loss 消融对比。

完整训练日志见 [docs/optimization_log.md](optimization_log.md)。


## 命令

所有命令均从仓库根目录执行。

准备目标语料：

```powershell
py -m src.prepare_data
```

准备外部预训练语料：

```powershell
py -m src.prepare_modern
py -m src.prepare_translation
py -m src.prepare_song
py -m src.prepare_tang
py -m src.prepare_pretrain
```

严格复现历史预训练前，还需准备人工冻结的 `vocab_benchmark.json` 与 `pretrain_benchmark.txt`。上述准备脚本不会自动重建这两个历史 benchmark 文件。

从零训练、预训练、微调与测试：

```powershell
py -m src.train
py -m src.pretrain
py -m src.fine_tuning
py -m src.test
```

为避免覆盖正式 V1 权重，当前微调脚本将新结果写入 `checkpoints/best_finetuned_model_new.pth`；App 与测试脚本继续读取已确认的 `best_finetuned_model.pth`。

启动 App：

```powershell
py -m streamlit run app.py
```

也可以在环境已正确激活时使用：

```powershell
streamlit run app.py
```

不要使用 `py app.py` 启动 Streamlit App。


## Data

Target Blind Verse split:

| Split | Sentences |
| --- | ---: |
| Train | 1,292 |
| Validation | 159 |
| Test | 160 |

The final large-scale pretraining collection is mainly modern poetry, mixed with Song ci poetry, classical poetry, and modern Chinese translations:

| Item | Value |
| --- | ---: |
| Pretraining train | 612,349 sentences |
| Pretraining validation | 32,228 sentences |
| Total | 644,577 sentences (about 645k) |
| Training characters | 5,134,711 |
| Frozen vocabulary | 5,160 |

To compare the two later corpus expansions under the same standard, `vocab_benchmark.json` and `pretrain_benchmark.txt` were manually frozen and saved during the experiments. They are not generated automatically by `prepare_pretrain.py`. Exact reproduction of the historical results requires these two frozen files and their original mapping.

External corpora are not redistributed with this repository. See [docs/data_sources.md](data_sources.md) for sources, processing purposes, and licensing notes.


## Commands

Run every command from the repository root.

Prepare the target corpus:

```powershell
py -m src.prepare_data
```

Prepare the external pretraining corpora:

```powershell
py -m src.prepare_modern
py -m src.prepare_translation
py -m src.prepare_song
py -m src.prepare_tang
py -m src.prepare_pretrain
```

Exact historical pretraining also requires the manually frozen `vocab_benchmark.json` and `pretrain_benchmark.txt`. The preparation scripts above do not automatically recreate these historical benchmark files.

Train from scratch, pretrain, fine-tune, and test:

```powershell
py -m src.train
py -m src.pretrain
py -m src.fine_tuning
py -m src.test
```

To avoid overwriting the confirmed V1 weights, the current fine-tuning script writes new results to `checkpoints/best_finetuned_model_new.pth`; the App and test script continue to read the confirmed `best_finetuned_model.pth`.

Start the App:

```powershell
py -m streamlit run app.py
```

When the environment is already active, this also works:

```powershell
streamlit run app.py
```

Do not start the Streamlit App with `py app.py`.
