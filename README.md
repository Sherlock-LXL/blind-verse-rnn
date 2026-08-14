# Blind Verse / 双盲诗生成器

Blind Verse 是一个字符级中文 LSTM 诗句生成项目。同一个模型从 `<BOS>` 出发独立采样两次，A/B 两句不共享 hidden state 或上下文，因此形成真正的“双盲”组合。

Blind Verse is a character-level Chinese LSTM poetry project. The same model samples twice from `<BOS>` with no shared hidden state or context between lines A and B, producing a genuinely “blind” pairing.

项目覆盖数据准备、从零训练、外部语料预训练、目标语料微调、严格测试评估与 Streamlit 界面，是一次从模型训练到产品化展示的完整学习实践。

The project covers data preparation, training from scratch, external-corpus pretraining, target-corpus fine-tuning, strict test evaluation, and a Streamlit interface. It is an end-to-end learning project from model training to a usable demo.

[中文](#中文) · [English](#english)

## 中文

### 项目概览

模型以字符为基本单位，使用 `<BOS> x...` 预测 `x... <EOS>`。训练采用自回归 teacher forcing；生成时逐字符采样，直到生成 `<EOS>` 或达到最大长度。

“双盲”不是两个不同模型，而是同一个模型的两次独立生成。每句都从新的 hidden state 开始，第二句看不到第一句。

### App 功能

- 独立生成两句双盲诗；
- A/B 诗句可分别收藏，再次点击可取消收藏；
- 一键复制整组诗；
- 输入最终 1～3 个汉字进行短续写；
- 续写结果可收藏、复制；
- 当前会话内查看、删除或清空收藏；
- 双盲诗和诗句续写分别拥有独立的采样模式。

### 采样模式

| 模式 | Temperature | Top-k | 说明 |
| --- | ---: | ---: | --- |
| 稳健 | 0.55 | 5 | 更稳定、自然，偏向常见表达 |
| 适中 | 0.70 | 15 | 在自然和意外之间取得平衡 |
| 激进 | 0.90 | 30 | 更自由、更偶然，也更容易出现惊喜 |

这三档参数是当前 V1 App 的固定产品设置。Top-k 和 temperature 只影响生成，不影响 train/val loss。

### 项目结构

```text
blind-verse-rnn/
├── app.py
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── prepare_data.py
│   ├── prepare_modern.py
│   ├── prepare_translation.py
│   ├── prepare_song.py
│   ├── prepare_tang.py
│   ├── prepare_pretrain.py
│   ├── check_vocab.py
│   ├── train.py
│   ├── pretrain.py
│   ├── fine_tuning.py
│   └── test.py
├── docs/
│   ├── optimization_log.md
│   └── data_sources.md
├── data/                 # 本地语料与处理结果；原始语料不随仓库发布
├── checkpoints/          # 本地模型文件；.pth 不进入普通 Git 历史
├── requirements.txt
├── requirements-data.txt
└── .gitignore
```

### 模型架构

正式模型为：

```text
Embedding → Dropout → 1-layer LSTM → Dropout → Linear
```

| 配置 | 数值 |
| --- | ---: |
| Vocabulary size | 5160 |
| Embedding dimension | 1536 |
| Hidden size | 1536 |
| LSTM layers | 1 |
| Dropout | 0.1 |

Embedding 与输出 Linear 不共享权重。项目曾测试 weight tying，但效果明显变差，因此最终模型保持 untied。

### 数据

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

外部语料不会随本仓库重新分发。来源、处理用途和许可注意事项见 [docs/data_sources.md](docs/data_sources.md)。

### 训练流程

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

### 结果与指标口径

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

完整训练日志见 [docs/optimization_log.md](docs/optimization_log.md)。

### 安装

建议使用独立虚拟环境。在 Windows PowerShell 中：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

如需运行外部数据准备脚本：

```powershell
py -m pip install -r requirements-data.txt
```

项目不锁死具体补丁版本；请安装与你的 Python、CUDA 和显卡环境兼容的 PyTorch。

### 运行前所需文件

运行 App 至少需要：

```text
data/processed/vocab_benchmark.json
checkpoints/best_finetuned_model.pth
```

`vocab_benchmark.json` 是模型对应的冻结字符映射。最终 checkpoint 不提交到普通 Git 历史，而是通过 GitHub Release 分发。

如果只使用 Windows 便携版，可直接下载 [BlindVerseRNN-v1.0.0.exe](https://github.com/Sherlock-LXL/blind-verse-rnn/releases/download/v1.0.0/BlindVerseRNN-v1.0.0.exe)，无需配置源码环境。使用源码或 Streamlit App 时，请下载 [best_finetuned_model.pth](https://github.com/Sherlock-LXL/blind-verse-rnn/releases/download/v1.0.0/best_finetuned_model.pth) 并放入 `checkpoints/`。

### 命令

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

### 复现注意事项

- 原始外部语料不在仓库中，完整复现需要自行从上游来源获取；
- benchmark 词表和 benchmark 集是历史实验中人工冻结的固定标准；
- 数据准备使用固定随机种子 42，但训练脚本没有承诺逐位确定性；
- 不同 PyTorch、CUDA、GPU 和随机状态可能造成小幅结果差异；
- `pretrain.py` 默认输出 `best_pretrained_model.pth`，历史文件如 `best_pretrained_model_645kdata.pth` 是实验后人工命名；
- `.pth` 文件体积较大，不应直接加入普通 Git 历史。

### 发布到 GitHub

本项目采用“代码进入普通 Git，最终模型通过 GitHub Release 分发”的方式。`checkpoints/*.pth`、外部语料、HF 缓存与敏感配置已由 `.gitignore` 排除；`data/processed/vocab_benchmark.json` 被保留为 App 所需的小型映射文件。

首次发布前先确认 `git status` 中没有 `.pth`、原始语料、缓存、token 或密钥。随后可执行：

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial release"
git remote add origin https://github.com/<your-username>/blind-verse-rnn.git
git push -u origin main
```

推送代码后，在 GitHub 创建 `v1.0.0` Release，并将本地 `checkpoints/best_finetuned_model.pth` 作为 Release asset 上传。不要解除 `.gitignore` 后把该文件直接加入普通 Git。

### 数据、版权与许可

本仓库只发布代码、文档和运行所需的冻结字符映射，不发布下载或整理后的原始外部语料。上游项目页面标示的许可证不当然替代对具体文本内容、作者权利和模型权重分发范围的独立判断。

除非另有说明，本仓库中的源代码和文档采用 [MIT License](LICENSE)。该许可不适用于原始或处理后的训练语料，也不自动适用于通过 GitHub Releases 分发的模型权重；如果程序包中包含模型权重，该权重仍受相同边界约束。冻结字符映射和第三方材料受其各自的权利与许可说明约束。

### 局限性

- 字符级 LSTM 不具备大型预训练 Transformer 的长程建模能力；
- 输出偶尔会重复、不通顺或语义跳跃；
- 目标语料规模较小，风格适配能力有限；
- top-k 与 temperature 能改变生成风格，但不能消除模型本身的能力边界；
- 训练数据可能影响输出措辞，公开使用前仍需人工检查。

### 未来方向

- 增加可复现的配置与命令行参数；
- 为 benchmark 冻结过程补充自动化和校验；
- 增加更系统的生成质量评估；
- 在不改变 V1 模型结论的前提下改进产品层交互。

### 截图

仓库目前没有正式截图。后续可将 App 截图保存为 `docs/images/app.png`，再在本节加入图片；在真实截图加入前不使用占位图冒充 Demo。

### 开发过程与 AI 协作

用户提出了“双盲诗生成器”的核心创意与产品定义，设计了独立生成、诗句续写、收藏、复制和三档采样模式等交互，并负责数据选择与扩展思路、数据划分原则、本地数据处理、全部训练实验及生成质量判断。模型架构、词表、预训练与微调流程、优化器、模型尺寸、batch size、学习率、dropout、early stopping 和采样参数等关键方案，均由用户在学习、讨论和实际实验后参与决定；最终的实验取舍、产品需求和项目方向也由用户确定。

ChatGPT 等 AI 工具在项目中承担教学、讨论、代码辅助和 debugging partner 的角色，解释语言模型与训练评估概念，并提供了相当一部分 Python、PyTorch 和 Streamlit 代码草稿、重构建议与实现示例。AI 也协助设计实验对照、分析训练日志、定位逻辑问题，以及改进训练、评估、采样、续写、UI 和项目文档；具体建议是否采纳及其效果，则由用户通过实际运行和结果比较来验证。

因此，本项目最准确的定位是一个由用户主导设计、实验和决策的 AI-assisted 端到端深度学习学习项目。实现过程大量借助了 AI 编程工具，但数据处理与模型训练的实际执行、实验结果判断、方案迭代、产品定义及相关知识的学习理解，均由用户持续参与和推动。项目的价值不仅在最终代码，也在于通过反复实验与调试建立对神经网络训练和语言模型工程的实际理解。

## English

### Project overview

The model operates on Chinese characters and learns to predict `x... <EOS>` from `<BOS> x...`. Training uses autoregressive teacher forcing; generation samples one character at a time until `<EOS>` is produced or the maximum length is reached.

“Blind” does not mean two different models. It means two independent generations from the same model. Each line starts with a fresh hidden state, so the second line cannot see the first.

### App features

- Generate two independent blind-verse lines;
- Favorite line A or B separately and click again to remove it;
- Copy the complete pair with one click;
- Continue from a final input of 1–3 Chinese characters;
- Favorite or copy continuation results;
- View, delete, or clear favorites within the current session;
- Use separate sampling-mode controls for blind verse and continuation.

### Sampling modes

| Mode | Temperature | Top-k | Description |
| --- | ---: | ---: | --- |
| Conservative | 0.55 | 5 | More stable and natural, favoring common expressions |
| Balanced | 0.70 | 15 | Balances naturalness and surprise |
| Exploratory | 0.90 | 30 | Freer and more random, with a higher chance of unexpected results |

These three settings are fixed product settings for the current V1 App. Top-k and temperature affect generation only; they do not affect train/validation loss.

### Project structure

```text
blind-verse-rnn/
├── app.py
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── prepare_data.py
│   ├── prepare_modern.py
│   ├── prepare_translation.py
│   ├── prepare_song.py
│   ├── prepare_tang.py
│   ├── prepare_pretrain.py
│   ├── check_vocab.py
│   ├── train.py
│   ├── pretrain.py
│   ├── fine_tuning.py
│   └── test.py
├── docs/
│   ├── optimization_log.md
│   └── data_sources.md
├── data/                 # Local corpora and processed files; raw corpora are not published
├── checkpoints/          # Local model files; .pth files stay out of normal Git history
├── requirements.txt
├── requirements-data.txt
└── .gitignore
```

### Model architecture

The final model is:

```text
Embedding → Dropout → 1-layer LSTM → Dropout → Linear
```

| Setting | Value |
| --- | ---: |
| Vocabulary size | 5160 |
| Embedding dimension | 1536 |
| Hidden size | 1536 |
| LSTM layers | 1 |
| Dropout | 0.1 |

The embedding and output linear layer do not share weights. Weight tying was tested during the project but performed substantially worse, so the final model remains untied.

### Data

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

External corpora are not redistributed with this repository. See [docs/data_sources.md](docs/data_sources.md) for sources, processing purposes, and licensing notes.

### Training pipeline

1. **Run 0: training from scratch**  
   Uses only the target corpus, vocab size 1555, and no external pretraining.
2. **Pretraining and ablations**  
   Compares model size, batch size, Adam/AdamW, dropout, weight tying, and data scale. Data scaling produced the largest improvement; 2048 did not clearly outperform 1536, while weight tying performed substantially worse.
3. **Final pretraining**  
   E1536/H1536/L1/dropout 0.1, batch 64, AdamW, lr `4e-4`, weight decay `1e-4`, and gradient clipping `1.0`; uses `ReduceLROnPlateau` with `mode="min"`, `factor=0.5`, `patience=1`, and `min_lr=1e-5`.
4. **Fine-tuning**  
   Loads `best_pretrained_model_645kdata.pth`, uses the target train/validation splits, batch 32, AdamW, lr `4e-5`, weight decay `1e-4`, gradient clipping `1.0`, up to 30 epochs, and early-stopping patience 3.
5. **Final testing**  
   Uses the summed loss over all non-PAD tokens divided by the total non-PAD token count, producing token-average CrossEntropy.

The `.pth` filenames used during the experiments were assigned manually to match the optimization stages recorded in `docs/optimization_log.md`. They are historical experiment labels rather than an automated versioning system.

### Results and metric definitions

Training logs use batch-average loss: each batch first produces an average loss, and those batch losses are then averaged.

| Result | Loss | Definition |
| --- | ---: | --- |
| Run 0 best validation | 5.3966 | Legacy batch-average |
| Before fine-tuning validation | 3.9668 | Legacy batch-average |
| Best fine-tuned validation | 3.7379 | Legacy batch-average |

Final evaluation uses the more rigorous token-average definition:

| Result | Value |
| --- | ---: |
| Final validation loss | 3.7171 |
| Final test loss | 3.3620 |
| Test perplexity | 28.85 |

The two loss definitions must not be merged into one directly comparable series. Run 0 uses vocab size 1555, while the final model uses vocab size 5160, so they are not a strictly controlled loss ablation either.

See [docs/optimization_log.md](docs/optimization_log.md) for the complete training log.

### Installation

An isolated virtual environment is recommended. In Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

To run the external-data preparation scripts:

```powershell
py -m pip install -r requirements-data.txt
```

The project does not pin patch versions. Install a PyTorch build compatible with your Python, CUDA, and GPU environment.

### Required files before running

The App requires at least:

```text
data/processed/vocab_benchmark.json
checkpoints/best_finetuned_model.pth
```

`vocab_benchmark.json` is the frozen character mapping associated with the model. The final checkpoint is distributed through a GitHub Release rather than committed to normal Git history.

To use only the portable Windows build, download [BlindVerseRNN-v1.0.0.exe](https://github.com/Sherlock-LXL/blind-verse-rnn/releases/download/v1.0.0/BlindVerseRNN-v1.0.0.exe) directly; no source environment setup is required. To run the source code or Streamlit App, download [best_finetuned_model.pth](https://github.com/Sherlock-LXL/blind-verse-rnn/releases/download/v1.0.0/best_finetuned_model.pth) and place it under `checkpoints/`.

### Commands

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

### Reproduction notes

- Raw external corpora are not included; full reproduction requires downloading them from their upstream sources;
- The benchmark vocabulary and benchmark set are fixed standards manually frozen during the historical experiments;
- Data preparation uses random seed 42, but the training scripts do not promise bitwise determinism;
- PyTorch, CUDA, GPU, and random-state differences may cause small numerical variations;
- `pretrain.py` writes `best_pretrained_model.pth` by default; historical names such as `best_pretrained_model_645kdata.pth` were assigned manually after experiments;
- `.pth` files are large and should not be added directly to normal Git history.

### Publishing to GitHub

This project uses normal Git for code and GitHub Releases for the final model. `checkpoints/*.pth`, external corpora, Hugging Face caches, and sensitive configuration are excluded by `.gitignore`; `data/processed/vocab_benchmark.json` remains available as the small mapping required by the App.

Before the first publication, confirm that `git status` does not stage any `.pth` file, raw corpus, cache, token, or secret. Then run:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial release"
git remote add origin https://github.com/<your-username>/blind-verse-rnn.git
git push -u origin main
```

After pushing the code, create a `v1.0.0` GitHub Release and upload the local `checkpoints/best_finetuned_model.pth` as a release asset. Do not remove the ignore rule and commit this file to normal Git.

### Data, copyright, and licensing

This repository publishes code, documentation, and the frozen character mapping required at runtime. It does not redistribute downloaded or processed raw external corpora. A license label shown by an upstream project does not automatically replace a separate review of individual text rights, author rights, or the permitted distribution scope of trained model weights.

Unless otherwise stated, the source code and documentation in this repository are licensed under the [MIT License](LICENSE). This license does not apply to raw or processed training corpora and does not automatically apply to model weights distributed through GitHub Releases; if a program package contains model weights, those weights remain subject to the same boundary. The frozen character mapping and third-party materials remain subject to their respective rights and license notices.

### Limitations

- A character-level LSTM has less long-range modeling capacity than a large pretrained Transformer;
- Outputs may occasionally repeat, become ungrammatical, or jump semantically;
- The target corpus is small, limiting style adaptation;
- Top-k and temperature change generation behavior but cannot remove the model's underlying capability limits;
- Training data may influence wording, so public-facing outputs still require human review.

### Future work

- Add reproducible configuration and command-line arguments;
- Automate and validate the benchmark-freezing process;
- Add more systematic generation-quality evaluation;
- Improve product-layer interaction without changing the V1 model conclusion.

### Screenshot

The repository does not currently contain an official screenshot. A future App screenshot can be saved as `docs/images/app.png` and added here. No placeholder image is presented as a real Demo before that asset exists.

### Development Process & AI Collaboration

The user originated the core idea and product definition of Blind Verse, designed interactions including independent generation, verse continuation, favorites, copying, and three sampling modes, and took responsibility for data selection and scaling strategy, split principles, local data processing, all training experiments, and generation-quality assessment. Key choices concerning the model architecture, vocabulary, pretraining and fine-tuning pipeline, optimizer, model size, batch size, learning rate, dropout, early stopping, and sampling parameters were made with the user's participation after study, discussion, and practical experimentation. The user also made the final decisions on experimental tradeoffs, product requirements, and project direction.

ChatGPT and other AI tools served as teaching, discussion, coding, and debugging partners throughout the project. They explained language-model, training, and evaluation concepts and provided a substantial portion of the Python, PyTorch, and Streamlit code drafts, refactoring suggestions, and implementation examples. AI also helped design controlled experiments, analyze training logs, identify logic problems, and improve training, evaluation, sampling, continuation, UI, and project documentation. The user decided which suggestions to adopt and validated their effects through actual runs and result comparisons.

The most accurate description is therefore an AI-assisted, end-to-end deep-learning learning project led by the user in its design, experimentation, and decision-making. The implementation received substantial support from AI coding tools, while the user continuously carried out and drove the data processing and model training, interpretation of experimental results, iterative choices, product definition, and learning of the underlying concepts. The project's value lies not only in the final code, but also in the practical understanding of neural-network training and language-model engineering built through repeated experimentation and debugging.
