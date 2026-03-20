# MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution

[![arXiv](https://img.shields.io/badge/arXiv-2603.18718-b31b1b.svg)](https://arxiv.org/abs/2603.18718)

Official implementation of **"MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution"**.

If you find this work helpful, please cite our paper:

```bibtex
@article{lin2026memma,
  title={MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution},
  author={Lin, Minhua and Zhang, Zhiwei and Lu, Hanqing and Liu, Hui and Tang, Xianfeng and He, Qi and Zhang, Xiang and Wang, Suhang},
  journal={arXiv preprint arXiv:2603.18718},
  year={2026}
}
```

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
- [Baselines](#baselines)
  - [Vanilla Single-Agent Baseline](#vanilla-single-agent-baseline)
  - [Memory Toolkit Baselines (A-MEM, NaiveRAG, LangMem, FullContext)](#memory-toolkit-baselines)
  - [LightMem Baseline](#lightmem-baseline)

## Overview

Existing memory-augmented LLM agents mainly treat memory construction, retrieval, and utilization as isolated subroutines, leading to **myopic construction** and **aimless retrieval** (shallow or repetitive searches). MemMA introduces a multi-agent framework that coordinates both the forward and backward paths of the memory cycle.

### Forward Path: Multi-Agent Coordination

- **Meta-Thinker** produces structured guidance that steers both construction and retrieval phases, identifying information importance, redundancy, conflicts, and evidence gaps.
- **Memory Manager** executes atomic memory edits (`ADD`, `UPDATE`, `DELETE`, `NONE`) conditioned on Meta-Thinker guidance and current context.
- **Query Reasoner** implements an iterative *Refine-and-Probe* loop, replacing one-shot retrieval with diagnosis-guided evidence refinement over multiple turns.
- **Answer Agent** generates final responses from accumulated evidence.

### Backward Path: In-Situ Self-Evolution

After each session, a three-stage mechanism repairs the provisional memory before commitment:
1. **Probe Generation** -- synthesizes QA pairs testing factual recall, cross-session reasoning, and temporal inference.
2. **In-situ Verification** -- retrieves evidence from provisional memory and generates answers.
3. **Evidence-Grounded Repair** -- failed probes trigger repair proposals (`SKIP`/`MERGE`/`INSERT`), consolidating and correcting memory.

## Project Structure

```
memma/
├── scripts/                  # Experiment runner scripts
│   ├── run_memma_self_refine_single*.py       # MemMA: Single-Agent backend
│   ├── run_memma_self_refine_lightmem*.py     # MemMA: LightMem backend
│   ├── run_memma_self_refine_amem*.py         # MemMA: A-MEM backend
│   ├── run_memma_lightmem*.py                 # LightMem (no self-refine)
│   └── run_vanilla_baseline*.py               # Vanilla baselines
│
├── data/                     # Dataset files
│   └── memory_rl_train_locomo_conv_26.parquet  # Pre-generated probe QA
│
├── data_preprocess/          # Dataset utilities
│   └── utils.py              # Data loading (load_locomo_dataset, etc.)
│
├── configs/memory_toolkits/  # Baseline toolkit configs (JSON)
│
├── examples/                 # Experiment shell scripts
│   ├── run_memma_*.sh        # MemMA experiments
│   └── run_baseline_*.sh     # Baseline experiments
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

```bash
git clone https://github.com/ventr1c/memma.git
cd memma

# Create conda environment
conda create -n memma python=3.11 -y
conda activate memma

# Install PyTorch (CUDA 12.8, adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

### External Dependencies

The **LightMem** and **FullContext/NaiveRAG/LangMem/A-MEM** baseline scripts rely on memory toolkits from [LightMem](https://github.com/zjunlp/LightMem). To run these baselines, install LightMem and set the path:

```bash
git clone https://github.com/zjunlp/LightMem.git
cd LightMem
pip install -e .
cd ..
export LIGHTMEM_ROOT=/path/to/LightMem
```

### Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (for GPT-4o-mini backbone) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude Haiku backbone) |


## Dataset Preparation

MemMA is evaluated on the [LoCoMo](https://github.com/LeoLjl/LoCoMo) dataset. Download `locomo10.json` and place it under the `data/` directory:

```bash
mkdir -p data
# Download LoCoMo dataset and place locomo10.json in data/
```

The self-evolution backward path uses a pre-generated probe QA parquet file, which is included in the repo:

```
data/memory_rl_train_locomo_conv_26.parquet
```

> The generation pipeline for this file will be documented in a future update.


## Quick Start

Each experiment follows a **two-phase** pipeline:

1. **Phase 1 -- Memory Construction**: Build memories from conversation sessions (with optional Meta-Thinker guidance and self-evolution).
2. **Phase 2 -- QA Evaluation**: Retrieve from constructed memories and answer questions (with optional iterative Query Reasoner).

Example with the **Single-Agent** backend using GPT-4o-mini:

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Phase 1: Build memories with Meta-Thinker + Self-Evolution
python scripts/run_memma_self_refine_single.py \
  --dataset data/locomo10.json \
  --memory-dir ./results/memory_bank/memma_single/ \
  --output_dir ./results/memma_single/ \
  --enable_construction_meta_guidance \
  --build_memories \
  --model gpt-4o-mini \
  --retrieve_k 30 \
  --ratio 0.1 \
  --self_refine_source parquet \
  --self_refine_parquet data/memory_rl_train_locomo_conv_26.parquet \
  --self_refine_log_jsonl ./results/memma_single/self_refine_log.jsonl

# Phase 2: QA Evaluation with iterative retrieval
python scripts/run_memma_self_refine_single.py \
  --dataset data/locomo10.json \
  --memory-dir ./results/memory_bank/memma_single/ \
  --output_dir ./results/memma_single/ \
  --model gpt-4o-mini \
  --retrieve_k 30 \
  --qr_max_turns 5 \
  --ratio 0.1
```

## Evaluation
We provide experiment scripts for three memory backends. Each script handles both memory construction and QA evaluation:

| Backend | Scripts |
|---------|---------|
| **Single-Agent** | `examples/run_memma_self_refine_single_gpt.sh`, `examples/run_memma_self_refine_single_claude.sh` |
| **LightMem** | `examples/run_memma_lightmem_gpt.sh`, `examples/run_memma_self_refine_lightmem_gpt.sh`, `examples/run_memma_self_refine_lightmem_claude.sh` |
| **A-MEM** | `examples/run_memma_self_refine_amem_gpt.sh`, `examples/run_memma_self_refine_amem_claude.sh` |

> LightMem backend requires [LightMem](https://github.com/zjunlp/LightMem) installed (see [External Dependencies](#external-dependencies)).

## Baselines

### Vanilla Single-Agent Baseline

Single-agent memory construction without Meta-Thinker or self-evolution:

```bash
# GPT-4o-mini
bash examples/run_baseline_single.sh

# Claude Haiku 4.5
bash examples/run_baseline_single_claude.sh
```

### Memory Toolkit Baselines

These baselines use memory toolkits from [LightMem](https://github.com/zjunlp/LightMem). Make sure LightMem is installed and `LIGHTMEM_ROOT` is set (see [Installation](#external-dependencies)).

Evaluate existing memory systems ([A-MEM](https://github.com/agiresearch/A-mem), NaiveRAG, [LangMem](https://github.com/langchain-ai/langmem), FullContext) using the toolkit configs in `configs/memory_toolkits/`:

```bash
# A-MEM
bash examples/run_baseline_amem.sh           # GPT-4o-mini
bash examples/run_baseline_amem_claude.sh    # Claude Haiku 4.5

# NaiveRAG (BM25 + embedding retrieval)
bash examples/run_baseline_naiverag_light.sh
bash examples/run_baseline_naiverag_light_claude.sh

# LangMem
bash examples/run_baseline_langmem.sh
bash examples/run_baseline_langmem_claude.sh

# Full Context
bash examples/run_baseline_full_text.sh
bash examples/run_baseline_full_text_light.sh
bash examples/run_baseline_full_text_light_claude.sh
bash examples/run_baseline_full_text_longmem.sh
```

### LightMem Baseline

LightMem without MemMA coordination (requires [LightMem](https://github.com/zjunlp/LightMem)):

```bash
bash examples/run_baseline_lightmem.sh                  # GPT-4o-mini
bash examples/run_baseline_lightmem_claude_bedrock.sh   # Claude via Bedrock
```
