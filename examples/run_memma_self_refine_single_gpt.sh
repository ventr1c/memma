cd "$(dirname "$0")/.."
echo $$

API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
# TIMESTAMP="20260311073553" # 0310

# Phase 1: Build memories using MemoryManagerAgent
python scripts/run_memma_self_refine_single.py \
  --dataset data/locomo10.json \
  --memory-dir ./results/memory_bank/memma_single_$TIMESTAMP/ \
  --output_dir ./results/memma-single_gpt4o-mini_$TIMESTAMP/ \
  --enable_construction_meta_guidance \
  --build_memories \
  --model gpt-4o-mini \
  --mm_max_turns 1 \
  --mm_retrieval_mode similarity \
  --mm_retrieve_k 10 \
  --retrieve_k 30 \
  --ratio 0.1 \
  --self_refine_source parquet \
  --self_refine_parquet data/memory_rl_train_locomo_conv_26.parquet \
  --self_refine_log_jsonl ./results/memma-single_gpt4o-mini_$TIMESTAMP/self_refine_log.jsonl \
  --self_refine_add_fact_prompt_version v3

# Phase 2: QA Evaluation (simple retrieve+answer, no meta-thinker)
python scripts/run_memma_self_refine_single.py \
  --dataset data/locomo10.json \
  --memory-dir ./results/memory_bank/memma_single_$TIMESTAMP/ \
  --output_dir ./results/memma-single_gpt4o-mini_$TIMESTAMP/ \
  --model gpt-4o-mini \
  --retrieve_k 30 \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --qa_max_workers 8 \
