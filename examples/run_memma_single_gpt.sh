cd "$(dirname "$0")/.."
echo $$

API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"



# Memory-R1-like multi-agent baseline: mm_retrieval_mode = similarity or time
python scripts/run_vanilla_baseline.py \
  --dataset data/locomo10.json \
  --meta_thinker_mode vanilla \
  --mm_retrieval_mode similarity \
  --mm_max_turns 1 \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --parser amem \
  --output_dir results/baseline_multi_agent_planner_free-mm-max-1-qr-max-5-meta-thinker-vanilla