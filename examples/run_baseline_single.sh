cd "$(dirname "$0")/.."
API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Memory-R1-like multi-agent baseline: mm_retrieval_mode = similarity or time
python scripts/run_vanilla_baseline.py \
  --dataset data/locomo10.json \
  --mm_retrieval_mode similarity \
  --mm_max_turns 1 \
  --qr_max_turns 0 \
  --qr_retrieve_k 30 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --parser amem \
  --output_dir results/vanilla_single-agent_$TIMESTAMP