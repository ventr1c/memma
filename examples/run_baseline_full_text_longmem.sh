cd "$(dirname "$0")/.."
echo $$

API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

MODEL="gpt-4o-mini"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Memory-R1-like multi-agent baseline: mm_retrieval_mode = similarity or time
python scripts/run_vanilla_baseline.py \
  --dataset data/longmemeval_s_cleaned.json \
  --mm_retrieval_mode similarity \
  --mm_max_turns 1 \
  --qr_max_turns 0 \
  --ratio 0.1 \
  --model "$MODEL" \
  --parser amem \
  --output_dir results/baseline_full_text_${MODEL}_${TIMESTAMP}