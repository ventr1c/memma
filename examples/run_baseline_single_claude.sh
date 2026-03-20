cd "$(dirname "$0")/.."
API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

BEDROCK_MODEL="${BEDROCK_MODEL:-us.anthropic.claude-haiku-4-5-20251001-v1:0}"
BEDROCK_REGION="${BEDROCK_REGION:-us-west-2}"
export AWS_DEFAULT_REGION="$BEDROCK_REGION"
export AWS_REGION="$BEDROCK_REGION"

TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Memory-R1-like multi-agent baseline: mm_retrieval_mode = similarity or time
python scripts/run_vanilla_baseline_claude.py \
  --dataset data/locomo10.json \
  --mm_retrieval_mode similarity \
  --mm_max_turns 1 \
  --qr_max_turns 0 \
  --qr_retrieve_k 30 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --backbone_model "$BEDROCK_MODEL" \
  --answer_model gpt-4o-mini \
  --region "$BEDROCK_REGION" \
  --parser amem \
  --output_dir results/vanilla_single-agent_claude_$TIMESTAMP
