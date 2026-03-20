cd "$(dirname "$0")/.."
echo $$

API_KEY='your-anthropic-api-key-here'

export ANTHROPIC_API_KEY="$API_KEY"

# API_KEY='your-openai-api-key-here'
API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

model_builder="claude-haiku-4-5-20251001"
model_answerer="gpt-4o-mini"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
# TIMESTAMP=20260227110351
# Choose mode: 
# --build_memories: Build memories from scratch using LightMem
# (remove --build_memories to run evaluation mode)

python scripts/run_memma_lightmem_claude.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model claude-haiku-4-5-20251001 \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --build_max_workers 3 \
  --qdrant-dir ./results/qdrant/memma_lightmem_qdrant_$TIMESTAMP/ \
  --output_dir ./results/memma-lightmem-qdrant_claude-haiku_$TIMESTAMP/ \
  --build_memories \

python scripts/run_memma_lightmem_claude.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model ${model_answerer} \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --qa_max_workers 8 \
  --qdrant-dir ./results/qdrant/memma_lightmem_qdrant_$TIMESTAMP/ \
  --output_dir ./results/memma-lightmem-qdrant_builder_${model_builder}_${model_answerer}_$TIMESTAMP/ \
  # --build_memories \
  # --disable_meta_thinker \
