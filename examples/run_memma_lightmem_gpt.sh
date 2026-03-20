cd "$(dirname "$0")/.."
echo $$

# API_KEY='your-openai-api-key-here'
API_KEY='your-openai-api-key-here'
BASE_URL="https://api.openai.com/v1"

export OPENAI_API_KEY="$API_KEY"
export OPENAI_API_BASE="$BASE_URL"

TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Choose mode: 
# --build_memories: Build memories from scratch using LightMem
# (remove --build_memories to run evaluation mode)

python scripts/run_memma_lightmem.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --build_max_workers 1 \
  --qdrant-dir ./results/qdrant/memma_lightmem_qdrant_$TIMESTAMP/ \
  --output_dir ./results/memma-lightmem-qdrant_gpt4o-mini_$TIMESTAMP/ \
  --build_memories \
  --disable_meta_thinker \

python scripts/run_memma_lightmem.py \
  --dataset data/locomo10.json \
  --qr_max_turns 5 \
  --ratio 0.1 \
  --model gpt-4o-mini \
  --retrieve_k 30 \
  --retrieval_mode combined \
  --qa_max_workers 8 \
  --qdrant-dir ./results/qdrant/memma_lightmem_qdrant_$TIMESTAMP/ \
  --output_dir ./results/memma-lightmem-qdrant_gpt4o-mini_$TIMESTAMP/ \