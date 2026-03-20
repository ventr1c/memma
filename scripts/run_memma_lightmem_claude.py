#!/usr/bin/env python3
"""
LightMem + Meta-Thinker Integration Script

Uses LightMem's pre-built memory (Qdrant) and original prompts, 
only adding Meta-Thinker for answerability checking.

Architecture:
- Memory Construction: LightMem (pre-built via add_locomo.py with topic segmentation)
- Entry Loading: LightMem QdrantEntryLoader
- Vector Retrieval: LightMem VectorRetriever (cosine similarity)
- Meta-Thinker: MEMMA MetaThinkerAgent (answerability checking) - ONLY ADDITION
- Answer Generation: LightMem's original ANSWER_PROMPT

Usage:
    python run_memma_lightmem.py \
        --dataset /path/to/locomo10.json \
        --qdrant-dir /path/to/qdrant_post_lighupdate \
        --output_dir results/memma_lightmem
"""

import argparse
import datetime
import json
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

try:
    import anthropic as _anthropic_mod
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Import reflection module
try:
    from memma_reflection import ReflectionEngine, ProceduralMemoryStore, ExperienceMemory
    HAS_REFLECTION = True
except ImportError:
    HAS_REFLECTION = False

# Import data loader with evidence support
try:
    from data_preprocess.utils import load_locomo_dataset
    HAS_AMEM_LOADER = True
except ImportError:
    HAS_AMEM_LOADER = False

# Resolve LightMem paths with env override and fsx default, then legacy fallbacks.
_script_dir = os.path.dirname(__file__)
_lightmem_root_candidates = []
if os.environ.get("LIGHTMEM_ROOT"):
    _lightmem_root_candidates.append(os.path.realpath(os.environ["LIGHTMEM_ROOT"]))
_lightmem_root_candidates.extend(
    [
        os.environ.get("LIGHTMEM_ROOT", "LightMem"),
        os.environ.get("LIGHTMEM_ROOT", "LightMem"),
        os.path.realpath(os.path.join(_script_dir, "../../../LightMem")),
        os.path.realpath(os.path.join(_script_dir, "../../../LightMem_locomo")),
    ]
)

LIGHTMEM_SRC_PATH = ""
LIGHTMEM_PATH = ""
for _root in _lightmem_root_candidates:
    _src = os.path.join(_root, "src")
    _exp = os.path.join(_root, "experiments", "locomo")
    if os.path.isdir(_src) and os.path.isdir(_exp):
        LIGHTMEM_SRC_PATH = _src
        LIGHTMEM_PATH = _exp
        break

if not LIGHTMEM_PATH:
    _default_root = _lightmem_root_candidates[0] if _lightmem_root_candidates else ""
    LIGHTMEM_SRC_PATH = os.path.join(_default_root, "src")
    LIGHTMEM_PATH = os.path.join(_default_root, "experiments", "locomo")

sys.path.insert(0, LIGHTMEM_SRC_PATH)  # Add src first for lightmem module
sys.path.insert(0, LIGHTMEM_PATH)

# Import LightMem utilities
try:
    from retrievers import QdrantEntryLoader, VectorRetriever, format_related_memories
    from llm_judge import evaluate_llm_judge as lightmem_llm_judge
    from prompts import ANSWER_PROMPT, METADATA_GENERATE_PROMPT_locomo
    HAS_LIGHTMEM = True
except ImportError as e:
    print(f"Warning: Cannot import LightMem utilities: {e}")
    print(f"LightMem path: {LIGHTMEM_PATH}")
    HAS_LIGHTMEM = False
    METADATA_GENERATE_PROMPT_locomo = None
    # Fallback prompt if import fails
    ANSWER_PROMPT = """You are an intelligent memory assistant.
Memories for user {speaker_1_name}:
{speaker_1_memories}

Memories for user {speaker_2_name}:
{speaker_2_memories}

Question: {question}

Answer:"""

try:
    from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
    from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
    from lightmem.memory.lightmem import LightMemory
    from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
    from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
    HAS_LIGHTMEM_EMBEDDER = True
except ImportError:
    HAS_LIGHTMEM_EMBEDDER = False
    LightMemory = None

import shutil
import sqlite3

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


def _simple_tokenize(text: str):
    text = str(text).lower()
    return text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_token_f1(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    pred_tokens = set(_simple_tokenize(prediction))
    ref_tokens = set(_simple_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_bleu1(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    if not _HAS_NLTK:
        return 0.0
    try:
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        smooth = SmoothingFunction().method1
        return float(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
    except Exception:
        return 0.0


# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logger(output_dir: str, name: str) -> logging.Logger:
    """Setup logger with file and console handlers."""
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{timestamp}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==============================================================================
# LightMem Memory Generation Utilities (from add_locomo.py)
# ==============================================================================

# Model paths (configurable via args)
DEFAULT_LLMLINGUA_MODEL = 'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def parse_locomo_timestamp(timestamp_str: str) -> str:
    """Parse LoCoMo timestamp format."""
    timestamp_str = timestamp_str.strip("()")
    try:
        dt = datetime.datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def extract_locomo_sessions(conversation_dict: Dict) -> tuple:
    """Extract sessions from LoCoMo conversation format."""
    speaker_a = conversation_dict.get('speaker_a', 'Speaker_A')
    speaker_b = conversation_dict.get('speaker_b', 'Speaker_B')
    
    session_nums = set()
    for key in conversation_dict.keys():
        if key.startswith('session_') and not key.endswith('_date_time'):
            try:
                num = int(key.split('_')[1])
                session_nums.add(num)
            except:
                continue
    
    sessions = []
    timestamps = []
    
    for num in sorted(session_nums):
        session_key = f'session_{num}'
        timestamp_key = f'{session_key}_date_time'
        
        if session_key not in conversation_dict:
            continue
            
        session_data = conversation_dict[session_key]
        timestamp = conversation_dict.get(timestamp_key, '')
        
        messages = []
        for turn in session_data:
            speaker_name = turn['speaker']
            speaker_id = 'speaker_a' if speaker_name == speaker_a else 'speaker_b'
            content = turn['text']
            if 'blip_caption' in turn and turn['blip_caption']:
                content = f"{content} (image description: {turn['blip_caption']})"
            
            messages.append({
                "role": "user",
                "content": content,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            })
            messages.append({
                "role": "assistant",
                "content": "",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            })
        
        sessions.append(messages)
        timestamps.append(parse_locomo_timestamp(timestamp))
    
    return sessions, timestamps, speaker_a, speaker_b


def _build_memory_manager_config(llm_model: str, api_key: str, api_base_url: str = None) -> dict:
    """Return the memory_manager config block, choosing provider based on model name."""
    if llm_model.startswith("claude"):
        return {
            "model_name": "anthropic",
            "configs": {
                "model": llm_model,
                "api_key": api_key,
                "anthropic_api_key": api_key,
                "max_tokens": 16000,
            },
        }
    return {
        "model_name": "openai",
        "configs": {
            "model": llm_model,
            "api_key": api_key,
            "max_tokens": 16000,
            "openai_base_url": api_base_url,
        },
    }


def get_lightmem_config(
    collection_name: str,
    api_key: str,
    qdrant_dir: str,
    llm_model: str = "gpt-4o-mini",
    llmlingua_model: str = None,
    embedding_model: str = None,
    api_base_url: str = None,
    log_dir: str = "./logs",
) -> dict:
    """Get LightMem configuration dictionary."""
    llmlingua_model = llmlingua_model or DEFAULT_LLMLINGUA_MODEL
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    
    return {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": llmlingua_model,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "instruction": "",
                    "rate": 0.6,
                    "target_token": -1
                },
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": _build_memory_manager_config(llm_model, api_key, api_base_url),
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": embedding_model,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f'{qdrant_dir}/{collection_name}',
            }
        },
        "update": "offline",
        "logging": {
            "level": "DEBUG",
            "file_enabled": True,
            "log_dir": log_dir,
        }
    }


def collection_entry_count(collection_name: str, base_dir: str) -> int:
    """Count entries in a Qdrant collection."""
    try:
        cfg = QdrantConfig(
            collection_name=collection_name,
            path=base_dir,
            embedding_model_dims=384,
            on_disk=True,
        )
        q = Qdrant(cfg)
        try:
            points = q.get_all(with_vectors=False, with_payload=False)
            if points:
                return len(points)
        except Exception:
            pass

        storage_sqlite = os.path.join(
            base_dir, collection_name, 'collection', collection_name, 'storage.sqlite'
        )
        if not os.path.exists(storage_sqlite):
            return 0

        try:
            conn = sqlite3.connect(storage_sqlite)
            cur = conn.execute("SELECT count(*) FROM points")
            row = cur.fetchone()
            conn.close()
            if row:
                return int(row[0])
            return 0
        except Exception:
            return -1
    except Exception:
        return -1


# ==============================================================================
# Meta-Thinker Construction Guidance Prompt
# ==============================================================================

META_THINKER_CONSTRUCTION_PROMPT = """You are a Meta-Thinker providing guidance for memory construction.

Given the current conversation chunk being processed, provide construction guidance to help 
the memory manager extract and store the most relevant facts.

Focus on:
1. Key entities, relationships, and events that should be captured
2. Temporal information that should be preserved (dates, durations, sequences)
3. Facts that may be important for answering future questions
4. Potential redundancy with existing memories that should be avoided
5. Relationships to existing stored facts that should be noted

Output format:
FOCUS POINTS:
1. <point 1>
2. <point 2>
...
"""


# ==============================================================================
# LLM Client
# ==============================================================================

class LLMClient:
    """LLM client supporting both OpenAI and Anthropic (Claude) models."""

    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.model = model
        self.is_claude = model.startswith("claude")

        if self.is_claude:
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package is required for Claude models: pip install anthropic")
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.client = _anthropic_mod.Anthropic(api_key=self.api_key)
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.base_url = base_url
            kwargs = {"api_key": self.api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)

    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str = None,
    ) -> str:
        if self.is_claude:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content


# ==============================================================================
# Meta-Thinker Agent (THE ONLY ADDITION TO LIGHTMEM)
# ==============================================================================

META_THINKER_ANSWERABILITY_PROMPT = """You are a Meta-Thinker agent that performs answerability checking for a memory-augmented QA system.

You will be given:
- Question
- Retrieved memories GROUPED BY SPEAKER (each with memory_id, timestamp, snippet)
- Previous queries used (if any)

Your job:
Decide whether the CURRENT evidence is sufficient to answer the Question correctly and completely.

Hard constraints:
- Use ONLY the provided evidence. Do NOT invent facts or rely on unstated assumptions.
- Be conservative: if you are not confident the final answer would be correct, choose NOT_ANSWERABLE.
- Pay attention to WHICH SPEAKER's memories contain relevant information.

Decision criteria:
Choose ANSWERABLE only if:
1) Coverage: All key facts required by the question are explicitly supported by the evidence.
2) Consistency: No unresolved contradictions among the evidence.
3) Specificity: Evidence contains the exact entity/attribute/time scope needed.
4) Completeness: A correct answer can be produced without guessing.

Choose NOT_ANSWERABLE if ANY of the above fails.

If NOT_ANSWERABLE, you MUST provide actionable guidance:
- Identify which speaker's memories are missing or incomplete.
- Target the missing entity/attribute/relation/time scope directly.
- If time references exist ("last year", "two months ago"), note both relative AND absolute time if calculable.

Output format (STRICT):
<decision>ANSWERABLE|NOT_ANSWERABLE</decision>

<reason>
1-4 sentences explaining the decision.
</reason>

<key_gaps>
- If NOT_ANSWERABLE: ranked bullet list of missing information (TOP-1 gap is most critical).
- If ANSWERABLE: write "NONE".
</key_gaps>

<missing_speaker>
- If NOT_ANSWERABLE: which speaker's memories are lacking (speaker_1 / speaker_2 / both / unknown)
- If ANSWERABLE: write "NONE".
</missing_speaker>

<time_need>
- If NOT_ANSWERABLE and question involves time: required absolute date/year/month OR conversion from relative time.
- Otherwise: write "NONE".
</time_need>

<retrieval_guidance>
- Only if NOT_ANSWERABLE.
1) goal: what to retrieve next (one sentence).
2) suggested_queries: 2-5 concrete query candidates, each targeting a different aspect.
3) keywords: 3-8 keywords/entities.
4) constraints: time range, entity, version filters if applicable.
5) avoid_terms: topics/phrases already exhausted in previous queries.
</retrieval_guidance>"""


class MetaThinkerAgent:
    """Meta-Thinker for answerability checking with procedural memory support."""
    
    def __init__(self, llm: LLMClient, logger: logging.Logger, procedural_memories: Dict[str, str] = None):
        self.llm = llm
        self.logger = logger
        # Procedural memories: {"meta": "...", "retrieval": "...", "answering": "...", "storage": "..."}
        self.procedural_memories = procedural_memories or {}
    
    def check_answerability(
        self,
        question: str,
        retrieved_memories_str: str,
        previous_queries: List[str] = None,
    ) -> Dict[str, Any]:
        """Check if question is answerable with current evidence."""
        prev_queries_str = "\n".join(previous_queries) if previous_queries else "[None]"
        
        prompt = f"""Question: {question}

Retrieved Memories:
{retrieved_memories_str if retrieved_memories_str.strip() else "[No memories retrieved]"}

Previous Queries Used:
{prev_queries_str}

Analyze whether the evidence is sufficient to answer the question."""

        # Inject PM_meta if available
        pm_section = ""
        if self.procedural_memories.get("meta"):
            pm_section = f"\n\n## Procedural Memory (Meta Policy)\n{self.procedural_memories['meta']}"
        
        system_prompt = META_THINKER_ANSWERABILITY_PROMPT + pm_section
        
        response = self.llm.get_completion(
            prompt,
            temperature=0.3,
            max_tokens=800,
            system_prompt=system_prompt,
        )
        
        self.logger.debug(f"Meta-Thinker Response:\n{response}")
        
        # Parse response
        decision_match = re.search(r'<decision>\s*(ANSWERABLE|NOT_ANSWERABLE)\s*</decision>', response, re.IGNORECASE)
        reason_match = re.search(r'<reason>(.*?)</reason>', response, re.DOTALL | re.IGNORECASE)
        key_gaps_match = re.search(r'<key_gaps>(.*?)</key_gaps>', response, re.DOTALL | re.IGNORECASE)
        guidance_match = re.search(r'<retrieval_guidance>(.*?)</retrieval_guidance>', response, re.DOTALL | re.IGNORECASE)
        
        # Default to NOT_ANSWERABLE for safety
        decision = decision_match.group(1).upper() if decision_match else "NOT_ANSWERABLE"
        
        # Fallback parsing
        if not decision_match:
            if "ANSWERABLE" in response.upper() and "NOT_ANSWERABLE" not in response.upper():
                decision = "ANSWERABLE"
            else:
                decision = "NOT_ANSWERABLE"
        
        reason = reason_match.group(1).strip() if reason_match else ""
        key_gaps = key_gaps_match.group(1).strip() if key_gaps_match else ""
        retrieval_guidance = guidance_match.group(1).strip() if guidance_match else ""
        
        # Parse new per-speaker fields
        missing_speaker_match = re.search(r'<missing_speaker>(.*?)</missing_speaker>', response, re.DOTALL | re.IGNORECASE)
        time_need_match = re.search(r'<time_need>(.*?)</time_need>', response, re.DOTALL | re.IGNORECASE)
        
        missing_speaker = missing_speaker_match.group(1).strip() if missing_speaker_match else "unknown"
        time_need = time_need_match.group(1).strip() if time_need_match else ""
        
        return {
            "decision": decision,
            "reason": reason,
            "key_gaps": key_gaps,
            "retrieval_guidance": retrieval_guidance,
            "missing_speaker": missing_speaker,
            "time_need": time_need,
            "raw_response": response,
        }
    
    def generate_orthogonal_query(
        self,
        question: str,
        key_gaps: str,
        retrieval_guidance: str,
        previous_queries: List[str],
        retrieved_memory_ids: List[str] = None,
        missing_speaker: str = None,
        time_need: str = None,
        speaker_1_name: str = "Speaker 1",
        speaker_2_name: str = "Speaker 2",
    ) -> Optional[str]:
        """Generate an orthogonal query based on Meta-Thinker guidance with speaker awareness."""
        if not key_gaps and not retrieval_guidance:
            return None
        
        # Build retrieval trace
        retrieval_trace = ""
        if retrieved_memory_ids:
            retrieval_trace = f"Already retrieved memory IDs: {', '.join(retrieved_memory_ids[:20])}"
        
        # Format missing speaker info
        speaker_info = ""
        if missing_speaker and missing_speaker.lower() not in ["none", "unknown"]:
            if missing_speaker == "speaker_1":
                speaker_info = f"Missing speaker: {speaker_1_name} (target this speaker's memories)"
            elif missing_speaker == "speaker_2":
                speaker_info = f"Missing speaker: {speaker_2_name} (target this speaker's memories)"
            elif missing_speaker == "both":
                speaker_info = f"Missing: both speakers ({speaker_1_name} and {speaker_2_name})"
        
        # Format time info
        time_info = ""
        if time_need and time_need.lower() != "none":
            time_info = f"Time requirement: {time_need}"
        
        prompt = f"""Question: {question}

Top Gap (most critical missing info):
{key_gaps.split(chr(10))[0] if key_gaps else "[None]"}

All Key Gaps:
{key_gaps or "[None]"}

{speaker_info}

{time_info}

Retrieval Guidance:
{retrieval_guidance or "[None]"}

Previous Queries:
{chr(10).join(f"- {q}" for q in previous_queries)}

{retrieval_trace}"""

        system_prompt = f"""You are an expert Query Rewriter for conversation memory retrieval.

You are given:
- The Question
- Retrieved memories grouped by speaker, each with: memory_id, timestamp, snippet
- Meta-Thinker diagnosis (structured):
  - top_gap (the single most blocking missing fact)
  - missing_speaker ({speaker_1_name} / {speaker_2_name} / both / unknown)
  - time_need: required absolute date/year/month OR conversion from relative time
  - constraints: entities/versions/time window that MUST appear
  - avoid_terms: topics/phrases already exhausted
  - suggested_query_angles (optional)
- Previous queries
- Retrieval trace: previous query -> retrieved memory_ids

## Rewrite Task
Generate EXACTLY ONE new retrieval query that targets the TOP_GAP and is maximally likely
to retrieve NEW evidence (new memory_ids), minimizing overlap with previous queries/evidence.

## Hard Rules
1. Do NOT repeat any previous query verbatim or near-verbatim.
2. MUST target the TOP_GAP only (do not broaden to multiple gaps).
3. MUST include all constraints (entity + time/version) exactly as provided.
4. If time_need is provided, include BOTH:
   (a) the relative phrase likely used in dialogue (e.g., "last year", "two months ago") AND
   (b) the computed absolute time (e.g., "2021", "March 2023") when possible.
5. MUST avoid avoid_terms.
6. Prefer disambiguation queries if contradiction exists (e.g., "most recent update", "latest", "changed to").
7. If missing_speaker is specified, phrase the query to target that speaker's perspective/experiences.

## Output Format (JSON)
{{
  "rewritten_query": "improved search query with expanded keywords",
  "strategy": "brief explanation of rewrite approach (1 sentence)",
  "target_speaker": "{speaker_1_name}" or "{speaker_2_name}" or "both"
}}

Generate rewrite:"""

        # Inject PM_retrieval if available
        if self.procedural_memories.get("retrieval"):
            pm_retrieval = self.procedural_memories['retrieval']
            system_prompt += f"\n\n## Procedural Memory (Retrieval Policy)\n{pm_retrieval}"

        response = self.llm.get_completion(
            prompt,
            temperature=0.4,
            max_tokens=200,
            system_prompt=system_prompt,
        )
        
        self.logger.debug(f"Orthogonal Query Response:\n{response}")
        
        # Parse query from JSON response
        query = None
        try:
            # Try to extract JSON
            json_match = re.search(r'\{[^{}]*"rewritten_query"[^{}]*\}', response, re.DOTALL)
            if json_match:
                import json
                parsed = json.loads(json_match.group(0))
                query = parsed.get("rewritten_query", "").strip()
                strategy = parsed.get("strategy", "")
                if strategy:
                    self.logger.info(f"Query strategy: {strategy}")
        except (json.JSONDecodeError, Exception) as e:
            self.logger.debug(f"JSON parse failed: {e}")
        
        # Fallback: try old XML format
        if not query:
            query_match = re.search(r'<query>(.*?)</query>', response, re.DOTALL | re.IGNORECASE)
            if query_match:
                query = query_match.group(1).strip()
        
        # Last resort: use cleaned response
        if not query:
            query = response.strip().strip('"').strip("'").split('\n')[0]
        
        # Avoid repeating previous queries
        if query and query.lower() in [q.lower() for q in previous_queries]:
            return None
        
        return query


# ==============================================================================
# LightMem + Meta-Thinker Evaluator
# ==============================================================================

class LightMemMetaThinkerEvaluator:
    """Evaluator using LightMem memory + Meta-Thinker for answerability with reflection support."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        base_url: str = None,
        qdrant_dir: str = None,
        embedding_model_path: str = None,
        retrieve_k: int = 60,
        qr_max_turns: int = 3,
        enable_meta_thinker: bool = True,
        retrieval_mode: str = "combined",  # "per-speaker" or "combined"
        procedural_memories: Dict[str, str] = None,  # For reflection mode
        qa_max_workers: int = 1,
        logger: logging.Logger = None,
    ):
        self.llm = LLMClient(model=model, api_key=api_key, base_url=base_url)
        self.qdrant_dir = qdrant_dir
        self.retrieve_k = retrieve_k
        self.qr_max_turns = qr_max_turns
        self.enable_meta_thinker = enable_meta_thinker
        self.retrieval_mode = retrieval_mode
        self.qa_max_workers = qa_max_workers
        self.logger = logger or logging.getLogger(__name__)
        
        # Procedural memories for reflection mode
        self.procedural_memories = procedural_memories or {}
        
        self.logger.info(f"Retrieval mode: {retrieval_mode}")
        if self.procedural_memories:
            self.logger.info(f"Loaded procedural memories: {list(self.procedural_memories.keys())}")
        
        # Initialize LightMem components
        if HAS_LIGHTMEM:
            self.entry_loader = QdrantEntryLoader(qdrant_dir) if qdrant_dir else None
        else:
            self.entry_loader = None
            self.logger.warning("LightMem not available, entry_loader is None")
        
        # Initialize embedder
        self.embedder = None
        self.retriever = None
        if HAS_LIGHTMEM_EMBEDDER and embedding_model_path:
            try:
                cfg = BaseTextEmbedderConfig(
                    model=embedding_model_path,
                    embedding_dims=384,
                    model_kwargs={"device": "cuda"},
                )
                self.embedder = TextEmbedderHuggingface(cfg)
                self.retriever = VectorRetriever(self.embedder)
            except Exception as e:
                self.logger.warning(f"Failed to initialize embedder: {e}")
        
        self._embed_lock = threading.Lock()
        
        # Initialize Meta-Thinker with procedural memories
        self.meta_thinker = MetaThinkerAgent(
            self.llm, self.logger, procedural_memories=self.procedural_memories
        ) if enable_meta_thinker else None
        
        # Reflection engine (for training mode)
        self.reflection_engine = None
        if HAS_REFLECTION:
            self.reflection_engine = ReflectionEngine(self.llm, self.logger)
        
        # Track retrieval history for reflection
        self.current_retrieval_history: List[Dict] = []
        self.current_storage_actions: List[Dict] = []
    
    def _format_memories_by_speaker(self, retrieved_entries: List[Dict]) -> tuple:
        """Format memories grouped by speaker using LightMem's format_related_memories."""
        speaker_groups = {}
        for entry in retrieved_entries:
            payload = entry.get('payload', {})
            speaker_name = payload.get('speaker_name', 'Unknown')
            if speaker_name not in speaker_groups:
                speaker_groups[speaker_name] = []
            speaker_groups[speaker_name].append(entry)
        
        speaker_names = list(speaker_groups.keys())
        
        if len(speaker_names) == 0:
            return "Speaker 1", "No memories available.", "Speaker 2", "No memories available."
        elif len(speaker_names) == 1:
            speaker_1_name = speaker_names[0]
            speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
            return speaker_1_name, speaker_1_memories, "Speaker 2", "No memories available."
        else:
            speaker_1_name = speaker_names[0]
            speaker_2_name = speaker_names[1]
            speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
            speaker_2_memories = format_related_memories(speaker_groups[speaker_2_name])
            return speaker_1_name, speaker_1_memories, speaker_2_name, speaker_2_memories
    
    def _retrieve_by_speaker(
        self,
        entries: List[Dict],
        query: str,
        limit_per_speaker: int,
    ) -> List[Dict]:
        """Retrieve top-k memories from each speaker separately.
        
        This ensures balanced representation of both speakers.
        """
        # Group entries by speaker
        speaker_groups = {}
        for entry in entries:
            payload = entry.get('payload', {})
            speaker_name = payload.get('speaker_name', 'Unknown')
            if speaker_name not in speaker_groups:
                speaker_groups[speaker_name] = []
            speaker_groups[speaker_name].append(entry)
        
        self.logger.info(f"Found {len(speaker_groups)} speakers: {list(speaker_groups.keys())}")
        
        # Retrieve from each speaker separately
        all_retrieved = []
        for speaker_name, group_entries in speaker_groups.items():
            speaker_retrieved = self.retriever.retrieve(
                group_entries, 
                query, 
                limit=limit_per_speaker
            )
            self.logger.info(f"  {speaker_name}: retrieved {len(speaker_retrieved)}/{len(group_entries)} entries")
            
            # Annotate with speaker name
            for entry in speaker_retrieved:
                entry['_retrieved_speaker'] = speaker_name
            
            all_retrieved.extend(speaker_retrieved)
        
        return all_retrieved
    
    def _thread_safe_retrieve(self, entries: List[Dict], query: str, limit: int) -> List[Dict]:
        """Thread-safe wrapper around retriever.retrieve() to serialize GPU embedder access."""
        with self._embed_lock:
            return self.retriever.retrieve(entries, query, limit=limit)
    
    def _thread_safe_retrieve_by_speaker(
        self, entries: List[Dict], query: str, limit_per_speaker: int
    ) -> List[Dict]:
        """Thread-safe wrapper around _retrieve_by_speaker."""
        with self._embed_lock:
            return self._retrieve_by_speaker(entries, query, limit_per_speaker)
    
    def answer_question(
        self,
        question: str,
        entries: List[Dict],
        category: int = 1,
    ) -> Dict:
        """Answer question using LightMem retrieval + Meta-Thinker answerability check."""
        result = {
            "question": question,
            "category": category,
            "retrieval_turns": [],
            "answer": "",
            "abstained": False,
        }
        
        if not self.retriever:
            self.logger.error("Retriever not initialized")
            result["answer"] = "Error: Retriever not initialized"
            return result
        
        # Initial retrieval using selected mode (thread-safe)
        if self.retrieval_mode == "per-speaker":
            limit_per_speaker = self.retrieve_k // 2
            all_retrieved = self._thread_safe_retrieve_by_speaker(entries, question, limit_per_speaker)
            self.logger.info(f"Initial per-speaker retrieval: {len(all_retrieved)} memories")
        else:
            all_retrieved = self._thread_safe_retrieve(entries, question, limit=self.retrieve_k)
            self.logger.info(f"Initial combined retrieval: {len(all_retrieved)} memories")
        
        previous_queries = [question]
        
        # Log initial retrieval for reflection
        self.current_retrieval_history.append({
            "query": question,
            "retrieved": [str(r.get('id', '')) for r in all_retrieved],
            "turn": 0,
        })
        
        # Format memories for display
        s1_name, s1_mem, s2_name, s2_mem = self._format_memories_by_speaker(all_retrieved)
        combined_memories_str = f"[{s1_name}]\n{s1_mem}\n\n[{s2_name}]\n{s2_mem}"
        
        meta_result = None
        turn = 0
        
        # Meta-Thinker loop (if enabled)
        if self.meta_thinker:
            while turn < self.qr_max_turns:
                # Check answerability
                meta_result = self.meta_thinker.check_answerability(
                    question=question,
                    retrieved_memories_str=combined_memories_str,
                    previous_queries=previous_queries,
                )
                
                decision = meta_result["decision"]
                
                turn_info = {
                    "turn": turn,
                    "decision": decision,
                    "reason": meta_result.get("reason", ""),
                    "missing_speaker": meta_result.get("missing_speaker", ""),
                    "num_memories": len(all_retrieved),
                }
                result["retrieval_turns"].append(turn_info)
                
                self.logger.info(f"[Turn {turn}] Meta-Thinker: {decision}")
                if meta_result.get("missing_speaker"):
                    self.logger.info(f"[Turn {turn}] Missing speaker: {meta_result.get('missing_speaker')}")
                
                if decision == "ANSWERABLE":
                    break
                
                # Generate orthogonal query with speaker awareness
                retrieved_memory_ids = [str(r.get('id', '')) for r in all_retrieved]
                orthogonal_query = self.meta_thinker.generate_orthogonal_query(
                    question=question,
                    key_gaps=meta_result.get("key_gaps", ""),
                    retrieval_guidance=meta_result.get("retrieval_guidance", ""),
                    previous_queries=previous_queries,
                    retrieved_memory_ids=retrieved_memory_ids,
                    missing_speaker=meta_result.get("missing_speaker", ""),
                    time_need=meta_result.get("time_need", ""),
                    speaker_1_name=s1_name,
                    speaker_2_name=s2_name,
                )
                
                if not orthogonal_query:
                    self.logger.info(f"[Turn {turn}] No orthogonal query generated, stopping")
                    break
                
                previous_queries.append(orthogonal_query)
                self.logger.info(f"[Turn {turn}] Orthogonal Query: {orthogonal_query}")
                
                # Retrieve more using orthogonal query (thread-safe)
                new_retrieved = self._thread_safe_retrieve(entries, orthogonal_query, limit=self.retrieve_k)
                
                # Merge avoiding duplicates
                existing_ids = {str(r.get('id', '')) for r in all_retrieved}
                new_count = 0
                for item in new_retrieved:
                    if str(item.get('id', '')) not in existing_ids:
                        all_retrieved.append(item)
                        existing_ids.add(str(item.get('id', '')))
                        new_count += 1
                
                self.logger.info(f"[Turn {turn}] Retrieved {len(new_retrieved)}, {new_count} new")
                
                # Log orthogonal query retrieval for reflection
                self.current_retrieval_history.append({
                    "query": orthogonal_query,
                    "retrieved": [str(r.get('id', '')) for r in new_retrieved],
                    "turn": turn + 1,
                })
                
                # Update formatted memories
                s1_name, s1_mem, s2_name, s2_mem = self._format_memories_by_speaker(all_retrieved)
                combined_memories_str = f"[{s1_name}]\n{s1_mem}\n\n[{s2_name}]\n{s2_mem}"
                
                turn += 1
            
            # Abstain if NOT_ANSWERABLE after budget AND at least one orthogonal query was tried
            # if meta_result and meta_result["decision"] != "ANSWERABLE":
            #     if len(previous_queries) > 1:  # At least one orthogonal query attempted
            #         self.logger.info(f">>> NOT_ANSWERABLE after {turn} turns - returning abstention")
            #         result["answer"] = "Not mentioned in the conversation"
            #         result["abstained"] = True
            #         result["abstention_reason"] = meta_result.get("reason", "")
            #         return result
                # If no orthogonal query was tried, fall through to answer (conservative)
        
        # Generate answer using LightMem's ANSWER_PROMPT
        s1_name, s1_mem, s2_name, s2_mem = self._format_memories_by_speaker(all_retrieved)
        
        prompt = ANSWER_PROMPT.format(
            speaker_1_name=s1_name,
            speaker_1_memories=s1_mem,
            speaker_2_name=s2_name,
            speaker_2_memories=s2_mem,
            question=question,
        )
        
        # # Inject PM_answering if available
        # if self.procedural_memories.get("answering"):
        #     pm_answering = self.procedural_memories['answering']
        #     prompt += f"\n\n## Procedural Memory (Answering Policy)\n{pm_answering}"
        
        result["answer"] = self.llm.get_completion(prompt, temperature=0.0).strip()
        result["num_memories_used"] = len(all_retrieved)
        
        return result
    
    def _process_single_qa(
        self,
        qa: Dict,
        entries: List[Dict],
    ) -> Dict:
        """Process a single QA pair (thread-safe for parallel execution)."""
        question = qa['question']
        reference = qa.get('answer', '')
        category = qa.get('category', 1)

        self.logger.info(f"\nQ: {question}")

        qa_result = self.answer_question(
            question=question,
            entries=entries,
            category=category,
        )
        qa_result["reference"] = reference

        try:
            if HAS_LIGHTMEM:
                if self.llm.is_claude:
                    accuracy = lightmem_llm_judge(
                        question, reference, qa_result["answer"],
                    )
                else:
                    accuracy = lightmem_llm_judge(
                        question, reference, qa_result["answer"],
                        client_obj=self.llm.client, model_name=self.llm.model
                    )
            else:
                accuracy = 0
            qa_result["accuracy"] = float(accuracy)
        except Exception as e:
            self.logger.warning(f"LLM judge failed: {e}")
            qa_result["accuracy"] = 0.0

        qa_result["token_f1"] = calculate_token_f1(qa_result["answer"], reference)
        qa_result["bleu1"] = calculate_bleu1(qa_result["answer"], reference)

        self.logger.info(f"A: {qa_result['answer']}")
        self.logger.info(f"Ref: {reference}")
        self.logger.info(f"J={qa_result['accuracy']:.0f}  F1={qa_result['token_f1']:.4f}  B1={qa_result['bleu1']:.4f}")

        return qa_result

    def process_sample(
        self,
        sample: Dict,
        allow_categories: List[int] = [1, 2, 3, 4, 5],
    ) -> Dict:
        """Process a single sample. Uses ThreadPoolExecutor when qa_max_workers > 1."""
        sample_id = sample['sample_id']
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing sample: {sample_id}")
        self.logger.info(f"{'='*60}")
        
        if not self.entry_loader:
            self.logger.error("Entry loader not initialized")
            return {"sample_id": sample_id, "error": "No entry loader", "results": []}
        
        try:
            entries = self.entry_loader.load_entries(sample_id, with_vectors=True)
            self.logger.info(f"Loaded {len(entries)} LightMem entries for sample {sample_id}")
        except Exception as e:
            self.logger.error(f"Failed to load entries: {e}")
            return {"sample_id": sample_id, "error": str(e), "results": []}
        
        if not entries:
            self.logger.warning(f"No entries found for sample {sample_id}")
            return {"sample_id": sample_id, "error": "No entries", "results": []}
        
        qa_list = [
            qa for qa in sample.get('qa', [])
            if qa.get('category', 1) in allow_categories
        ]
        
        if self.qa_max_workers > 1 and len(qa_list) > 1:
            self.logger.info(f"Parallel QA processing: {len(qa_list)} questions, {self.qa_max_workers} workers")
            qa_results = [None] * len(qa_list)
            with ThreadPoolExecutor(max_workers=self.qa_max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._process_single_qa, qa, entries): idx
                    for idx, qa in enumerate(qa_list)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        qa_results[idx] = future.result()
                    except Exception as e:
                        self.logger.error(f"QA {idx} failed: {e}")
                        qa_results[idx] = {
                            "question": qa_list[idx]['question'],
                            "answer": f"Error: {e}",
                            "accuracy": 0.0,
                            "token_f1": 0.0,
                            "bleu1": 0.0,
                        }
        else:
            qa_results = []
            for qa in qa_list:
                qa_results.append(self._process_single_qa(qa, entries))
        
        return {
            "sample_id": sample_id,
            "num_entries": len(entries),
            "results": qa_results,
        }
    
    def evaluate_dataset(
        self,
        samples: List[Dict],
        allow_categories: List[int] = [1, 2, 3, 4, 5],
        max_samples: int = None,
    ) -> Dict:
        """Evaluate full dataset."""
        if max_samples:
            samples = samples[:max_samples]
        
        all_results = []
        
        for sample in tqdm(samples, desc="Evaluating"):
            result = self.process_sample(sample, allow_categories)
            all_results.append(result)
        
        # Aggregate metrics
        all_qa_results = []
        for r in all_results:
            all_qa_results.extend(r.get("results", []))
        
        category_metrics = defaultdict(lambda: {"accuracy": [], "token_f1": [], "bleu1": []})
        for qa in all_qa_results:
            cat = qa.get("category", 0)
            category_metrics[cat]["accuracy"].append(qa.get("accuracy", 0))
            category_metrics[cat]["token_f1"].append(qa.get("token_f1", 0))
            category_metrics[cat]["bleu1"].append(qa.get("bleu1", 0))
        
        aggregate = {}
        for cat, m in category_metrics.items():
            aggregate[f"category_{cat}"] = {
                "accuracy_mean": float(np.mean(m["accuracy"])) if m["accuracy"] else 0,
                "accuracy_std": float(np.std(m["accuracy"])) if m["accuracy"] else 0,
                "token_f1_mean": float(np.mean(m["token_f1"])) if m["token_f1"] else 0,
                "token_f1_std": float(np.std(m["token_f1"])) if m["token_f1"] else 0,
                "bleu1_mean": float(np.mean(m["bleu1"])) if m["bleu1"] else 0,
                "bleu1_std": float(np.std(m["bleu1"])) if m["bleu1"] else 0,
                "count": len(m["accuracy"]),
            }
        
        all_accs = [qa.get("accuracy", 0) for qa in all_qa_results]
        all_f1s = [qa.get("token_f1", 0) for qa in all_qa_results]
        all_b1s = [qa.get("bleu1", 0) for qa in all_qa_results]
        aggregate["overall"] = {
            "accuracy_mean": float(np.mean(all_accs)) if all_accs else 0,
            "accuracy_std": float(np.std(all_accs)) if all_accs else 0,
            "token_f1_mean": float(np.mean(all_f1s)) if all_f1s else 0,
            "token_f1_std": float(np.std(all_f1s)) if all_f1s else 0,
            "bleu1_mean": float(np.mean(all_b1s)) if all_b1s else 0,
            "bleu1_std": float(np.std(all_b1s)) if all_b1s else 0,
            "count": len(all_accs),
        }
        
        # Count abstentions
        abstentions = sum(1 for qa in all_qa_results if qa.get("abstained", False))
        aggregate["abstention_rate"] = abstentions / len(all_qa_results) if all_qa_results else 0
        
        return {
            "model": self.llm.model,
            "qdrant_dir": self.qdrant_dir,
            "retrieve_k": self.retrieve_k,
            "qr_max_turns": self.qr_max_turns,
            "meta_thinker_enabled": self.enable_meta_thinker,
            "total_samples": len(all_results),
            "total_questions": len(all_qa_results),
            "abstentions": abstentions,
            "aggregate_metrics": aggregate,
            "sample_results": all_results,
        }
    
    def run_reflection_training(
        self,
        training_samples: List[Dict],
        allow_categories: List[int] = [1, 2, 3, 4, 5],
    ) -> Dict[str, str]:
        """Run reflection training on training samples to learn procedural memory.
        
        Processes each sample's QA, reflects on successes/failures, and builds
        per-agent procedural memories (storage, retrieval, answering, meta).
        
        Returns:
            Dict mapping phase names to procedural memory strings.
        """
        if not self.reflection_engine:
            self.logger.warning("Reflection engine not available, skipping training")
            return {}
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("REFLECTION TRAINING MODE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Training samples: {len(training_samples)}")
        
        total_qa = 0
        correct_qa = 0
        
        for sample in tqdm(training_samples, desc="Reflection Training"):
            sample_id = sample['sample_id']
            self.logger.info(f"\n--- Training on sample: {sample_id} ---")
            
            # Load entries
            if not self.entry_loader:
                continue
            
            try:
                entries = self.entry_loader.load_entries(sample_id, with_vectors=True)
            except Exception as e:
                self.logger.error(f"Failed to load entries for {sample_id}: {e}")
                continue
            
            if not entries:
                continue
            
            # Process each QA for reflection
            for qa in sample.get('qa', []):
                category = qa.get('category', 1)
                if category not in allow_categories:
                    continue
                
                question = qa['question']
                reference = qa.get('answer', '')
                
                # Reset retrieval tracking
                self.current_retrieval_history = []
                
                # Answer
                qa_result = self.answer_question(
                    question=question,
                    entries=entries,
                    category=category,
                )
                
                # Judge correctness (always use OpenAI for judging)
                try:
                    if HAS_LIGHTMEM:
                        if self.llm.is_claude:
                            accuracy = lightmem_llm_judge(
                                question, reference, qa_result["answer"],
                            )
                        else:
                            accuracy = lightmem_llm_judge(
                                question, reference, qa_result["answer"],
                                client_obj=self.llm.client, model_name=self.llm.model
                            )
                    else:
                        accuracy = 0
                except Exception:
                    accuracy = 0
                
                is_correct = accuracy > 0.5
                total_qa += 1
                if is_correct:
                    correct_qa += 1
                
                # Reflect and update procedural memories
                self.reflection_engine.reflect_on_trajectory(
                    storage_actions=self.current_storage_actions,
                    retrieval_history=self.current_retrieval_history,
                    question=question,
                    generated_answer=qa_result["answer"],
                    reference_answer=reference,
                    is_correct=is_correct,
                    enable_validation=True,
                )
                
                self.logger.info(f"Q: {question[:80]}...")
                self.logger.info(f"{'✓' if is_correct else '✗'} | Gen: {qa_result['answer'][:50]}... | Ref: {reference[:50]}...")
        
        # Get aggregated procedural memories
        pm = self.reflection_engine.pm_store.memories
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("REFLECTION TRAINING COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total QA: {total_qa}, Correct: {correct_qa} ({100*correct_qa/total_qa:.1f}%)" if total_qa else "No QA processed")
        self.logger.info(f"Experiences extracted: {len(self.reflection_engine.pm_store.experiences)}")
        for phase, content in pm.items():
            self.logger.info(f"PM_{phase}: {len(content)} chars")
        
        return pm
    
    @staticmethod
    def _parse_evidence_id(eid: str) -> Optional[tuple]:
        """Parse evidence ID like 'D3:13' into (session_num, turn_num)."""
        match = re.match(r'D(\d+):(\d+)', eid.strip())
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None


# ==============================================================================
# Dataset Parsing
# ==============================================================================

def parse_locomo_dataset(data_path: str) -> List[Dict]:
    """Parse LoCoMo dataset with evidence support."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        sample = {
            'sample_id': item['sample_id'],
            'conversation': item.get('conversation', {}),
            'qa': []
        }
        
        for qa_item in item.get('qa', []):
            answer = qa_item.get('answer') or qa_item.get('adversarial_answer', '')
            sample['qa'].append({
                'question': qa_item['question'],
                'answer': answer,
                'category': qa_item.get('category', 1),
                'evidence': qa_item.get('evidence', ''),  # Include evidence for turn-alignment
            })
        
        samples.append(sample)
    
    return samples


# ==============================================================================
# Memory Building with Meta-Thinker Guidance
# ==============================================================================

def build_memories_for_sample(
    sample: Dict,
    qdrant_dir: str,
    api_key: str,
    llm_model: str = "gpt-4o-mini",
    api_base_url: str = None,
    embedding_model: str = None,
    llmlingua_model: str = None,
    log_dir: str = "./logs",
    enable_meta_guidance: bool = True,
    llm_client: LLMClient = None,
    logger: logging.Logger = None,
) -> Dict:
    """
    Build memories for a single sample using LightMem with optional Meta-Thinker guidance.
    
    This mirrors add_locomo.py but adds Meta-Thinker construction guidance.
    """
    sample_id = sample['sample_id']
    logger = logger or logging.getLogger(__name__)
    
    if enable_meta_guidance and llm_client is None:
        llm_client = LLMClient(model=llm_model, api_key=api_key, base_url=api_base_url)
    
    try:
        logger.info(f"{'='*60}")
        logger.info(f"Building memories for: {sample_id}")
        logger.info(f"{'='*60}")
        
        conversation = sample['conversation']
        sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(conversation)
        
        logger.info(f"  Sessions: {len(sessions)}")
        logger.info(f"  Speakers: {speaker_a}, {speaker_b}")
        
        # Initialize LightMem
        config = get_lightmem_config(
            collection_name=sample_id,
            api_key=api_key,
            qdrant_dir=qdrant_dir,
            llm_model=llm_model,
            llmlingua_model=llmlingua_model,
            embedding_model=embedding_model,
            api_base_url=api_base_url,
            log_dir=log_dir,
        )
        
        lightmem = LightMemory.from_config(config)
        
        start_time = time.time()
        
        # Process each session turn by turn
        for session_idx, (session, timestamp) in enumerate(zip(sessions, timestamps)):
            while session and session[0]["role"] != "user":
                session.pop(0)
            
            num_turns = len(session) // 2
            logger.info(f"\n  Session {session_idx + 1}: {num_turns} turns")
            
            for turn_idx in range(num_turns):
                turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
                if len(turn_messages) < 2:
                    continue
                if turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                    continue
                
                for msg in turn_messages:
                    msg["time_stamp"] = timestamp
                
                # Get Meta-Thinker construction guidance (if enabled)
                guidance = None
                if enable_meta_guidance and llm_client:
                    chunk_content = turn_messages[0].get("content", "")
                    speaker = turn_messages[0].get("speaker_name", "Unknown")
                    
                    guidance_prompt = f"""Current conversation chunk being processed:
[{timestamp}] {speaker}: {chunk_content}

Provide construction guidance for the memory manager."""

                    try:
                        guidance = llm_client.get_completion(
                            guidance_prompt,
                            temperature=0.3,
                            max_tokens=300,
                            system_prompt=META_THINKER_CONSTRUCTION_PROMPT,
                        )
                        logger.debug(f"Meta-Thinker Construction Guidance:\n{guidance}")
                    except Exception as e:
                        guidance = None
                        logger.warning(f"Meta-Thinker guidance failed: {e}")
                
                # Build dynamic extraction prompt with Meta-Thinker guidance
                dynamic_extraction_prompt = None
                if enable_meta_guidance and guidance and METADATA_GENERATE_PROMPT_locomo:
                    dynamic_extraction_prompt = (
                        METADATA_GENERATE_PROMPT_locomo
                        + f"\n\n## Construction Focus Guidance\n{guidance}"
                    )
                
                # Add memory using LightMem
                is_last_turn = (session is sessions[-1] and turn_idx == num_turns - 1)
                lightmem.add_memory(
                    turn_messages,
                    dynamic_extraction_prompt,
                    force_segment=is_last_turn,
                    force_extract=is_last_turn,
                )
        
        add_memory_time = time.time() - start_time
        after_add_count = collection_entry_count(sample_id, qdrant_dir)
        logger.info(f"\n✓ Add memory completed: {after_add_count} entries in {add_memory_time:.2f}s")
        
        # Perform offline update
        logger.info(f"\n{'─'*60}")
        logger.info("Performing offline update")
        logger.info(f"{'─'*60}")
        
        update_start_time = time.time()
        lightmem.construct_update_queue_all_entries()
        lightmem.offline_update_all_entries(score_threshold=0.9)
        update_time = time.time() - update_start_time
        
        post_update_count = collection_entry_count(sample_id, qdrant_dir)
        logger.info(f"✓ Update completed: {post_update_count} entries in {update_time:.2f}s")
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: {sample_id}")
        logger.info(f"{'='*60}")
        logger.info(f"  Entries: {post_update_count}")
        logger.info(f"  Total time: {total_time:.2f}s")
        
        return {
            'sample_id': sample_id,
            'status': 'success',
            'entry_count': post_update_count,
            'total_duration': total_time,
            'add_memory_duration': add_memory_time,
            'update_duration': update_time,
        }
        
    except Exception as e:
        logger.error(f"✗ {sample_id} failed: {str(e)}", exc_info=True)
        return {
            'sample_id': sample_id,
            'status': 'failed',
            'error': str(e)
        }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="LightMem + Meta-Thinker Integration")
    
    # Mode selection
    parser.add_argument("--build_memories", action="store_true",
                       help="Build memories from scratch using LightMem (instead of evaluating)")
    
    # Data paths
    parser.add_argument("--dataset", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--qdrant-dir", type=str, required=True, help="Path to LightMem Qdrant data")
    parser.add_argument("--output_dir", type=str, default="results/memma_lightmem", help="Output directory")
    
    # Model settings
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name (e.g. gpt-4o-mini, claude-haiku-4-5-20251001)")
    parser.add_argument("--api_key", type=str, default=None, help="API key (OpenAI or Anthropic)")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI base URL (ignored for Claude models)")
    
    # Retrieval settings (same as LightMem default)
    parser.add_argument("--retrieve_k", type=int, default=60, help="Number of memories to retrieve")
    parser.add_argument("--qr_max_turns", type=int, default=3, help="Maximum Meta-Thinker turns")
    parser.add_argument("--retrieval_mode", type=str, default="combined", 
                       choices=["per-speaker", "combined"],
                       help="Retrieval mode: 'per-speaker' (balanced per speaker) or 'combined' (global similarity)")
    
    # Meta-Thinker toggle
    parser.add_argument("--disable_meta_thinker", action="store_true", 
                       help="Disable Meta-Thinker (run pure LightMem baseline)")
    
    # Parallel processing
    parser.add_argument("--qa_max_workers", type=int, default=1,
                       help="Max parallel threads for QA processing (1=sequential)")
    parser.add_argument("--build_max_workers", type=int, default=1,
                       help="Max parallel processes for memory building (1=sequential)")
    
    # Dataset settings
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of samples to evaluate")
    parser.add_argument("--categories", type=str, default="1,2,3,4", help="Comma-separated categories")
    
    # Embedding model
    parser.add_argument("--embedding_model", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Path to embedding model")
    
    # LLMLingua model (for memory building)
    parser.add_argument("--llmlingua_model", type=str,
                       default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                       help="Path to LLMLingua model (for memory building)")
    
    # Reflection mode settings
    parser.add_argument("--meta_thinker_mode", type=str, default="vanilla",
                       choices=["none", "vanilla", "reflection"],
                       help="Meta-Thinker mode: 'none' (disabled), 'vanilla' (enabled w/o learning), 'reflection' (with procedural learning)")
    parser.add_argument("--reflection_test_samples", type=int, default=1,
                       help="Number of samples to use for reflection training (remaining used for test)")
    parser.add_argument("--procedural_memory_path", type=str, default=None,
                       help="Path to save/load procedural memory JSON")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, "memma_lightmem")
    
    if args.model.startswith("claude"):
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    # Load dataset
    logger.info(f"\nLoading dataset from {args.dataset}")
    samples = parse_locomo_dataset(args.dataset)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Apply ratio
    if args.ratio < 1.0:
        num_samples = max(1, int(len(samples) * args.ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({args.ratio*100:.1f}%)")
    
    # Apply max_samples
    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")
    
    # =========================================================================
    # MODE: Build Memories
    # =========================================================================
    if args.build_memories:
        logger.info("=" * 60)
        logger.info("MEMORY BUILDING MODE")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Qdrant Dir: {args.qdrant_dir}")
        logger.info(f"Meta-Thinker Guidance: {'ENABLED' if not args.disable_meta_thinker else 'DISABLED'}")
        
        if not HAS_LIGHTMEM_EMBEDDER or LightMemory is None:
            logger.error("LightMem not available. Cannot build memories.")
            return
        
        build_kwargs_common = dict(
            qdrant_dir=args.qdrant_dir,
            api_key=api_key,
            llm_model=args.model,
            api_base_url=args.base_url,
            embedding_model=args.embedding_model,
            llmlingua_model=args.llmlingua_model,
            log_dir=os.path.join(args.output_dir, 'logs'),
            enable_meta_guidance=not args.disable_meta_thinker,
        )
        
        results = []
        if args.build_max_workers > 1 and len(samples) > 1:
            logger.info(f"Parallel build: {len(samples)} samples, {args.build_max_workers} workers")
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.build_max_workers) as executor:
                future_to_sample = {}
                for sample in samples:
                    future = executor.submit(
                        build_memories_for_sample,
                        sample=sample,
                        llm_client=None,
                        logger=None,
                        **build_kwargs_common,
                    )
                    future_to_sample[future] = sample
                
                with tqdm(total=len(samples), desc="Building memories") as pbar:
                    for future in as_completed(future_to_sample):
                        sample = future_to_sample[future]
                        try:
                            result = future.result()
                            results.append(result)
                            status = "ok" if result.get("status") == "success" else "FAIL"
                            pbar.set_postfix_str(f"{status} {sample['sample_id']}")
                        except Exception as e:
                            logger.error(f"Build failed for {sample['sample_id']}: {e}")
                            results.append({
                                "sample_id": sample['sample_id'],
                                "status": "failed",
                                "error": str(e),
                            })
                        pbar.update(1)
        else:
            llm_client = LLMClient(model=args.model, api_key=api_key, base_url=args.base_url)
            for sample in tqdm(samples, desc="Building memories"):
                result = build_memories_for_sample(
                    sample=sample,
                    llm_client=llm_client if not args.disable_meta_thinker else None,
                    logger=logger,
                    **build_kwargs_common,
                )
                results.append(result)
        
        # Summary
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        logger.info("\n" + "=" * 60)
        logger.info("MEMORY BUILDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if successful:
            total_entries = sum(r.get('entry_count', 0) for r in successful)
            total_time = sum(r.get('total_duration', 0) for r in successful)
            logger.info(f"Total entries: {total_entries}")
            logger.info(f"Total time: {total_time:.2f}s")
        
        return
    
    # =========================================================================
    # MODE: Evaluate (default) with optional Reflection Training
    # =========================================================================
    logger.info("=" * 60)
    logger.info("LightMem + Meta-Thinker Integration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Qdrant Dir: {args.qdrant_dir}")
    logger.info(f"Retrieve K: {args.retrieve_k}")
    logger.info(f"Retrieval Mode: {args.retrieval_mode}")
    logger.info(f"Meta-Thinker Mode: {args.meta_thinker_mode}")
    logger.info(f"Max Turns: {args.qr_max_turns}")
    
    # Parse categories
    allow_categories = [int(c) for c in args.categories.split(',')]
    logger.info(f"Categories: {allow_categories}")
    
    # Determine enable_meta_thinker and procedural_memories
    enable_meta_thinker = args.meta_thinker_mode in ["vanilla", "reflection"]
    procedural_memories = {}
    
    # Load existing procedural memory if provided and mode is vanilla/reflection
    if args.procedural_memory_path and os.path.exists(args.procedural_memory_path):
        try:
            with open(args.procedural_memory_path, 'r') as f:
                pm_data = json.load(f)
            procedural_memories = pm_data.get("memories", {})
            logger.info(f"Loaded procedural memories from {args.procedural_memory_path}")
        except Exception as e:
            logger.warning(f"Failed to load procedural memories: {e}")
    
    # Split samples for reflection training if needed
    training_samples = None
    if args.meta_thinker_mode == "reflection" and len(samples) > args.reflection_test_samples:
        test_count = args.reflection_test_samples
        training_samples = samples[test_count:]  # Use all but first N for training
        samples = samples[:test_count]  # Use first N for testing
        logger.info(f"Reflection mode: {len(training_samples)} training sample(s), {len(samples)} test sample(s)")
    
    # Create evaluator
    evaluator = LightMemMetaThinkerEvaluator(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        qdrant_dir=args.qdrant_dir,
        embedding_model_path=args.embedding_model,
        retrieve_k=args.retrieve_k,
        qr_max_turns=args.qr_max_turns,
        enable_meta_thinker=enable_meta_thinker,
        retrieval_mode=args.retrieval_mode,
        procedural_memories=procedural_memories,
        qa_max_workers=args.qa_max_workers,
        logger=logger,
    )
    
    # Run reflection training if in reflection mode with training samples
    if args.meta_thinker_mode == "reflection" and training_samples:
        learned_pm = evaluator.run_reflection_training(
            training_samples=training_samples,
            allow_categories=allow_categories,
        )
        
        # Update evaluator's procedural memories for inference
        if learned_pm:
            evaluator.procedural_memories = learned_pm
            if evaluator.meta_thinker:
                evaluator.meta_thinker.procedural_memories = learned_pm
            
            # Save procedural memories if path provided
            if args.procedural_memory_path:
                pm_save_path = args.procedural_memory_path
            else:
                pm_save_path = os.path.join(args.output_dir, "procedural_memories.json")
            
            evaluator.reflection_engine.save_procedural_memories(pm_save_path)
    
    # Run evaluation on test samples
    results = evaluator.evaluate_dataset(
        samples=samples,
        allow_categories=allow_categories,
        max_samples=args.max_samples,
    )
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Total questions: {results['total_questions']}")
    logger.info(f"Abstentions: {results['abstentions']}")
    
    logger.info(f"\n{'Category':<20s} {'J':>8s} {'F1':>8s} {'B1':>8s} {'n':>6s}")
    logger.info("-" * 60)
    for key, metrics in sorted(results['aggregate_metrics'].items()):
        if isinstance(metrics, dict) and 'accuracy_mean' in metrics:
            j  = metrics['accuracy_mean']
            f1 = metrics.get('token_f1_mean', 0)
            b1 = metrics.get('bleu1_mean', 0)
            n  = metrics['count']
            logger.info(f"{key:<20s} {j:>8.4f} {f1:>8.4f} {b1:>8.4f} {n:>6d}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
