#!/usr/bin/env python3
"""
A-Mem + Meta-Thinker Integration Script (MeMMA Self-Refine)

Uses A-Mem (ChromaDB + AgenticMemorySystem + evolution) for memory
construction/retrieval, with MeMMA Meta-Thinker for answerability checking
and self-refine for memory improvement.

Architecture:
- Memory Construction: A-Mem (AMEMLayer with LLM metadata extraction + evolution)
- Storage: ChromaDB + pickle persistence
- Retrieval: A-Mem search_agentic (vector + linked neighbors)
- Meta-Thinker: MEMMA MetaThinkerAgent (answerability checking)
- Answer Generation: LightMem's ANSWER_PROMPT (reused for formatting)

Usage:
    python run_memma_self_refine_amem_0310.py \\
        --dataset /path/to/locomo10.json \\
        --amem-dir /path/to/amem_storage \\
        --output_dir results/memma_amem
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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

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

SPACY_SHIM_ACTIVE = False
LIGHTMEM_LLM_JUDGE_AVAILABLE = False
lightmem_llm_judge = None


def _install_spacy_shim() -> None:
    import importlib.machinery
    import types

    if "spacy" in sys.modules:
        return

    shim = types.ModuleType("spacy")
    shim.__version__ = "0.0-shim"
    shim.__spec__ = importlib.machinery.ModuleSpec("spacy", loader=None)

    def _unsupported(*_args, **_kwargs):
        raise RuntimeError(
            "spacy is not installed. A minimal shim is active for retriever import compatibility."
        )

    shim.load = _unsupported
    shim.blank = _unsupported
    sys.modules["spacy"] = shim

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

# Import the LightMem prompt assets used by this script.
# Avoid importing the retriever helpers here because they pull in optional
# qdrant/spacy dependencies that A-Mem mode does not need.
try:
    from prompts import ANSWER_PROMPT, METADATA_GENERATE_PROMPT_locomo
    HAS_LIGHTMEM = True
except ImportError as e:
    print(f"Warning: Cannot import LightMem prompts: {type(e).__name__}: {e}")
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

if HAS_LIGHTMEM:
    try:
        from llm_judge import evaluate_llm_judge as lightmem_llm_judge
        LIGHTMEM_LLM_JUDGE_AVAILABLE = True
    except Exception as e:
        LIGHTMEM_LLM_JUDGE_AVAILABLE = False
        print(
            "Warning: Cannot import LightMem llm_judge; "
            f"fallback judging will be used ({type(e).__name__}: {e})."
        )

try:
    from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
    from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
    HAS_LIGHTMEM_EMBEDDER = True
except ImportError:
    HAS_LIGHTMEM_EMBEDDER = False

try:
    from lightmem.memory_toolkits.memories.layers.amem import AMEMLayer, AMEMConfig
    HAS_AMEM = True
except ImportError:
    HAS_AMEM = False

import shutil
import uuid

HAS_MEMORY_ENTRY = False

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


def prepare_messages_for_lightmem(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a turn payload for LightMem ingestion.

    LightMem's sensory buffer segmentation expects an even-length message list.
    Under `messages_use=user_only`, assistant messages can be filtered out and
    trigger odd-length states internally. We remap assistant role to user for
    ingestion-time stability and pad to even length when needed.
    """
    prepared: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        normalized = dict(msg)
        if normalized.get("role") == "assistant":
            normalized["role"] = "user"
        normalized["content"] = str(normalized.get("content", ""))
        prepared.append(normalized)

    if len(prepared) % 2 == 1:
        filler = dict(prepared[-1]) if prepared else {"role": "user", "content": ""}
        filler["role"] = "user"
        filler["content"] = ""
        prepared.append(filler)

    return prepared


def get_amem_config(
    user_id: str,
    amem_dir: str,
    llm_model: str = "gpt-4o-mini",
    llm_backend: str = "openai",
    embedder_provider: str = "openai",
    retriever_model: str = "text-embedding-3-small",
    evo_threshold: int = 100,
    api_key: str = None,
    base_url: str = None,
) -> "AMEMConfig":
    """Get A-Mem configuration."""
    return AMEMConfig(
        user_id=user_id,
        save_dir=os.path.join(amem_dir, user_id),
        llm_backend=llm_backend,
        llm_model=llm_model,
        embedder_provider=embedder_provider,
        retriever_name_or_path=retriever_model,
        evo_threshold=evo_threshold,
        api_key=api_key,
        base_url=base_url,
    )


def amem_entry_count(amem_layer) -> int:
    """Count entries in an A-Mem layer."""
    try:
        return len(amem_layer.memory_layer.memories)
    except Exception:
        return -1


# ==============================================================================
# Meta-Thinker Construction Guidance Prompt
# ==============================================================================

META_THINKER_CONSTRUCTION_PROMPT = """You are a Meta-Thinker providing guidance for memory construction.

Given the current conversation chunk being processed, return STRICT JSON
describing the memory metadata that should be stored.

Return JSON only:
{
  "keywords": ["..."],
  "context": "one sentence summary",
  "tags": ["..."]
}

Rules:
- Use only explicit information from the chunk.
- keywords: 3-8 short retrieval-critical anchors such as entities, events, dates, or attributes.
- context: one concise sentence summarizing the salient facts.
- tags: 3-8 broad categorical labels useful for retrieval.
- Keep list items short and specific.
- No markdown, bullets, explanations, or extra keys.
"""

CONSTRUCTION_METADATA_MAX_KEYWORDS = 8
CONSTRUCTION_METADATA_MAX_TAGS = 8
CONSTRUCTION_METADATA_MAX_ITEM_WORDS = 8
CONSTRUCTION_METADATA_MAX_ITEM_CHARS = 80
CONSTRUCTION_METADATA_MAX_CONTEXT_CHARS = 240


LLM_CAP_SELECTOR_SYSTEM_PROMPT = """You are a memory subset selector under a strict budget.

Task:
Select memory entry IDs to keep so a downstream QA model can answer QUESTION correctly.

You are NOT answering the question.
You only select IDs from provided candidates.

Hard constraints:
1) Output STRICT JSON only:
   {"keep_ids":["..."], "reason":"..."}
2) keep_ids must be unique and length <= BUDGET_K.
3) Use only candidate IDs. Do not invent IDs.
4) If uncertain, still return your best keep_ids (never empty unless no candidates).

Selection priorities (highest to lowest):
A) Exact answer-bearing evidence for the asked slot:
   - time/date/number/name/location/relationship/cause
B) Specificity over genericity:
   - Prefer "trans woman" over "LGBTQ+"
   - Prefer exact date/day over month-only wording
C) Event alignment:
   - Keep entries about the SAME event asked in QUESTION
   - Avoid near-topic but different events
D) Consistency:
   - Prefer entries with explicit anchors (date, entity, quantity)
   - If conflict exists, prefer the more specific and directly grounded evidence
E) OLD/NEW balance:
   - Do NOT drop all OLD evidence just because NEW exists
   - Preserve old anchor evidence when it directly supports the question

Critical anti-failure rules:
- Never replace a precise old fact with a broader new paraphrase.
- For temporal questions, prioritize entries with explicit date/time anchors.
- For identity/attribute questions, prefer canonical explicit phrasing over umbrella terms.
- Avoid redundant duplicates; keep the most specific representative.

Self-check before output:
- Did I keep direct evidence for the exact asked slot?
- Did I keep at least some OLD anchor evidence when relevant?
- Did I avoid generic substitutes for specific facts?
- Is keep_ids <= BUDGET_K and JSON valid?

Return JSON only.
"""


# ==============================================================================
# LLM Client
# ==============================================================================

class LLMClient:
    """OpenAI-compatible LLM client."""
    
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.model = model
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
SELF_REFINE_ADD_FACT_SYSTEM_PROMPT_v1 = """You are a memory repair assistant for a conversation memory system.

You will be given:
- Question
- Gold Answer
- Model Answer
- retrieved evidence snippets (may be empty)

Your job:
Decide whether we should write ONE new memory fact to help answer the question in the future.

Return STRICT JSON only:
{
  "op": "ADD_FACT" | "NOOP",
  "target_speaker": "speaker_a" | "speaker_b" | "both" | "unknown",
  "fact": "a single canonical fact sentence (no QA format, no 'Question:', no 'Answer:')",
  "dedup_key": "a short stable key like PREF:food or EVENT:trip:date or REL:person:relation",
  "evidence_span": "copy one supporting span from the evidence snippets if available, else \"\"",
  "confidence": 0.0,
  "reason": "one short sentence"
}

Hard rules:
- If Gold Answer is 'Not mentioned in the conversation' or equivalent, output op='NOOP'.
- Do NOT add a fact that is not supported by evidence_span unless evidence is empty.
- The fact must be concrete and retrieval-friendly.
- The fact must NOT contain evaluation artifacts (e.g., 'gold answer', 'model answer', 'verified').
- Output JSON only. No markdown.
"""


SELF_REFINE_ADD_FACT_SYSTEM_PROMPT_v2 = """You are a memory-repair assistant for a two-speaker conversation memory system.

You will receive:
- Question
- Gold Answer
- Model Answer
- Retrieved evidence snippets (each snippet includes id and speaker)

Your task:
Return exactly ONE action:
- ADD_FACT: add one canonical fact to memory, or
- NOOP: do not write memory

Return STRICT JSON only:
{
  "op": "ADD_FACT" | "NOOP",
  "target_speaker": "speaker_a" | "speaker_b" | "unknown",
  "fact": "single canonical fact sentence",
  "dedup_key": "stable short key",
  "evidence_span": "verbatim span from provided evidence snippets, or empty string",
  "confidence": 0.0,
  "reason": "one short sentence"
}

Hard rules:
1) If Gold Answer is unanswerable (e.g., "Not mentioned in the conversation"), output NOOP.
2) For ADD_FACT, target_speaker MUST be speaker_a or speaker_b. Never use unknown for ADD_FACT.
3) If speaker ownership is ambiguous or unsupported, output NOOP with target_speaker="unknown".
4) For ADD_FACT, evidence_span must be copied verbatim from provided evidence snippets (unless evidence is empty).
5) Fact must be concrete and retrieval-friendly, not generic labels replacing specific slots.
6) Fact must NOT contain evaluation artifacts (e.g., "gold answer", "model answer", "verified").
7) Output JSON only. No markdown. No extra keys.

Speaker policy:
- Use speaker_a/speaker_b only when evidence explicitly supports ownership.
- If both speakers are mentioned but ownership is unclear, output NOOP.
- Do not infer speaker from world knowledge or stereotypes.

Few-shot examples:

Example 1
Input:
Question: What is Caroline's identity?
Gold Answer: Transgender woman
Model Answer: LGBTQ+ community member
Retrieved evidence snippets:
- [id=e1][speaker=speaker_a] I am a transgender woman and proud of it.
Output:
{"op":"ADD_FACT","target_speaker":"speaker_a","fact":"Caroline is a transgender woman.","dedup_key":"IDENTITY:caroline:gender","evidence_span":"I am a transgender woman and proud of it.","confidence":0.96,"reason":"Gold answer is explicitly supported with clear speaker ownership."}

Example 2
Input:
Question: When did Melanie go camping in July?
Gold Answer: two weekends before 17 July 2023
Model Answer: July 2023
Retrieved evidence snippets:
- [id=e2][speaker=speaker_b] We went camping two weekends before July 17, 2023.
Output:
{"op":"ADD_FACT","target_speaker":"speaker_b","fact":"Melanie went camping two weekends before 17 July 2023.","dedup_key":"EVENT:melanie:camping:date","evidence_span":"We went camping two weekends before July 17, 2023.","confidence":0.93,"reason":"Date detail is explicit and tied to speaker_b evidence."}

Example 3
Input:
Question: What is Melanie's passport number?
Gold Answer: Not mentioned in the conversation
Model Answer: Not sure
Retrieved evidence snippets:
- [id=e3][speaker=speaker_b] I renewed my passport recently.
Output:
{"op":"NOOP","target_speaker":"unknown","fact":"","dedup_key":"NOOP:unanswerable:passport_number","evidence_span":"","confidence":0.0,"reason":"Gold answer is unanswerable."}

Example 4
Input:
Question: Who recommended the book \"Becoming Nicole\"?
Gold Answer: Caroline recommended it to Melanie
Model Answer: Unknown
Retrieved evidence snippets:
- [id=e4][speaker=speaker_a] I read Becoming Nicole last year.
- [id=e5][speaker=speaker_b] That book was impactful for me.
Output:
{"op":"NOOP","target_speaker":"unknown","fact":"","dedup_key":"NOOP:ambiguous_speaker:becoming_nicole","evidence_span":"","confidence":0.35,"reason":"Evidence mentions the book but does not explicitly attribute recommendation ownership."}
"""


SELF_REFINE_ADD_FACT_SYSTEM_PROMPT_v3 = """You are a memory-repair assistant for a two-speaker conversation memory system.

You will receive:
- Question
- Gold Answer (ground truth from the conversation)
- Model Answer (what the memory system produced — may be wrong or incomplete)
- Retrieved evidence snippets (from current memory; may be irrelevant if the info is missing)

Your task:
Decide whether to ADD one fact to memory so the system can answer correctly next time.

Return STRICT JSON only:
{
  "op": "ADD_FACT" | "NOOP",
  "target_speaker": "speaker_a" | "speaker_b" | "unknown",
  "fact": "single canonical fact sentence",
  "dedup_key": "stable short key like PREF:food or EVENT:trip:date",
  "evidence_span": "verbatim span from evidence if relevant evidence exists, else empty string",
  "confidence": 0.0,
  "reason": "one short sentence"
}

Decision rules (in priority order):
1) If Gold Answer is unanswerable (e.g., "Not mentioned in the conversation"), output NOOP.
2) If Gold Answer is answerable and Model Answer is wrong or incomplete, output ADD_FACT.
   The fact should capture the key information from the Gold Answer needed to answer correctly.
3) If Gold Answer and Model Answer are essentially equivalent, output NOOP.

Evidence policy:
- If evidence snippets contain a relevant span, copy it verbatim into evidence_span.
- If evidence snippets are irrelevant or do not cover the needed information, set evidence_span to "".
  This is expected — the whole point of ADD_FACT is to add MISSING information.

Speaker policy:
- Assign target_speaker based on who the fact is about, using ALL available context:
  question wording (e.g., "What did I do" → the asker), Gold Answer content, and evidence.
- Use "unknown" only as a last resort; it is acceptable for ADD_FACT when speaker cannot be determined.
- Do not refuse to add a fact solely because speaker ownership is uncertain.

Quality rules:
- Fact must be concrete, specific, and retrieval-friendly (include names, dates, details from Gold Answer).
- Fact must NOT contain evaluation artifacts (e.g., "gold answer", "model answer", "verified").
- When the Gold Answer contains relative date expressions (e.g., "last Friday before August 14",
  "two weekends before July 17", "the week before 3 July"), preserve them VERBATIM in the fact.
  Do NOT calculate or convert to absolute dates — day-of-week arithmetic is error-prone.
  Good: "Caroline attended a pride parade last Friday before August 14, 2023"
  Bad:  "Caroline attended a pride parade on August 9, 2023"
- Output JSON only. No markdown. No extra keys.

Few-shot examples:

Example 1 — evidence supports the fact
Input:
Question: What is Caroline's identity?
Gold Answer: Transgender woman
Model Answer: LGBTQ+ community member
Retrieved evidence snippets:
- [id=e1][speaker=speaker_a] I am a transgender woman and proud of it.
Output:
{"op":"ADD_FACT","target_speaker":"speaker_a","fact":"Caroline is a transgender woman.","dedup_key":"IDENTITY:caroline:gender","evidence_span":"I am a transgender woman and proud of it.","confidence":0.96,"reason":"Gold answer is explicitly supported with clear speaker ownership."}

Example 2 — information completely missing from memory
Input:
Question: What does the necklace from Caroline's grandma symbolize?
Gold Answer: Love, faith, and strength. It reminds her of her roots and family support.
Model Answer: The memories do not contain information about a necklace.
Retrieved evidence snippets:
- [id=e2][speaker=speaker_a] I painted a sunrise at the lake last summer.
- [id=e3][speaker=speaker_b] That painting captures beautiful colors.
Output:
{"op":"ADD_FACT","target_speaker":"speaker_a","fact":"Caroline's grandma gave her a necklace from Sweden that symbolizes love, faith, and strength, reminding her of her roots and family support.","dedup_key":"OBJECT:caroline:necklace:grandma","evidence_span":"","confidence":0.88,"reason":"Information is completely absent from memory; fact derived from Gold Answer to fill the gap."}

Example 3 — unanswerable gold answer
Input:
Question: What is Melanie's passport number?
Gold Answer: Not mentioned in the conversation
Model Answer: Not sure
Retrieved evidence snippets:
- [id=e4][speaker=speaker_b] I renewed my passport recently.
Output:
{"op":"NOOP","target_speaker":"unknown","fact":"","dedup_key":"NOOP:unanswerable:passport_number","evidence_span":"","confidence":0.0,"reason":"Gold answer is unanswerable."}
"""


SELF_REFINE_ADD_FACT_PROMPT_BY_VERSION = {
    "v1": SELF_REFINE_ADD_FACT_SYSTEM_PROMPT_v1,
    "v2": SELF_REFINE_ADD_FACT_SYSTEM_PROMPT_v2,
    "v3": SELF_REFINE_ADD_FACT_SYSTEM_PROMPT_v3,
}
DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION = "v3"


SELF_REFINE_DEDUP_SYSTEM_PROMPT = """You are a memory deduplication assistant.

You will receive:
- A NEW proposed fact to be added to memory
- One or more EXISTING memory entries that are semantically similar (with similarity scores)

Decide how to handle the new fact:

Return STRICT JSON only:
{
  "action": "SKIP" | "MERGE" | "INSERT",
  "merge_target_index": -1,
  "merged_fact": "",
  "reason": "one short sentence"
}

Decision rules:
1) SKIP: An existing entry already fully covers the information in the new fact. Nothing new to add.
   Set merge_target_index to -1, merged_fact to "".
2) MERGE: The new fact describes the EXACT SAME single event or attribute as an existing entry
   and adds a missing non-temporal detail (e.g., adding a feeling, a companion, or a location
   to an event already stored). Combine into one entry.
   Set merge_target_index to the 0-based index of the target entry. Set merged_fact to the merged text.
3) INSERT: The new fact is about a different topic, a different event, a different time period,
   or a different occurrence of the same activity. Each distinct event must be a separate entry
   for precise retrieval.
   Set merge_target_index to -1, merged_fact to "".

CRITICAL — temporal/event rule:
- Different dates, different time periods, or different occurrences of the same activity = INSERT, NEVER MERGE.
- Example: "camping trip in June" and "camping trip in July" are DIFFERENT events → INSERT.
- Example: "attended pride parade June 30" and "attended pride parade Aug 11" are DIFFERENT events → INSERT.
- Only MERGE when both texts refer to the exact same single event at the same time.

Merge guidelines:
- The merged_fact must preserve ALL concrete details from BOTH the existing entry and the new fact.
- Keep it as one or two concise sentences. Do not lose any names, dates, or specifics.
- Do not add information that is not in either source.

Few-shot examples:

Example 1 — different trips, same topic → INSERT
New fact: "Melanie's family went camping at the beach on July 6, 2023."
Existing: [0] (score=0.87) "Melanie's family went camping in the mountains around June 20-23, 2023."
Output: {"action":"INSERT","merge_target_index":-1,"merged_fact":"","reason":"Different camping trips at different dates."}

Example 2 — same event, adding detail → MERGE
New fact: "Caroline felt inspired and empowered at the pride parade."
Existing: [0] (score=0.91) "Caroline attended a pride parade on June 30, 2023."
Output: {"action":"MERGE","merge_target_index":0,"merged_fact":"Caroline attended a pride parade on June 30, 2023, and felt inspired and empowered.","reason":"Same parade, adding emotional detail."}

Example 3 — already covered → SKIP
New fact: "Caroline is a transgender woman."
Existing: [0] (score=0.95) "Caroline is a transgender woman and proud of it."
Output: {"action":"SKIP","merge_target_index":-1,"merged_fact":"","reason":"Existing entry already covers this fact."}

Output JSON only. No markdown. No extra keys.
"""


REALTIME_SESSION_QA_SYSTEM_PROMPT = """You generate probe QA pairs to verify what facts are explicitly stated in ONE conversation session.

Return STRICT JSON only:
{
  "questions": [
    {
      "question": "...",
      "answer": "...",
      "type": "single-session" | "temporal" | "preference" | "event" | "unanswerable",
      "target_speaker": "speaker_a" | "speaker_b" | "both" | "unknown",
      "evidence_span": "verbatim substring from the session text that supports the answer (<= 25 words). If unanswerable, use \"\"."
    }
  ]
}

Rules:
- Generate exactly N QA pairs, where N is specified in the user message.
- Each question must test a distinct fact.
- Keep answers short (<= 12 words) unless the source itself is a list.
- For answerable items, evidence_span must appear in the provided session text.
- If facts are insufficient, use type='unanswerable' and answer='Not mentioned in the conversation'.
- Output JSON only. No markdown. No extra keys.
"""


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _extract_first_json_block(text: str) -> Optional[str]:
    if not text:
        return None

    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [pos for pos in [start_obj, start_arr] if pos >= 0]
    if not starts:
        return None

    start = min(starts)
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_str = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _safe_parse_json_response(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fenced block
    if text.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        stripped = re.sub(r"\s*```\s*$", "", stripped).strip()
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # First JSON block
    candidate = _extract_first_json_block(text)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _to_list_like(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return []


def _normalize_construction_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _sanitize_construction_list(raw_values: Any, max_items: int) -> List[str]:
    if isinstance(raw_values, (list, tuple, np.ndarray)):
        source_items = _to_list_like(raw_values)
    else:
        source_items = [raw_values]

    cleaned: List[str] = []
    seen: Set[str] = set()
    for item in source_items:
        text = _normalize_construction_text(item, CONSTRUCTION_METADATA_MAX_ITEM_CHARS)
        if not text:
            continue
        words = text.split()
        if len(words) > CONSTRUCTION_METADATA_MAX_ITEM_WORDS:
            text = " ".join(words[:CONSTRUCTION_METADATA_MAX_ITEM_WORDS])
        normalized_key = text.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _sanitize_construction_metadata(raw_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_obj, dict):
        return None

    metadata = {
        "keywords": _sanitize_construction_list(
            raw_obj.get("keywords", []),
            CONSTRUCTION_METADATA_MAX_KEYWORDS,
        ),
        "context": _normalize_construction_text(
            raw_obj.get("context", ""),
            CONSTRUCTION_METADATA_MAX_CONTEXT_CHARS,
        ),
        "tags": _sanitize_construction_list(
            raw_obj.get("tags", []),
            CONSTRUCTION_METADATA_MAX_TAGS,
        ),
    }
    if not metadata["keywords"] or not metadata["context"] or not metadata["tags"]:
        return None
    return metadata


def _build_construction_metadata(
    llm_client: "LLMClient",
    timestamp: str,
    speaker_name: str,
    content: str,
) -> Optional[Dict[str, Any]]:
    prompt = (
        "Current conversation chunk:\n"
        f"[{timestamp}] {speaker_name}: {content}\n\n"
        "Return JSON only."
    )
    response = llm_client.get_completion(
        prompt,
        temperature=0.0,
        max_tokens=260,
        system_prompt=META_THINKER_CONSTRUCTION_PROMPT,
    )
    parsed = _safe_parse_json_response(response)
    return _sanitize_construction_metadata(parsed)


def _normalize_session_questions(raw_questions: Any) -> List[Dict[str, Any]]:
    questions_src = raw_questions
    if isinstance(questions_src, dict) and "questions" in questions_src:
        questions_src = questions_src.get("questions")

    normalized: List[Dict[str, Any]] = []
    for item in _to_list_like(questions_src):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        normalized.append(
            {
                "question": question,
                "answer": answer,
                "type": str(item.get("type", "single-session")),
                "source": str(item.get("source", "current_session")),
                "target_speaker": _normalize_target_speaker(str(item.get("target_speaker", "unknown"))),
                "evidence_span": str(item.get("evidence_span", "")),
            }
        )
    return normalized


def _canonicalize_questions(questions: List[Dict[str, Any]]) -> str:
    return json.dumps(questions, sort_keys=True, ensure_ascii=False)


def _normalize_question_key(question: str) -> str:
    return re.sub(r"\s+", " ", str(question).strip().lower())


def _is_unanswerable_text(text: str) -> bool:
    normalized = _normalize_question_key(text)
    return normalized in {
        "not mentioned in the conversation",
        "not mentioned in conversation",
        "not mentioned",
        "not answerable",
        "unknown",
        "none",
    }


def _normalize_target_speaker(
    value: str,
    speaker_a_name: str = "",
    speaker_b_name: str = "",
) -> str:
    lowered = _normalize_question_key(value)
    if lowered in {"speaker_a", "speaker 1", "speaker_1", "a"}:
        return "speaker_a"
    if lowered in {"speaker_b", "speaker 2", "speaker_2", "b"}:
        return "speaker_b"
    if lowered in {"both", "speaker_a_and_b", "speaker_a & speaker_b"}:
        return "both"

    if speaker_a_name and lowered == _normalize_question_key(speaker_a_name):
        return "speaker_a"
    if speaker_b_name and lowered == _normalize_question_key(speaker_b_name):
        return "speaker_b"

    return "unknown"


def _make_noop_action(question: str, reason: str = "", **extra: Any) -> Dict[str, Any]:
    payload = {
        "op": "NOOP",
        "question": question,
        "question_key": _normalize_question_key(question),
        "reason": reason,
        "confidence": 0.0,
    }
    payload.update(extra)
    return payload


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _evidence_contains_span(evidence_items: List[Dict[str, Any]], evidence_span: str) -> bool:
    span = _normalize_match_text(evidence_span)
    if not span:
        return True
    for item in evidence_items:
        snippet = _normalize_match_text(str(item.get("snippet", "")))
        if span in snippet:
            return True
    return False


def _token_overlap_score(text_a: str, text_b: str) -> float:
    def _tokens(text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+", _normalize_match_text(text)))

    a = _tokens(text_a)
    b = _tokens(text_b)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / max(1, len(a))


def _infer_target_speaker_from_evidence(
    evidence_items: List[Dict[str, Any]],
    fact: str,
    evidence_span: str,
    speaker_evidence_ids: Optional[List[str]] = None,
    min_confidence: float = 0.75,
    min_margin: float = 0.15,
) -> Dict[str, Any]:
    scores = {"speaker_a": 0.0, "speaker_b": 0.0}
    selected_ids = {str(x) for x in (speaker_evidence_ids or []) if str(x)}
    span_norm = _normalize_match_text(evidence_span)

    for rank, item in enumerate(evidence_items):
        speaker = _normalize_target_speaker(str(item.get("speaker", "unknown")))
        if speaker not in {"speaker_a", "speaker_b"}:
            continue

        snippet = str(item.get("snippet", ""))
        base = 1.0 / float(rank + 1)
        score = base
        snippet_norm = _normalize_match_text(snippet)

        if span_norm and span_norm in snippet_norm:
            score += 1.0

        overlap = _token_overlap_score(fact, snippet)
        if overlap > 0:
            score += min(0.5, 0.5 * overlap)

        if str(item.get("id", "")) in selected_ids:
            score += 0.3

        scores[speaker] += score

    total = scores["speaker_a"] + scores["speaker_b"]
    if total <= 0:
        return {"target_speaker": "unknown", "confidence": 0.0, "scores": scores}

    if scores["speaker_a"] >= scores["speaker_b"]:
        top = "speaker_a"
        second = "speaker_b"
    else:
        top = "speaker_b"
        second = "speaker_a"

    top_score = scores[top]
    second_score = scores[second]
    confidence = top_score / total
    margin = top_score - second_score

    if confidence < min_confidence or margin < min_margin:
        return {"target_speaker": "unknown", "confidence": confidence, "scores": scores}

    return {"target_speaker": top, "confidence": confidence, "scores": scores}


def load_parquet_session_qa(
    parquet_path: str,
    fail_on_empty: bool = True,
) -> Tuple[Dict[Tuple[str, int], List[Dict[str, Any]]], set]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas is required for --self_refine_source parquet. "
            "Install pandas/pyarrow in current environment."
        ) from exc

    if not parquet_path:
        raise ValueError("Parquet path is empty")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "meta" not in df.columns or "reward_model" not in df.columns:
        raise ValueError(
            f"Parquet missing required columns. Expected: meta,reward_model; got: {list(df.columns)}"
        )

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row_idx, row in enumerate(df.to_dict(orient="records")):
        meta = row.get("meta")
        reward_model = row.get("reward_model")
        if not isinstance(meta, dict) or not isinstance(reward_model, dict):
            raise ValueError(
                f"Invalid row {row_idx}: meta/reward_model must be dict. "
                f"meta={type(meta).__name__}, reward_model={type(reward_model).__name__}"
            )
        if "qa_questions" not in reward_model:
            raise ValueError(f"Invalid row {row_idx}: reward_model.qa_questions is missing")

        conv_id = meta.get("conversation_id", reward_model.get("conversation_id"))
        sess_idx_raw = meta.get("session_index", reward_model.get("session_index"))
        if conv_id is None or sess_idx_raw is None:
            raise ValueError(
                f"Invalid row {row_idx}: missing conversation_id/session_index in meta or reward_model"
            )
        try:
            session_index = int(sess_idx_raw)
        except Exception as exc:
            raise ValueError(
                f"Invalid row {row_idx}: session_index must be int-compatible, got {sess_idx_raw!r}"
            ) from exc

        agent_type = str(meta.get("agent_type", reward_model.get("agent_type", "")))
        qa_questions = _normalize_session_questions(reward_model.get("qa_questions"))

        grouped[(str(conv_id), session_index)].append(
            {
                "row_idx": row_idx,
                "agent_type": agent_type,
                "qa_questions": qa_questions,
            }
        )

    if not grouped:
        raise ValueError(f"No session records found in parquet: {parquet_path}")

    session_qa: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    conv_ids = set()
    for key, rows in grouped.items():
        conv_ids.add(key[0])
        non_empty = [r for r in rows if r["qa_questions"]]
        canonical_non_empty = {_canonicalize_questions(r["qa_questions"]) for r in non_empty}
        if len(canonical_non_empty) > 1:
            raise ValueError(
                f"Conflicting qa_questions for {key}. "
                f"Rows={[r['row_idx'] for r in rows]} contain inconsistent non-empty QA payloads."
            )

        core_row = next((r for r in rows if r.get("agent_type") == "core"), None)
        if core_row is not None:
            selected = core_row
        elif non_empty:
            selected = non_empty[0]
        else:
            selected = rows[0]

        selected_questions = selected.get("qa_questions", [])
        if fail_on_empty and not selected_questions:
            raise ValueError(
                f"Empty qa_questions for session {key}. "
                f"Set --self_refine_fail_on_empty false to skip this strict check."
            )
        session_qa[key] = selected_questions

    return session_qa, conv_ids


def init_self_refine_runtime(
    llm_client: LLMClient,
    model_name: str,
    retrieve_k: int,
    self_refine_inference_mode: str,
    amem_dir: str,
    amem_config_kwargs: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    if not HAS_AMEM:
        raise RuntimeError("A-Mem is unavailable; cannot run self refinement.")

    requested_mode = str(self_refine_inference_mode or "simple")
    effective_mode = requested_mode

    runtime: Dict[str, Any] = {
        "llm_client": llm_client,
        "model_name": model_name,
        "retrieve_k": retrieve_k,
        "inference_mode_requested": requested_mode,
        "inference_mode_effective": effective_mode,
        "full_memma_evaluator": None,
        "runtime_binding": "template",
    }

    if requested_mode == "full_memma":
        logger.warning(
            "Build-time self-refine currently supports only simple inference; "
            "forcing self_refine_inference_mode from full_memma to simple."
        )
        effective_mode = "simple"

    runtime["inference_mode_effective"] = effective_mode
    return runtime


def _bind_self_refine_runtime(
    runtime_template: Dict[str, Any],
    amem_layer,
    sample_id: str,
) -> Dict[str, Any]:
    runtime = dict(runtime_template or {})
    runtime["entry_loader"] = amem_layer
    runtime["retriever"] = amem_layer
    runtime["amem_layer"] = amem_layer
    runtime["sample_id"] = sample_id
    runtime["runtime_binding"] = "live_amem_layer"
    return runtime


def _format_amem_entries_as_text(entries: List[Dict[str, Any]]) -> str:
    """Format A-Mem retrieved entries into a readable text block."""
    if not entries:
        return "No memories available."
    lines = []
    for entry in entries:
        used = entry.get("used_content", "")
        if used:
            lines.append(used)
        else:
            lines.append(str(entry.get("content", "")))
    return "\n\n".join(lines) if lines else "No memories available."


def _format_memories_by_speaker_for_refine(retrieved_entries: List[Dict[str, Any]]) -> tuple:
    """Format A-Mem retrieved entries grouped by speaker for self-refine QA."""
    speaker_groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in retrieved_entries:
        content = str(entry.get("content", ""))
        speaker_name = "Unknown"
        if content.startswith("Speaker ") and " says: " in content:
            speaker_name = content.split(" says: ", 1)[0].replace("Speaker ", "")
        speaker_groups.setdefault(speaker_name, []).append(entry)

    speaker_names = list(speaker_groups.keys())
    if len(speaker_names) == 0:
        return "Speaker 1", "No memories available.", "Speaker 2", "No memories available."
    if len(speaker_names) == 1:
        s1 = speaker_names[0]
        return s1, _format_amem_entries_as_text(speaker_groups[s1]), "Speaker 2", "No memories available."

    s1 = speaker_names[0]
    s2 = speaker_names[1]
    return (
        s1,
        _format_amem_entries_as_text(speaker_groups[s1]),
        s2,
        _format_amem_entries_as_text(speaker_groups[s2]),
    )


def _extract_evidence_snippet(entry: Dict[str, Any], max_len: int = 220) -> str:
    text = entry.get("content", "") or entry.get("used_content", "") or ""
    snippet = re.sub(r"\s+", " ", str(text)).strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet


def _answer_question_for_self_refine(
    question: str,
    entries,
    retriever,
    llm_client: LLMClient,
    retrieve_k: int,
    inference_mode: str = "simple",
    full_memma_evaluator: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    if retriever is not None and hasattr(retriever, 'retrieve'):
        retrieved = retriever.retrieve(question, k=retrieve_k)
    elif isinstance(entries, list):
        retrieved = entries[:retrieve_k]
    else:
        retrieved = []

    evidence = []
    for entry in retrieved:
        metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        content = str(entry.get("content", ""))
        speaker = "Unknown"
        if content.startswith("Speaker ") and " says: " in content:
            speaker = content.split(" says: ", 1)[0].replace("Speaker ", "")
        evidence.append(
            {
                "id": str(metadata.get("id", "")),
                "speaker": speaker,
                "snippet": _extract_evidence_snippet(entry),
            }
        )

    inference_mode_used = "simple"
    answer = ""

    if inference_mode == "full_memma" and full_memma_evaluator is not None:
        try:
            full_result = full_memma_evaluator.answer_question(
                question=question,
                entries=entries,
                category=1,
            )
            answer = str(full_result.get("answer", "")).strip()
            if answer:
                inference_mode_used = "full_memma"
        except Exception as exc:
            logger.warning(f"Self-refine full_memma answer failed, fallback to simple: {exc}")

    if not answer:
        speaker_1_name, speaker_1_memories, speaker_2_name, speaker_2_memories = (
            _format_memories_by_speaker_for_refine(retrieved)
        )
        prompt = ANSWER_PROMPT.format(
            speaker_1_name=speaker_1_name,
            speaker_1_memories=speaker_1_memories,
            speaker_2_name=speaker_2_name,
            speaker_2_memories=speaker_2_memories,
            question=question,
        )
        answer = llm_client.get_completion(prompt, temperature=0.0).strip()

    return {
        "answer": answer,
        "retrieved_count": len(retrieved),
        "evidence": evidence,
        "inference_mode_used": inference_mode_used,
    }


def _judge_for_self_refine(
    question: str,
    reference: str,
    generated_answer: str,
    llm_client: LLMClient,
    model_name: str,
    logger: logging.Logger,
) -> float:
    try:
        if LIGHTMEM_LLM_JUDGE_AVAILABLE and lightmem_llm_judge is not None:
            score = lightmem_llm_judge(
                question,
                reference,
                generated_answer,
                client_obj=llm_client.client,
                model_name=model_name,
            )
            return float(score)
    except Exception as exc:
        logger.warning(f"Self-refine judge failed, fallback to exact match: {exc}")

    # Fallback when LLM judge is unavailable.
    return float(_normalize_question_key(reference) == _normalize_question_key(generated_answer))


def _evaluate_questions_parallel(
    questions: List[Dict[str, Any]],
    entries,
    runtime: Dict[str, Any],
    batch_size: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    retriever = runtime.get("retriever")
    llm_client: LLMClient = runtime["llm_client"]
    model_name = str(runtime.get("model_name", llm_client.model))
    retrieve_k = int(runtime.get("retrieve_k", 30))
    inference_mode = str(runtime.get("inference_mode_effective", "simple"))
    full_memma_evaluator = runtime.get("full_memma_evaluator")

    indexed_questions = list(enumerate(questions))
    workers = max(1, min(len(indexed_questions), batch_size))
    # full_memma evaluator keeps internal mutable state; use single worker for determinism/safety.
    if inference_mode == "full_memma":
        workers = 1

    def worker(idx: int, qa: Dict[str, Any]) -> Dict[str, Any]:
        question = str(qa.get("question", ""))
        reference = str(qa.get("answer", ""))
        answer_payload = _answer_question_for_self_refine(
            question=question,
            entries=entries,
            retriever=retriever,
            llm_client=llm_client,
            retrieve_k=retrieve_k,
            inference_mode=inference_mode,
            full_memma_evaluator=full_memma_evaluator,
            logger=logger,
        )
        accuracy = _judge_for_self_refine(
            question=question,
            reference=reference,
            generated_answer=str(answer_payload.get("answer", "")),
            llm_client=llm_client,
            model_name=model_name,
            logger=logger,
        )
        answer_payload.update(
            {
                "index": idx,
                "question": question,
                "reference": reference,
                "accuracy": accuracy,
                "target_speaker": _normalize_target_speaker(str(qa.get("target_speaker", "unknown"))),
                "evidence_span": str(qa.get("evidence_span", "")),
            }
        )
        return answer_payload

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(worker, idx, qa): idx
            for idx, qa in indexed_questions
        }
        for future in as_completed(future_map):
            results.append(future.result())

    results.sort(key=lambda x: x.get("index", 0))
    return results


def _propose_add_fact_action(
    failure_record: Dict[str, Any],
    llm_client: LLMClient,
    speaker_a_name: str,
    speaker_b_name: str,
    self_refine_add_fact_prompt_version: str = DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION,
) -> Dict[str, Any]:
    question = str(failure_record.get("question", ""))
    reference = str(failure_record.get("reference", ""))
    generated = str(failure_record.get("answer", ""))

    # Avoid polluting memory for unanswerable questions.
    if _is_unanswerable_text(reference):
        return _make_noop_action(question, reason="unanswerable_gold")

    evidence_items = failure_record.get("evidence", []) or []
    normalized_evidence_items: List[Dict[str, Any]] = []
    evidence_lines = []
    for item in evidence_items:
        evidence_id = str(item.get("id", "")).strip() or "unknown"
        evidence_speaker = _normalize_target_speaker(
            str(item.get("speaker", "unknown")),
            speaker_a_name=speaker_a_name,
            speaker_b_name=speaker_b_name,
        )
        snippet = str(item.get("snippet", "")).strip()
        normalized_item = dict(item)
        normalized_item["speaker"] = evidence_speaker
        normalized_evidence_items.append(normalized_item)
        if snippet:
            evidence_lines.append(
                f"- [id={evidence_id}][speaker={evidence_speaker}] {snippet}"
            )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "- [No evidence snippets]"

    prompt = f"""Known speakers:
- speaker_a = {speaker_a_name}
- speaker_b = {speaker_b_name}

Question: {question}
Gold answer: {reference}
Generated answer: {generated}
Retrieved evidence snippets:
{evidence_text}

Generate one memory-repair action."""
    prompt_version = str(self_refine_add_fact_prompt_version or DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION)
    system_prompt = SELF_REFINE_ADD_FACT_PROMPT_BY_VERSION.get(
        prompt_version,
        SELF_REFINE_ADD_FACT_SYSTEM_PROMPT_v2,
    )
    response = llm_client.get_completion(
        prompt,
        temperature=0.0,
        max_tokens=360,
        system_prompt=system_prompt,
    )
    parsed = _safe_parse_json_response(response)
    op = "NOOP"
    parsed_target_speaker = "unknown"
    fact = ""
    dedup_key = ""
    evidence_span = ""
    speaker_evidence_ids: List[str] = []
    reason = ""
    confidence = 0.5
    speaker_confidence = 0.0
    if isinstance(parsed, dict):
        op = str(parsed.get("op", op)).strip().upper()
        parsed_target_speaker = _normalize_target_speaker(
            str(parsed.get("target_speaker", parsed_target_speaker))
        )
        fact = str(parsed.get("fact", "")).strip()
        dedup_key = str(parsed.get("dedup_key", "")).strip()
        evidence_span = str(parsed.get("evidence_span", "")).strip()
        speaker_evidence_ids = [str(x) for x in _to_list_like(parsed.get("speaker_evidence_ids")) if str(x)]
        reason = str(parsed.get("reason", "")).strip()
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        try:
            speaker_confidence = float(parsed.get("speaker_confidence", 0.0))
        except Exception:
            speaker_confidence = 0.0

    if op != "ADD_FACT" or not fact:
        return _make_noop_action(question, reason=reason or "model_noop_or_invalid_action")
    if _is_unanswerable_text(fact):
        return _make_noop_action(question, reason="fact_is_unanswerable_text")
    if evidence_span and not _evidence_contains_span(evidence_items, evidence_span):
        return _make_noop_action(
            question,
            reason="invalid_evidence_span",
            invalid_evidence_span_noop=True,
            target_speaker="unknown",
        )

    confidence = max(0.0, min(1.0, confidence))
    question_key = _normalize_question_key(question)
    dedup_key = dedup_key or question_key
    speaker_confidence = max(0.0, min(1.0, speaker_confidence))

    final_target_speaker = parsed_target_speaker
    speaker_inferred = False
    if final_target_speaker not in {"speaker_a", "speaker_b"}:
        inferred = _infer_target_speaker_from_evidence(
            evidence_items=evidence_items,
            fact=fact,
            evidence_span=evidence_span,
            speaker_evidence_ids=speaker_evidence_ids,
        )
        final_target_speaker = inferred.get("target_speaker", "unknown")
        speaker_confidence = max(float(inferred.get("confidence", 0.0)), speaker_confidence)
        speaker_inferred = final_target_speaker in {"speaker_a", "speaker_b"}
        if not speaker_inferred:
            if prompt_version in ("v3",):
                final_target_speaker = "speaker_a"
                speaker_inferred = True
                speaker_confidence = 0.1
            else:
                return _make_noop_action(
                    question,
                    reason="speaker_ambiguous_or_unknown",
                    speaker_ambiguous_noop=True,
                    target_speaker="unknown",
                    prompt_version=prompt_version,
                )

    return {
        "op": "ADD_FACT",
        "question": question,
        "question_key": question_key,
        "value": fact,
        "target_speaker": final_target_speaker,
        "dedup_key": dedup_key,
        "evidence_span": evidence_span,
        "confidence": confidence,
        "speaker_confidence": speaker_confidence,
        "speaker_inferred": speaker_inferred,
        "prompt_version": prompt_version,
        "reason": reason,
        "source_answer": reference,
        "evidence": evidence_items,
    }


def _aggregate_refine_actions(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    selected: Dict[str, Dict[str, Any]] = {}
    dropped = 0
    for action in actions:
        if str(action.get("op", "")).upper() != "ADD_FACT":
            continue
        key = str(action.get("dedup_key", "") or action.get("question_key", ""))
        if not key:
            dropped += 1
            continue
        current = selected.get(key)
        if current is None:
            selected[key] = action
            continue
        if float(action.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            dropped += 1
            selected[key] = action
        else:
            dropped += 1
    return list(selected.values()), dropped


def _parse_session_timestamp(raw_ts: str):
    """Parse a session timestamp like '2023/05/20 (Sat) 00:44' into (iso_str, float_ts, weekday)."""
    import re as _re
    _SESSION_RE = _re.compile(
        r'(?P<date>\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*\((?P<weekday>[^)]+)\)\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?)'
    )
    m = _SESSION_RE.search(raw_ts)
    if m:
        date_str = m.group('date').replace('-', '/')
        time_str = m.group('time')
        weekday = m.group('weekday')
        fmt = "%Y/%m/%d %H:%M:%S" if time_str.count(':') == 2 else "%Y/%m/%d %H:%M"
        base_dt = datetime.datetime.strptime(f"{date_str} {time_str}", fmt)
        return base_dt.isoformat(timespec="milliseconds"), base_dt.timestamp(), weekday
    try:
        dt = datetime.datetime.fromisoformat(raw_ts)
        return dt.isoformat(timespec="milliseconds"), dt.timestamp(), dt.strftime("%a")
    except Exception:
        return raw_ts, 0.0, ""


def _check_and_merge_fact(
    amem_layer,
    fact: str,
    llm_client,
    similarity_threshold: float = 0.8,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Check if a proposed fact duplicates or complements an existing A-Mem entry.

    Returns (action, merged_fact_or_none, target_entry_id_or_none)
    where action is "SKIP", "MERGE", or "INSERT".
    """
    logger = logger or logging.getLogger(__name__)
    try:
        hits = amem_layer.retrieve(fact, k=3)
    except Exception as exc:
        logger.warning(f"[SemanticDedup] A-Mem search failed, falling back to INSERT: {exc}")
        return ("INSERT", None, None)

    if not hits:
        return ("INSERT", None, None)

    candidates = hits
    candidate_lines = []
    for idx, c in enumerate(candidates):
        memory_text = str(c.get("content", "")).strip()
        candidate_lines.append(
            f"[{idx}] {memory_text}"
        )
    candidates_text = "\n".join(candidate_lines)

    prompt = f"""New proposed fact:
{fact}

Existing memory entries:
{candidates_text}

Decide: SKIP, MERGE, or INSERT."""

    try:
        response = llm_client.get_completion(
            prompt,
            temperature=0.0,
            max_tokens=300,
            system_prompt=SELF_REFINE_DEDUP_SYSTEM_PROMPT,
        )
        parsed = _safe_parse_json_response(response)
    except Exception as exc:
        logger.warning(f"[SemanticDedup] LLM call failed, falling back to INSERT: {exc}")
        return ("INSERT", None, None)

    if not isinstance(parsed, dict):
        logger.warning("[SemanticDedup] LLM returned non-dict, falling back to INSERT")
        return ("INSERT", None, None)

    action = str(parsed.get("action", "INSERT")).strip().upper()
    reason = str(parsed.get("reason", "")).strip()

    if action == "SKIP":
        logger.info(f"[SemanticDedup] SKIP: {reason}")
        return ("SKIP", None, None)

    if action == "MERGE":
        merged_fact = str(parsed.get("merged_fact", "")).strip()
        merge_idx = -1
        try:
            merge_idx = int(parsed.get("merge_target_index", -1))
        except (TypeError, ValueError):
            merge_idx = -1
        if not merged_fact or merge_idx < 0 or merge_idx >= len(candidates):
            logger.warning(
                f"[SemanticDedup] MERGE response invalid (idx={merge_idx}, "
                f"fact_len={len(merged_fact)}), falling back to INSERT"
            )
            return ("INSERT", None, None)
        target_metadata = candidates[merge_idx].get("metadata", {})
        target_id = str(target_metadata.get("id", ""))
        if not target_id:
            logger.warning("[SemanticDedup] MERGE target has no id, falling back to INSERT")
            return ("INSERT", None, None)
        logger.info(f"[SemanticDedup] MERGE into {target_id}: {reason}")
        return ("MERGE", merged_fact, target_id)

    return ("INSERT", None, None)


def _apply_refine_actions(
    amem_layer,
    actions: List[Dict[str, Any]],
    session_timestamp: str,
    session_idx: int,
    speaker_a_name: str,
    speaker_b_name: str,
    llm_client=None,
    similarity_threshold: float = 0.8,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    logger = logger or logging.getLogger(__name__)
    iso_ts, float_ts, weekday = _parse_session_timestamp(session_timestamp)

    applied: List[Dict[str, Any]] = []
    dedup_stats: Dict[str, int] = {"skip": 0, "merge": 0, "insert": 0}

    for action in actions:
        if str(action.get("op", "")).upper() != "ADD_FACT":
            continue
        fact = str(action.get("value", "")).strip()
        if not fact:
            continue

        target_speaker = _normalize_target_speaker(str(action.get("target_speaker", "unknown")))
        if target_speaker == "speaker_a":
            speaker_id, speaker_name = "speaker_a", speaker_a_name
        elif target_speaker == "speaker_b":
            speaker_id, speaker_name = "speaker_b", speaker_b_name
        else:
            continue

        if llm_client is not None:
            decision, merged_fact, target_id = _check_and_merge_fact(
                amem_layer=amem_layer,
                fact=fact,
                llm_client=llm_client,
                similarity_threshold=similarity_threshold,
                logger=logger,
            )

            if decision == "SKIP":
                dedup_stats["skip"] += 1
                applied_action = dict(action)
                applied_action["dedup_action"] = "SKIP"
                applied.append(applied_action)
                continue

            if decision == "MERGE" and merged_fact and target_id:
                try:
                    amem_layer.update(target_id, content=merged_fact)
                    dedup_stats["merge"] += 1
                    applied_action = dict(action)
                    applied_action["dedup_action"] = "MERGE"
                    applied_action["merge_target_id"] = target_id
                    applied_action["merged_fact"] = merged_fact
                    applied.append(applied_action)
                    continue
                except Exception as exc:
                    logger.warning(f"[SemanticDedup] MERGE update failed, falling back to INSERT: {exc}")

        dedup_stats["insert"] += 1

        fact_text = f"Speaker {speaker_name} says: {fact}"
        amem_layer.memory_layer.add_note(fact_text, time=iso_ts)

        applied_action = dict(action)
        applied_action["dedup_action"] = "INSERT"
        applied_action["applied_speaker_id"] = speaker_id
        applied_action["applied_speaker_name"] = speaker_name
        applied.append(applied_action)
    return applied, dedup_stats


def _append_jsonl_record(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _format_session_for_realtime_qa(session_messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in session_messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        speaker = str(msg.get("speaker_name") or msg.get("role") or "Speaker")
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "[Empty session]"


def _generate_realtime_session_questions(
    session_messages: List[Dict[str, Any]],
    session_timestamp: str,
    llm_client: LLMClient,
    num_questions: int,
) -> List[Dict[str, Any]]:
    session_text = _format_session_for_realtime_qa(session_messages)
    prompt = f"""Session timestamp: {session_timestamp}
Session text:
{session_text}

Generate exactly {num_questions} QA pairs."""
    response = llm_client.get_completion(
        prompt,
        temperature=0.0,
        max_tokens=1200,
        system_prompt=REALTIME_SESSION_QA_SYSTEM_PROMPT,
    )
    parsed = _safe_parse_json_response(response)
    questions = _normalize_session_questions(parsed)
    if num_questions > 0:
        questions = questions[:num_questions]
    return questions


def _get_session_self_refine_questions(
    self_refine_source: str,
    sample_id: str,
    session_idx: int,
    session_messages: List[Dict[str, Any]],
    session_timestamp: str,
    llm_client: Optional[LLMClient],
    session_qa_index: Optional[Dict[Tuple[str, int], List[Dict[str, Any]]]],
    max_questions: int,
    fail_on_empty: bool,
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    if self_refine_source == "none":
        return questions

    if self_refine_source == "parquet":
        lookup = session_qa_index or {}
        questions = lookup.get((sample_id, session_idx), [])
    elif self_refine_source == "realtime":
        if llm_client is None:
            raise ValueError("Realtime self-refine requires an initialized llm_client")
        requested = max_questions if max_questions > 0 else 5
        questions = _generate_realtime_session_questions(
            session_messages=session_messages,
            session_timestamp=session_timestamp,
            llm_client=llm_client,
            num_questions=requested,
        )

    questions = _normalize_session_questions(questions)
    if max_questions > 0:
        questions = questions[:max_questions]
    if fail_on_empty and not questions:
        raise ValueError(
            f"Empty session QA for sample={sample_id}, session={session_idx}, source={self_refine_source}"
        )
    return questions


def _estimate_self_refine_question_count(
    self_refine_source: str,
    sample_id: str,
    session_idx: int,
    session_qa_index: Optional[Dict[Tuple[str, int], List[Dict[str, Any]]]],
    self_refine_max_questions: int,
) -> int:
    if self_refine_source == "none":
        return 0
    if self_refine_source == "parquet":
        size = len((session_qa_index or {}).get((sample_id, session_idx), []))
        if self_refine_max_questions > 0:
            return min(size, self_refine_max_questions)
        return size
    # realtime source defaults to 5 probes unless max_questions overrides it.
    return self_refine_max_questions if self_refine_max_questions > 0 else 5


def run_session_self_refinement(
    sample_id: str,
    session_idx: int,
    session_messages: List[Dict[str, Any]],
    session_timestamp: str,
    amem_layer,
    runtime: Dict[str, Any],
    llm_client: Optional[LLMClient],
    self_refine_source: str,
    session_qa_index: Optional[Dict[Tuple[str, int], List[Dict[str, Any]]]] = None,
    self_refine_max_questions: int = 0,
    self_refine_batch_size: int = 4,
    self_refine_fail_on_empty: bool = True,
    speaker_a_name: str = "Speaker A",
    speaker_b_name: str = "Speaker B",
    self_refine_inference_mode: str = "simple",
    self_refine_apply_threshold: float = 0.0,
    self_refine_add_fact_prompt_version: str = DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION,
    self_refine_skip_empty_entries: bool = True,
    self_refine_log_jsonl: str = "",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    try:
        apply_threshold = float(self_refine_apply_threshold)
    except Exception:
        apply_threshold = 0.88
    apply_threshold = max(0.0, min(1.0, apply_threshold))

    inference_mode_requested = str(self_refine_inference_mode or "simple")
    inference_mode_effective = str(runtime.get("inference_mode_effective", "simple"))
    add_fact_prompt_version = str(
        self_refine_add_fact_prompt_version or DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION
    )
    if self_refine_source == "none":
        return {
            "sample_id": sample_id,
            "session_index": session_idx,
            "self_refine_source": self_refine_source,
            "prompt_version": add_fact_prompt_version,
            "inference_mode_requested": inference_mode_requested,
            "inference_mode_effective": inference_mode_effective,
            "apply_threshold": apply_threshold,
            "qa_total": 0,
            "qa_correct_before": 0,
            "qa_correct_after": 0,
            "actions_applied": 0,
            "skipped_batches": 0,
            "noop_count": 0,
            "speaker_inferred_count": 0,
            "speaker_ambiguous_noop_count": 0,
            "invalid_evidence_span_noop_count": 0,
            "filtered_low_confidence_count": 0,
            "dedup_dropped_count": 0,
            "deferred_due_to_empty_entries": False,
            "deferred_question_count": 0,
            "batches": [],
        }

    live_amem_layer = amem_layer or runtime.get("amem_layer") or runtime.get("entry_loader")
    if live_amem_layer is None:
        raise RuntimeError("Self-refine runtime missing amem_layer")
    amem_rt = live_amem_layer

    runtime_amem_layer = runtime.get("amem_layer")
    live_entries_probe_count = amem_entry_count(live_amem_layer)
    runtime_entries_probe_count = (
        amem_entry_count(runtime_amem_layer) if runtime_amem_layer is not None else live_entries_probe_count
    )
    if (
        runtime_amem_layer is not None
        and runtime_amem_layer is not live_amem_layer
        and live_entries_probe_count > 0
        and runtime_entries_probe_count <= 0
    ):
        logger.error(
            "[SelfRefine] runtime/live A-Mem mismatch: sample=%s session=%d "
            "live_entries=%d runtime_entries=%d. Using live amem_layer.",
            sample_id,
            session_idx,
            live_entries_probe_count,
            runtime_entries_probe_count,
        )

    questions = _get_session_self_refine_questions(
        self_refine_source=self_refine_source,
        sample_id=sample_id,
        session_idx=session_idx,
        session_messages=session_messages,
        session_timestamp=session_timestamp,
        llm_client=llm_client,
        session_qa_index=session_qa_index,
        max_questions=self_refine_max_questions,
        fail_on_empty=self_refine_fail_on_empty,
    )
    if not questions:
        return {
            "sample_id": sample_id,
            "session_index": session_idx,
            "self_refine_source": self_refine_source,
            "prompt_version": add_fact_prompt_version,
            "inference_mode_requested": inference_mode_requested,
            "inference_mode_effective": inference_mode_effective,
            "apply_threshold": apply_threshold,
            "qa_total": 0,
            "qa_correct_before": 0,
            "qa_correct_after": 0,
            "actions_applied": 0,
            "skipped_batches": 0,
            "noop_count": 0,
            "speaker_inferred_count": 0,
            "speaker_ambiguous_noop_count": 0,
            "invalid_evidence_span_noop_count": 0,
            "filtered_low_confidence_count": 0,
            "dedup_dropped_count": 0,
            "deferred_due_to_empty_entries": False,
            "deferred_question_count": 0,
            "batches": [],
        }

    batch_size = max(1, int(self_refine_batch_size))
    if live_entries_probe_count <= 0:
        if self_refine_skip_empty_entries:
            skipped_batches = (len(questions) + batch_size - 1) // batch_size if questions else 0
            report = {
                "sample_id": sample_id,
                "session_index": session_idx,
                "self_refine_source": self_refine_source,
                "prompt_version": add_fact_prompt_version,
                "inference_mode_requested": inference_mode_requested,
                "inference_mode_effective": inference_mode_effective,
                "apply_threshold": apply_threshold,
                "qa_total": len(questions),
                "qa_correct_before": 0,
                "qa_correct_after": 0,
                "actions_applied": 0,
                "skipped_batches": skipped_batches,
                "noop_count": 0,
                "speaker_inferred_count": 0,
                "speaker_ambiguous_noop_count": 0,
                "invalid_evidence_span_noop_count": 0,
                "filtered_low_confidence_count": 0,
                "dedup_dropped_count": 0,
                "dedup_dropped_within_batch_count": 0,
                "dedup_dropped_seen_count": 0,
                "deferred_due_to_empty_entries": True,
                "deferred_reason": "empty_live_entries",
                "deferred_question_count": len(questions),
                "live_entries_probe_count": live_entries_probe_count,
                "runtime_entries_probe_count": runtime_entries_probe_count,
                "batches": [
                    {
                        "batch_index": 0,
                        "batch_size": len(questions),
                        "skipped_reason": "empty_live_entries_deferred",
                    }
                ],
            }
            logger.warning(
                f"[SelfRefine] sample={sample_id} session={session_idx} "
                f"deferred: empty live entries qa={len(questions)}"
            )
            _append_jsonl_record(self_refine_log_jsonl, report)
            return report
        raise ValueError(f"No entries found for sample={sample_id} during self refinement")

    total_before = 0
    total_after = 0
    total_actions = 0
    noop_count = 0
    speaker_inferred_count = 0
    speaker_ambiguous_noop_count = 0
    invalid_evidence_span_noop_count = 0
    filtered_low_confidence_count = 0
    dedup_dropped_count = 0
    dedup_dropped_seen_count = 0
    total_dedup_skip = 0
    total_dedup_merge = 0
    total_dedup_insert = 0
    skipped_batches = 0
    batch_reports: List[Dict[str, Any]] = []

    for batch_start in range(0, len(questions), batch_size):
        batch_questions = questions[batch_start : batch_start + batch_size]

        entries_before_count = amem_entry_count(live_amem_layer)
        if entries_before_count <= 0:
            if self_refine_skip_empty_entries:
                skipped_batches += 1
                logger.warning(
                    f"[SelfRefine] sample={sample_id} session={session_idx} "
                    f"batch={batch_start // batch_size} skipped: empty live entries"
                )
                batch_reports.append(
                    {
                        "batch_index": batch_start // batch_size,
                        "batch_size": len(batch_questions),
                        "skipped_reason": "empty_live_entries",
                    }
                )
                continue
            raise ValueError(f"No entries found for sample={sample_id} during self refinement")

        before_results = _evaluate_questions_parallel(
            questions=batch_questions,
            entries=amem_rt,
            runtime=runtime,
            batch_size=batch_size,
            logger=logger,
        )
        before_correct = sum(1 for r in before_results if float(r.get("accuracy", 0.0)) > 0.5)

        failure_cases = [r for r in before_results if float(r.get("accuracy", 0.0)) <= 0.5]
        local_actions = [
            _propose_add_fact_action(
                r,
                llm_client=runtime["llm_client"],
                speaker_a_name=speaker_a_name,
                speaker_b_name=speaker_b_name,
                self_refine_add_fact_prompt_version=add_fact_prompt_version,
            )
            for r in failure_cases
        ]

        noop_count += sum(1 for action in local_actions if str(action.get("op", "")).upper() != "ADD_FACT")
        speaker_inferred_count += sum(1 for action in local_actions if bool(action.get("speaker_inferred")))
        speaker_ambiguous_noop_count += sum(
            1 for action in local_actions if bool(action.get("speaker_ambiguous_noop"))
        )
        invalid_evidence_span_noop_count += sum(
            1 for action in local_actions if bool(action.get("invalid_evidence_span_noop"))
        )

        threshold_filtered_actions: List[Dict[str, Any]] = []
        for action in local_actions:
            if str(action.get("op", "")).upper() != "ADD_FACT":
                continue
            try:
                confidence = float(action.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            if confidence < apply_threshold:
                filtered_low_confidence_count += 1
                continue
            threshold_filtered_actions.append(action)

        final_actions = threshold_filtered_actions

        applied_actions, batch_dedup_stats = _apply_refine_actions(
            amem_layer=amem_layer,
            actions=final_actions,
            session_timestamp=session_timestamp,
            session_idx=session_idx,
            speaker_a_name=speaker_a_name,
            speaker_b_name=speaker_b_name,
            llm_client=runtime["llm_client"],
            logger=logger,
        )

        after_results = _evaluate_questions_parallel(
            questions=batch_questions,
            entries=amem_rt,
            runtime=runtime,
            batch_size=batch_size,
            logger=logger,
        )
        after_correct = sum(1 for r in after_results if float(r.get("accuracy", 0.0)) > 0.5)

        total_before += before_correct
        total_after += after_correct
        total_actions += len(applied_actions)
        total_dedup_skip += batch_dedup_stats.get("skip", 0)
        total_dedup_merge += batch_dedup_stats.get("merge", 0)
        total_dedup_insert += batch_dedup_stats.get("insert", 0)
        batch_reports.append(
            {
                "batch_index": batch_start // batch_size,
                "batch_size": len(batch_questions),
                "before_correct": before_correct,
                "after_correct": after_correct,
                "actions_applied": len(applied_actions),
                "prompt_version": add_fact_prompt_version,
                "noop_count": sum(1 for action in local_actions if str(action.get("op", "")).upper() != "ADD_FACT"),
                "speaker_inferred_count": sum(1 for action in local_actions if bool(action.get("speaker_inferred"))),
                "speaker_ambiguous_noop_count": sum(
                    1 for action in local_actions if bool(action.get("speaker_ambiguous_noop"))
                ),
                "invalid_evidence_span_noop_count": sum(
                    1 for action in local_actions if bool(action.get("invalid_evidence_span_noop"))
                ),
                "filtered_low_confidence_count": (
                    len([a for a in local_actions if str(a.get("op", "")).upper() == "ADD_FACT"]) -
                    len(threshold_filtered_actions)
                ),
                "semantic_dedup_stats": batch_dedup_stats,
                "actions": applied_actions,
                "before_results": before_results,
                "after_results": after_results,
            }
        )

    report = {
        "sample_id": sample_id,
        "session_index": session_idx,
        "self_refine_source": self_refine_source,
        "prompt_version": add_fact_prompt_version,
        "inference_mode_requested": inference_mode_requested,
        "inference_mode_effective": inference_mode_effective,
        "apply_threshold": apply_threshold,
        "qa_total": len(questions),
        "qa_correct_before": total_before,
        "qa_correct_after": total_after,
        "actions_applied": total_actions,
        "skipped_batches": skipped_batches,
        "noop_count": noop_count,
        "speaker_inferred_count": speaker_inferred_count,
        "speaker_ambiguous_noop_count": speaker_ambiguous_noop_count,
        "invalid_evidence_span_noop_count": invalid_evidence_span_noop_count,
        "filtered_low_confidence_count": filtered_low_confidence_count,
        "dedup_dropped_count": dedup_dropped_count + dedup_dropped_seen_count,
        "dedup_dropped_within_batch_count": dedup_dropped_count,
        "dedup_dropped_seen_count": dedup_dropped_seen_count,
        "semantic_dedup_skip_count": total_dedup_skip,
        "semantic_dedup_merge_count": total_dedup_merge,
        "semantic_dedup_insert_count": total_dedup_insert,
        "deferred_due_to_empty_entries": False,
        "deferred_reason": "",
        "deferred_question_count": 0,
        "live_entries_probe_count": live_entries_probe_count,
        "runtime_entries_probe_count": runtime_entries_probe_count,
        "batches": batch_reports,
    }
    logger.info(
        f"[SelfRefine] sample={sample_id} session={session_idx} "
        f"qa={len(questions)} correct={total_before}->{total_after} actions={total_actions} "
        f"dedup(skip={total_dedup_skip},merge={total_dedup_merge},insert={total_dedup_insert})"
    )
    _append_jsonl_record(self_refine_log_jsonl, report)
    return report
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

META_THINKER_ANSWERABILITY_PROMPT_v2 = """You are a Meta-Thinker agent for answerability checking in a memory-augmented QA system.

You will be given:
- Question
- Retrieved memories grouped by speaker (memory_id, timestamp, snippet)
- Previous queries

Primary objective:
Decide whether evidence is SUFFICIENT for a downstream answer agent to produce a correct answer.
Do NOT require perfect completeness if the asked slot is already answerable.

Hard constraints:
- Use ONLY provided evidence; no fabricated facts.
- Be calibrated (not overly conservative): avoid false NOT_ANSWERABLE when core evidence is already enough.
- Focus on the exact asked slot, not peripheral details.

Decision rule:
Choose ANSWERABLE when all are true:
1) Core-slot support: The key slot asked by the question is directly supported (or strongly entailed) by evidence.
2) No blocking contradiction: No unresolved conflict that would change the final answer.
3) Needed granularity met: Evidence granularity matches what the question explicitly requires.
4) No guesswork on the core slot: Final answer can be produced with high confidence.

Choose NOT_ANSWERABLE only if any core requirement above fails.

Granularity calibration:
A) Time questions:
- If question explicitly asks exact day/date (e.g., "exact date", "what day", "on which date"), require day-level evidence.
- Otherwise, month/year-level evidence is sufficient if unambiguous.
- Relative time is sufficient only when anchored clearly by timestamp context.

B) Likelihood / hypothetical yes-no questions (e.g., "would likely"):
- Limited inference is allowed ONLY from explicit evidence patterns (stated preferences, repeated behavior, explicit plans).
- If evidence supports a clear direction and no contradiction exists, choose ANSWERABLE.

C) List/plural questions:
- If question does NOT explicitly require exhaustive/count-complete output, at least one clearly supported item can be sufficient.
- If question asks total count/exhaustive set ("how many", "all", "exactly which"), require corresponding completeness.

D) Identity/status/attribute questions:
- Explicit statements or strong direct evidence are sufficient; do not reject for missing non-essential context.

Anti-overconservative rules:
- Do NOT require extra details that were not asked.
- Do NOT reject only because wording is paraphrased rather than identical.
- If the answer can be short, correct, and evidence-grounded, choose ANSWERABLE.

If NOT_ANSWERABLE, provide actionable retrieval guidance:
- Name the single most critical missing gap first.
- Specify missing speaker precisely (speaker_1 / speaker_2 / both / unknown).
- For time gaps, specify required granularity (day vs month/year) and absolute/relative need.

Output format (STRICT):
<decision>ANSWERABLE|NOT_ANSWERABLE</decision>

<reason>
1-3 sentences explaining the decision on the core asked slot.
</reason>

<key_gaps>
- If NOT_ANSWERABLE: ranked bullets, TOP-1 is the blocking gap.
- If ANSWERABLE: NONE
</key_gaps>

<missing_speaker>
- If NOT_ANSWERABLE: speaker_1 / speaker_2 / both / unknown
- If ANSWERABLE: NONE
</missing_speaker>

<time_need>
- If NOT_ANSWERABLE and time-related: required granularity and needed anchor
- Otherwise: NONE
</time_need>

<retrieval_guidance>
- Only if NOT_ANSWERABLE.
1) goal: one sentence about what to retrieve next.
2) suggested_queries: 2-4 concrete queries, each targeting a different angle.
3) keywords: 3-8 entities/terms.
4) constraints: required speaker/time/entity scope.
5) avoid_terms: exhausted or overly generic terms.
</retrieval_guidance>"""

META_THINKER_ANSWERABILITY_PROMPT_v3 = """You are a Meta-Thinker agent for answerability checking in a memory-augmented QA system.

You will be given:
- Question
- Retrieved memories grouped by speaker (memory_id, timestamp, snippet)
- Previous queries

Primary objective:
Minimize false NOT_ANSWERABLE while staying evidence-grounded.
Decide whether the CURRENT evidence is sufficient for a downstream answer agent to produce a correct answer to the asked slot.

Hard constraints:
- Use ONLY provided evidence; do not invent facts.
- Focus on the core asked slot.
- Do NOT require unasked details.
- If the best-supported answer is already clear, choose ANSWERABLE.

Decision rule:
Choose ANSWERABLE when all are true:
1) Core-slot support exists (directly stated OR strongly entailed by explicit evidence).
2) No blocking contradiction that would change the final answer.
3) Required granularity is satisfied by the question wording.
4) The answer can be produced without guessing the core slot.

Granularity policy (critical):
A) Time questions:
- "When did/when is ..." does NOT automatically require exact day-level date.
- Require exact day/date only if question explicitly asks: "exact date", "what day", "on which date", "specific date".
- Otherwise, use best available unambiguous granularity (day > month > year > anchored relative time).
- Relative time is sufficient when anchored by memory timestamp (e.g., "yesterday" + dated message).
- Do NOT mark NOT_ANSWERABLE only because a finer granularity is missing but not asked.

B) Who/what/which questions:
- One directly supported item/entity is sufficient unless the question explicitly asks exhaustive set/count
  (e.g., "all", "how many", "exactly which", "list every").

C) Hypothetical / likely yes-no questions:
- Limited inference is allowed from explicit preferences, repeated behavior, stated plans, and explicit causal statements.
- If direction is clear and uncontradicted, choose ANSWERABLE.

D) Status/attribute questions:
- Explicit statement OR strong direct evidence is sufficient.
- Paraphrase-equivalent evidence is acceptable.

Multi-turn anti-stall rule:
- If retrieval has already been attempted multiple times and evidence repeatedly supports the same core answer with no blocker, prefer ANSWERABLE at best-supported granularity.
- Do NOT keep returning NOT_ANSWERABLE for the same non-blocking "missing detail".

If NOT_ANSWERABLE:
- Provide the single most critical blocking gap first.
- Missing speaker must be one of: speaker_1 / speaker_2 / both / unknown.
- For time gaps, state required granularity explicitly (day vs month vs year) and whether an anchor is missing.

Output format (STRICT):
<decision>ANSWERABLE|NOT_ANSWERABLE</decision>

<reason>
1-3 sentences focused on the core asked slot only.
</reason>

<key_gaps>
- If NOT_ANSWERABLE: ranked bullets; TOP-1 must be the blocker.
- If ANSWERABLE: NONE
</key_gaps>

<missing_speaker>
- If NOT_ANSWERABLE: speaker_1 / speaker_2 / both / unknown
- If ANSWERABLE: NONE
</missing_speaker>

<time_need>
- If NOT_ANSWERABLE and time-related: required granularity and missing anchor
- Otherwise: NONE
</time_need>

<retrieval_guidance>
- Only if NOT_ANSWERABLE.
1) goal: one sentence.
2) suggested_queries: 2-4 concrete queries.
3) keywords: 3-8 entities/terms.
4) constraints: required speaker/time/entity scope.
5) avoid_terms: exhausted or generic terms.
</retrieval_guidance>"""

META_THINKER_ANSWERABILITY_PROMPT_v4 = """You are a Meta-Thinker agent for answerability checking in a memory-augmented QA system.

You will be given:
- Question
- Retrieved memories grouped by speaker (memory_id, timestamp, snippet)
- Previous queries

Goal:
Minimize false NOT_ANSWERABLE while staying evidence-grounded.

Blocking-gap test (critical):
Return NOT_ANSWERABLE only if a missing fact or unresolved contradiction would CHANGE the final short answer.
If a best-supported answer is already stable, return ANSWERABLE.

Rules:
1) Use only provided evidence. No fabrication.
2) Focus only on the asked slot. Do not require unasked details.
3) Do not reject just because evidence is paraphrased (not verbatim).

Granularity policy:
A) Time questions:
- "When did/when is ..." does NOT automatically require exact day.
- Require exact day/date only if question explicitly asks: "exact date", "exact day", "on which date", "specific date".
- Otherwise accept best unambiguous granularity (day > month > year > anchored relative time).
- Relative time is valid when anchored by timestamp context.

B) Who/what/which:
- One clearly supported item/entity is enough unless question explicitly requests exhaustive output
  (e.g., all/every/how many/exactly which/list every).

C) Hypothetical/likely:
- Limited inference is allowed from explicit preferences, repeated behavior, stated plans, or explicit causal statements.
- If direction is evidence-supported and uncontradicted, choose ANSWERABLE.

D) Contradictions:
- Only contradictions that change the final answer are blocking.

Anti-stall:
- If multiple rewrites were already attempted (>=3 previous queries) and the same non-blocking gap repeats, prefer ANSWERABLE at best-supported granularity.

If NOT_ANSWERABLE:
- Give the single most critical blocker first.
- Specify missing speaker exactly as one token: speaker_1 | speaker_2 | both | unknown.
- For time gaps, state required granularity and missing anchor.

Output format (STRICT):
<decision>ANSWERABLE|NOT_ANSWERABLE</decision>

<reason>
1-3 sentences about the asked slot only.
</reason>

<key_gaps>
- If NOT_ANSWERABLE: ranked bullets, TOP-1 is the blocker.
- If ANSWERABLE: NONE
</key_gaps>

<missing_speaker>
- If NOT_ANSWERABLE: speaker_1 / speaker_2 / both / unknown
- If ANSWERABLE: NONE
</missing_speaker>

<time_need>
- If NOT_ANSWERABLE and time-related: required granularity and missing anchor
- Otherwise: NONE
</time_need>

<retrieval_guidance>
- Only if NOT_ANSWERABLE.
1) goal: one sentence.
2) suggested_queries: 2-4 concrete non-overlapping queries.
3) keywords: 3-8 entities/terms.
4) constraints: required speaker/time/entity scope.
5) avoid_terms: exhausted or generic terms.
</retrieval_guidance>"""

META_THINKER_ANSWERABILITY_PROMPT_v5 = """You are a Meta-Thinker agent for answerability checking in a memory-augmented QA system.

You will be given:
- Question
- Retrieved memories grouped by speaker (memory_id, timestamp, snippet)
- Previous queries

Goal:
Minimize false NOT_ANSWERABLE while staying evidence-grounded.

Blocking-gap test (critical):
Return NOT_ANSWERABLE only if a missing fact or unresolved contradiction would CHANGE the final short answer.
If a best-supported answer is already stable, return ANSWERABLE.

Rules:
1) Use only provided evidence. No fabrication.
2) Focus only on the asked slot. Do not require unasked details.
3) Do not reject just because evidence is paraphrased (not verbatim).

Granularity policy:
A) Time questions:
- "When did/when is ..." does NOT automatically require exact day.
- Require exact day/date only if question explicitly asks: "exact date", "exact day", "on which date", "specific date".
- Otherwise accept best unambiguous granularity (day > month > year > anchored relative time).
- Relative time is valid when anchored by timestamp context.
- For plain "When did/when is" wording, month/year or anchored relative time is sufficient; do NOT force calendar-day precision.

B) Who/what/which:
- One clearly supported item/entity is enough unless question explicitly requests exhaustive output
  (e.g., all/every/how many/exactly which/list every).
- Questions like "What X has ...?" are non-exhaustive by default unless exhaustive wording is explicit.

C) Hypothetical/likely:
- Limited inference is allowed from explicit preferences, repeated behavior, stated plans, or explicit causal statements.
- If direction is evidence-supported and uncontradicted, choose ANSWERABLE.

D) Contradictions:
- Only contradictions that change the final answer are blocking.

Anti-stall:
- If multiple rewrites were already attempted (>=3 previous queries) and the same non-blocking gap repeats, prefer ANSWERABLE at best-supported granularity.
- If previous queries already contain the original question plus >=3 rewrites and no NEW blocker appears, output ANSWERABLE.
- Repeating the same non-blocking gap across turns is not sufficient for NOT_ANSWERABLE.

If NOT_ANSWERABLE:
- Give the single most critical blocker first.
- Specify missing speaker exactly as one token: speaker_1 | speaker_2 | both | unknown.
- For time gaps, state required granularity and missing anchor.
- The blocker must be NEW and answer-changing; if it is the same non-blocking detail requested before, choose ANSWERABLE.

Output format (STRICT):
<decision>ANSWERABLE|NOT_ANSWERABLE</decision>

<reason>
1-3 sentences about the asked slot only.
</reason>

<key_gaps>
- If NOT_ANSWERABLE: ranked bullets, TOP-1 is the blocker.
- If ANSWERABLE: NONE
</key_gaps>

<missing_speaker>
- If NOT_ANSWERABLE: speaker_1 / speaker_2 / both / unknown
- If ANSWERABLE: NONE
</missing_speaker>

<time_need>
- If NOT_ANSWERABLE and time-related: required granularity and missing anchor
- Otherwise: NONE
</time_need>

<retrieval_guidance>
- Only if NOT_ANSWERABLE.
1) goal: one sentence.
2) suggested_queries: 2-4 concrete non-overlapping queries.
3) keywords: 3-8 entities/terms.
4) constraints: required speaker/time/entity scope.
5) avoid_terms: exhausted or generic terms.
</retrieval_guidance>"""

ANSWERABILITY_PROMPT_BY_VERSION = {
    "v1": META_THINKER_ANSWERABILITY_PROMPT,
    "v2": META_THINKER_ANSWERABILITY_PROMPT_v2,
    "v3": META_THINKER_ANSWERABILITY_PROMPT_v3,
    "v4": META_THINKER_ANSWERABILITY_PROMPT_v4,
    "v5": META_THINKER_ANSWERABILITY_PROMPT_v5,
}


class MetaThinkerAgent:
    """Meta-Thinker for answerability checking with procedural memory support."""
    
    def __init__(
        self,
        llm: LLMClient,
        logger: logging.Logger,
        procedural_memories: Dict[str, str] = None,
        answerability_prompt: str = META_THINKER_ANSWERABILITY_PROMPT,
    ):
        self.llm = llm
        self.logger = logger
        # Procedural memories: {"meta": "...", "retrieval": "...", "answering": "...", "storage": "..."}
        self.procedural_memories = procedural_memories or {}
        self.answerability_prompt = answerability_prompt or META_THINKER_ANSWERABILITY_PROMPT
    
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
        
        system_prompt = self.answerability_prompt + pm_section
        
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
# A-Mem + Meta-Thinker Evaluator
# ==============================================================================

class AMemMetaThinkerEvaluator:
    """Evaluator using A-Mem memory + Meta-Thinker for answerability with reflection support."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        base_url: str = None,
        amem_dir: str = None,
        amem_config_kwargs: Dict[str, Any] = None,
        retrieve_k: int = 60,
        qr_max_turns: int = 3,
        enable_meta_thinker: bool = True,
        retrieval_mode: str = "combined",
        procedural_memories: Dict[str, str] = None,
        qa_max_workers: int = 1,
        rewrite_rerank_strategy: str = "keep_new_prune_old",
        answerability_prompt: str = META_THINKER_ANSWERABILITY_PROMPT_v4,
        answerability_prompt_version: str = "v4",
        logger: logging.Logger = None,
        **_extra,
    ):
        self.llm = LLMClient(model=model, api_key=api_key, base_url=base_url)
        self.amem_dir = amem_dir
        self.amem_config_kwargs = amem_config_kwargs or {}
        self.retrieve_k = retrieve_k
        self.qr_max_turns = qr_max_turns
        self.enable_meta_thinker = enable_meta_thinker
        self.retrieval_mode = retrieval_mode
        self.qa_max_workers = qa_max_workers
        self.rewrite_rerank_strategy = rewrite_rerank_strategy
        self.logger = logger or logging.getLogger(__name__)
        self.answerability_prompt = answerability_prompt
        self.answerability_prompt_version = answerability_prompt_version
        
        self.procedural_memories = procedural_memories or {}
        
        self.logger.info(f"Retrieval mode: {retrieval_mode}")
        self.logger.info(f"Answerability Prompt Version: {self.answerability_prompt_version}")
        valid_rerank_strategies = {"keep_new_prune_old", "original_topk", "llm_select", "no_cap"}
        if self.rewrite_rerank_strategy not in valid_rerank_strategies:
            self.logger.warning(
                "Unknown rewrite_rerank_strategy=%s, fallback to no_cap for A-Mem",
                self.rewrite_rerank_strategy,
            )
            self.rewrite_rerank_strategy = "no_cap"
        if self.rewrite_rerank_strategy == "no_cap":
            self.logger.info("Rewrite truncation: disabled (strategy=no_cap)")
        else:
            self.logger.info(
                "Rewrite truncation: enabled (cap=%d), strategy=%s",
                self.retrieve_k,
                self.rewrite_rerank_strategy,
            )
        if self.procedural_memories:
            self.logger.info(f"Loaded procedural memories: {list(self.procedural_memories.keys())}")
        
        self._amem_layers: Dict[str, AMEMLayer] = {}
        self.retriever = True
        
        self._embed_lock = threading.Lock()
        
        self.meta_thinker = MetaThinkerAgent(
            self.llm,
            self.logger,
            procedural_memories=self.procedural_memories,
            answerability_prompt=self.answerability_prompt,
        ) if enable_meta_thinker else None
        
        self.reflection_engine = None
        if HAS_REFLECTION:
            self.reflection_engine = ReflectionEngine(self.llm, self.logger)
        
        self.current_retrieval_history: List[Dict] = []
        self.current_storage_actions: List[Dict] = []

    def _get_amem_layer(self, sample_id: str) -> Optional[AMEMLayer]:
        """Load or return cached AMEMLayer for a sample."""
        if sample_id in self._amem_layers:
            return self._amem_layers[sample_id]
        if not self.amem_dir or not HAS_AMEM:
            return None
        try:
            cfg = get_amem_config(
                user_id=sample_id,
                amem_dir=self.amem_dir,
                **self.amem_config_kwargs,
            )
            layer = AMEMLayer(cfg)
            loaded = layer.load_memory(sample_id)
            if not loaded:
                self.logger.warning(f"No saved A-Mem data for {sample_id}")
                return None
            self._amem_layers[sample_id] = layer
            return layer
        except Exception as exc:
            self.logger.error(f"Failed to load A-Mem for {sample_id}: {exc}")
            return None

    def _amem_retrieve(self, amem_layer: AMEMLayer, query: str, k: int) -> List[Dict]:
        """Retrieve from A-Mem with thread safety."""
        with self._embed_lock:
            return amem_layer.retrieve(query, k=k)

    def _format_memories_by_speaker(self, retrieved_entries: List[Dict]) -> tuple:
        """Format A-Mem retrieved entries grouped by speaker."""
        speaker_groups: Dict[str, List[Dict]] = {}
        for entry in retrieved_entries:
            content = str(entry.get("content", ""))
            speaker_name = "Unknown"
            if content.startswith("Speaker ") and " says: " in content:
                speaker_name = content.split(" says: ", 1)[0].replace("Speaker ", "")
            speaker_groups.setdefault(speaker_name, []).append(entry)
        
        speaker_names = list(speaker_groups.keys())
        
        if len(speaker_names) == 0:
            return "Speaker 1", "No memories available.", "Speaker 2", "No memories available."
        elif len(speaker_names) == 1:
            speaker_1_name = speaker_names[0]
            speaker_1_memories = _format_amem_entries_as_text(speaker_groups[speaker_1_name])
            return speaker_1_name, speaker_1_memories, "Speaker 2", "No memories available."
        else:
            speaker_1_name = speaker_names[0]
            speaker_2_name = speaker_names[1]
            speaker_1_memories = _format_amem_entries_as_text(speaker_groups[speaker_1_name])
            speaker_2_memories = _format_amem_entries_as_text(speaker_groups[speaker_2_name])
            return speaker_1_name, speaker_1_memories, speaker_2_name, speaker_2_memories
    
    def _thread_safe_retrieve(self, amem_layer: AMEMLayer, query: str, limit: int) -> List[Dict]:
        """Thread-safe A-Mem retrieval."""
        return self._amem_retrieve(amem_layer, query, limit)

    def _thread_safe_retrieve_by_speaker(
        self, amem_layer: AMEMLayer, query: str, limit_per_speaker: int
    ) -> List[Dict]:
        """A-Mem doesn't natively support per-speaker retrieval; retrieve combined and split."""
        total_limit = limit_per_speaker * 2
        return self._amem_retrieve(amem_layer, query, total_limit)

    @staticmethod
    def _entry_id(entry: Dict) -> str:
        metadata = entry.get("metadata", {})
        return str(metadata.get("id", ""))

    def _unique_by_id(self, entries: List[Dict]) -> List[Dict]:
        unique_entries = []
        seen_ids = set()
        for entry in entries:
            entry_id = self._entry_id(entry)
            if entry_id in seen_ids:
                continue
            unique_entries.append(entry)
            seen_ids.add(entry_id)
        return unique_entries

    def _entry_brief(self, entry: Dict, max_len: int = 180) -> str:
        content = str(entry.get("content", ""))
        metadata = entry.get("metadata", {}) or {}
        ts = str(metadata.get("timestamp", ""))
        speaker = "Unknown"
        if content.startswith("Speaker ") and " says: " in content:
            speaker = content.split(" says: ", 1)[0].replace("Speaker ", "")
        memory = content.replace("\n", " ").strip()
        if len(memory) > max_len:
            memory = memory[:max_len] + "..."
        return f"({speaker}, {ts}) {memory}"

    def _build_llm_selector_prompt(
        self,
        question: str,
        meta_result: Optional[Dict[str, Any]],
        old_unique: List[Dict],
        new_unique: List[Dict],
        cap: int,
    ) -> tuple:
        meta_result = meta_result or {}
        top_gap = str(meta_result.get("key_gaps", "") or "").strip()
        if "\n" in top_gap:
            top_gap = top_gap.split("\n", 1)[0].strip()
        missing_speaker = str(meta_result.get("missing_speaker", "") or "").strip()
        time_need = str(meta_result.get("time_need", "") or "").strip()

        candidates = [("OLD", e) for e in old_unique] + [("NEW", e) for e in new_unique]
        max_candidates = max(2 * cap, cap)
        candidates = candidates[: min(len(candidates), max_candidates)]

        lines = [
            f"QUESTION: {question}",
            f"TOP_GAP: {top_gap or 'NONE'}",
            f"MISSING_SPEAKER: {missing_speaker or 'NONE'}",
            f"TIME_NEED: {time_need or 'NONE'}",
            f"BUDGET_K: {cap}",
            "",
            "CANDIDATES:",
        ]
        for idx, (tag, entry) in enumerate(candidates):
            entry_id = self._entry_id(entry)
            lines.append(
                f"[{idx}] id={entry_id} tag={tag} {self._entry_brief(entry)}"
            )
        return "\n".join(lines), candidates

    def _parse_selector_output(
        self,
        raw: str,
        valid_ids: set,
        cap: int,
    ) -> tuple:
        reason = ""
        stats = {
            "llm_returned_n": 0,
            "validated_n": 0,
            "invalid_n": 0,
            "dropped_by_cap_n": 0,
        }
        if not raw:
            return [], reason, False, stats

        try:
            json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not json_match:
                return [], reason, False, stats
            parsed = json.loads(json_match.group(0))
            keep_ids = parsed.get("keep_ids", [])
            reason = str(parsed.get("reason", "")).strip()
            if not isinstance(keep_ids, list):
                return [], reason, False, stats
            stats["llm_returned_n"] = len(keep_ids)
            cleaned_ids = []
            seen = set()
            for raw_id in keep_ids:
                entry_id = str(raw_id).strip()
                if not entry_id or entry_id in seen or entry_id not in valid_ids:
                    stats["invalid_n"] += 1
                    continue
                if len(cleaned_ids) >= cap:
                    stats["dropped_by_cap_n"] += 1
                    continue
                cleaned_ids.append(entry_id)
                seen.add(entry_id)
            stats["validated_n"] = len(cleaned_ids)
            if not cleaned_ids:
                return [], reason, False, stats
            return cleaned_ids, reason, True, stats
        except Exception:
            return [], reason, False, stats

    def _llm_select_cap(
        self,
        question: str,
        meta_result: Optional[Dict[str, Any]],
        old_unique: List[Dict],
        new_unique: List[Dict],
        cap: int,
    ) -> tuple:
        prompt, candidates = self._build_llm_selector_prompt(
            question=question,
            meta_result=meta_result,
            old_unique=old_unique,
            new_unique=new_unique,
            cap=cap,
        )
        id_to_entry = {self._entry_id(e): e for _, e in candidates}
        valid_ids = set(id_to_entry.keys())
        diag = {
            "cap_selector_source": "fallback_original_topk",
            "cap_selector_parse_ok": False,
            "cap_selector_reason": "",
            "cap_selector_llm_returned_n": 0,
            "cap_selector_validated_n": 0,
            "cap_selector_invalid_n": 0,
            "cap_selector_dropped_by_cap_n": 0,
        }

        if not candidates:
            return None, diag

        try:
            raw = self.llm.get_completion(
                prompt,
                temperature=0.0,
                max_tokens=500,
                system_prompt=LLM_CAP_SELECTOR_SYSTEM_PROMPT,
            )
        except Exception as e:
            self.logger.warning(f"LLM selector call failed: {e}")
            return None, diag

        keep_ids, reason, parse_ok, parse_stats = self._parse_selector_output(raw, valid_ids, cap)
        diag["cap_selector_parse_ok"] = parse_ok
        diag["cap_selector_reason"] = reason
        diag["cap_selector_llm_returned_n"] = int(parse_stats.get("llm_returned_n", 0))
        diag["cap_selector_validated_n"] = int(parse_stats.get("validated_n", 0))
        diag["cap_selector_invalid_n"] = int(parse_stats.get("invalid_n", 0))
        diag["cap_selector_dropped_by_cap_n"] = int(parse_stats.get("dropped_by_cap_n", 0))

        self.logger.info(
            "LLM selector stats: returned=%d validated=%d invalid=%d dropped_by_cap=%d parse_ok=%s",
            diag["cap_selector_llm_returned_n"],
            diag["cap_selector_validated_n"],
            diag["cap_selector_invalid_n"],
            diag["cap_selector_dropped_by_cap_n"],
            diag["cap_selector_parse_ok"],
        )
        if not parse_ok:
            self.logger.warning("LLM selector parse failed, fallback to original_topk")
            return None, diag

        selected = []
        for entry_id in keep_ids:
            entry = id_to_entry.get(entry_id)
            if entry is not None:
                selected.append(entry)
        selected = self._unique_by_id(selected)
        if not selected:
            self.logger.warning("LLM selector returned empty/invalid keep_ids, fallback to original_topk")
            return None, diag

        diag["cap_selector_source"] = "llm"
        return selected[:cap], diag

    def _rerank_by_original_question(
        self,
        question: str,
        candidate_entries: List[Dict],
        limit: int,
    ) -> List[Dict]:
        if limit <= 0:
            return []
        unique_candidates = self._unique_by_id(candidate_entries)
        if not unique_candidates:
            return []
        return unique_candidates[:limit]

    def _apply_rewrite_truncation(
        self,
        question: str,
        old_pool: List[Dict],
        new_retrieved: List[Dict],
        meta_result: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        cap = max(1, int(self.retrieve_k))
        old_unique = self._unique_by_id(old_pool)
        old_ids = {self._entry_id(item) for item in old_unique}

        new_unique = []
        seen_new_ids = set()
        for item in new_retrieved:
            item_id = self._entry_id(item)
            if item_id in old_ids or item_id in seen_new_ids:
                continue
            new_unique.append(item)
            seen_new_ids.add(item_id)

        merged_before_cap = self._unique_by_id(old_unique + new_unique)
        before_cap = len(merged_before_cap)

        if self.rewrite_rerank_strategy == "no_cap":
            return merged_before_cap, {
                "before_cap": before_cap,
                "after_cap": before_cap,
                "new_unique": len(new_unique),
                "new_kept_after_cap": len(new_unique),
                "pruned": 0,
                "strategy": self.rewrite_rerank_strategy,
                "cap_selector_source": "non_cap",
                "cap_selector_parse_ok": True,
                "cap_selector_reason": "",
                "cap_selector_llm_returned_n": 0,
                "cap_selector_validated_n": 0,
                "cap_selector_invalid_n": 0,
                "cap_selector_dropped_by_cap_n": 0,
            }

        if before_cap <= cap:
            return merged_before_cap, {
                "before_cap": before_cap,
                "after_cap": before_cap,
                "new_unique": len(new_unique),
                "new_kept_after_cap": len(new_unique),
                "pruned": 0,
                "strategy": self.rewrite_rerank_strategy,
                "cap_selector_source": "non_cap",
                "cap_selector_parse_ok": True,
                "cap_selector_reason": "",
                "cap_selector_llm_returned_n": 0,
                "cap_selector_validated_n": 0,
                "cap_selector_invalid_n": 0,
                "cap_selector_dropped_by_cap_n": 0,
            }

        selector_diag = {
            "cap_selector_source": "non_cap",
            "cap_selector_parse_ok": True,
            "cap_selector_reason": "",
            "cap_selector_llm_returned_n": 0,
            "cap_selector_validated_n": 0,
            "cap_selector_invalid_n": 0,
            "cap_selector_dropped_by_cap_n": 0,
        }
        if self.rewrite_rerank_strategy == "keep_new_prune_old":
            kept_new = new_unique[:cap]
            remain = cap - len(kept_new)
            kept_old = []
            if remain > 0:
                kept_old = self._rerank_by_original_question(question, old_unique, remain)

            capped_pool = self._unique_by_id(kept_new + kept_old)
            if len(capped_pool) < cap:
                # Backfill from old pool by stable order if rerank returned fewer rows.
                kept_ids = {self._entry_id(item) for item in capped_pool}
                for item in old_unique:
                    item_id = self._entry_id(item)
                    if item_id in kept_ids:
                        continue
                    capped_pool.append(item)
                    kept_ids.add(item_id)
                    if len(capped_pool) >= cap:
                        break
        elif self.rewrite_rerank_strategy == "original_topk":
            capped_pool = self._rerank_by_original_question(question, merged_before_cap, cap)
        else:  # llm_select
            llm_selected, selector_diag = self._llm_select_cap(
                question=question,
                meta_result=meta_result,
                old_unique=old_unique,
                new_unique=new_unique,
                cap=cap,
            )
            if llm_selected is not None:
                capped_pool = llm_selected
            else:
                capped_pool = self._rerank_by_original_question(question, merged_before_cap, cap)
                selector_diag["cap_selector_source"] = "fallback_original_topk"

        new_ids = {self._entry_id(item) for item in new_unique}
        kept_new_after_cap = sum(
            1 for item in capped_pool if self._entry_id(item) in new_ids
        )
        after_cap = len(capped_pool)
        pruned = max(0, before_cap - after_cap)

        return capped_pool, {
            "before_cap": before_cap,
            "after_cap": after_cap,
            "new_unique": len(new_unique),
            "new_kept_after_cap": kept_new_after_cap,
            "pruned": pruned,
            "strategy": self.rewrite_rerank_strategy,
            "cap_selector_source": selector_diag["cap_selector_source"],
            "cap_selector_parse_ok": selector_diag["cap_selector_parse_ok"],
            "cap_selector_reason": selector_diag["cap_selector_reason"],
            "cap_selector_llm_returned_n": selector_diag["cap_selector_llm_returned_n"],
            "cap_selector_validated_n": selector_diag["cap_selector_validated_n"],
            "cap_selector_invalid_n": selector_diag["cap_selector_invalid_n"],
            "cap_selector_dropped_by_cap_n": selector_diag["cap_selector_dropped_by_cap_n"],
        }
    
    def answer_question(
        self,
        question: str,
        amem_layer: AMEMLayer,
        category: int = 1,
    ) -> Dict:
        """Answer question using A-Mem retrieval + Meta-Thinker answerability check."""
        result = {
            "question": question,
            "category": category,
            "retrieval_turns": [],
            "answer": "",
            "abstained": False,
            "cap_strategy": self.rewrite_rerank_strategy,
            "cap_applied": False,
            "num_memories_pre_cap": 0,
            "num_memories_post_cap": 0,
            "num_pruned_by_cap": 0,
            "num_new_kept_after_cap": 0,
            "cap_selector_source": "non_cap",
            "cap_selector_parse_ok": True,
            "cap_selector_reason": "",
            "cap_selector_llm_returned_n": 0,
            "cap_selector_validated_n": 0,
            "cap_selector_invalid_n": 0,
            "cap_selector_dropped_by_cap_n": 0,
        }
        
        if amem_layer is None:
            self.logger.error("A-Mem layer not available")
            result["answer"] = "Error: A-Mem layer not available"
            return result
        
        if self.retrieval_mode == "per-speaker":
            limit_per_speaker = self.retrieve_k // 2
            all_retrieved = self._thread_safe_retrieve_by_speaker(amem_layer, question, limit_per_speaker)
            self.logger.info(f"Initial per-speaker retrieval: {len(all_retrieved)} memories")
        else:
            all_retrieved = self._thread_safe_retrieve(amem_layer, question, limit=self.retrieve_k)
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
        total_pruned_by_cap = 0
        total_new_kept_after_cap = 0
        last_memories_pre_cap = len(all_retrieved)
        selector_source = "non_cap"
        selector_parse_ok = True
        selector_reason = ""
        selector_llm_returned_n = 0
        selector_validated_n = 0
        selector_invalid_n = 0
        selector_dropped_by_cap_n = 0
        
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
                
                new_retrieved = self._thread_safe_retrieve(amem_layer, orthogonal_query, limit=self.retrieve_k)

                old_pool = all_retrieved
                all_retrieved, cap_stats = self._apply_rewrite_truncation(
                    question=question,
                    old_pool=old_pool,
                    new_retrieved=new_retrieved,
                    meta_result=meta_result,
                )

                last_memories_pre_cap = cap_stats["before_cap"]
                total_pruned_by_cap += cap_stats["pruned"]
                total_new_kept_after_cap += cap_stats["new_kept_after_cap"]
                if cap_stats["pruned"] > 0:
                    result["cap_applied"] = True
                if cap_stats.get("cap_selector_source", "non_cap") != "non_cap":
                    selector_source = cap_stats.get("cap_selector_source", "non_cap")
                if not cap_stats.get("cap_selector_parse_ok", True):
                    selector_parse_ok = False
                if cap_stats.get("cap_selector_reason"):
                    selector_reason = cap_stats["cap_selector_reason"]
                if cap_stats.get("cap_selector_llm_returned_n", 0):
                    selector_llm_returned_n = cap_stats.get("cap_selector_llm_returned_n", 0)
                    selector_validated_n = cap_stats.get("cap_selector_validated_n", 0)
                    selector_invalid_n = cap_stats.get("cap_selector_invalid_n", 0)
                    selector_dropped_by_cap_n = cap_stats.get("cap_selector_dropped_by_cap_n", 0)

                self.logger.info(
                    "[Turn %d] Retrieved %d, %d new | before_cap=%d after_cap=%d pruned=%d strategy=%s selector=%s parse_ok=%s llm_returned=%d validated=%d invalid=%d dropped_by_cap=%d",
                    turn,
                    len(new_retrieved),
                    cap_stats["new_unique"],
                    cap_stats["before_cap"],
                    cap_stats["after_cap"],
                    cap_stats["pruned"],
                    cap_stats["strategy"],
                    cap_stats.get("cap_selector_source", "non_cap"),
                    cap_stats.get("cap_selector_parse_ok", True),
                    cap_stats.get("cap_selector_llm_returned_n", 0),
                    cap_stats.get("cap_selector_validated_n", 0),
                    cap_stats.get("cap_selector_invalid_n", 0),
                    cap_stats.get("cap_selector_dropped_by_cap_n", 0),
                )
                turn_info["num_memories_before_cap"] = cap_stats["before_cap"]
                turn_info["num_memories_after_cap"] = cap_stats["after_cap"]
                turn_info["new_added_before_cap"] = cap_stats["new_unique"]
                turn_info["new_kept_after_cap"] = cap_stats["new_kept_after_cap"]
                turn_info["pruned_this_turn"] = cap_stats["pruned"]
                turn_info["cap_strategy"] = cap_stats["strategy"]
                turn_info["cap_selector_source"] = cap_stats.get("cap_selector_source", "non_cap")
                turn_info["cap_selector_parse_ok"] = cap_stats.get("cap_selector_parse_ok", True)
                turn_info["cap_selector_reason"] = cap_stats.get("cap_selector_reason", "")
                turn_info["cap_selector_llm_returned_n"] = cap_stats.get("cap_selector_llm_returned_n", 0)
                turn_info["cap_selector_validated_n"] = cap_stats.get("cap_selector_validated_n", 0)
                turn_info["cap_selector_invalid_n"] = cap_stats.get("cap_selector_invalid_n", 0)
                turn_info["cap_selector_dropped_by_cap_n"] = cap_stats.get("cap_selector_dropped_by_cap_n", 0)
                
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
        result["num_memories_pre_cap"] = last_memories_pre_cap
        result["num_memories_post_cap"] = len(all_retrieved)
        result["num_pruned_by_cap"] = total_pruned_by_cap
        result["num_new_kept_after_cap"] = total_new_kept_after_cap
        result["cap_selector_source"] = selector_source
        result["cap_selector_parse_ok"] = selector_parse_ok
        result["cap_selector_reason"] = selector_reason
        result["cap_selector_llm_returned_n"] = selector_llm_returned_n
        result["cap_selector_validated_n"] = selector_validated_n
        result["cap_selector_invalid_n"] = selector_invalid_n
        result["cap_selector_dropped_by_cap_n"] = selector_dropped_by_cap_n
        result["num_memories_used"] = len(all_retrieved)
        
        return result
    
    def _process_single_qa(
        self,
        qa: Dict,
        amem_layer: AMEMLayer,
    ) -> Dict:
        """Process a single QA pair (thread-safe for parallel execution)."""
        question = qa['question']
        reference = qa.get('answer', '')
        category = qa.get('category', 1)

        self.logger.info(f"\nQ: {question}")

        qa_result = self.answer_question(
            question=question,
            amem_layer=amem_layer,
            category=category,
        )
        qa_result["reference"] = reference

        try:
            if LIGHTMEM_LLM_JUDGE_AVAILABLE and lightmem_llm_judge is not None:
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
        
        amem_layer = self._get_amem_layer(sample_id)
        if amem_layer is None:
            self.logger.error(f"A-Mem layer not available for {sample_id}")
            return {"sample_id": sample_id, "error": "No A-Mem layer", "results": []}
        
        num_entries = amem_entry_count(amem_layer)
        self.logger.info(f"Loaded {num_entries} A-Mem entries for sample {sample_id}")
        
        if num_entries <= 0:
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
                    executor.submit(self._process_single_qa, qa, amem_layer): idx
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
                qa_results.append(self._process_single_qa(qa, amem_layer))
        
        return {
            "sample_id": sample_id,
            "num_entries": num_entries,
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
            "amem_dir": self.amem_dir,
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
            
            amem_layer = self._get_amem_layer(sample_id)
            if amem_layer is None:
                continue
            
            if amem_entry_count(amem_layer) <= 0:
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
                
                qa_result = self.answer_question(
                    question=question,
                    amem_layer=amem_layer,
                    category=category,
                )
                
                try:
                    if LIGHTMEM_LLM_JUDGE_AVAILABLE and lightmem_llm_judge is not None:
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
    amem_dir: str,
    api_key: str,
    llm_model: str = "gpt-4o-mini",
    api_base_url: str = None,
    llm_backend: str = "openai",
    embedder_provider: str = "openai",
    retriever_model: str = "text-embedding-3-small",
    evo_threshold: int = 100,
    log_dir: str = "./logs",
    enable_meta_guidance: bool = False,
    llm_client: LLMClient = None,
    self_refine_source: str = "none",
    session_qa_index: Optional[Dict[Tuple[str, int], List[Dict[str, Any]]]] = None,
    self_refine_runtime: Optional[Dict[str, Any]] = None,
    self_refine_max_questions: int = 0,
    self_refine_batch_size: int = 4,
    self_refine_fail_on_empty: bool = True,
    self_refine_inference_mode: str = "simple",
    self_refine_apply_threshold: float = 0.88,
    self_refine_add_fact_prompt_version: str = DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION,
    self_refine_skip_empty_entries: bool = True,
    self_refine_log_jsonl: str = "",
    max_sessions: int = 0,
    logger: logging.Logger = None,
    **_extra,
) -> Dict:
    """Build memories for a single sample using A-Mem with optional Meta-Thinker guidance."""
    sample_id = sample['sample_id']
    logger = logger or logging.getLogger(__name__)
    
    if (enable_meta_guidance or self_refine_source != "none") and llm_client is None:
        llm_client = LLMClient(model=llm_model, api_key=api_key, base_url=api_base_url)
    
    try:
        logger.info(f"{'='*60}")
        logger.info(f"Building memories for: {sample_id}")
        logger.info(f"{'='*60}")
        
        conversation = sample['conversation']
        sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(conversation)
        
        logger.info(f"  Sessions: {len(sessions)}")
        logger.info(f"  Speakers: {speaker_a}, {speaker_b}")
        logger.info(
            f"  Construction Meta Guidance: {'ENABLED' if enable_meta_guidance else 'DISABLED'}"
        )
        
        amem_config = get_amem_config(
            user_id=sample_id,
            amem_dir=amem_dir,
            llm_model=llm_model,
            llm_backend=llm_backend,
            embedder_provider=embedder_provider,
            retriever_model=retriever_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
            base_url=api_base_url,
        )
        amem_layer = AMEMLayer(amem_config)
        
        start_time = time.time()
        self_refine_reports: List[Dict[str, Any]] = []
        pending_self_refine_sessions: List[Dict[str, Any]] = []
        pending_self_refine_qa_est = 0
        self_refine_no_new_extraction_deferred_sessions = 0
        self_refine_no_new_extraction_deferred_qa = 0
        self_refine_live_empty_deferred_attempts = 0
        self_refine_live_empty_deferred_qa = 0
        self_refine_live_empty_deferred_session_ids: Set[int] = set()
        self_refine_replayed_sessions = 0
        self_refine_replayed_qa = 0
        self_refine_executed_session_ids: Set[int] = set()

        def _run_single_self_refine_job(job: Dict[str, Any]) -> Dict[str, Any]:
            runtime_view = _bind_self_refine_runtime(
                runtime_template=self_refine_runtime,
                amem_layer=amem_layer,
                sample_id=sample_id,
            )
            return run_session_self_refinement(
                sample_id=sample_id,
                session_idx=int(job["session_idx"]),
                session_messages=job["session_messages"],
                session_timestamp=str(job["session_timestamp"]),
                amem_layer=amem_layer,
                runtime=runtime_view,
                llm_client=llm_client,
                self_refine_source=self_refine_source,
                session_qa_index=session_qa_index,
                self_refine_max_questions=self_refine_max_questions,
                self_refine_batch_size=self_refine_batch_size,
                self_refine_fail_on_empty=self_refine_fail_on_empty,
                speaker_a_name=speaker_a,
                speaker_b_name=speaker_b,
                self_refine_inference_mode=self_refine_inference_mode,
                self_refine_apply_threshold=self_refine_apply_threshold,
                self_refine_add_fact_prompt_version=self_refine_add_fact_prompt_version,
                self_refine_skip_empty_entries=self_refine_skip_empty_entries,
                self_refine_log_jsonl=self_refine_log_jsonl,
                logger=logger,
            )
        
        session_pairs = list(zip(sessions, timestamps))
        if max_sessions and max_sessions > 0:
            session_pairs = session_pairs[:max_sessions]
            logger.info(f"  Session limit applied: {len(session_pairs)}/{len(sessions)}")

        for session_idx, (session, timestamp) in enumerate(session_pairs):
            while session and session[0]["role"] != "user":
                session.pop(0)
            
            num_turns = len(session) // 2
            logger.info(f"\n  Session {session_idx + 1}: {num_turns} turns")
            
            entries_before_session = amem_entry_count(amem_layer)

            for turn_idx in range(num_turns):
                turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
                if len(turn_messages) < 2:
                    continue
                if turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                    continue
                
                for msg in turn_messages:
                    speaker_name = msg.get("speaker_name", msg.get("role", "user"))
                    content = str(msg.get("content", "")).strip()
                    if not content:
                        continue

                    add_message_kwargs: Dict[str, Any] = {"timestamp": timestamp}
                    if enable_meta_guidance and llm_client:
                        try:
                            construction_metadata = _build_construction_metadata(
                                llm_client=llm_client,
                                timestamp=timestamp,
                                speaker_name=speaker_name,
                                content=content,
                            )
                            if construction_metadata is not None:
                                add_message_kwargs.update(construction_metadata)
                                logger.debug(
                                    "Construction metadata for %s: %s",
                                    speaker_name,
                                    json.dumps(construction_metadata, ensure_ascii=False),
                                )
                            else:
                                logger.debug(
                                    "Construction metadata missing or invalid for %s; "
                                    "falling back to default A-Mem analysis.",
                                    speaker_name,
                                )
                        except Exception as exc:
                            logger.warning(f"Meta-Thinker construction guidance failed: {exc}")

                    amem_layer.add_message(
                        {"role": speaker_name, "content": content},
                        **add_message_kwargs,
                    )

            if self_refine_source != "none":
                if self_refine_runtime is None:
                    raise ValueError(
                        "self_refine_runtime is required when self_refine_source is enabled"
                    )
                qa_estimate = _estimate_self_refine_question_count(
                    self_refine_source=self_refine_source,
                    sample_id=sample_id,
                    session_idx=session_idx,
                    session_qa_index=session_qa_index,
                    self_refine_max_questions=self_refine_max_questions,
                )
                pending_self_refine_sessions.append(
                    {
                        "session_idx": session_idx,
                        "session_messages": session,
                        "session_timestamp": timestamp,
                        "qa_estimate": qa_estimate,
                    }
                )
                pending_self_refine_qa_est += qa_estimate

                entries_now = amem_entry_count(amem_layer)
                if entries_now <= entries_before_session:
                    self_refine_no_new_extraction_deferred_sessions += 1
                    self_refine_no_new_extraction_deferred_qa += qa_estimate
                    logger.info(
                        "[SelfRefine] deferred (no new extraction): sample=%s session=%d "
                        "entries=%d session_qa_est=%d pending_sessions=%d pending_qa_est=%d",
                        sample_id,
                        session_idx,
                        entries_now,
                        qa_estimate,
                        len(pending_self_refine_sessions),
                        pending_self_refine_qa_est,
                    )
                    continue

                logger.info(
                    "[SelfRefine] replaying pending sessions: sample=%s entries=%d pending_sessions=%d "
                    "pending_qa_est=%d",
                    sample_id,
                    entries_now,
                    len(pending_self_refine_sessions),
                    pending_self_refine_qa_est,
                )
                next_pending_sessions: List[Dict[str, Any]] = []
                next_pending_qa_est = 0
                for pending_job in pending_self_refine_sessions:
                    pending_session_idx = int(pending_job.get("session_idx", -1))
                    report = _run_single_self_refine_job(pending_job)
                    if bool(report.get("deferred_due_to_empty_entries")):
                        self_refine_live_empty_deferred_attempts += 1
                        self_refine_live_empty_deferred_qa += int(report.get("qa_total", 0))
                        if pending_session_idx >= 0:
                            self_refine_live_empty_deferred_session_ids.add(pending_session_idx)
                        next_pending_sessions.append(pending_job)
                        next_pending_qa_est += int(pending_job.get("qa_estimate", 0))
                        continue
                    if pending_session_idx >= 0:
                        self_refine_executed_session_ids.add(pending_session_idx)
                    if pending_session_idx >= 0 and pending_session_idx < session_idx:
                        self_refine_replayed_sessions += 1
                        self_refine_replayed_qa += int(report.get("qa_total", 0))
                    self_refine_reports.append(report)
                pending_self_refine_sessions = next_pending_sessions
                pending_self_refine_qa_est = next_pending_qa_est
                
        if self_refine_source != "none" and pending_self_refine_sessions:
            entries_now = amem_entry_count(amem_layer)
            logger.info(
                "[SelfRefine] final pending replay attempt: sample=%s entries=%d pending_sessions=%d "
                "pending_qa_est=%d",
                sample_id,
                entries_now,
                len(pending_self_refine_sessions),
                pending_self_refine_qa_est,
            )
            if entries_now > 0:
                next_pending_sessions: List[Dict[str, Any]] = []
                next_pending_qa_est = 0
                for pending_job in pending_self_refine_sessions:
                    pending_session_idx = int(pending_job.get("session_idx", -1))
                    report = _run_single_self_refine_job(pending_job)
                    if bool(report.get("deferred_due_to_empty_entries")):
                        self_refine_live_empty_deferred_attempts += 1
                        self_refine_live_empty_deferred_qa += int(report.get("qa_total", 0))
                        if pending_session_idx >= 0:
                            self_refine_live_empty_deferred_session_ids.add(pending_session_idx)
                        next_pending_sessions.append(pending_job)
                        next_pending_qa_est += int(pending_job.get("qa_estimate", 0))
                        continue
                    if pending_session_idx >= 0:
                        self_refine_executed_session_ids.add(pending_session_idx)
                    self_refine_replayed_sessions += 1
                    self_refine_replayed_qa += int(report.get("qa_total", 0))
                    self_refine_reports.append(report)
                pending_self_refine_sessions = next_pending_sessions
                pending_self_refine_qa_est = next_pending_qa_est

            if pending_self_refine_sessions:
                unresolved_session_ids = sorted(
                    int(job.get("session_idx", -1))
                    for job in pending_self_refine_sessions
                    if int(job.get("session_idx", -1)) >= 0
                )
                logger.warning(
                    "[SelfRefine] unresolved deferred sessions after final replay: sample=%s "
                    "pending_sessions=%d pending_qa_est=%d session_ids=%s",
                    sample_id,
                    len(pending_self_refine_sessions),
                    pending_self_refine_qa_est,
                    unresolved_session_ids,
                )

        add_memory_time = time.time() - start_time
        after_add_count = amem_entry_count(amem_layer)
        logger.info(f"\n  Add memory completed: {after_add_count} entries in {add_memory_time:.2f}s")
        
        logger.info(f"\n{'─'*60}")
        logger.info("Consolidating and saving A-Mem memories")
        logger.info(f"{'─'*60}")
        
        update_start_time = time.time()
        amem_layer.consolidate_memories()
        amem_layer.save_memory()
        update_time = time.time() - update_start_time
        
        post_update_count = amem_entry_count(amem_layer)
        logger.info(f"  Consolidate+save completed: {post_update_count} entries in {update_time:.2f}s")
        
        total_time = time.time() - start_time
        self_refine_summary: Dict[str, Any] = {}
        if self_refine_source != "none":
            unresolved_session_ids = sorted(
                int(job.get("session_idx", -1))
                for job in pending_self_refine_sessions
                if int(job.get("session_idx", -1)) >= 0
            )
            total_refine_qa = sum(int(r.get("qa_total", 0)) for r in self_refine_reports)
            total_actions_applied = sum(int(r.get("actions_applied", 0)) for r in self_refine_reports)
            total_skipped_batches = sum(int(r.get("skipped_batches", 0)) for r in self_refine_reports)
            total_filtered_low_confidence = sum(
                int(r.get("filtered_low_confidence_count", 0)) for r in self_refine_reports
            )
            total_speaker_ambiguous_noop = sum(
                int(r.get("speaker_ambiguous_noop_count", 0)) for r in self_refine_reports
            )
            total_invalid_evidence_noop = sum(
                int(r.get("invalid_evidence_span_noop_count", 0)) for r in self_refine_reports
            )
            total_noop_count = sum(int(r.get("noop_count", 0)) for r in self_refine_reports)
            executed_reports_count = len(self_refine_reports)
            self_refine_summary = {
                "deferred_no_new_extraction_sessions": self_refine_no_new_extraction_deferred_sessions,
                "deferred_no_new_extraction_qa_est": self_refine_no_new_extraction_deferred_qa,
                "deferred_live_empty_attempts": self_refine_live_empty_deferred_attempts,
                "deferred_live_empty_unique_sessions": len(self_refine_live_empty_deferred_session_ids),
                "deferred_live_empty_session_ids": sorted(self_refine_live_empty_deferred_session_ids),
                "deferred_live_empty_qa_total": self_refine_live_empty_deferred_qa,
                "deferred_due_to_empty_entries_sessions": len(self_refine_live_empty_deferred_session_ids),
                "deferred_due_to_empty_entries_qa_est": self_refine_live_empty_deferred_qa,
                "replayed_deferred_sessions": self_refine_replayed_sessions,
                "replayed_deferred_qa": self_refine_replayed_qa,
                "unresolved_deferred_sessions": len(pending_self_refine_sessions),
                "unresolved_deferred_qa_est": pending_self_refine_qa_est,
                "unresolved_deferred_session_ids": unresolved_session_ids,
                "reports_count": executed_reports_count,
                "executed_reports_count": executed_reports_count,
                "executed_session_count": len(self_refine_executed_session_ids),
                "qa_total": total_refine_qa,
                "actions_applied_total": total_actions_applied,
                "noop_count_total": total_noop_count,
                "skipped_batches_total": total_skipped_batches,
                "filtered_low_confidence_total": total_filtered_low_confidence,
                "speaker_ambiguous_noop_total": total_speaker_ambiguous_noop,
                "invalid_evidence_span_noop_total": total_invalid_evidence_noop,
            }
            logger.info(
                "  Self-Refine deferred(no new extraction): sessions=%d qa_est=%d",
                self_refine_no_new_extraction_deferred_sessions,
                self_refine_no_new_extraction_deferred_qa,
            )
            logger.info(
                "  Self-Refine deferred(live empty): attempts=%d unique_sessions=%d qa=%d",
                self_refine_live_empty_deferred_attempts,
                len(self_refine_live_empty_deferred_session_ids),
                self_refine_live_empty_deferred_qa,
            )
            logger.info(
                "  Self-Refine replayed deferred: sessions=%d qa=%d unresolved_sessions=%d unresolved_qa_est=%d",
                self_refine_replayed_sessions,
                self_refine_replayed_qa,
                len(pending_self_refine_sessions),
                pending_self_refine_qa_est,
            )
            logger.info(
                "  Self-Refine executed totals: reports=%d qa=%d actions=%d noop=%d filtered_low_conf=%d "
                "speaker_amb_noop=%d invalid_span_noop=%d skipped_batches=%d",
                executed_reports_count,
                total_refine_qa,
                total_actions_applied,
                total_noop_count,
                total_filtered_low_confidence,
                total_speaker_ambiguous_noop,
                total_invalid_evidence_noop,
                total_skipped_batches,
            )
        
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
            'self_refine_reports': self_refine_reports,
            'self_refine_summary': self_refine_summary,
        }
        
    except Exception as e:
        logger.error(f"  {sample_id} failed: {str(e)}", exc_info=True)
        return {
            'sample_id': sample_id,
            'status': 'failed',
            'error': str(e)
        }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="A-Mem + Meta-Thinker Integration (MeMMA Self-Refine)")
    
    parser.add_argument("--build_memories", action="store_true",
                       help="Build memories from scratch using A-Mem (instead of evaluating)")
    
    parser.add_argument("--dataset", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--amem-dir", type=str, required=True, help="Path to A-Mem storage directory")
    parser.add_argument("--output_dir", type=str, default="results/memma_amem", help="Output directory")
    
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI base URL")
    
    parser.add_argument("--llm_backend", type=str, default="openai",
                       choices=["openai", "ollama", "anthropic"],
                       help="LLM backend for A-Mem memory construction")
    parser.add_argument("--embedder_provider", type=str, default="openai",
                       choices=["sentence-transformers", "openai"],
                       help="Embedding provider for A-Mem")
    parser.add_argument("--retriever_model", type=str, default="text-embedding-3-small",
                       help="Retriever model name or path for A-Mem")
    parser.add_argument("--evo_threshold", type=int, default=100,
                       help="A-Mem evolution threshold")
    
    # Retrieval settings (same as LightMem default)
    parser.add_argument("--retrieve_k", type=int, default=60, help="Number of memories to retrieve")
    parser.add_argument("--qr_max_turns", type=int, default=3, help="Maximum Meta-Thinker turns")
    parser.add_argument("--retrieval_mode", type=str, default="combined", 
                       choices=["per-speaker", "combined"],
                       help="Retrieval mode: 'per-speaker' (balanced per speaker) or 'combined' (global similarity)")
    parser.add_argument(
        "--rewrite_rerank_strategy",
        type=str,
        default="keep_new_prune_old",
        choices=["keep_new_prune_old", "original_topk", "llm_select", "no_cap"],
        help=(
            "Rewrite truncation strategy after each orthogonal retrieval: "
            "'keep_new_prune_old' keeps newly retrieved memories and prunes old pool first; "
            "'original_topk' reranks merged pool by original question and keeps top-K; "
            "'llm_select' uses LLM evidence selection under the same top-K cap; "
            "'no_cap' keeps all unique old+new memories without truncation."
        ),
    )
    parser.add_argument(
        "--answerability_prompt_version",
        type=str,
        default="v4",
        choices=["v1", "v2", "v3", "v4", "v5"],
        help="Meta-Thinker answerability prompt version to use for evaluation.",
    )
    
    # Meta-Thinker toggle
    parser.add_argument("--disable_meta_thinker", action="store_true", 
                       help="Disable QA/eval Meta-Thinker (run pure A-Mem baseline for evaluation)")
    parser.add_argument("--enable_construction_meta_guidance", action="store_true",
                       help="Enable construction-time Meta-Thinker guidance during A-Mem memory building")
    
    # Parallel processing
    parser.add_argument("--qa_max_workers", type=int, default=1,
                       help="Max parallel threads for QA processing (1=sequential)")
    parser.add_argument("--build_max_workers", type=int, default=1,
                       help="Max parallel processes for memory building (1=sequential)")
    
    # Dataset settings
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=0,
        help="Maximum sessions to process per sample in build mode (0 means all sessions)",
    )
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of samples to evaluate")
    parser.add_argument("--categories", type=str, default="1,2,3,4", help="Comma-separated categories")
    
    parser.add_argument("--embedding_model", type=str, default=None,
                       help="(Legacy, unused by A-Mem) Path to embedding model")

    # Build-time self-refinement settings
    parser.add_argument(
        "--self_refine_source",
        type=str,
        default="none",
        choices=["none", "parquet", "realtime"],
        help="Self refinement QA source for --build_memories: none|parquet|realtime",
    )
    parser.add_argument(
        "--self_refine_parquet",
        type=str,
        default="",
        help="Parquet path used when --self_refine_source parquet",
    )
    parser.add_argument(
        "--self_refine_max_questions",
        type=int,
        default=0,
        help="Max questions per session for self refinement (0 means all available)",
    )
    parser.add_argument(
        "--self_refine_batch_size",
        type=int,
        default=4,
        help="Batch size for parallel self-refinement QA evaluation",
    )
    parser.add_argument(
        "--self_refine_fail_on_empty",
        type=str_to_bool,
        default=True,
        help="Whether empty/missing session QA should fail immediately (true/false)",
    )
    parser.add_argument(
        "--self_refine_log_jsonl",
        type=str,
        default="",
        help="Optional JSONL path for self-refinement session reports",
    )
    parser.add_argument(
        "--self_refine_inference_mode",
        type=str,
        default="simple",
        choices=["simple", "full_memma"],
        help="Inference mode for self-refine QA answering: simple|full_memma",
    )
    parser.add_argument(
        "--self_refine_apply_threshold",
        type=float,
        default=0.0,
        help="Minimum confidence required to apply an ADD_FACT action",
    )
    parser.add_argument(
        "--self_refine_add_fact_prompt_version",
        type=str,
        default=DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION,
        choices=["v1", "v2", "v3"],
        help="Prompt version used by self-refine add-fact proposer",
    )
    parser.add_argument(
        "--self_refine_skip_empty_entries",
        type=str_to_bool,
        default=True,
        help="Skip self-refine batch when entries are empty (true/false)",
    )
    
    # Reflection mode settings
    parser.add_argument("--meta_thinker_mode", type=str, default="vanilla",
                       choices=["none", "vanilla", "reflection"],
                       help="Meta-Thinker mode: 'none' (disabled), 'vanilla' (enabled w/o learning), 'reflection' (with procedural learning)")
    parser.add_argument("--reflection_test_samples", type=int, default=1,
                       help="Number of samples to use for reflection training (remaining used for test)")
    parser.add_argument("--procedural_memory_path", type=str, default=None,
                       help="Path to save/load procedural memory JSON")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, "memma_amem")
    if SPACY_SHIM_ACTIVE:
        logger.warning(
            "spacy shim is active: LightMem retrievers were imported without a real spacy installation."
        )
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    answerability_prompt = ANSWERABILITY_PROMPT_BY_VERSION.get(
        args.answerability_prompt_version,
        META_THINKER_ANSWERABILITY_PROMPT_v4,
    )
    
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
        logger.info("MEMORY BUILDING MODE (A-Mem)")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model}")
        logger.info(f"A-Mem Dir: {args.amem_dir}")
        logger.info(f"LLM Backend: {args.llm_backend}")
        logger.info(f"Embedder Provider: {args.embedder_provider}")
        logger.info(f"Retriever Model: {args.retriever_model}")
        logger.info(f"Evo Threshold: {args.evo_threshold}")
        logger.info(f"QA Meta-Thinker: {'ENABLED' if not args.disable_meta_thinker else 'DISABLED'}")
        logger.info(
            f"Construction Meta Guidance: {'ENABLED' if args.enable_construction_meta_guidance else 'DISABLED'}"
        )
        logger.info(f"Self-Refine Source: {args.self_refine_source}")
        logger.info(f"Self-Refine Inference Mode: {args.self_refine_inference_mode}")
        logger.info(f"Self-Refine Apply Threshold: {args.self_refine_apply_threshold}")
        logger.info(f"Self-Refine Add-Fact Prompt Version: {args.self_refine_add_fact_prompt_version}")
        
        if not HAS_AMEM:
            logger.error("A-Mem not available. Cannot build memories.")
            return

        llm_client = None
        if args.enable_construction_meta_guidance or args.self_refine_source != "none":
            llm_client = LLMClient(model=args.model, api_key=api_key, base_url=args.base_url)

        session_qa_index: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        if args.self_refine_source == "parquet":
            if not args.self_refine_parquet:
                raise ValueError("--self_refine_parquet is required when --self_refine_source parquet")
            session_qa_index, parquet_conv_ids = load_parquet_session_qa(
                parquet_path=args.self_refine_parquet,
                fail_on_empty=args.self_refine_fail_on_empty,
            )
            sample_ids = [str(s.get("sample_id", "")) for s in samples]
            missing_sample_ids = sorted([sid for sid in sample_ids if sid not in parquet_conv_ids])
            if missing_sample_ids:
                raise ValueError(
                    "sample_id must exactly match parquet conversation_id for self refinement. "
                    f"Missing IDs: {missing_sample_ids}"
                )
            logger.info(
                f"Loaded parquet self-refine QA: {len(session_qa_index)} session entries "
                f"from {args.self_refine_parquet}"
            )

        amem_config_kwargs = dict(
            llm_model=args.model,
            llm_backend=args.llm_backend,
            embedder_provider=args.embedder_provider,
            retriever_model=args.retriever_model,
            evo_threshold=args.evo_threshold,
            api_key=api_key,
            base_url=args.base_url,
        )

        self_refine_runtime = None
        if args.self_refine_source != "none":
            self_refine_runtime = init_self_refine_runtime(
                llm_client=llm_client,
                model_name=args.model,
                retrieve_k=args.retrieve_k,
                self_refine_inference_mode=args.self_refine_inference_mode,
                amem_dir=args.amem_dir,
                amem_config_kwargs=amem_config_kwargs,
                logger=logger,
            )

        effective_build_max_workers = args.build_max_workers
        if args.self_refine_source != "none" and effective_build_max_workers > 1:
            logger.warning(
                "Self-refine is enabled; forcing build_max_workers from %d to 1 to avoid "
                "runtime serialization/order issues.",
                effective_build_max_workers,
            )
            effective_build_max_workers = 1

        build_kwargs_common = dict(
            amem_dir=args.amem_dir,
            api_key=api_key,
            llm_model=args.model,
            api_base_url=args.base_url,
            llm_backend=args.llm_backend,
            embedder_provider=args.embedder_provider,
            retriever_model=args.retriever_model,
            evo_threshold=args.evo_threshold,
            log_dir=os.path.join(args.output_dir, 'logs'),
            enable_meta_guidance=args.enable_construction_meta_guidance,
            self_refine_source=args.self_refine_source,
            session_qa_index=session_qa_index,
            self_refine_runtime=self_refine_runtime,
            self_refine_max_questions=args.self_refine_max_questions,
            self_refine_batch_size=args.self_refine_batch_size,
            self_refine_fail_on_empty=args.self_refine_fail_on_empty,
            self_refine_inference_mode=args.self_refine_inference_mode,
            self_refine_apply_threshold=args.self_refine_apply_threshold,
            self_refine_add_fact_prompt_version=args.self_refine_add_fact_prompt_version,
            self_refine_skip_empty_entries=args.self_refine_skip_empty_entries,
            self_refine_log_jsonl=args.self_refine_log_jsonl,
            max_sessions=args.max_sessions,
        )
        
        results = []
        if effective_build_max_workers > 1 and len(samples) > 1:
            logger.info(f"Parallel build: {len(samples)} samples, {effective_build_max_workers} workers")
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=effective_build_max_workers) as executor:
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
            for sample in tqdm(samples, desc="Building memories"):
                should_pass_llm_client = (
                    args.enable_construction_meta_guidance
                    or (args.self_refine_source != "none")
                )
                result = build_memories_for_sample(
                    sample=sample,
                    llm_client=llm_client if should_pass_llm_client else None,
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
    logger.info("A-Mem + Meta-Thinker Integration")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"A-Mem Dir: {args.amem_dir}")
    logger.info(f"Retrieve K: {args.retrieve_k}")
    logger.info(f"Retrieval Mode: {args.retrieval_mode}")
    logger.info(f"Rewrite Rerank Strategy: {args.rewrite_rerank_strategy}")
    logger.info(f"Answerability Prompt Version: {args.answerability_prompt_version}")
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
    
    amem_config_kwargs = dict(
        llm_model=args.model,
        llm_backend=args.llm_backend,
        embedder_provider=args.embedder_provider,
        retriever_model=args.retriever_model,
        evo_threshold=args.evo_threshold,
        api_key=api_key,
        base_url=args.base_url,
    )

    evaluator = AMemMetaThinkerEvaluator(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        amem_dir=args.amem_dir,
        amem_config_kwargs=amem_config_kwargs,
        retrieve_k=args.retrieve_k,
        qr_max_turns=args.qr_max_turns,
        enable_meta_thinker=enable_meta_thinker,
        retrieval_mode=args.retrieval_mode,
        procedural_memories=procedural_memories,
        qa_max_workers=args.qa_max_workers,
        rewrite_rerank_strategy=args.rewrite_rerank_strategy,
        answerability_prompt=answerability_prompt,
        answerability_prompt_version=args.answerability_prompt_version,
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
    results["answerability_prompt_version"] = args.answerability_prompt_version
    
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
    logger.info(f"Answerability prompt version: {args.answerability_prompt_version}")
    
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
