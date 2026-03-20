#!/usr/bin/env python3
"""
Single-Agent MemoryManager + Meta-Thinker Self-Refine Script

Uses MemoryManagerAgent (LLM-decided ADD/UPDATE/DELETE) for memory construction,
with optional self-refinement and three QA evaluation modes.

Architecture:
- Memory Construction: MemoryManagerAgent (LLM decides ADD/UPDATE/DELETE per chunk)
- Memory Storage: In-memory MemoryBank with JSON persistence
- Self-Refinement: Parquet/realtime QA-driven fact injection
- QA Modes: Single (retrieve+answer), QR loop, or Meta-Thinker answerability

Usage:
    python run_memma_self_refine_single_0310.py \\
        --dataset /path/to/locomo10.json \\
        --memory-dir ./results/memory_bank/ \\
        --output_dir results/memma_single \\
        --build_memories
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

import uuid
from dataclasses import dataclass, field

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    SentenceTransformer = None

ANSWER_PROMPT = """You are an intelligent memory assistant.
Memories for user {speaker_1_name}:
{speaker_1_memories}

Memories for user {speaker_2_name}:
{speaker_2_memories}

Question: {question}

Answer:"""


# ==============================================================================
# Memory Bank (ported from run_vanilla_baseline.py)
# ==============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry in the memory bank."""
    id: str
    content: str
    timestamp: str
    source_turn_id: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    access_count: int = 0
    speaker: str = ""
    weekday: str = ""
    float_timestamp: float = 0.0
    category: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "source_turn_id": self.source_turn_id,
            "keywords": self.keywords,
            "speaker": self.speaker,
            "weekday": self.weekday,
            "float_timestamp": self.float_timestamp,
            "category": self.category,
            "access_count": self.access_count,
        }


class MemoryBank:
    """In-memory bank with embedding-based retrieval and JSON persistence."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", **kwargs):
        self.entries: Dict[str, MemoryEntry] = {}
        self.next_id = 0
        self.storage_type = "memory"

        if HAS_SBERT:
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            self.encoder = None
            self.embedding_dim = 384

        self.embeddings: Optional[np.ndarray] = None
        self.entry_ids: List[str] = []

    def _parse_timestamp(self, timestamp: str) -> Tuple[float, str]:
        from datetime import datetime as dt
        weekday = ""
        float_ts = 0.0
        if not timestamp:
            return float_ts, weekday
        match = re.search(
            r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*\(([^)]+)\)\s*(\d{1,2}:\d{2}(?::\d{2})?)",
            timestamp,
        )
        if match:
            date_str = match.group(1).replace("-", "/")
            weekday = match.group(2)
            time_str = match.group(3)
            fmt = "%Y/%m/%d %H:%M:%S" if time_str.count(":") == 2 else "%Y/%m/%d %H:%M"
            try:
                parsed_dt = dt.strptime(f"{date_str} {time_str}", fmt)
                float_ts = parsed_dt.timestamp()
            except ValueError:
                pass
        return float_ts, weekday

    def _generate_id(self) -> str:
        self.next_id += 1
        return f"mem_{self.next_id:04d}"

    def add(
        self,
        content: str,
        timestamp: str = "",
        source_turn_id: str = None,
        keywords: List[str] = None,
        speaker: str = "",
    ) -> str:
        entry_id = self._generate_id()
        float_ts, weekday = self._parse_timestamp(timestamp)
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            timestamp=timestamp,
            source_turn_id=source_turn_id,
            keywords=keywords or [],
            speaker=speaker,
            weekday=weekday,
            float_timestamp=float_ts,
        )
        if self.encoder:
            entry.embedding = self.encoder.encode([content])[0]
        self.entries[entry_id] = entry
        self._rebuild_index()
        return entry_id

    def update(self, entry_id: str, new_content: str) -> bool:
        if entry_id not in self.entries:
            return False
        self.entries[entry_id].content = new_content
        if self.encoder:
            self.entries[entry_id].embedding = self.encoder.encode([new_content])[0]
        self._rebuild_index()
        return True

    def delete(self, entry_id: str) -> bool:
        if entry_id not in self.entries:
            return False
        del self.entries[entry_id]
        self._rebuild_index()
        return True

    def _rebuild_index(self):
        if not self.entries or not self.encoder:
            self.embeddings = None
            self.entry_ids = []
            return
        self.entry_ids = list(self.entries.keys())
        embeddings_list = []
        for eid in self.entry_ids:
            entry = self.entries[eid]
            if entry.embedding is not None:
                embeddings_list.append(entry.embedding)
            else:
                embeddings_list.append(self.encoder.encode([entry.content])[0])
        self.embeddings = np.array(embeddings_list)

    def retrieve_by_time(self, k: int) -> List[MemoryEntry]:
        sorted_entries = sorted(self.entries.values(), key=lambda e: e.id, reverse=True)
        return sorted_entries[:k]

    def retrieve_by_similarity(self, query: str, k: int) -> List[MemoryEntry]:
        if not self.entries or self.embeddings is None or self.encoder is None:
            return list(self.entries.values())[:k]
        query_embedding = self.encoder.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_k_indices:
            if idx < len(self.entry_ids):
                entry_id = self.entry_ids[idx]
                results.append(self.entries[entry_id])
        return results

    def get_all(self) -> List[MemoryEntry]:
        return list(self.entries.values())

    def format_for_context(self, entries: List[MemoryEntry], max_entries: int = 20) -> str:
        lines = []
        for entry in entries[:max_entries]:
            ts = f"[{entry.timestamp}] " if entry.timestamp else ""
            lines.append(f"{ts}({entry.id}) {entry.content}")
        return "\n".join(lines)

    def size(self) -> int:
        return len(self.entries)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {"next_id": self.next_id, "storage_type": self.storage_type, "entries": []}
        for entry in self.entries.values():
            entry_dict = entry.to_dict()
            if entry.embedding is not None:
                entry_dict["embedding"] = entry.embedding.tolist()
            data["entries"].append(entry_dict)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str, embedding_model: str = "all-MiniLM-L6-v2") -> "MemoryBank":
        with open(filepath, "r") as f:
            data = json.load(f)
        bank = cls(embedding_model)
        bank.next_id = data.get("next_id", 0)
        for entry_dict in data.get("entries", []):
            embedding = None
            if "embedding" in entry_dict:
                embedding = np.array(entry_dict["embedding"])
            entry = MemoryEntry(
                id=entry_dict["id"],
                content=entry_dict["content"],
                timestamp=entry_dict.get("timestamp", ""),
                source_turn_id=entry_dict.get("source_turn_id"),
                keywords=entry_dict.get("keywords", []),
                embedding=embedding,
                access_count=entry_dict.get("access_count", 0),
                speaker=entry_dict.get("speaker", ""),
                weekday=entry_dict.get("weekday", ""),
                float_timestamp=entry_dict.get("float_timestamp", 0.0),
                category=entry_dict.get("category", ""),
            )
            bank.entries[entry.id] = entry
        bank._rebuild_index()
        return bank


def create_memory_bank(
    embedding_model: str = "all-MiniLM-L6-v2",
) -> MemoryBank:
    return MemoryBank(embedding_model=embedding_model)


# ==============================================================================
# Agent Prompts (MemoryManager + QueryRewriter from run_vanilla_baseline.py)
# ==============================================================================

MEMORY_MANAGER_SYSTEM_PROMPT = """You are a smart Memory Manager that controls the memory of a conversation system.

You can perform four operations:
1. **ADD** - Add new information to memory as a new element
2. **UPDATE** - Update an existing memory element with new/more detailed information
3. **DELETE** - Delete an existing memory element (contradictory information)
4. **NONE** - Make no change (information already present or irrelevant)

## Output Format
Return a JSON object with a "memory" array containing all memory elements with their events:
```json
{
  "memory": [
    {"id": "0", "text": "Existing memory", "event": "NONE"},
    {"id": "1", "text": "New important fact", "event": "ADD"}
  ]
}
```

Focus on storing facts that will help answer future questions about the conversation."""


QUERY_REWRITER_SYSTEM_PROMPT = """You are a Query Rewriter agent. Your task is to decide if retrieved memories are sufficient to answer a question, or if queries need to be rewritten for better retrieval.

Given:
- A question
- Currently retrieved memories
- Previous rewrite attempts (if any)

Decide:
- ANSWERABLE - The retrieved memories contain enough information to answer the question
- REWRITE - The memories are insufficient; provide better search queries

Output format:
<decision>ANSWERABLE|REWRITE</decision>
<content>If REWRITE: query1 ## query2 ## query3 (multiple queries separated by ##). If ANSWERABLE: leave empty</content>

Be strategic with rewrites - consider synonyms, related concepts, and temporal aspects."""


ANSWER_AGENT_SYSTEM_PROMPT = """You are an Answer Agent. Given a question and relevant memories, provide a short, accurate answer.

Focus on:
- Answering directly from the provided memories
- Being concise (short phrase or sentence)
- Using exact words from memories when possible
- Saying "Not mentioned in the conversation" if the answer cannot be found"""


# ==============================================================================
# MemoryManagerAgent (ported from run_vanilla_baseline.py)
# ==============================================================================

class MemoryManagerAgent:
    """Memory Manager agent that decides ADD/UPDATE/DELETE/NONE per chunk."""

    def __init__(self, llm, logger: logging.Logger):
        self.llm = llm
        self.logger = logger

    def query(
        self,
        chunk: str,
        memory_bank: MemoryBank,
        previous_actions: List[str],
        retrieval_mode: str = "similarity",
        retrieve_k: int = 10,
        meta_guidance: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if retrieval_mode == "time":
            recent_memories = memory_bank.retrieve_by_time(retrieve_k)
        else:
            recent_memories = memory_bank.retrieve_by_similarity(chunk, retrieve_k)

        memory_json = self._format_memory_as_json(recent_memories)

        meta_section = ""
        if meta_guidance:
            meta_section = (
                f"\n## Extracted Facts (ensure each is stored — ADD if missing, UPDATE if partial):\n"
                f"{meta_guidance}\n"
            )

        prompt = f"""Current Memory Bank:
{memory_json}

New Conversation Chunk:
{chunk}
{meta_section}
Analyze the new conversation chunk and compare it with the existing memory.
For each piece of new information, decide whether to ADD, UPDATE, DELETE, or mark as NONE.
When extracted facts are listed above, make sure every fact has a corresponding ADD or UPDATE entry.

Return your response as a JSON object with a "memory" array."""

        response = self.llm.get_completion(
            prompt, temperature=0.3, max_tokens=1024,
            system_prompt=MEMORY_MANAGER_SYSTEM_PROMPT,
        )
        operations = self._parse_json_response(response, memory_bank)
        num_ops = sum(1 for op in operations if op.get("event", "NONE") != "NONE")
        self.logger.info(f"MM returned {len(operations)} operations ({num_ops} active)")
        return operations

    def _format_memory_as_json(self, memories: List[MemoryEntry]) -> str:
        if not memories:
            return "[]"
        memory_list = []
        for mem in memories:
            ts_prefix = f"[{mem.timestamp}] " if mem.timestamp else ""
            memory_list.append({"id": mem.id, "timestamp": mem.timestamp or "", "text": f"{ts_prefix}{mem.content}"})
        return json.dumps(memory_list, indent=2)

    def _parse_json_response(self, response: str, memory_bank: MemoryBank) -> List[Dict[str, Any]]:
        operations: List[Dict[str, Any]] = []
        try:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{[^{}]*"memory"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
                json_str = json_match.group(0) if json_match else response.strip()
            data = json.loads(json_str)
            if isinstance(data, dict) and "memory" in data:
                for item in data["memory"]:
                    if isinstance(item, dict) and "event" in item:
                        operations.append({
                            "id": item.get("id", ""),
                            "text": item.get("text", ""),
                            "event": item.get("event", "NONE").upper(),
                            "old_memory": item.get("old_memory", ""),
                        })
        except (json.JSONDecodeError, TypeError, AttributeError):
            operations = self._fallback_parse(response)
        return operations

    def _fallback_parse(self, response: str) -> List[Dict[str, Any]]:
        operations: List[Dict[str, Any]] = []
        response_upper = response.upper()
        if 'EVENT": "ADD' in response_upper or '"ADD"' in response:
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
            if text_match:
                operations.append({"id": "", "text": text_match.group(1), "event": "ADD", "old_memory": ""})
        elif 'EVENT": "UPDATE' in response_upper:
            id_match = re.search(r'"id"\s*:\s*"([^"]+)"', response)
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
            old_match = re.search(r'"old_memory"\s*:\s*"([^"]+)"', response)
            if text_match:
                operations.append({
                    "id": id_match.group(1) if id_match else "",
                    "text": text_match.group(1),
                    "event": "UPDATE",
                    "old_memory": old_match.group(1) if old_match else "",
                })
        elif 'EVENT": "DELETE' in response_upper:
            id_match = re.search(r'"id"\s*:\s*"([^"]+)"', response)
            if id_match:
                operations.append({"id": id_match.group(1), "text": "", "event": "DELETE", "old_memory": ""})
        else:
            operations.append({"id": "", "text": "", "event": "NONE", "old_memory": ""})
        return operations


# ==============================================================================
# QueryRewriterAgent (ported from run_vanilla_baseline.py)
# ==============================================================================

class QueryRewriterAgent:
    """Query Rewriter that decides ANSWERABLE/REWRITE."""

    def __init__(self, llm, logger: logging.Logger):
        self.llm = llm
        self.logger = logger

    def query(
        self,
        question: str,
        retrieved_memories: List[MemoryEntry],
        previous_rewrites: List[str],
        meta_guidance: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        memories_str = (
            "\n".join([f"({m.id}) [{m.timestamp}] {m.content}" for m in retrieved_memories])
            if retrieved_memories
            else "[No memories retrieved]"
        )
        prev_rewrites_str = (
            "\n".join([f"- {rw}" for rw in previous_rewrites])
            if previous_rewrites
            else "[No previous queries]"
        )
        meta_section = ""
        if meta_guidance:
            meta_section = f"\n## Meta-Thinker Guidance:\n{meta_guidance}\n"

        prompt = f"""Question: {question}

Retrieved Memories:
{memories_str}

Previous Rewrite Attempts:
{prev_rewrites_str}
{meta_section}
Are these memories sufficient to answer the question, or should we search with different queries?

Output format:
<decision>ANSWERABLE|REWRITE</decision>
<content>If REWRITE: query1 ## query2 (separated by ##). If ANSWERABLE: leave empty</content>"""

        response = self.llm.get_completion(
            prompt, temperature=0.3, max_tokens=200,
            system_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
        )
        decision, queries = self._parse_response(response)
        self.logger.info(f"QR Decision: {decision}, Queries: {queries}")
        return decision, queries

    def _parse_response(self, response: str) -> Tuple[str, List[str]]:
        decision_match = re.search(r"<decision>\s*(ANSWERABLE|REWRITE)\s*</decision>", response, re.IGNORECASE)
        content_match = re.search(r"<content>(.*?)</content>", response, re.DOTALL | re.IGNORECASE)
        decision = decision_match.group(1).upper() if decision_match else "ANSWERABLE"
        content = content_match.group(1).strip() if content_match else ""
        queries: List[str] = []
        if content and decision == "REWRITE":
            queries = [q.strip() for q in content.split("##") if q.strip()]
        if not decision_match:
            if "REWRITE" in response.upper():
                decision = "REWRITE"
                if "##" in response:
                    queries = [p.strip() for p in response.split("##") if p.strip() and len(p.strip()) > 5]
            else:
                decision = "ANSWERABLE"
        return decision, queries


# ==============================================================================
# LLM Judge and Metrics (ported from run_vanilla_baseline.py)
# ==============================================================================

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
    (1) a question, (2) a 'gold' answer, (3) a generated answer.

The gold answer is usually concise. The generated answer might be longer, but be generous -
as long as it touches on the same topic, count it as CORRECT.

For time-related questions, be generous with format differences (e.g., "May 7th" vs "7 May").

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Provide a short explanation, then finish with CORRECT or WRONG.
Return the label in JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}
"""


def evaluate_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    client=None,
    model_name: str = "gpt-4o-mini",
) -> int:
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer,
                ),
            }],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        json_str = match.group(1) if match else text
        result = json.loads(json_str)
        label = result.get("label", "WRONG").upper()
        return 1 if label == "CORRECT" else 0
    except Exception:
        return 0


def calculate_simple_metrics(prediction: str, reference: str) -> Dict[str, float]:
    pred_lower = str(prediction).lower().strip() if prediction else ""
    ref_lower = str(reference).lower().strip() if reference else ""
    exact_match = 1.0 if pred_lower == ref_lower else 0.0
    containment = 1.0 if (ref_lower in pred_lower or pred_lower in ref_lower) else 0.0
    pred_tokens = set(pred_lower.split())
    ref_tokens = set(ref_lower.split())
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"exact_match": exact_match, "containment": containment, "token_f1": f1}

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
# LoCoMo Utilities
# ==============================================================================

DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


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




# ==============================================================================
# Meta-Thinker Construction Guidance Prompt
# ==============================================================================

META_THINKER_CONSTRUCTION_PROMPT = """You are a quality-control checker for a memory construction system.

Given one conversation utterance, list every distinct factual statement it contains.
Each fact must be an atomic, self-contained statement that could answer a WHO/WHAT/WHEN/WHERE/HOW MANY question.

Rules:
1. Extract EVERY fact — do not skip anything. Err on the side of over-extraction.
2. Use the speaker's exact words for names, objects, dates, places, and quantities.
   Good: "cup with a dog face"   Bad: "creative pottery"
   Good: "guinea pig named Oscar"   Bad: "pets that bring comfort"
   Good: "August 27, 2023"   Bad: "recently"
3. One fact per line. Do NOT merge multiple facts into one line.
4. Prefix each fact with the correct speaker name.
5. Do NOT interpret emotions, themes, values, or symbolism.
6. Do NOT paraphrase — preserve the original phrasing.

Output format:
FACTS:
- [Speaker] fact 1
- [Speaker] fact 2
- ...
"""


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


def _is_bedrock_model(model: str) -> bool:
    return "." in model and "anthropic" in model


class _BedrockFakeMessage:
    def __init__(self, content: str):
        self.content = content


class _BedrockFakeChoice:
    def __init__(self, content: str):
        self.message = _BedrockFakeMessage(content)


class _BedrockFakeResponse:
    def __init__(self, content: str):
        self.choices = [_BedrockFakeChoice(content)]


class BedrockOpenAICompat:
    """Adapter wrapping a boto3 Bedrock client to expose .chat.completions.create()."""

    def __init__(self, bedrock_client, model_id: str):
        self._bedrock = bedrock_client
        self._model_id = model_id
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent):
            self.completions = parent._Completions(parent)

    class _Completions:
        def __init__(self, parent):
            self._parent = parent

        def create(self, model=None, messages=None, temperature=0.0, max_tokens=1024, **kwargs):
            system_blocks = []
            converse_msgs = []
            for message in messages or []:
                if message["role"] == "system":
                    system_blocks.append({"text": message["content"]})
                else:
                    converse_msgs.append(
                        {"role": message["role"], "content": [{"text": message["content"]}]}
                    )

            params = {
                "modelId": model or self._parent._model_id,
                "messages": converse_msgs,
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
            }
            if system_blocks:
                params["system"] = system_blocks

            response = self._parent._bedrock.converse(**params)
            content_blocks = response.get("output", {}).get("message", {}).get("content", [])
            text = "".join(block["text"] for block in content_blocks if "text" in block)
            return _BedrockFakeResponse(text)


class BedrockLLMClient:
    """Bedrock Converse API LLM client with the same interface as LLMClient."""

    def __init__(self, model_id: str, region: str = "us-west-2"):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for Bedrock Claude models. "
                "Install boto3 in the current environment and configure AWS credentials."
            ) from exc

        self.model = model_id
        self._region = region
        self._bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.client = BedrockOpenAICompat(self._bedrock, model_id)

    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str = None,
    ) -> str:
        system_blocks = []
        if system_prompt:
            system_blocks.append({"text": system_prompt})

        params = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system_blocks:
            params["system"] = system_blocks

        response = self._bedrock.converse(**params)
        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(block["text"] for block in content_blocks if "text" in block)


def _create_llm_client(
    model: str,
    region: str = "us-west-2",
    api_key: str = None,
    base_url: str = None,
):
    """Create the appropriate LLM client based on model name."""
    if _is_bedrock_model(model):
        return BedrockLLMClient(model_id=model, region=region)
    return LLMClient(model=model, api_key=api_key, base_url=base_url)


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
    logger: logging.Logger,
) -> Dict[str, Any]:
    requested_mode = str(self_refine_inference_mode or "simple")
    effective_mode = requested_mode
    runtime: Dict[str, Any] = {
        "llm_client": llm_client,
        "model_name": model_name,
        "retrieve_k": retrieve_k,
        "inference_mode_requested": requested_mode,
        "inference_mode_effective": effective_mode,
        "full_memma_evaluator": None,
    }

    if requested_mode == "full_memma":
        logger.info("full_memma self-refine mode is not supported for single-agent; falling back to simple.")
        effective_mode = "simple"

    runtime["inference_mode_effective"] = effective_mode
    return runtime


def _format_memories_by_speaker_for_refine(retrieved_entries) -> tuple:
    """Format memory entries (MemoryEntry or dict) grouped by speaker for the ANSWER_PROMPT."""
    speaker_groups: Dict[str, List[str]] = {}
    for entry in retrieved_entries:
        if isinstance(entry, MemoryEntry):
            speaker_name = entry.speaker or "Unknown"
            ts = f"[{entry.timestamp}] " if entry.timestamp else ""
            text = f"{ts}{entry.content}"
        else:
            payload = entry.get("payload", {}) if isinstance(entry, dict) else {}
            speaker_name = payload.get("speaker_name", "Unknown")
            text = str(payload.get("memory") or payload.get("content") or "")
        speaker_groups.setdefault(speaker_name, []).append(text)

    speaker_names = list(speaker_groups.keys())
    if len(speaker_names) == 0:
        return "Speaker 1", "No memories available.", "Speaker 2", "No memories available."

    def _join(items):
        return "\n".join(items) if items else "No memories available."

    if len(speaker_names) == 1:
        return speaker_names[0], _join(speaker_groups[speaker_names[0]]), "Speaker 2", "No memories available."
    s1, s2 = speaker_names[0], speaker_names[1]
    return s1, _join(speaker_groups[s1]), s2, _join(speaker_groups[s2])


def _extract_evidence_snippet(entry, max_len: int = 220) -> str:
    if isinstance(entry, MemoryEntry):
        text = entry.content or ""
    else:
        payload = entry.get("payload", {}) if isinstance(entry, dict) else {}
        text = (
            payload.get("memory") or payload.get("text")
            or payload.get("content") or payload.get("summary") or ""
        )
    snippet = re.sub(r"\s+", " ", str(text)).strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet


def _answer_question_for_self_refine(
    question: str,
    memory_bank: MemoryBank,
    llm_client: LLMClient,
    retrieve_k: int,
    inference_mode: str = "simple",
    full_memma_evaluator: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    retrieved = memory_bank.retrieve_by_similarity(question, k=retrieve_k)

    evidence = []
    for entry in retrieved:
        evidence.append({
            "id": entry.id,
            "speaker": entry.speaker or "Unknown",
            "snippet": _extract_evidence_snippet(entry),
        })

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
        "inference_mode_used": "simple",
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
        score = evaluate_llm_judge(
            question=question,
            gold_answer=reference,
            generated_answer=generated_answer,
            client=llm_client.client,
            model_name=model_name,
        )
        return float(score)
    except Exception as exc:
        logger.warning(f"Self-refine judge failed, fallback to exact match: {exc}")
    return float(_normalize_question_key(reference) == _normalize_question_key(generated_answer))


def _evaluate_questions_parallel(
    questions: List[Dict[str, Any]],
    memory_bank: MemoryBank,
    runtime: Dict[str, Any],
    batch_size: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    llm_client: LLMClient = runtime["llm_client"]
    model_name = str(runtime.get("model_name", llm_client.model))
    retrieve_k = int(runtime.get("retrieve_k", 30))

    indexed_questions = list(enumerate(questions))
    workers = max(1, min(len(indexed_questions), batch_size))

    def worker(idx: int, qa: Dict[str, Any]) -> Dict[str, Any]:
        question = str(qa.get("question", ""))
        reference = str(qa.get("answer", ""))
        answer_payload = _answer_question_for_self_refine(
            question=question,
            memory_bank=memory_bank,
            llm_client=llm_client,
            retrieve_k=retrieve_k,
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
    memory_bank: MemoryBank,
    fact: str,
    llm_client,
    similarity_threshold: float = 0.8,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Check if a proposed fact duplicates or complements an existing MemoryBank entry.

    Returns (action, merged_fact_or_none, target_entry_id_or_none)
    where action is "SKIP", "MERGE", or "INSERT".
    """
    logger = logger or logging.getLogger(__name__)

    hits = memory_bank.retrieve_by_similarity(fact, k=3)
    if not hits:
        return ("INSERT", None, None)

    if memory_bank.encoder is not None:
        fact_vec = memory_bank.encoder.encode([fact])[0]
        scored_hits = []
        for h in hits:
            if h.embedding is not None:
                score = float(np.dot(fact_vec, h.embedding))
            else:
                score = 0.0
            if score >= similarity_threshold:
                scored_hits.append((h, score))
    else:
        scored_hits = [(h, 1.0) for h in hits]

    if not scored_hits:
        return ("INSERT", None, None)

    candidate_lines = []
    for idx, (h, score) in enumerate(scored_hits):
        candidate_lines.append(f"[{idx}] (score={score:.3f}) {h.content}")

    candidates_text = "\n".join(candidate_lines)
    prompt = f"""New proposed fact:
{fact}

Existing memory entries:
{candidates_text}

Decide: SKIP, MERGE, or INSERT."""

    try:
        response = llm_client.get_completion(
            prompt, temperature=0.0, max_tokens=300,
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
        if not merged_fact or merge_idx < 0 or merge_idx >= len(scored_hits):
            logger.warning(
                f"[SemanticDedup] MERGE response invalid (idx={merge_idx}, "
                f"fact_len={len(merged_fact)}), falling back to INSERT"
            )
            return ("INSERT", None, None)
        target_id = scored_hits[merge_idx][0].id
        if not target_id:
            logger.warning("[SemanticDedup] MERGE target has no id, falling back to INSERT")
            return ("INSERT", None, None)
        logger.info(f"[SemanticDedup] MERGE into {target_id}: {reason}")
        return ("MERGE", merged_fact, target_id)

    return ("INSERT", None, None)


def _apply_refine_actions(
    memory_bank: MemoryBank,
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
                memory_bank=memory_bank,
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
                ok = memory_bank.update(target_id, merged_fact)
                if ok:
                    dedup_stats["merge"] += 1
                    applied_action = dict(action)
                    applied_action["dedup_action"] = "MERGE"
                    applied_action["merge_target_id"] = target_id
                    applied_action["merged_fact"] = merged_fact
                    applied.append(applied_action)
                    continue
                else:
                    logger.warning(f"[SemanticDedup] MERGE update failed for {target_id}, falling back to INSERT")

        dedup_stats["insert"] += 1
        memory_bank.add(
            content=fact,
            timestamp=iso_ts,
            speaker=speaker_name,
        )

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
    memory_bank: MemoryBank,
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
    if memory_bank.size() == 0:
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
                "deferred_question_count": len(questions),
                "batches": [
                    {
                        "batch_index": 0,
                        "batch_size": len(questions),
                        "skipped_reason": "empty_entries_deferred",
                    }
                ],
            }
            logger.warning(
                f"[SelfRefine] sample={sample_id} session={session_idx} "
                f"deferred: empty memory_bank qa={len(questions)}"
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

        if memory_bank.size() == 0:
            if self_refine_skip_empty_entries:
                skipped_batches += 1
                logger.warning(
                    f"[SelfRefine] sample={sample_id} session={session_idx} "
                    f"batch={batch_start // batch_size} skipped: empty memory_bank"
                )
                batch_reports.append(
                    {
                        "batch_index": batch_start // batch_size,
                        "batch_size": len(batch_questions),
                        "skipped_reason": "empty_entries",
                    }
                )
                continue
            raise ValueError(f"No entries found for sample={sample_id} during self refinement")

        before_results = _evaluate_questions_parallel(
            questions=batch_questions,
            memory_bank=memory_bank,
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
            memory_bank=memory_bank,
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
            memory_bank=memory_bank,
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
        "deferred_question_count": 0,
        "batches": batch_reports,
    }
    logger.info(
        f"[SelfRefine] sample={sample_id} session={session_idx} "
        f"qa={len(questions)} correct={total_before}->{total_after} actions={total_actions} "
        f"dedup(skip={total_dedup_skip},merge={total_dedup_merge},insert={total_dedup_insert})"
    )
    _append_jsonl_record(self_refine_log_jsonl, report)
    return report
# Meta-Thinker Agent
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
# Single-Agent Evaluator (replaces LightMemMetaThinkerEvaluator)
# ==============================================================================

class SingleAgentEvaluator:
    """Evaluator using MemoryBank + optional Meta-Thinker / QueryRewriter.

    Three QA modes:
      - Single (qr_max_turns=0, meta-thinker off): retrieve + answer
      - QR loop (qr_max_turns>0, meta-thinker off): QueryRewriterAgent loop + answer
      - Meta-Thinker (qr_max_turns>0, meta-thinker on): answerability check + orthogonal queries
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        region: str = "us-west-2",
        api_key: str = None,
        base_url: str = None,
        memory_dir: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        retrieve_k: int = 60,
        qr_max_turns: int = 3,
        enable_meta_thinker: bool = True,
        rewrite_rerank_strategy: str = "no_cap",
        procedural_memories: Dict[str, str] = None,
        qa_max_workers: int = 1,
        answerability_prompt: str = None,
        answerability_prompt_version: str = "v4",
        meta_thinker_llm=None,
        logger: logging.Logger = None,
    ):
        self.llm = _create_llm_client(
            model=model, region=region, api_key=api_key, base_url=base_url
        )
        self.meta_thinker_llm = meta_thinker_llm if meta_thinker_llm is not None else self.llm
        self.memory_dir = memory_dir
        self.embedding_model = embedding_model
        self.retrieve_k = retrieve_k
        self.qr_max_turns = qr_max_turns
        self.enable_meta_thinker = enable_meta_thinker
        self.rewrite_rerank_strategy = rewrite_rerank_strategy
        self.qa_max_workers = qa_max_workers
        self.logger = logger or logging.getLogger(__name__)
        self.answerability_prompt_version = answerability_prompt_version
        self.procedural_memories = procedural_memories or {}

        if answerability_prompt is None:
            answerability_prompt = ANSWERABILITY_PROMPT_BY_VERSION.get(
                answerability_prompt_version, META_THINKER_ANSWERABILITY_PROMPT_v4
            )
        self.answerability_prompt = answerability_prompt

        self.meta_thinker = MetaThinkerAgent(
            self.meta_thinker_llm, self.logger,
            procedural_memories=self.procedural_memories,
            answerability_prompt=self.answerability_prompt,
        ) if enable_meta_thinker else None

        self.query_rewriter = QueryRewriterAgent(self.llm, self.logger) if not enable_meta_thinker and qr_max_turns > 0 else None

        self.logger.info(
            "SingleAgentEvaluator: model=%s retrieve_k=%d qr_max_turns=%d meta_thinker=%s rerank=%s",
            model, retrieve_k, qr_max_turns, enable_meta_thinker, rewrite_rerank_strategy,
        )
        self.logger.info(
            "Answer LLM: %s (model=%s)",
            type(self.llm).__name__,
            self.llm.model,
        )
        self.logger.info(
            "Meta-thinker LLM: %s (model=%s)",
            type(self.meta_thinker_llm).__name__,
            self.meta_thinker_llm.model,
        )

    # ------------------------------------------------------------------
    def _load_memory_bank(self, sample_id: str) -> Optional[MemoryBank]:
        if not self.memory_dir:
            return None
        path = os.path.join(self.memory_dir, sample_id, "memory_bank.json")
        if not os.path.exists(path):
            self.logger.warning(f"No memory_bank.json for {sample_id} at {path}")
            return None
        return MemoryBank.load(path, embedding_model=self.embedding_model)

    # ------------------------------------------------------------------
    @staticmethod
    def _format_memories_by_speaker(entries: List[MemoryEntry]) -> tuple:
        speaker_groups: Dict[str, List[str]] = {}
        for e in entries:
            name = e.speaker or "Unknown"
            ts = f"[{e.timestamp}] " if e.timestamp else ""
            speaker_groups.setdefault(name, []).append(f"{ts}{e.content}")
        names = list(speaker_groups.keys())
        def _join(items):
            return "\n".join(items) if items else "No memories available."
        if len(names) == 0:
            return "Speaker 1", "No memories available.", "Speaker 2", "No memories available."
        if len(names) == 1:
            return names[0], _join(speaker_groups[names[0]]), "Speaker 2", "No memories available."
        return names[0], _join(speaker_groups[names[0]]), names[1], _join(speaker_groups[names[1]])

    # ------------------------------------------------------------------
    def _merge_unique(self, old: List[MemoryEntry], new: List[MemoryEntry]) -> List[MemoryEntry]:
        seen = {e.id for e in old}
        merged = list(old)
        for e in new:
            if e.id not in seen:
                merged.append(e)
                seen.add(e.id)
        return merged

    # ------------------------------------------------------------------
    def answer_question_single(self, question: str, memory_bank: MemoryBank, category: int = 1) -> Dict:
        """Single mode: retrieve + answer directly."""
        retrieved = memory_bank.retrieve_by_similarity(question, k=self.retrieve_k)
        s1, s1m, s2, s2m = self._format_memories_by_speaker(retrieved)
        prompt = ANSWER_PROMPT.format(
            speaker_1_name=s1, speaker_1_memories=s1m,
            speaker_2_name=s2, speaker_2_memories=s2m,
            question=question,
        )
        answer = self.llm.get_completion(prompt, temperature=0.0).strip()
        return {"question": question, "category": category, "answer": answer,
                "num_memories_used": len(retrieved), "retrieval_turns": [], "mode": "single"}

    # ------------------------------------------------------------------
    def answer_question_with_rewriter(self, question: str, memory_bank: MemoryBank, category: int = 1) -> Dict:
        """QR-loop mode: QueryRewriterAgent decides ANSWERABLE/REWRITE."""
        all_retrieved = memory_bank.retrieve_by_similarity(question, k=self.retrieve_k)
        previous_rewrites: List[str] = []
        rewrite_turns: List[Dict] = []

        turn = 0
        while turn < self.qr_max_turns:
            decision, queries = self.query_rewriter.query(
                question=question,
                retrieved_memories=all_retrieved,
                previous_rewrites=previous_rewrites,
            )
            rewrite_turns.append({"turn": turn, "decision": decision, "queries": queries, "num_memories": len(all_retrieved)})
            if decision == "ANSWERABLE" or not queries:
                break
            for q in queries:
                previous_rewrites.append(q)
                new_memories = memory_bank.retrieve_by_similarity(q, k=self.retrieve_k)
                all_retrieved = self._merge_unique(all_retrieved, new_memories)
            turn += 1

        s1, s1m, s2, s2m = self._format_memories_by_speaker(all_retrieved[:self.retrieve_k])
        prompt = ANSWER_PROMPT.format(
            speaker_1_name=s1, speaker_1_memories=s1m,
            speaker_2_name=s2, speaker_2_memories=s2m,
            question=question,
        )
        answer = self.llm.get_completion(prompt, temperature=0.0).strip()
        return {"question": question, "category": category, "answer": answer,
                "num_memories_used": len(all_retrieved), "retrieval_turns": rewrite_turns, "mode": "qr_loop"}

    # ------------------------------------------------------------------
    def answer_question_with_meta_thinker(self, question: str, memory_bank: MemoryBank, category: int = 1) -> Dict:
        """Meta-Thinker mode: answerability check + orthogonal queries."""
        all_retrieved = memory_bank.retrieve_by_similarity(question, k=self.retrieve_k)
        previous_queries = [question]
        retrieval_turns: List[Dict] = []

        s1, s1m, s2, s2m = self._format_memories_by_speaker(all_retrieved)
        combined = f"[{s1}]\n{s1m}\n\n[{s2}]\n{s2m}"

        turn = 0
        while turn < self.qr_max_turns:
            meta_result = self.meta_thinker.check_answerability(
                question=question,
                retrieved_memories_str=combined,
                previous_queries=previous_queries,
            )
            decision = meta_result["decision"]
            turn_info = {
                "turn": turn, "decision": decision,
                "reason": meta_result.get("reason", ""),
                "missing_speaker": meta_result.get("missing_speaker", ""),
                "num_memories": len(all_retrieved),
            }
            retrieval_turns.append(turn_info)
            self.logger.info(f"[Turn {turn}] Meta-Thinker: {decision}")

            if decision == "ANSWERABLE":
                break

            retrieved_ids = [e.id for e in all_retrieved]
            orthogonal_query = self.meta_thinker.generate_orthogonal_query(
                question=question,
                key_gaps=meta_result.get("key_gaps", ""),
                retrieval_guidance=meta_result.get("retrieval_guidance", ""),
                previous_queries=previous_queries,
                retrieved_memory_ids=retrieved_ids,
                missing_speaker=meta_result.get("missing_speaker", ""),
                time_need=meta_result.get("time_need", ""),
                speaker_1_name=s1,
                speaker_2_name=s2,
            )
            if not orthogonal_query:
                self.logger.info(f"[Turn {turn}] No orthogonal query generated, stopping")
                break

            previous_queries.append(orthogonal_query)
            self.logger.info(f"[Turn {turn}] Orthogonal Query: {orthogonal_query}")

            new_retrieved = memory_bank.retrieve_by_similarity(orthogonal_query, k=self.retrieve_k)
            all_retrieved = self._merge_unique(all_retrieved, new_retrieved)

            if self.rewrite_rerank_strategy != "no_cap" and len(all_retrieved) > self.retrieve_k:
                all_retrieved = memory_bank.retrieve_by_similarity(question, k=self.retrieve_k)

            s1, s1m, s2, s2m = self._format_memories_by_speaker(all_retrieved)
            combined = f"[{s1}]\n{s1m}\n\n[{s2}]\n{s2m}"
            turn += 1

        s1, s1m, s2, s2m = self._format_memories_by_speaker(all_retrieved)
        prompt = ANSWER_PROMPT.format(
            speaker_1_name=s1, speaker_1_memories=s1m,
            speaker_2_name=s2, speaker_2_memories=s2m,
            question=question,
        )
        answer = self.llm.get_completion(prompt, temperature=0.0).strip()
        return {"question": question, "category": category, "answer": answer,
                "num_memories_used": len(all_retrieved), "retrieval_turns": retrieval_turns, "mode": "meta_thinker"}

    # ------------------------------------------------------------------
    def answer_question(self, question: str, memory_bank: MemoryBank, category: int = 1) -> Dict:
        """Dispatch to the appropriate QA mode."""
        if self.enable_meta_thinker and self.qr_max_turns > 0:
            return self.answer_question_with_meta_thinker(question, memory_bank, category)
        elif self.qr_max_turns > 0 and self.query_rewriter is not None:
            return self.answer_question_with_rewriter(question, memory_bank, category)
        else:
            return self.answer_question_single(question, memory_bank, category)

    # ------------------------------------------------------------------
    def _process_single_qa(self, qa: Dict, memory_bank: MemoryBank) -> Dict:
        question = qa["question"]
        reference = qa.get("answer", "")
        category = qa.get("category", 1)
        self.logger.info(f"\nQ: {question}")

        qa_result = self.answer_question(question, memory_bank, category)
        qa_result["reference"] = reference

        try:
            accuracy = evaluate_llm_judge(
                question, reference, qa_result["answer"],
                client=self.llm.client, model_name=self.llm.model,
            )
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

    # ------------------------------------------------------------------
    def process_sample(self, sample: Dict, allow_categories: List[int] = None) -> Dict:
        if allow_categories is None:
            allow_categories = [1, 2, 3, 4, 5]
        sample_id = sample["sample_id"]
        self.logger.info(f"\n{'='*60}\nProcessing sample: {sample_id}\n{'='*60}")

        memory_bank = self._load_memory_bank(sample_id)
        if memory_bank is None:
            self.logger.error(f"No MemoryBank for {sample_id}")
            return {"sample_id": sample_id, "error": "No MemoryBank", "results": []}

        self.logger.info(f"Loaded MemoryBank with {memory_bank.size()} entries for {sample_id}")

        qa_list = [qa for qa in sample.get("qa", []) if qa.get("category", 1) in allow_categories]

        if self.qa_max_workers > 1 and len(qa_list) > 1:
            qa_results = [None] * len(qa_list)
            with ThreadPoolExecutor(max_workers=self.qa_max_workers) as executor:
                futs = {executor.submit(self._process_single_qa, qa, memory_bank): i for i, qa in enumerate(qa_list)}
                for future in as_completed(futs):
                    idx = futs[future]
                    try:
                        qa_results[idx] = future.result()
                    except Exception as e:
                        qa_results[idx] = {"question": qa_list[idx]["question"], "answer": f"Error: {e}",
                                           "accuracy": 0.0, "token_f1": 0.0, "bleu1": 0.0}
        else:
            qa_results = [self._process_single_qa(qa, memory_bank) for qa in qa_list]

        return {"sample_id": sample_id, "num_entries": memory_bank.size(), "results": qa_results}

    # ------------------------------------------------------------------
    def evaluate_dataset(self, samples: List[Dict], allow_categories: List[int] = None, max_samples: int = None) -> Dict:
        if allow_categories is None:
            allow_categories = [1, 2, 3, 4, 5]
        if max_samples:
            samples = samples[:max_samples]

        all_results = []
        for sample in tqdm(samples, desc="Evaluating"):
            all_results.append(self.process_sample(sample, allow_categories))

        all_qa = []
        for r in all_results:
            all_qa.extend(r.get("results", []))

        cat_metrics = defaultdict(lambda: {"accuracy": [], "token_f1": [], "bleu1": []})
        for qa in all_qa:
            cat = qa.get("category", 0)
            cat_metrics[cat]["accuracy"].append(qa.get("accuracy", 0))
            cat_metrics[cat]["token_f1"].append(qa.get("token_f1", 0))
            cat_metrics[cat]["bleu1"].append(qa.get("bleu1", 0))

        aggregate = {}
        for cat, m in cat_metrics.items():
            aggregate[f"category_{cat}"] = {
                "accuracy_mean": float(np.mean(m["accuracy"])) if m["accuracy"] else 0,
                "token_f1_mean": float(np.mean(m["token_f1"])) if m["token_f1"] else 0,
                "bleu1_mean": float(np.mean(m["bleu1"])) if m["bleu1"] else 0,
                "count": len(m["accuracy"]),
            }

        all_accs = [qa.get("accuracy", 0) for qa in all_qa]
        all_f1s = [qa.get("token_f1", 0) for qa in all_qa]
        all_b1s = [qa.get("bleu1", 0) for qa in all_qa]
        aggregate["overall"] = {
            "accuracy_mean": float(np.mean(all_accs)) if all_accs else 0,
            "token_f1_mean": float(np.mean(all_f1s)) if all_f1s else 0,
            "bleu1_mean": float(np.mean(all_b1s)) if all_b1s else 0,
            "count": len(all_accs),
        }

        return {
            "model": self.llm.model,
            "memory_dir": self.memory_dir,
            "retrieve_k": self.retrieve_k,
            "qr_max_turns": self.qr_max_turns,
            "meta_thinker_enabled": self.enable_meta_thinker,
            "total_samples": len(all_results),
            "total_questions": len(all_qa),
            "aggregate_metrics": aggregate,
            "sample_results": all_results,
        }


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
    memory_dir: str,
    api_key: str,
    llm_model: str = "gpt-4o-mini",
    api_base_url: str = None,
    aws_region: str = "us-west-2",
    embedding_model: str = "all-MiniLM-L6-v2",
    log_dir: str = "./logs",
    enable_meta_guidance: bool = False,
    llm_client: LLMClient = None,
    mm_max_turns: int = 1,
    mm_retrieval_mode: str = "similarity",
    mm_retrieve_k: int = 10,
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
) -> Dict:
    """Build memories using MemoryManagerAgent + MemoryBank with optional self-refinement."""
    sample_id = sample["sample_id"]
    logger = logger or logging.getLogger(__name__)

    if llm_client is None:
        llm_client = _create_llm_client(
            model=llm_model,
            region=aws_region,
            api_key=api_key,
            base_url=api_base_url,
        )

    try:
        logger.info(f"{'='*60}")
        logger.info(f"Building memories for: {sample_id}")
        logger.info(f"{'='*60}")

        conversation = sample["conversation"]
        sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(conversation)
        logger.info(f"  Sessions: {len(sessions)}, Speakers: {speaker_a}, {speaker_b}")
        logger.info(f"  Construction Meta Guidance: {'ENABLED' if enable_meta_guidance else 'DISABLED'}")

        memory_bank = create_memory_bank(embedding_model=embedding_model)
        mm_agent = MemoryManagerAgent(llm_client, logger)

        start_time = time.time()
        self_refine_reports: List[Dict[str, Any]] = []

        def _apply_mm_operations(operations: List[Dict[str, Any]], ts: str, spk: str):
            for op in operations:
                event = op.get("event", "NONE").upper()
                text = op.get("text", "").strip()
                op_id = op.get("id", "")
                if event == "ADD" and text:
                    memory_bank.add(content=text, timestamp=ts, speaker=spk)
                elif event == "UPDATE" and text and op_id:
                    if not memory_bank.update(op_id, text):
                        memory_bank.add(content=text, timestamp=ts, speaker=spk)
                elif event == "DELETE" and op_id:
                    memory_bank.delete(op_id)

        def _run_single_self_refine_job(job: Dict[str, Any]) -> Dict[str, Any]:
            job_runtime = dict(self_refine_runtime or {})
            job_runtime["sample_id"] = sample_id
            job_runtime["session_index"] = int(job["session_idx"])
            return run_session_self_refinement(
                sample_id=sample_id,
                session_idx=int(job["session_idx"]),
                session_messages=job["session_messages"],
                session_timestamp=str(job["session_timestamp"]),
                memory_bank=memory_bank,
                runtime=job_runtime,
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

            entries_before = memory_bank.size()
            user_turns = [m for m in session if m.get("role") == "user" and m.get("content", "").strip()]
            logger.info(f"\n  Session {session_idx + 1}: {len(user_turns)} user turns")

            for turn in user_turns:
                speaker_name = turn.get("speaker_name", "Unknown")
                content = turn.get("content", "").strip()
                if not content:
                    continue

                chunk = f"[{timestamp}] {speaker_name}: {content}"
                guidance = None
                if enable_meta_guidance and llm_client:
                    guidance_prompt = f"""Current conversation chunk being processed:
[{timestamp}] {speaker_name}: {content}

Provide construction guidance for the memory manager."""
                    try:
                        guidance = llm_client.get_completion(
                            guidance_prompt,
                            temperature=0.0,
                            max_tokens=300,
                            system_prompt=META_THINKER_CONSTRUCTION_PROMPT,
                        )
                        logger.debug(f"Meta-Thinker Construction Guidance:\n{guidance}")
                    except Exception as exc:
                        guidance = None
                        logger.warning(f"Meta-Thinker construction guidance failed: {exc}")
                for _ in range(mm_max_turns):
                    operations = mm_agent.query(
                        chunk=chunk,
                        memory_bank=memory_bank,
                        previous_actions=[],
                        retrieval_mode=mm_retrieval_mode,
                        retrieve_k=mm_retrieve_k,
                        meta_guidance=guidance,
                    )
                    has_active = any(op.get("event", "NONE").upper() not in ("NONE", "") for op in operations)
                    _apply_mm_operations(operations, timestamp, speaker_name)
                    if not has_active:
                        break

            logger.info(f"    Entries after session {session_idx + 1}: {memory_bank.size()}")

            if self_refine_source != "none":
                if self_refine_runtime is None:
                    raise ValueError("self_refine_runtime required when self_refine_source is enabled")
                report = _run_single_self_refine_job({
                    "session_idx": session_idx,
                    "session_messages": session,
                    "session_timestamp": timestamp,
                })
                self_refine_reports.append(report)

        build_time = time.time() - start_time
        final_count = memory_bank.size()
        logger.info(f"\nMemory construction complete: {final_count} entries in {build_time:.2f}s")

        save_path = os.path.join(memory_dir, sample_id, "memory_bank.json")
        memory_bank.save(save_path)
        logger.info(f"Saved MemoryBank to {save_path}")

        total_time = time.time() - start_time
        self_refine_summary: Dict[str, Any] = {}
        if self_refine_source != "none":
            total_reports = len(self_refine_reports)
            total_refine_qa = sum(int(r.get("qa_total", 0)) for r in self_refine_reports)
            total_actions = sum(int(r.get("actions_applied", 0)) for r in self_refine_reports)
            deferred_reports = sum(
                1 for r in self_refine_reports if bool(r.get("deferred_due_to_empty_entries"))
            )
            deferred_qa = sum(
                int(r.get("deferred_question_count", 0))
                for r in self_refine_reports
                if bool(r.get("deferred_due_to_empty_entries"))
            )
            skipped_batches_total = sum(int(r.get("skipped_batches", 0)) for r in self_refine_reports)
            self_refine_summary = {
                "reports_count": total_reports,
                "executed_reports_count": max(0, total_reports - deferred_reports),
                "deferred_empty_memory_sessions": deferred_reports,
                "deferred_empty_memory_qa": deferred_qa,
                "skipped_batches_total": skipped_batches_total,
                "qa_total": total_refine_qa,
                "actions_applied_total": total_actions,
            }

        logger.info(f"\n{'='*60}\nSUMMARY: {sample_id}\n{'='*60}")
        logger.info(f"  Entries: {final_count}")
        logger.info(f"  Total time: {total_time:.2f}s")
        if self_refine_summary:
            logger.info(
                "  Self-Refine reports: %d executed=%d qa=%d actions=%d",
                self_refine_summary["reports_count"],
                self_refine_summary["executed_reports_count"],
                self_refine_summary["qa_total"],
                self_refine_summary["actions_applied_total"],
            )
            logger.info(
                "  Self-Refine deferred(empty memory_bank): sessions=%d qa=%d",
                self_refine_summary["deferred_empty_memory_sessions"],
                self_refine_summary["deferred_empty_memory_qa"],
            )
            logger.info(
                "  Self-Refine skipped batches: %d",
                self_refine_summary["skipped_batches_total"],
            )

        return {
            "sample_id": sample_id,
            "status": "success",
            "entry_count": final_count,
            "total_duration": total_time,
            "self_refine_reports": self_refine_reports,
            "self_refine_summary": self_refine_summary,
        }

    except Exception as e:
        logger.error(f"Failed {sample_id}: {e}", exc_info=True)
        return {"sample_id": sample_id, "status": "failed", "error": str(e)}


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Single-Agent MemoryManager + Meta-Thinker Self-Refine")

    parser.add_argument("--build_memories", action="store_true",
                       help="Build memories using MemoryManagerAgent (instead of evaluating)")

    parser.add_argument("--dataset", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--memory-dir", type=str, required=True, help="Directory for MemoryBank JSON files")
    parser.add_argument("--output_dir", type=str, default="results/memma_single", help="Output directory")

    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="LLM model name (Bedrock model IDs auto-detected)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI base URL")
    parser.add_argument("--region", type=str, default="us-west-2", help="AWS region for Bedrock")
    parser.add_argument("--meta_thinker_model", type=str, default=None,
                       help="Separate model for meta-thinker (e.g. Bedrock model ID). If unset, uses --model.")

    parser.add_argument("--retrieve_k", type=int, default=60, help="Number of memories to retrieve at QA time")
    parser.add_argument("--qr_max_turns", type=int, default=3, help="Maximum query rewrite / Meta-Thinker turns")
    parser.add_argument("--rewrite_rerank_strategy", type=str, default="no_cap",
                       choices=["no_cap", "keep_new_prune_old", "original_topk"],
                       help="Rewrite truncation strategy")
    parser.add_argument("--answerability_prompt_version", type=str, default="v4",
                       choices=["v1", "v2", "v3", "v4", "v5"],
                       help="Meta-Thinker answerability prompt version")
    parser.add_argument("--disable_meta_thinker", action="store_true",
                       help="Disable Meta-Thinker (single or QR-loop mode instead)")
    parser.add_argument("--enable_construction_meta_guidance", action="store_true",
                       help="Enable construction-time Meta-Thinker guidance during memory building")
    parser.add_argument("--mm_max_turns", type=int, default=1,
                       help="Max MemoryManagerAgent turns per conversation chunk")
    parser.add_argument("--mm_retrieval_mode", type=str, default="similarity",
                       choices=["similarity", "time"],
                       help="Retrieval mode for MemoryManagerAgent context")
    parser.add_argument("--mm_retrieve_k", type=int, default=10,
                       help="Number of memories to retrieve for MemoryManagerAgent context")

    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence-transformers model name for embeddings")

    parser.add_argument("--qa_max_workers", type=int, default=1, help="Max parallel QA workers")
    parser.add_argument("--build_max_workers", type=int, default=1, help="Max parallel build workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples")
    parser.add_argument("--max_sessions", type=int, default=0, help="Max sessions per sample (0=all)")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of samples to use")
    parser.add_argument("--categories", type=str, default="1,2,3,4", help="Comma-separated categories")

    parser.add_argument("--self_refine_source", type=str, default="none",
                       choices=["none", "parquet", "realtime"], help="Self refinement QA source")
    parser.add_argument("--self_refine_parquet", type=str, default="", help="Parquet path for self-refine QA")
    parser.add_argument("--self_refine_max_questions", type=int, default=0)
    parser.add_argument("--self_refine_batch_size", type=int, default=4)
    parser.add_argument("--self_refine_fail_on_empty", type=str_to_bool, default=True)
    parser.add_argument("--self_refine_log_jsonl", type=str, default="")
    parser.add_argument("--self_refine_inference_mode", type=str, default="simple",
                       choices=["simple"], help="Inference mode for self-refine QA")
    parser.add_argument("--self_refine_apply_threshold", type=float, default=0.0)
    parser.add_argument("--self_refine_add_fact_prompt_version", type=str,
                       default=DEFAULT_SELF_REFINE_ADD_FACT_PROMPT_VERSION, choices=["v1", "v2", "v3"])
    parser.add_argument("--self_refine_skip_empty_entries", type=str_to_bool, default=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, "memma_single_agent")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    answerability_prompt = ANSWERABILITY_PROMPT_BY_VERSION.get(
        args.answerability_prompt_version, META_THINKER_ANSWERABILITY_PROMPT_v4,
    )

    logger.info(f"\nLoading dataset from {args.dataset}")
    samples = parse_locomo_dataset(args.dataset)
    logger.info(f"Loaded {len(samples)} samples")

    if args.ratio < 1.0:
        num_samples = max(1, int(len(samples) * args.ratio))
        samples = samples[:num_samples]
    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]

    # =========================================================================
    # MODE: Build Memories
    # =========================================================================
    if args.build_memories:
        logger.info("=" * 60)
        logger.info("MEMORY BUILDING MODE (MemoryManagerAgent)")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Memory Dir: {args.memory_dir}")
        logger.info(f"MM max turns: {args.mm_max_turns}")
        logger.info(
            f"Construction Meta Guidance: {'ENABLED' if args.enable_construction_meta_guidance else 'DISABLED'}"
        )
        logger.info(f"Self-Refine Source: {args.self_refine_source}")

        llm_client = _create_llm_client(
            model=args.model,
            region=args.region,
            api_key=api_key,
            base_url=args.base_url,
        )
        logger.info(f"Build LLM client: {type(llm_client).__name__} (model={args.model})")

        session_qa_index: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        if args.self_refine_source == "parquet":
            if not args.self_refine_parquet:
                raise ValueError("--self_refine_parquet is required when --self_refine_source parquet")
            session_qa_index, parquet_conv_ids = load_parquet_session_qa(
                parquet_path=args.self_refine_parquet,
                fail_on_empty=args.self_refine_fail_on_empty,
            )
            logger.info(f"Loaded parquet self-refine QA: {len(session_qa_index)} session entries")

        self_refine_runtime = None
        if args.self_refine_source != "none":
            self_refine_runtime = init_self_refine_runtime(
                llm_client=llm_client,
                model_name=args.model,
                retrieve_k=args.retrieve_k,
                self_refine_inference_mode=args.self_refine_inference_mode,
                logger=logger,
            )

        results = []
        for sample in tqdm(samples, desc="Building memories"):
            result = build_memories_for_sample(
                sample=sample,
                memory_dir=args.memory_dir,
                api_key=api_key,
                llm_model=args.model,
                api_base_url=args.base_url,
                aws_region=args.region,
                embedding_model=args.embedding_model,
                log_dir=os.path.join(args.output_dir, "logs"),
                enable_meta_guidance=args.enable_construction_meta_guidance,
                llm_client=llm_client,
                mm_max_turns=args.mm_max_turns,
                mm_retrieval_mode=args.mm_retrieval_mode,
                mm_retrieve_k=args.mm_retrieve_k,
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
                logger=logger,
            )
            results.append(result)

        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        logger.info("\n" + "=" * 60)
        logger.info("MEMORY BUILDING COMPLETE")
        logger.info(f"Successful: {len(successful)}, Failed: {len(failed)}")
        if successful:
            logger.info(f"Total entries: {sum(r.get('entry_count', 0) for r in successful)}")
        return

    # =========================================================================
    # MODE: Evaluate (default)
    # =========================================================================
    enable_meta_thinker = not args.disable_meta_thinker

    logger.info("=" * 60)
    logger.info("Single-Agent Evaluator")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Memory Dir: {args.memory_dir}")
    logger.info(f"Retrieve K: {args.retrieve_k}")
    logger.info(f"QR Max Turns: {args.qr_max_turns}")
    logger.info(f"Meta-Thinker: {'ENABLED' if enable_meta_thinker else 'DISABLED'}")
    logger.info(f"Answerability Prompt: {args.answerability_prompt_version}")

    allow_categories = [int(c) for c in args.categories.split(",")]

    mt_llm = None
    if args.meta_thinker_model and enable_meta_thinker:
        mt_llm = _create_llm_client(
            model=args.meta_thinker_model,
            region=args.region,
            api_key=api_key,
            base_url=args.base_url,
        )
        logger.info(f"Meta-thinker LLM override: {type(mt_llm).__name__} (model={args.meta_thinker_model})")

    evaluator = SingleAgentEvaluator(
        model=args.model,
        region=args.region,
        api_key=api_key,
        base_url=args.base_url,
        memory_dir=args.memory_dir,
        embedding_model=args.embedding_model,
        retrieve_k=args.retrieve_k,
        qr_max_turns=args.qr_max_turns,
        enable_meta_thinker=enable_meta_thinker,
        rewrite_rerank_strategy=args.rewrite_rerank_strategy,
        qa_max_workers=args.qa_max_workers,
        answerability_prompt=answerability_prompt,
        answerability_prompt_version=args.answerability_prompt_version,
        meta_thinker_llm=mt_llm,
        logger=logger,
    )

    results = evaluator.evaluate_dataset(
        samples=samples,
        allow_categories=allow_categories,
        max_samples=args.max_samples,
    )
    results["answerability_prompt_version"] = args.answerability_prompt_version

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_file}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Total questions: {results['total_questions']}")

    logger.info(f"\n{'Category':<20s} {'J':>8s} {'F1':>8s} {'B1':>8s} {'n':>6s}")
    logger.info("-" * 60)
    for key, metrics in sorted(results["aggregate_metrics"].items()):
        if isinstance(metrics, dict) and "accuracy_mean" in metrics:
            j = metrics["accuracy_mean"]
            f1 = metrics.get("token_f1_mean", 0)
            b1 = metrics.get("bleu1_mean", 0)
            n = metrics["count"]
            logger.info(f"{key:<20s} {j:>8.4f} {f1:>8.4f} {b1:>8.4f} {n:>6d}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
