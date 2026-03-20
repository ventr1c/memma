#!/usr/bin/env python3
"""Multi-Agent Memory Cycle Baseline on LoCoMo Dataset.

This script implements the full multi-agent memory cycle framework:

Phase 1 - Memory Construction (Memory Manager Agent):
- For each conversation chunk (turn), iteratively query the Memory Manager
- Memory Manager decides: ADD, UPDATE, DELETE, or STOP
- Continue until STOP or max_turns reached
- Memory bank grows from conversation chunks

Phase 2 - Answering (Query Rewriter Agent):
- For each question, retrieve top-k memories
- Query Rewriter decides: ANSWERABLE or REWRITE
- If REWRITE, use rewritten queries to re-retrieve
- If ANSWERABLE, use frozen Answer Agent to generate response
- Continue until ANSWERABLE or max_turns reached

Based on A-Mem and LightMem patterns.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocess.utils import (
    parse_locomo_dataset,
    load_locomo_dataset,
)

# Try to import optional dependencies
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    SentenceTransformer = None


# ==============================================================================
# Logging Setup
# ==============================================================================

def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{run_name}_{timestamp}.log")
    
    logger = logging.getLogger('multi_agent_baseline')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# ==============================================================================
# LLM Client
# ==============================================================================

class LLMClient:
    """Simple LLM client wrapper."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
    
    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Get completion from LLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR: {e}]"


def _is_bedrock_model(model: str) -> bool:
    return "." in str(model or "") and "anthropic" in str(model or "")


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
    """Expose Bedrock Converse through an OpenAI-like client interface."""

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

        def create(self, model=None, messages=None, temperature=0.0, max_tokens=1024, **_kwargs):
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
    """Bedrock Converse API client with the same interface as LLMClient."""

    def __init__(self, model_id: str, region: str = "us-west-2"):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for Bedrock Claude models. Install boto3 in the current environment."
            ) from exc

        self.model = model_id
        self._region = region
        self._bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.client = BedrockOpenAICompat(self._bedrock, model_id)

    def get_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
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
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    if _is_bedrock_model(model):
        return BedrockLLMClient(model_id=model, region=region)
    return LLMClient(model=model, api_key=api_key, base_url=base_url)


# ==============================================================================
# Memory Bank
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
    # New fields for richer metadata (like A-MEM and LightMem)
    speaker: str = ""  # Who said this
    weekday: str = ""  # Day of week (Mon, Tue, etc.)
    float_timestamp: float = 0.0  # Unix timestamp for sorting
    category: str = ""  # Memory category
    
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

# Check for Qdrant availability
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class MemoryBank:
    """Memory bank with retrieval capabilities (in-memory storage)."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', **kwargs):
        self.entries: Dict[str, MemoryEntry] = {}
        self.next_id = 0
        self.storage_type = "memory"
        
        if HAS_SBERT:
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            self.encoder = None
            self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        self.embeddings: Optional[np.ndarray] = None
        self.entry_ids: List[str] = []
    
    def _parse_timestamp(self, timestamp: str) -> Tuple[float, str]:
        """Parse timestamp string to float and weekday."""
        import re
        from datetime import datetime as dt
        
        weekday = ""
        float_ts = 0.0
        
        if not timestamp:
            return float_ts, weekday
        
        # Parse format like "2023/05/20 (Sat) 00:44"
        match = re.search(r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*\(([^)]+)\)\s*(\d{1,2}:\d{2}(?::\d{2})?)', timestamp)
        if match:
            date_str = match.group(1).replace('-', '/')
            weekday = match.group(2)
            time_str = match.group(3)
            fmt = "%Y/%m/%d %H:%M:%S" if time_str.count(':') == 2 else "%Y/%m/%d %H:%M"
            try:
                parsed_dt = dt.strptime(f"{date_str} {time_str}", fmt)
                float_ts = parsed_dt.timestamp()
            except ValueError:
                pass
        
        return float_ts, weekday
    
    def _generate_id(self) -> str:
        self.next_id += 1
        return f"mem_{self.next_id:04d}"
    
    def add(self, content: str, timestamp: str = "", source_turn_id: str = None, 
            keywords: List[str] = None, speaker: str = "") -> str:
        """Add a new memory entry with enhanced metadata."""
        entry_id = self._generate_id()
        
        # Parse timestamp for float value and weekday
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
        """Update an existing memory entry."""
        if entry_id not in self.entries:
            return False
        
        self.entries[entry_id].content = new_content
        if self.encoder:
            self.entries[entry_id].embedding = self.encoder.encode([new_content])[0]
        
        self._rebuild_index()
        return True
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id not in self.entries:
            return False
        
        del self.entries[entry_id]
        self._rebuild_index()
        return True
    
    def _rebuild_index(self):
        """Rebuild the embedding index."""
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
        """Retrieve most recent k entries by timestamp (or insertion order)."""
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.id,  # Using ID as proxy for recency
            reverse=True
        )
        return sorted_entries[:k]
    
    def retrieve_by_similarity(self, query: str, k: int) -> List[MemoryEntry]:
        """Retrieve top-k entries by embedding similarity."""
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
        """Get all memory entries."""
        return list(self.entries.values())
    
    def format_for_context(self, entries: List[MemoryEntry], max_entries: int = 20) -> str:
        """Format memory entries as context string."""
        lines = []
        for entry in entries[:max_entries]:
            ts = f"[{entry.timestamp}] " if entry.timestamp else ""
            lines.append(f"{ts}({entry.id}) {entry.content}")
        return "\n".join(lines)
    
    def size(self) -> int:
        return len(self.entries)
    
    def save(self, filepath: str) -> None:
        """Save memory bank to JSON file for visualization."""
        data = {
            "next_id": self.next_id,
            "storage_type": self.storage_type,
            "entries": []
        }
        for entry in self.entries.values():
            entry_dict = entry.to_dict()
            # Convert embedding to list for JSON serialization
            if entry.embedding is not None:
                entry_dict["embedding"] = entry.embedding.tolist()
            data["entries"].append(entry_dict)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, embedding_model: str = 'all-MiniLM-L6-v2') -> 'MemoryBank':
        """Load memory bank from JSON file."""
        with open(filepath, 'r') as f:
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


class QdrantMemoryBank(MemoryBank):
    """Memory bank with Qdrant vector database storage."""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 collection_name: str = "memory_bank",
                 qdrant_path: str = None,
                 qdrant_url: str = None,
                 **kwargs):
        """
        Initialize QdrantMemoryBank.
        
        Args:
            embedding_model: Sentence transformer model name
            collection_name: Name of the Qdrant collection
            qdrant_path: Path for local Qdrant storage (mutually exclusive with qdrant_url)
            qdrant_url: URL for remote Qdrant server (mutually exclusive with qdrant_path)
        """
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is required for QdrantMemoryBank. Install with: pip install qdrant-client")
        
        # Initialize encoder first
        if HAS_SBERT:
            self.encoder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            raise ImportError("sentence-transformers is required for QdrantMemoryBank")
        
        self.entries: Dict[str, MemoryEntry] = {}  # Local cache
        self.next_id = 0
        self.storage_type = "qdrant"
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        
        # Initialize Qdrant client
        if qdrant_path:
            os.makedirs(qdrant_path, exist_ok=True)
            self.client = QdrantClient(path=qdrant_path)
        elif qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            # In-memory Qdrant (ephemeral)
            self.client = QdrantClient(":memory:")
        
        # Create collection if not exists
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
    
    def add(self, content: str, timestamp: str = "", source_turn_id: str = None,
            keywords: List[str] = None, speaker: str = "") -> str:
        """Add a new memory entry to Qdrant."""
        entry_id = self._generate_id()
        
        # Parse timestamp
        float_ts, weekday = self._parse_timestamp(timestamp)
        
        # Create embedding
        embedding = self.encoder.encode([content])[0]
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            timestamp=timestamp,
            source_turn_id=source_turn_id,
            keywords=keywords or [],
            speaker=speaker,
            weekday=weekday,
            float_timestamp=float_ts,
            embedding=embedding,
        )
        
        # Store in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=self.next_id,  # Qdrant uses numeric IDs
                    vector=embedding.tolist(),
                    payload={
                        "entry_id": entry_id,
                        "content": content,
                        "timestamp": timestamp,
                        "source_turn_id": source_turn_id,
                        "keywords": keywords or [],
                        "speaker": speaker,
                        "weekday": weekday,
                        "float_timestamp": float_ts,
                        "access_count": 0,
                        "category": "",
                    }
                )
            ]
        )
        
        # Keep local cache
        self.entries[entry_id] = entry
        
        return entry_id
    
    def update(self, entry_id: str, new_content: str) -> bool:
        """Update an existing memory entry in Qdrant."""
        if entry_id not in self.entries:
            return False
        
        entry = self.entries[entry_id]
        entry.content = new_content
        new_embedding = self.encoder.encode([new_content])[0]
        entry.embedding = new_embedding
        
        # Find the numeric ID in Qdrant
        numeric_id = int(entry_id.split("_")[1])
        
        # Update in Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=numeric_id,
                    vector=new_embedding.tolist(),
                    payload={
                        "entry_id": entry_id,
                        "content": new_content,
                        "timestamp": entry.timestamp,
                        "source_turn_id": entry.source_turn_id,
                        "keywords": entry.keywords,
                        "speaker": entry.speaker,
                        "weekday": entry.weekday,
                        "float_timestamp": entry.float_timestamp,
                        "access_count": entry.access_count,
                        "category": entry.category,
                    }
                )
            ]
        )
        
        return True
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry from Qdrant."""
        if entry_id not in self.entries:
            return False
        
        numeric_id = int(entry_id.split("_")[1])
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[numeric_id]
        )
        
        del self.entries[entry_id]
        return True
    
    def retrieve_by_similarity(self, query: str, k: int) -> List[MemoryEntry]:
        """Retrieve top-k entries by embedding similarity using Qdrant."""
        if not self.entries:
            return []
        
        query_embedding = self.encoder.encode([query])[0]
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )
        
        entries = []
        for result in results:
            payload = result.payload
            entry_id = payload.get("entry_id")
            if entry_id in self.entries:
                entries.append(self.entries[entry_id])
        
        return entries
    
    def size(self) -> int:
        """Get the number of entries in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count


def create_memory_bank(
    storage_backend: str = "memory",
    embedding_model: str = 'all-MiniLM-L6-v2',
    collection_name: str = "memory_bank",
    qdrant_path: str = None,
    qdrant_url: str = None,
) -> MemoryBank:
    """
    Factory function to create a memory bank with the specified storage backend.
    
    Args:
        storage_backend: "memory" for in-memory storage, "qdrant" for Qdrant vector DB
        embedding_model: Sentence transformer model name
        collection_name: Name of Qdrant collection (only used for qdrant backend)
        qdrant_path: Path for local Qdrant storage
        qdrant_url: URL for remote Qdrant server
    
    Returns:
        MemoryBank instance (either MemoryBank or QdrantMemoryBank)
    """
    if storage_backend == "qdrant":
        if not HAS_QDRANT:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
        return QdrantMemoryBank(
            embedding_model=embedding_model,
            collection_name=collection_name,
            qdrant_path=qdrant_path,
            qdrant_url=qdrant_url,
        )
    else:
        return MemoryBank(embedding_model=embedding_model)


# ==============================================================================
# Agent Prompts
# ==============================================================================

MEMORY_MANAGER_SYSTEM_PROMPT = """You are a smart Memory Manager that controls the memory of a conversation system.

You can perform four operations:
1. **ADD** - Add new information to memory as a new element
2. **UPDATE** - Update an existing memory element with new/more detailed information
3. **DELETE** - Delete an existing memory element (contradictory information)
4. **NONE** - Make no change (information already present or irrelevant)

## Operation Guidelines:

### ADD
Add if the conversation contains new information not present in memory.
- Generate a new unique ID for the new memory element
- Example:
  Old Memory: [{"id": "0", "text": "Caroline is a software engineer"}]
  New Fact: "Caroline lives in Seattle"
  Output: {"id": "1", "text": "Caroline lives in Seattle", "event": "ADD"}

### UPDATE  
Update if the new information is about the SAME topic but adds more detail or changes it.
- Keep the same ID and preserve old_memory
- IMPORTANT: The "text" field MUST contain the NEW UPDATED content that COMBINES old and new info
- If new info conveys the same meaning with similar detail, use NONE instead (do NOT update)
- Example (a): Memory has "User likes cricket" → New fact "Loves to play cricket with friends on weekends"
  Output: {"id": "0", "text": "User loves to play cricket with friends on weekends", "event": "UPDATE", "old_memory": "User likes cricket"}
- Example (b): Memory has "Likes cheese pizza" → New fact "Loves cheese pizza" → Use NONE (same meaning)

### DELETE
Delete if the new information contradicts existing memory.
- Return the same ID with event DELETE
- Example: Memory has "Loves cheese pizza" → New fact "Dislikes cheese pizza" → DELETE
- Output format: {"id": "0", "text": "Loves cheese pizza", "event": "DELETE"}

### NONE
No change if fact is already present or not important enough to store.
- Output format: {"id": "0", "text": "Existing text", "event": "NONE"}

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


# Enhanced Query Reasoner for reasoning-aware retrieval (used when qr_mode='reasoning')
QUERY_REASONER_SYSTEM_PROMPT = """You are a Query Reasoner agent with enhanced reasoning capabilities. Your task is to analyze retrieved memories and decide the best strategy for answering a question.

Given:
- A question
- Currently retrieved memories
- Previous query attempts (if any) and their retrieved results

You have THREE options:
1. **ANSWERABLE**: The retrieved memories contain sufficient information to answer the question directly.
2. **REWRITE**: The memories are relevant but incomplete - refine the query phrasing to retrieve better matches.
3. **EXPAND**: The memories are missing key information - generate NEW queries to explore different retrieval paths.

## When to use EXPAND (vs REWRITE):
- **Multi-hop questions**: Need to decompose into sub-questions (e.g., "What did X do after meeting Y?" → query for X's actions, query for meeting details)
- **Missing entity context**: Need background on entities mentioned (e.g., query who a person is before asking what they did)
- **Temporal gaps**: Need to retrieve information from different time periods
- **Implicit information**: Need to query for unstated prerequisites or context
- **Counter-factual reasoning**: Need alternative perspectives or related events

## EXPAND Strategies:
- **Decomposition**: Break multi-hop into atomic sub-queries
- **Entity Focus**: Query for specific entity details
- **Temporal Focus**: Query for specific time periods or sequences
- **Context Enrichment**: Query for background/prerequisite information
- **Relation Extraction**: Query for connections between entities

Output format:
<reasoning>Brief analysis of what information is available vs. missing, and why the chosen strategy is appropriate</reasoning>
<decision>ANSWERABLE|REWRITE|EXPAND</decision>
<content>If REWRITE/EXPAND: query1 ## query2 ## query3 (separated by ##). If ANSWERABLE: leave empty</content>

Be strategic:
- For REWRITE: Focus on synonyms, paraphrasing, related terms
- For EXPAND: Generate diverse queries that target different aspects of the missing information"""


ANSWER_AGENT_SYSTEM_PROMPT = """You are an Answer Agent. Given a question and relevant memories, provide a short, accurate answer.

Focus on:
- Answering directly from the provided memories
- Being concise (short phrase or sentence)
- Using exact words from memories when possible
- Saying "Not mentioned in the conversation" if the answer cannot be found"""


# ==============================================================================
# Meta-Thinker Agent Prompts
# ==============================================================================

META_THINKER_CONSTRUCTION_PROMPT = """You are a Meta-Thinker agent that provides high-level strategic guidance for memory construction.

Given:
- A new conversation chunk to process
- Recent memory entries (ordered by time)
- Similar memory entries (found by semantic similarity)

Your task is to analyze and provide KEY FOCUS POINTS that the Memory Manager should pay attention to when deciding how to update the memory bank.

Consider:
1. **Information Importance**: What facts in this chunk are likely to be important for future questions?
2. **Potential Redundancy**: Does this chunk contain information similar to existing memories that might need consolidation?
3. **Temporal Context**: Are there dates, times, or temporal relationships that should be carefully captured?
4. **Entity Relationships**: Are there new relationships between people, places, or things that should be explicitly stored?
5. **Potential Conflicts**: Does any new information potentially contradict existing memories?

Output a short list of 2-5 KEY FOCUS POINTS for the Memory Manager.

Format:
```
FOCUS POINTS:
1. [First key point to pay attention to]
2. [Second key point to pay attention to]
3. [Optional third point]
...
```

Be concise and actionable."""


META_THINKER_QA_PROMPT = """You are a Meta-Thinker agent that provides high-level strategic guidance for question answering and memory retrieval.

Given:
- A question to answer
- Currently retrieved memories

Your task is to analyze and provide KEY FOCUS POINTS for the Query Rewriter to help determine:
- Whether the retrieved memories are sufficient
- What additional information might be needed
- How to improve retrieval if needed

Consider:
1. **Question Type**: Is this a factual, temporal, comparison, or reasoning question?
2. **Information Coverage**: Do the memories cover all aspects of the question?
3. **Missing Information**: What specific information is clearly missing?
4. **Query Strategy**: What alternative queries could help find missing information?
5. **Temporal Aspects**: Does the question require time-based reasoning?

Output a short list of 2-5 KEY FOCUS POINTS for the Query Rewriter.

Format:
```
FOCUS POINTS:
1. [First key point about what to check or retrieve]
2. [Second key point about gaps or strategies]
3. [Optional third point]
```

Be concise and actionable."""


META_THINKER_REFLECTION_PROMPT = """You are a Meta-Thinker agent performing reflection to generate procedural guidance for future similar tasks.

You have just observed a complete memory cycle:
1. Memory Construction: A conversation was processed and memories were stored
2. Memory Retrieval: Queries were used to retrieve relevant memories
3. Question Answering: An answer was generated and judged

Your task is to REFLECT on what went well and what could be improved, then generate PROCEDURAL GUIDANCE that can help in future similar situations.

Consider:
1. **Memory Quality**: Were the right facts stored? Were there redundancies or gaps?
2. **Retrieval Effectiveness**: Did the queries retrieve the most relevant memories?
3. **Answer Accuracy**: What led to success or failure in generating the correct answer?
4. **Key Patterns**: What patterns or heuristics could be useful in future?

If PRIOR PROCEDURAL MEMORY is provided, you should:
- KEEP lessons that are still valid and helpful
- UPDATE lessons if new evidence refines or contradicts them
- APPEND new lessons learned from this cycle
- REMOVE outdated or incorrect lessons

Output your reflection and procedural guidance.

Format:
```
REFLECTION:
[Brief analysis of what happened in this cycle and why the answer was correct/wrong]

PROCEDURAL GUIDANCE:
1. [First lesson learned or heuristic for future]
2. [Second lesson learned or heuristic]
3. [Third lesson learned or heuristic]
(up to 5 key lessons)
```

Focus on extracting generalizable insights that apply beyond this specific case."""

META_THINKER_QA_PROMPT_V2 = """You are a Meta-Thinker agent that provides high-level strategic guidance for question answering and memory retrieval.

Given:
- A question to answer
- Currently retrieved memories

Your task is to analyze and provide KEY FOCUS POINTS for the Query Rewriter to help determine:
- Whether the retrieved memories are sufficient
- What additional information might be needed
- How to improve retrieval if needed

Critical rule:
- If the retrieved memories already directly answer the question, recommend ANSWERABLE and advise against further retrieval.

Otherwise:
- Identify what is missing (e.g., missing entity, missing time, missing location, missing relation, missing second hop).
- Suggest retrieval angles (aliases/synonyms, decomposition, temporal phrasing, related entities).

Consider:
1. **Question Type**: Is this a factual, temporal, comparison, or reasoning question?
2. **Information Coverage**: Do the memories cover all aspects of the question?
3. **Missing Information**: What specific information is clearly missing?
4. **Query Strategy**: What alternative queries could help find missing information?
5. **Temporal Aspects**: Does the question require time-based reasoning?

Output a short list of 2-5 KEY FOCUS POINTS for the Query Rewriter.

Format:
```
FOCUS POINTS:
1. [First key point about what to check or retrieve]
2. [Second key point about gaps or strategies]
3. [Optional third point]
```

Be concise and actionable."""

META_THINKER_REFLECTION_PROMPT_V2 = """You are a Meta-Thinker agent performing reflection to generate procedural guidance for future similar tasks.

You observed:
1) Memory Construction (ADD/UPDATE/DELETE)
2) Memory Retrieval (queries and retrieved entries)
3) Question Answering (answer + judgment)

Your task:
- Diagnose what caused success/failure.
- Update procedural guidance as compact, general IF–THEN rules (max 5).
- Avoid overfitting to specific proper nouns unless necessary.

First output a DIAGNOSIS:
- outcome: CORRECT/WRONG
- failure_type: storage_gap | retrieval_gap | reasoning_gap | none
  - storage_gap: needed fact never stored
  - retrieval_gap: fact stored but not retrieved by queries
  - reasoning_gap: evidence retrieved but answer still wrong

Then output PROCEDURAL GUIDANCE (IF–THEN rules).
If PRIOR PROCEDURAL MEMORY is provided:
- keep valid rules
- refine or remove invalid ones
- add only truly new rules

Format:
DIAGNOSIS:
- outcome: ...
- failure_type: ...
- key_reason: ...

PROCEDURAL GUIDANCE:
1. IF ... THEN ...
2. IF ... THEN ...
(up to 5)
"""

# ==============================================================================
# Meta-Thinker Agent
# ==============================================================================

class MetaThinkerAgent:
    """Meta-Thinker agent that provides high-level strategic guidance.
    
    Operates in two modes:
    1. Vanilla (Baseline 1): Provides real-time guidance during construction and QA
    2. Reflection (Baseline 2): Uses procedural memory from prior reflection
    """
    
    def __init__(self, llm: LLMClient, logger: logging.Logger, procedural_memory: Optional[str] = None):
        self.llm = llm
        self.logger = logger
        self.procedural_memory = procedural_memory  # For reflection-based mode
        self.reflections: List[Dict[str, Any]] = []  # Store reflections during training
    
    def get_construction_guidance(
        self,
        chunk: str,
        recent_memories: List[MemoryEntry],
        similar_memories: List[MemoryEntry],
    ) -> str:
        """Get high-level focus points for memory construction.
        
        Returns guidance string to incorporate into Memory Manager prompt.
        """
        # Format memories for context
        recent_ctx = "\n".join([f"- ({m.id}) [{m.timestamp}] {m.content}" for m in recent_memories[:5]]) if recent_memories else "[No recent memories]"
        similar_ctx = "\n".join([f"- ({m.id}) [{m.timestamp}] {m.content}" for m in similar_memories[:5]]) if similar_memories else "[No similar memories]"
        
        prompt = f"""New Conversation Chunk:
{chunk}

Recent Memories (by time):
{recent_ctx}

Similar Memories (by similarity):
{similar_ctx}

{f'Procedural Guidance from Previous Reflection:{chr(10)}{self.procedural_memory}' if self.procedural_memory else ''}

Provide KEY FOCUS POINTS for the Memory Manager."""

        response = self.llm.get_completion(
            prompt,
            temperature=0.3,
            max_tokens=512,
            system_prompt=META_THINKER_CONSTRUCTION_PROMPT,
        )
        
        self.logger.debug(f"Meta-Thinker Construction Guidance: {response}")
        return response
    
    def get_qa_guidance(
        self,
        question: str,
        retrieved_memories: List[MemoryEntry],
    ) -> str:
        """Get high-level focus points for QA/retrieval.
        
        Returns guidance string to incorporate into Query Rewriter prompt.
        """
        memories_ctx = "\n".join([f"- ({m.id}) [{m.timestamp}] {m.content}" for m in retrieved_memories[:10]]) if retrieved_memories else "[No memories retrieved]"
        
        prompt = f"""Question:
{question}

Retrieved Memories:
{memories_ctx}

{f'Procedural Guidance from Previous Reflection:{chr(10)}{self.procedural_memory}' if self.procedural_memory else ''}

Provide KEY FOCUS POINTS for the Query Rewriter."""

        response = self.llm.get_completion(
            prompt,
            temperature=0.3,
            max_tokens=512,
            system_prompt=META_THINKER_QA_PROMPT,
        )
        
        self.logger.debug(f"Meta-Thinker QA Guidance: {response}")
        return response
    
    def reflect(
        self,
        memory_actions: List[Dict],
        queries_and_retrieved: List[Dict],
        question: str,
        generated_answer: str,
        reference_answer: str,
        is_correct: bool,
    ) -> str:
        """Reflect on a completed cycle to generate procedural memory.
        
        Args:
            memory_actions: List of ADD/UPDATE/DELETE actions taken
            queries_and_retrieved: List of {query, retrieved_memories} dicts
            question: The question that was asked
            generated_answer: The answer produced by the system
            reference_answer: The ground-truth answer
            is_correct: Whether the answer was judged correct
            
        Returns:
            Procedural guidance string for future use
        """
        # Format memory actions
        actions_str = "\n".join([
            f"- {a.get('event', 'UNKNOWN')}: {a.get('text', '')[:100]}..." 
            for a in memory_actions if a.get('event') != 'NONE'
        ][:10]) if memory_actions else "[No memory actions]"
        
        # Format retrieval info - FULL content, no truncation
        retrieval_str = ""
        # for qr in queries_and_retrieved[:5]:
        for qr in queries_and_retrieved:
            query = qr.get('query', 'N/A')
            retrieved = qr.get('retrieved', [])
            # ret_texts = "\n  ".join([
            #     f"• [{r.id}] {r.content}" if hasattr(r, 'content') and hasattr(r, 'id') 
            #     else f"• {str(r)}" 
            #     # for r in retrieved[:5]
            #     for r in retrieved
            # ])
            ret_texts = "\n  ".join([
                f"• [{r.id}] [{r.timestamp}] {r.content}"
                # for r in retrieved[:5]
                for r in retrieved
            ])
            retrieval_str += f"Query: {query}\nRetrieved:\n  {ret_texts}\n\n"
        if not retrieval_str:
            retrieval_str = "[No retrieval information]"
        
        # Include prior procedural memory if available for incremental learning
        prior_memory_section = ""
        if self.procedural_memory:
            prior_memory_section = f"""## Existing Procedural Memory:
{self.procedural_memory}

You may UPDATE/REFINE the above guidance or APPEND new learnings based on the current cycle.

"""
        
        prompt = f"""{prior_memory_section}## Memory Construction Actions:
{actions_str}

## Retrieval History:
{retrieval_str}

## Question & Answer:
Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}
Judgment: {'CORRECT' if is_correct else 'WRONG'}

Analyze this cycle and generate procedural guidance for future similar tasks."""

        response = self.llm.get_completion(
            prompt,
            temperature=0.3,
            max_tokens=1024,
            system_prompt=META_THINKER_REFLECTION_PROMPT_V2,
        )
        
        # Store reflection
        self.reflections.append({
            "question": question,
            "is_correct": is_correct,
            "reflection": response,
        })
        
        # Update procedural memory incrementally for next reflection
        if 'PROCEDURAL GUIDANCE:' in response:
            idx = response.find('PROCEDURAL GUIDANCE:')
            new_guidance = response[idx:].strip()
            # The new guidance becomes the prior for the next reflection
            self.procedural_memory = new_guidance
        
        # Log full reflection without truncation
        self.logger.info(f"Meta-Thinker Reflection: {response}")
        return response
    
    def get_aggregated_procedural_memory(self) -> str:
        """Aggregate all reflections into a single procedural memory string."""
        if not self.reflections:
            return ""
        
        # Extract just the procedural guidance parts
        guidance_points = []
        for r in self.reflections:
            reflection = r.get('reflection', '')
            # Try to extract PROCEDURAL GUIDANCE section
            if 'PROCEDURAL GUIDANCE:' in reflection:
                idx = reflection.find('PROCEDURAL GUIDANCE:')
                guidance = reflection[idx:].strip()
                guidance_points.append(guidance)
        
        if not guidance_points:
            return ""
        
        # Combine into single guidance
        combined = "Based on prior reflections, here are key lessons learned:\n\n"
        combined += "\n\n---\n\n".join(guidance_points[:5])  # Limit to 5 reflections
        
        return combined


# ==============================================================================
# Memory Manager Agent
# ==============================================================================

class MemoryManagerAgent:
    """Memory Manager agent that decides ADD/UPDATE/DELETE/NONE operations.
    
    Uses Memory-R1 style structured JSON output format.
    """
    
    def __init__(self, llm: LLMClient, logger: logging.Logger):
        self.llm = llm
        self.logger = logger
    
    def query(
        self,
        chunk: str,
        memory_bank: MemoryBank,
        previous_actions: List[str],
        retrieval_mode: str = "time",
        retrieve_k: int = 10,
        meta_guidance: Optional[str] = None,  # From Meta-Thinker
    ) -> List[Dict[str, Any]]:
        """Query the memory manager for operations on the memory bank.
        
        Uses Memory-R1 style JSON output format.
        
        Args:
            meta_guidance: Optional high-level focus points from Meta-Thinker
        
        Returns:
            List of memory operation dicts with keys: id, text, event, old_memory (optional)
        """
        # Get recent memories for context
        if retrieval_mode == "time":
            recent_memories = memory_bank.retrieve_by_time(retrieve_k)
        else:  # similarity
            recent_memories = memory_bank.retrieve_by_similarity(chunk, retrieve_k)
        
        # Format memory as JSON array for Memory-R1 style
        memory_json = self._format_memory_as_json(recent_memories)
        
        # Build prompt with optional meta-guidance
        meta_section = ""
        if meta_guidance:
            meta_section = f"""
## Meta-Thinker Guidance (Pay attention to these focus points):
{meta_guidance}

"""
        
        prompt = f"""Current Memory Bank:
{memory_json}

New Conversation Chunk:
{chunk}
{meta_section}
Analyze the new conversation chunk and compare it with the existing memory.
For each piece of new information, decide whether to ADD, UPDATE, DELETE, or mark as NONE.

Return your response as a JSON object with a "memory" array."""

        response = self.llm.get_completion(
            prompt,
            temperature=0.3,
            max_tokens=1024,
            system_prompt=MEMORY_MANAGER_SYSTEM_PROMPT,
        )
        
        # Parse JSON response
        operations = self._parse_json_response(response, memory_bank)
        
        # Log full LLM response for debugging
        self.logger.debug(f"MM Raw Response:\n{response}")
        
        # Log summary of operations
        num_ops = sum(1 for op in operations if op.get('event', 'NONE') != 'NONE')
        self.logger.info(f"MM returned {len(operations)} operations ({num_ops} active)")
        
        return operations
    
    def _format_memory_as_json(self, memories: List[MemoryEntry]) -> str:
        """Format memories as JSON array for Memory-R1 style prompt."""
        if not memories:
            return "[]"
        
        memory_list = []
        for mem in memories:
            # Embed timestamp in text for better LLM visibility
            ts_prefix = f"[{mem.timestamp}] " if mem.timestamp else ""
            entry = {
                "id": mem.id,
                "timestamp": mem.timestamp or "",
                "text": f"{ts_prefix}{mem.content}",
            }
            memory_list.append(entry)
        
        return json.dumps(memory_list, indent=2)
    
    def _parse_json_response(self, response: str, memory_bank: MemoryBank) -> List[Dict[str, Any]]:
        """Parse JSON response from Memory Manager.
        
        Expected format:
        {
            "memory": [
                {"id": "0", "text": "...", "event": "NONE"},
                {"id": "1", "text": "...", "event": "ADD"},
                {"id": "2", "text": "...", "event": "UPDATE", "old_memory": "..."}
            ]
        }
        """
        import re
        
        operations = []
        
        try:
            # Try to extract JSON from response
            # Look for JSON block in markdown code fence
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_match = re.search(r'\{[^{}]*"memory"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback: try the whole response
                    json_str = response.strip()
            
            data = json.loads(json_str)
            
            if isinstance(data, dict) and 'memory' in data:
                for item in data['memory']:
                    if isinstance(item, dict) and 'event' in item:
                        operations.append({
                            'id': item.get('id', ''),
                            'text': item.get('text', ''),
                            'event': item.get('event', 'NONE').upper(),
                            'old_memory': item.get('old_memory', ''),
                        })
            
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            # Fallback: try to extract operations from text
            operations = self._fallback_parse(response)
        
        return operations
    
    def _fallback_parse(self, response: str) -> List[Dict[str, Any]]:
        """Fallback parser when JSON parsing fails."""
        import re
        
        operations = []
        response_upper = response.upper()
        
        # Check for simple ADD/UPDATE/DELETE/NONE patterns
        if '"ADD"' in response or "'ADD'" in response or 'EVENT": "ADD' in response_upper:
            # Try to extract the text being added
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
            if text_match:
                operations.append({
                    'id': '',
                    'text': text_match.group(1),
                    'event': 'ADD',
                    'old_memory': '',
                })
        elif '"UPDATE"' in response or 'EVENT": "UPDATE' in response_upper:
            id_match = re.search(r'"id"\s*:\s*"([^"]+)"', response)
            text_match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
            old_match = re.search(r'"old_memory"\s*:\s*"([^"]+)"', response)
            if text_match:
                operations.append({
                    'id': id_match.group(1) if id_match else '',
                    'text': text_match.group(1),
                    'event': 'UPDATE',
                    'old_memory': old_match.group(1) if old_match else '',
                })
        elif '"DELETE"' in response or 'EVENT": "DELETE' in response_upper:
            id_match = re.search(r'"id"\s*:\s*"([^"]+)"', response)
            if id_match:
                operations.append({
                    'id': id_match.group(1),
                    'text': '',
                    'event': 'DELETE',
                    'old_memory': '',
                })
        else:
            # Default to NONE (no operation needed)
            operations.append({
                'id': '',
                'text': '',
                'event': 'NONE',
                'old_memory': '',
            })
        
        return operations


# ==============================================================================
# Query Rewriter Agent
# ==============================================================================

class QueryRewriterAgent:
    """Query Rewriter agent that decides ANSWERABLE/REWRITE/EXPAND.
    
    Supports two modes:
    - 'simple': Original rewrite-only mode (ANSWERABLE/REWRITE)
    - 'reasoning': Enhanced reasoning-aware mode (ANSWERABLE/REWRITE/EXPAND)
    """
    
    def __init__(self, llm: LLMClient, logger: logging.Logger, qr_mode: str = "simple"):
        self.llm = llm
        self.logger = logger
        self.qr_mode = qr_mode  # "simple" or "reasoning"
    
    def query(
        self,
        question: str,
        retrieved_memories: List[MemoryEntry],
        previous_rewrites: List[str],
        meta_guidance: Optional[str] = None,  # From Meta-Thinker
    ) -> Tuple[str, List[str]]:
        """Query the rewriter for the next action.
        
        Args:
            meta_guidance: Optional high-level focus points from Meta-Thinker
        
        Returns:
            (decision, queries) tuple where decision is ANSWERABLE/REWRITE/EXPAND
            and queries is a list of new/rewritten queries
        """
        # Format retrieved memories
        memories_str = "\n".join([
            f"({m.id}) [{m.timestamp}] {m.content}" for m in retrieved_memories
        ]) if retrieved_memories else "[No memories retrieved]"
        
        # Format previous rewrites
        prev_rewrites_str = "\n".join([f"- {rw}" for rw in previous_rewrites]) if previous_rewrites else "[No previous queries]"
        
        # Build prompt with optional meta-guidance
        meta_section = ""
        if meta_guidance:
            meta_section = f"""
## Meta-Thinker Guidance (Pay attention to these focus points):
{meta_guidance}

"""
        
        if self.qr_mode == "reasoning":
            # Enhanced reasoning-aware prompt
            prompt = f"""Question: {question}

Retrieved Memories:
{memories_str}

Previous Query Attempts:
{prev_rewrites_str}
{meta_section}
Analyze these memories and decide the best strategy:
- ANSWERABLE: Memories contain sufficient information
- REWRITE: Need to refine query phrasing for better retrieval
- EXPAND: Need to add NEW queries to explore different retrieval paths (for multi-hop, missing context, etc.)

Output format:
<reasoning>Brief reasoning about what information is available/missing</reasoning>
<decision>ANSWERABLE|REWRITE|EXPAND</decision>
<content>If REWRITE/EXPAND: query1 ## query2 (separated by ##). If ANSWERABLE: leave empty</content>"""
            
            system_prompt = QUERY_REASONER_SYSTEM_PROMPT
            max_tokens = 300
        else:
            # Simple rewrite-only prompt
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
            
            system_prompt = QUERY_REWRITER_SYSTEM_PROMPT
            max_tokens = 200

        response = self.llm.get_completion(
            prompt,
            temperature=0.3,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        
        # Parse response
        decision, queries = self._parse_response(response)
        self.logger.debug(f"QR Response: {response}")
        self.logger.info(f"QR Decision: {decision}, Queries: {queries}")
        
        return decision, queries
    
    def _parse_response(self, response: str) -> Tuple[str, List[str]]:
        """Parse the decision-content format from response."""
        import re
        
        # Support both simple mode (ANSWERABLE|REWRITE) and reasoning mode (ANSWERABLE|REWRITE|EXPAND)
        decision_match = re.search(r'<decision>\s*(ANSWERABLE|REWRITE|EXPAND)\s*</decision>', response, re.IGNORECASE)
        content_match = re.search(r'<content>(.*?)</content>', response, re.DOTALL | re.IGNORECASE)
        
        decision = decision_match.group(1).upper() if decision_match else "ANSWERABLE"
        content = content_match.group(1).strip() if content_match else ""
        
        # Parse queries from content (for both REWRITE and EXPAND)
        queries = []
        if content and decision in ("REWRITE", "EXPAND"):
            queries = [q.strip() for q in content.split("##") if q.strip()]
        
        # Fallback parsing
        if not decision_match:
            response_upper = response.upper()
            if "EXPAND" in response_upper:
                decision = "EXPAND"
            elif "REWRITE" in response_upper:
                decision = "REWRITE"
            else:
                decision = "ANSWERABLE"
            
            # Try to extract queries
            if decision in ("REWRITE", "EXPAND") and "##" in response:
                parts = response.split("##")
                queries = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
        
        return decision, queries


# ==============================================================================
# Answer Agent (Frozen)
# ==============================================================================

class AnswerAgent:
    """Frozen Answer Agent that generates answers from retrieved memories."""
    
    def __init__(self, llm: LLMClient, logger: logging.Logger):
        self.llm = llm
        self.logger = logger
    
    def answer(
        self,
        question: str,
        memories: List[MemoryEntry],
        category: int = 1,
    ) -> str:
        """Generate answer from memories."""
        memories_str = "\n".join([
            f"({m.id}) [{m.timestamp}] {m.content}" if m.timestamp else f"({m.id}) {m.content}"
            for m in memories
        ]) if memories else "[No memories available]"
        
        # === Visualize Retrieved Memories ===
        self.logger.info(f"\n{'═'*70}")
        self.logger.info(f"📦 ANSWER AGENT - RETRIEVED MEMORIES (n={len(memories)})")
        self.logger.info(f"{'═'*70}")
        self.logger.info(f"Question: {question}")
        self.logger.info(f"Category: {category}")
        self.logger.info(f"{'─'*70}")
        self.logger.info(f"Memories used in prompt:")
        for i, m in enumerate(memories[:10], 1):  # Show max 10
            ts = f"[{m.timestamp}]" if m.timestamp else ""
            speaker = f"({m.speaker})" if hasattr(m, 'speaker') and m.speaker else ""
            self.logger.info(f"  [{i}] ({m.id}) {ts} {speaker}")
            # self.logger.info(f"      {m.content[:100]}{'...' if len(m.content) > 100 else ''}")
        if len(memories) > 10:
            self.logger.info(f"  ... and {len(memories) - 10} more memories")
        self.logger.info(f"{'═'*70}")
        
        # Category-specific prompts (from A-Mem)
        if category == 5:
            prompt = f"""Memories:
{memories_str}

Question: {question}

If the answer is mentioned in the memories, provide it. Otherwise, say "Not mentioned in the conversation".

Short answer:"""
        elif category == 2:
            prompt = f"""Memories:
{memories_str}

Question: {question}

Answer with an approximate date based on the memories. Be concise.

Short answer:"""
        else:
            prompt = f"""Memories:
{memories_str}

Question: {question}

Answer in a short phrase using exact words from the memories when possible.

Short answer:"""

        response = self.llm.get_completion(
            prompt,
            temperature=0.3,
            max_tokens=100,
            system_prompt=ANSWER_AGENT_SYSTEM_PROMPT,
        )
        
        self.logger.info(f"Answer: {response}")
        return response


# ==============================================================================
# Multi-Agent Cycle Evaluator
# ==============================================================================

class MultiAgentCycleEvaluator:
    """Full multi-agent memory cycle evaluator with optional Meta-Thinker.
    
    Meta-Thinker modes:
    - "none": No meta-thinker (original baseline)
    - "vanilla": Vanilla LLM meta-thinker provides real-time guidance (Baseline 1)
    - "reflection": Reflection-based meta-thinker with procedural memory (Baseline 2)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        backbone_model: Optional[str] = None,
        answer_model: Optional[str] = None,
        region: str = "us-west-2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mm_retrieval_mode: str = "time",  # "time" or "similarity"
        mm_retrieve_k: int = 10,
        mm_max_turns: int = 5,
        qr_retrieve_k: int = 10,
        qr_max_turns: int = 3,
        meta_thinker_mode: str = "none",  # "none", "vanilla", "reflection"
        qr_mode: str = "simple",  # "simple" or "reasoning"
        logger: Optional[logging.Logger] = None,
    ):
        self.answer_model = answer_model or model
        self.backbone_model = backbone_model or model
        self.answer_llm = _create_llm_client(
            model=self.answer_model,
            region=region,
            api_key=api_key,
            base_url=base_url,
        )
        self.backbone_llm = _create_llm_client(
            model=self.backbone_model,
            region=region,
            api_key=api_key,
            base_url=base_url,
        )
        self.llm = self.backbone_llm
        self.mm_retrieval_mode = mm_retrieval_mode
        self.mm_retrieve_k = mm_retrieve_k
        self.mm_max_turns = mm_max_turns
        self.qr_retrieve_k = qr_retrieve_k
        self.qr_max_turns = qr_max_turns
        self.meta_thinker_mode = meta_thinker_mode
        self.qr_mode = qr_mode
        self.logger = logger or logging.getLogger('multi_agent_baseline')
        self.logger.info(f"Backbone model: {self.backbone_model}")
        self.logger.info(f"Answer/Judge model: {self.answer_model}")
        
        # Create agents
        self.memory_manager = MemoryManagerAgent(self.backbone_llm, self.logger)
        self.query_rewriter = QueryRewriterAgent(self.backbone_llm, self.logger, qr_mode=qr_mode)
        self.answer_agent = AnswerAgent(self.answer_llm, self.logger)
        
        # Create Meta-Thinker if needed
        self.meta_thinker: Optional[MetaThinkerAgent] = None
        if meta_thinker_mode in ("vanilla", "reflection"):
            self.meta_thinker = MetaThinkerAgent(self.backbone_llm, self.logger)
            self.logger.info(f"Meta-Thinker enabled (mode: {meta_thinker_mode})")
        
        # Log Query Rewriter mode
        if qr_mode == "reasoning":
            self.logger.info(f"Query Rewriter in reasoning mode (EXPAND enabled)")
        
        # For reflection mode: store all actions for post-sample reflection
        self.sample_memory_actions: List[Dict] = []
        self.sample_retrieval_history: List[Dict] = []
    
    def process_chunk_with_memory_manager(
        self,
        chunk: str,
        memory_bank: MemoryBank,
        timestamp: str = "",
    ) -> List[Dict]:
        """Process a conversation chunk through the Memory Manager loop.
        
        Uses iterative loop like Query Rewriter:
        - Keep querying until MM thinks it's done (all NONE/no operations)
        - Or until max_turns reached
        
        Returns list of actions taken across all turns.
        """
        all_actions = []
        turn = 0
        
        # Get Meta-Thinker construction guidance (only on first turn)
        meta_guidance = None
        if self.meta_thinker:
            recent_memories = memory_bank.retrieve_by_time(5)
            similar_memories = memory_bank.retrieve_by_similarity(chunk, 5)
            meta_guidance = self.meta_thinker.get_construction_guidance(
                chunk=chunk,
                recent_memories=recent_memories,
                similar_memories=similar_memories,
            )
            self.logger.info(f"Meta-Thinker Construction Guidance:\n{meta_guidance}")
        
        while turn < self.mm_max_turns:
            # Get previous actions for context
            previous_action_strs = [a.get('action', '') for a in all_actions if a.get('action')]
            
            # Query Memory Manager with optional meta-guidance
            operations = self.memory_manager.query(
                chunk=chunk,
                memory_bank=memory_bank,
                previous_actions=previous_action_strs,
                retrieval_mode=self.mm_retrieval_mode,
                retrieve_k=self.mm_retrieve_k,
                meta_guidance=meta_guidance if turn == 0 else None,  # Only first turn
            )
            
            # Count non-NONE operations in this turn
            active_ops_count = 0
            turn_actions = []
            
            # Process each operation
            for op in operations:
                event = op.get('event', 'NONE').upper()
                text = op.get('text', '')
                op_id = op.get('id', '')
                old_memory = op.get('old_memory', '')
                
                action = {
                    "turn": turn,
                    "event": event,
                    "text": text,
                    "id": op_id,
                    "old_memory": old_memory,
                }
                
                if event == "ADD" and text:
                    entry_id = memory_bank.add(
                        content=text,
                        timestamp=timestamp,
                        source_turn_id=None,
                    )
                    action["new_id"] = entry_id
                    action["action"] = f"ADD: {entry_id}"
                    self.logger.info(f"[Turn {turn}] ✚ ADD {entry_id}")
                    self.logger.info(f"    Content: {text}")
                    active_ops_count += 1
                    
                elif event == "UPDATE" and text and op_id:
                    # Get old content before update
                    old_content = memory_bank.entries.get(op_id).content if op_id in memory_bank.entries else "[not found]"
                    
                    # # IMPORTANT: Skip update if new text equals old text (LLM mistake)
                    # if text.strip() == old_content.strip():
                    #     self.logger.warning(f"[Turn {turn}] Skipping no-change UPDATE on {op_id} (new==old)")
                    #     action["action"] = "UPDATE_SKIPPED"
                    #     action["reason"] = "new_equals_old"
                    #     turn_actions.append(action)
                    #     continue
                    
                    success = memory_bank.update(op_id, text)
                    action["success"] = success
                    action["old_memory"] = old_content
                    action["action"] = f"UPDATE: {op_id}"
                    if success:
                        self.logger.info(f"[Turn {turn}] ⟳ UPDATE {op_id}")
                        self.logger.info(f"    Old: {old_content}")
                        self.logger.info(f"    New: {text}")
                        active_ops_count += 1
                    else:
                        self.logger.warning(f"[Turn {turn}] Failed to update memory {op_id}")
                        
                elif event == "DELETE" and op_id:
                    success = memory_bank.delete(op_id)
                    action["success"] = success
                    action["action"] = f"DELETE: {op_id}"
                    if success:
                        self.logger.info(f"[Turn {turn}] ✖ DELETE {op_id}")
                        active_ops_count += 1
                    else:
                        self.logger.warning(f"[Turn {turn}] Failed to delete memory {op_id}")
                    
                elif event == "NONE":
                    action["action"] = "NONE"
                    # No operation needed - don't count as active
                
                else:
                    action["action"] = "UNKNOWN"
                
                turn_actions.append(action)
            
            all_actions.extend(turn_actions)
            turn += 1
            
            # Stop condition: no active operations (all NONE or empty)
            if active_ops_count == 0:
                self.logger.info(f"[Turn {turn-1}] Memory Manager done - no more operations needed")
                break
            
            # Also stop if no operations returned
            if not operations:
                break
        
        if turn >= self.mm_max_turns:
            self.logger.info(f"Memory Manager reached max turns ({self.mm_max_turns})")
        
        return all_actions
    
    def answer_question_with_rewriter(
        self,
        question: str,
        memory_bank: MemoryBank,
        category: int = 1,
    ) -> Dict:
        """Answer a question using the Query Rewriter loop.
        
        Returns answer result with metadata.
        """
        # Category labels for logging
        CATEGORY_LABELS = {
            1: "Single-hop",
            2: "Temporal",
            3: "Multi-hop",
            4: "Open-domain",
            5: "Adversarial/Unanswerable",
        }
        result = {
            "question": question,
            "category": category,
            "rewrite_turns": [],
            "final_memories_used": [],
            "answer": "",
        }
        
        # Initial retrieval
        current_query = question
        all_retrieved = memory_bank.retrieve_by_similarity(current_query, self.qr_retrieve_k)
        previous_rewrites = []
        
        # Get Meta-Thinker QA guidance (only on first turn)
        meta_guidance = None
        if self.meta_thinker:
            meta_guidance = self.meta_thinker.get_qa_guidance(
                question=question,
                retrieved_memories=all_retrieved,
            )
            self.logger.info(f"Meta-Thinker QA Guidance:\n{meta_guidance}")
        
        # Store retrieval for reflection mode
        retrieval_history = []
        retrieval_history.append({
            "query": question,
            "retrieved": list(all_retrieved),  # Copy
        })
        
        turn = 0
        while turn < self.qr_max_turns:
            decision, queries = self.query_rewriter.query(
                question=question,
                retrieved_memories=all_retrieved,
                previous_rewrites=previous_rewrites,
                meta_guidance=meta_guidance if turn == 0 else None,  # Only first turn
            )
            
            rewrite_info = {
                "turn": turn,
                "decision": decision,
                "queries": queries,
                "num_memories": len(all_retrieved),
            }
            result["rewrite_turns"].append(rewrite_info)
            
            if decision == "ANSWERABLE" or not queries:
                break
            
            # Rewrite: retrieve with new queries
            for query in queries:
                previous_rewrites.append(query)
                new_memories = memory_bank.retrieve_by_similarity(query, self.qr_retrieve_k // len(queries) + 1)
                # Store for reflection
                retrieval_history.append({
                    "query": query,
                    "retrieved": list(new_memories),
                })
                # Merge with existing, avoiding duplicates
                existing_ids = {m.id for m in all_retrieved}
                for mem in new_memories:
                    if mem.id not in existing_ids:
                        all_retrieved.append(mem)
                        existing_ids.add(mem.id)
            
            # Limit total retrieved
            all_retrieved = all_retrieved[:self.qr_retrieve_k * 2]
            turn += 1
        
        # Store retrieval history for reflection mode
        self.sample_retrieval_history.extend(retrieval_history)
        
        # Generate answer with frozen Answer Agent
        final_memories = all_retrieved[:self.qr_retrieve_k]
        result["final_memories_used"] = [m.to_dict() for m in final_memories]
        
        # Log retrieved memories and category
        cat_label = CATEGORY_LABELS.get(category, f"Unknown ({category})")
        self.logger.info(f"\n>>> Category {category} ({cat_label})")
        self.logger.info(f">>> Retrieved {len(final_memories)} memories:")
        for i, mem in enumerate(final_memories, 1):
            ts = f"[{mem.timestamp}] " if mem.timestamp else ""
            content_preview = mem.content[:100] + "..." if len(mem.content) > 100 else mem.content
            self.logger.info(f"    [{i}] ID: {mem.id} | {ts}{content_preview}")
        
        result["answer"] = self.answer_agent.answer(
            question=question,
            memories=final_memories,
            category=category,
        )
        
        return result
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        allow_categories: List[int] = [1, 2, 3, 4, 5],
        storage_backend: str = "memory",
        qdrant_path: str = None,
    ) -> Dict:
        """Evaluate a single sample through the full cycle."""
        sample_result = {
            "sample_id": sample.get("sample_id", "unknown"),
            "memory_construction": {"turns_processed": 0, "final_memory_size": 0},
            "qa_results": [],
        }
        
        # Reset tracking for this sample (used for reflection mode)
        self.sample_memory_actions = []
        self.sample_retrieval_history = []
        
        # Create fresh memory bank for this sample using the specified backend
        sample_id = sample.get("sample_id", "unknown")
        if storage_backend == "qdrant":
            collection_name = f"memory_{sample_id}"
            sample_qdrant_path = os.path.join(qdrant_path, sample_id) if qdrant_path else None
            memory_bank = create_memory_bank(
                storage_backend="qdrant",
                collection_name=collection_name,
                qdrant_path=sample_qdrant_path,
            )
            self.logger.info(f"Using Qdrant storage backend (path: {sample_qdrant_path})")
        else:
            memory_bank = create_memory_bank(storage_backend="memory")
            self.logger.info("Using in-memory storage backend")
        
        # === Phase 1: Memory Construction ===
        self.logger.info(f"\n=== Phase 1: Memory Construction ===")
        
        conv = sample.get('conversation', {})
        session_keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')])
        
        all_mm_actions = []
        for sess_key in session_keys:
            timestamp = conv.get(f"{sess_key}_date_time", "")
            turns = conv.get(sess_key, [])
            
            for turn in turns:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                
                if not text:
                    continue
                
                chunk = f"[{timestamp}] {speaker}: {text}" if timestamp else f"{speaker}: {text}"
                
                self.logger.info(f"\nProcessing chunk: {chunk}...")
                actions = self.process_chunk_with_memory_manager(chunk, memory_bank, timestamp)
                all_mm_actions.extend(actions)
                sample_result["memory_construction"]["turns_processed"] += 1
        
        # Store all memory actions for reflection
        self.sample_memory_actions = all_mm_actions
        
        sample_result["memory_construction"]["final_memory_size"] = memory_bank.size()
        sample_result["memory_construction"]["total_actions"] = len(all_mm_actions)
        self.logger.info(f"\nMemory bank size after construction: {memory_bank.size()}")
        
        # Save memory bank for visualization
        memory_save_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "results", "memory_banks", f"memory_{sample_id}.json"
        )
        os.makedirs(os.path.dirname(memory_save_path), exist_ok=True)
        memory_bank.save(memory_save_path)
        self.logger.info(f"Saved memory bank to {memory_save_path}")
        
        # === Phase 2: Question Answering ===
        self.logger.info(f"\n=== Phase 2: Question Answering ===")
        
        for qa in sample.get('qa', []):
            category = qa.get('category', 1)
            if category not in allow_categories:
                continue
            
            question = qa.get('question', '')
            reference = str(qa.get('answer', ''))
            
            if not question:
                continue
            
            self.logger.info(f"\nQuestion: {question}...")
            
            qa_result = self.answer_question_with_rewriter(
                question=question,
                memory_bank=memory_bank,
                category=category,
            )
            
            qa_result["reference"] = reference
            qa_result["metrics"] = calculate_simple_metrics(qa_result["answer"], reference)
            
            # LLM Judge accuracy (from LightMem)
            accuracy = evaluate_llm_judge(
                question=question,
                gold_answer=reference,
                generated_answer=qa_result["answer"],
                client=self.answer_llm.client,
                model_name=self.answer_llm.model,
            )
            qa_result["metrics"]["accuracy"] = float(accuracy)
            
            self.logger.info(f"Answer: {qa_result['answer']}")
            self.logger.info(f"Reference: {reference}")
            self.logger.info(f"Accuracy: {accuracy} ({'CORRECT' if accuracy else 'WRONG'})")
            self.logger.info(f"Metrics: {qa_result['metrics']}")
            
            sample_result["qa_results"].append(qa_result)
        
        return sample_result
    
    def run_reflection_training(
        self,
        training_samples: List[Dict[str, Any]],
        allow_categories: List[int] = [1, 2, 3, 4, 5],
        storage_backend: str = "memory",
        qdrant_path: str = None,
    ) -> str:
        """Run reflection training on samples to build procedural memory.
        
        This is the TRAINING STAGE for reflection mode (Baseline 2).
        
        INTERLEAVED FLOW:
        For each conversation turn:
          1. Process the turn with Memory Manager + Meta-Thinker guidance
          2. If there are QA questions linked to this turn (via evidence field),
             immediately run QA phase and generate reflection
        
        Args:
            training_samples: List of samples to use for reflection training
            allow_categories: Categories to evaluate
            
        Returns:
            Aggregated procedural memory string for use in inference
        """
        if not self.meta_thinker:
            self.logger.warning("Meta-Thinker not enabled, skipping reflection training")
            return ""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"REFLECTION TRAINING STAGE (Interleaved)")
        self.logger.info(f"Training on {len(training_samples)} sample(s)")
        self.logger.info(f"{'='*60}")
        
        # Clear any previous reflections
        self.meta_thinker.reflections = []
        self.meta_thinker.procedural_memory = None
        
        for sample_idx, sample in enumerate(training_samples):
            self.logger.info(f"\n--- Training Sample {sample_idx + 1}/{len(training_samples)} ---")
            
            # Reset tracking
            self.sample_memory_actions = []
            self.sample_retrieval_history = []
            
            # Create memory bank for this sample
            sample_id = sample.get("sample_id", f"train_{sample_idx}")
            if storage_backend == "qdrant":
                collection_name = f"train_memory_{sample_id}"
                sample_qdrant_path = os.path.join(qdrant_path, sample_id) if qdrant_path else None
                memory_bank = create_memory_bank(
                    storage_backend="qdrant",
                    collection_name=collection_name,
                    qdrant_path=sample_qdrant_path,
                )
            else:
                memory_bank = create_memory_bank(storage_backend="memory")
            
            # === Build turn -> QA mapping from evidence field ===
            # Evidence format: "D1:3" means session_1, turn index 3
            # Only link QA to the LATEST evidence turn (max session, max turn)
            # This ensures QA is asked only when all evidence is available
            
            def _parse_evidence_id(eid: str) -> tuple:
                """Parse evidence ID like 'D3:13' into (session_num, turn_num)."""
                try:
                    parts = str(eid).split(":")
                    if len(parts) == 2:
                        session = int(parts[0].replace("D", "").replace("d", ""))
                        turn = int(parts[1])
                        return (session, turn)
                except (ValueError, IndexError):
                    pass
                return (0, 0)
            
            turn_to_qas = {}
            print(sample['qa'])
            for qa in sample.get('qa', []):
                category = qa.get('category', 1)
                if category not in allow_categories:
                    continue
                
                evidences = qa.get('evidence', [])
                if not evidences:
                    continue
                
                # Flatten any multi-evidence strings like "D8:6; D9:17"
                all_evidence_ids = []
                for ev in evidences:
                    ev_str = str(ev).replace(' ', '')
                    for ev_part in ev_str.split(';'):
                        if ':' in ev_part:
                            all_evidence_ids.append(ev_part)
                
                if not all_evidence_ids:
                    continue
                
                # Find the LATEST evidence turn (max session, then max turn)
                latest_evidence = max(all_evidence_ids, key=_parse_evidence_id)
                session_num, turn_idx = _parse_evidence_id(latest_evidence)
                
                if session_num > 0:
                    session_key = f"session_{session_num}"
                    key = (session_key, turn_idx)
                    if key not in turn_to_qas:
                        turn_to_qas[key] = []
                    if qa not in turn_to_qas[key]:
                        turn_to_qas[key].append(qa)
            
            self.logger.info(f"[Training] Built turn->QA mapping: {len(turn_to_qas)} turns have linked QAs")
            
            # === Interleaved Processing ===
            conv = sample.get('conversation', {})
            session_keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')])
            
            total_turns_processed = 0
            total_qa_processed = 0
            
            for sess_key in session_keys:
                timestamp = conv.get(f"{sess_key}_date_time", "")
                turns = conv.get(sess_key, [])
                
                for turn_idx, turn in enumerate(turns):
                    speaker = turn.get('speaker', 'Unknown')
                    text = turn.get('text', '')
                    if not text:
                        continue
                    
                    # === Phase 1: Memory Construction for this turn ===
                    chunk = f"[{timestamp}] {speaker}: {text}" if timestamp else f"{speaker}: {text}"
                    actions = self.process_chunk_with_memory_manager(chunk, memory_bank, timestamp)
                    self.sample_memory_actions.extend(actions)
                    total_turns_processed += 1
                    
                    # === Phase 2: If this turn has linked QAs, do QA + Reflection ===
                    linked_qas = turn_to_qas.get((sess_key, turn_idx), [])
                    
                    if linked_qas:
                        self.logger.info(f"\n[Training] Turn {sess_key}:{turn_idx} has {len(linked_qas)} linked QA(s)")
                        
                        for qa in linked_qas:
                            question = qa.get('question', '')
                            reference = str(qa.get('answer', ''))
                            category = qa.get('category', 1)
                            
                            if not question:
                                continue
                            
                            self.logger.info(f"[Training] Q: {question[:60]}...")
                            
                            # Answer question with current memory state
                            qa_result = self.answer_question_with_rewriter(
                                question=question,
                                memory_bank=memory_bank,
                                category=category,
                            )
                            
                            # Judge accuracy
                            accuracy = evaluate_llm_judge(
                                question=question,
                                gold_answer=reference,
                                generated_answer=qa_result["answer"],
                                client=self.answer_llm.client,
                                model_name=self.answer_llm.model,
                            )
                            
                            self.logger.info(f"[Training] A: {qa_result['answer'][:60]}...")
                            self.logger.info(f"[Training] Ref: {reference[:60]}...")
                            self.logger.info(f"[Training] {'✓ CORRECT' if accuracy else '✗ WRONG'}")
                            
                            # Generate reflection immediately
                            reflection = self.meta_thinker.reflect(
                                memory_actions=self.sample_memory_actions,
                                queries_and_retrieved=self.sample_retrieval_history,
                                question=question,
                                generated_answer=qa_result["answer"],
                                reference_answer=reference,
                                is_correct=(accuracy == 1),
                            )
                            
                            total_qa_processed += 1
                            
                            # Clear retrieval history for next QA (but keep memory actions)
                            self.sample_retrieval_history = []
            
            self.logger.info(f"\n[Training] Sample complete: {total_turns_processed} turns, {total_qa_processed} QAs, {memory_bank.size()} memories")
        
        # Aggregate all reflections into procedural memory
        procedural_memory = self.meta_thinker.get_aggregated_procedural_memory()
        self.meta_thinker.procedural_memory = procedural_memory
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"REFLECTION TRAINING COMPLETE")
        self.logger.info(f"Generated {len(self.meta_thinker.reflections)} reflections")
        self.logger.info(f"Procedural memory length: {len(procedural_memory)} chars")
        self.logger.info(f"{'='*60}")
        
        if procedural_memory:
            self.logger.info(f"\n--- Learned Procedural Memory ---\n{procedural_memory[:1000]}...")
        
        return procedural_memory
    
    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        allow_categories: List[int] = [1, 2, 3, 4, 5],
        max_samples: Optional[int] = None,
        storage_backend: str = "memory",
        qdrant_path: str = None,
        training_samples: Optional[List[Dict[str, Any]]] = None,  # For reflection mode
    ) -> Dict[str, Any]:
        """Evaluate full dataset with per-sample and aggregated metrics.
        
        For reflection mode:
        - If training_samples provided, use them for reflection training first
        - Then run inference on the main samples with learned procedural memory
        """
        all_results = []
        
        if max_samples:
            samples = samples[:max_samples]
        
        # === REFLECTION MODE: Two-Stage Process ===
        if self.meta_thinker_mode == "reflection" and self.meta_thinker:
            if training_samples:
                # Stage 1: Training - Learn from training samples
                self.logger.info(f"\n*** REFLECTION MODE: Using {len(training_samples)} training sample(s) ***")
                self.run_reflection_training(
                    training_samples=training_samples,
                    allow_categories=allow_categories,
                    storage_backend=storage_backend,
                    qdrant_path=qdrant_path,
                )
                self.logger.info(f"\n*** REFLECTION MODE: Now running inference on {len(samples)} test sample(s) ***\n")
            else:
                # No training samples provided - use first sample for training, rest for inference
                if len(samples) > 1:
                    self.logger.info(f"\n*** REFLECTION MODE: No training samples provided ***")
                    self.logger.info(f"*** Using sample 0 for training, samples 1-{len(samples)-1} for inference ***")
                    self.run_reflection_training(
                        training_samples=[samples[0]],
                        allow_categories=allow_categories,
                        storage_backend=storage_backend,
                        qdrant_path=qdrant_path,
                    )
                    samples = samples[1:]  # Remove training sample from test set
                    self.logger.info(f"\n*** REFLECTION MODE: Now running inference on {len(samples)} test sample(s) ***\n")
                else:
                    self.logger.warning("Only 1 sample available - running inference without training")
        
        # Track running metrics for per-sample output
        running_metrics = defaultdict(list)
        metric_names = ['exact_match', 'containment', 'token_f1', 'accuracy']
        
        for sample_idx, sample in enumerate(tqdm(samples, desc="Evaluating (Inference)")):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"[Inference] Sample {sample_idx + 1}/{len(samples)}")
            self.logger.info(f"{'='*60}")
            
            result = self.evaluate_sample(
                sample, 
                allow_categories,
                storage_backend=storage_backend,
                qdrant_path=qdrant_path,
            )
            all_results.append(result)
            
            # Collect metrics from this sample
            sample_metrics = defaultdict(list)
            for qa in result["qa_results"]:
                if "metrics" in qa:
                    for metric_name in metric_names:
                        val = qa["metrics"].get(metric_name)
                        if val is not None:
                            sample_metrics[metric_name].append(val)
                            running_metrics[metric_name].append(val)
            
            # Output per-sample metrics (mean ± std)
            self.logger.info(f"\n--- Sample {sample_idx + 1} Metrics ---")
            for metric_name in metric_names:
                vals = sample_metrics[metric_name]
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    self.logger.info(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(vals)})")
            
            # Output running aggregate metrics
            self.logger.info(f"\n--- Running Aggregate (Samples 1-{sample_idx + 1}) ---")
            for metric_name in metric_names:
                vals = running_metrics[metric_name]
                if vals:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    self.logger.info(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={len(vals)})")
        
        # Final aggregate metrics
        all_qa_results = []
        for r in all_results:
            all_qa_results.extend(r["qa_results"])
        
        aggregate = {}
        for metric_name in metric_names:
            values = [r['metrics'].get(metric_name) for r in all_qa_results if 'metrics' in r and r['metrics'].get(metric_name) is not None]
            if values:
                aggregate[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values),
                }
        
        # Per-category metrics
        category_metrics = defaultdict(lambda: defaultdict(list))
        for qa in all_qa_results:
            cat = qa.get('category', 0)
            if 'metrics' in qa:
                for metric_name in metric_names:
                    val = qa['metrics'].get(metric_name)
                    if val is not None:
                        category_metrics[cat][metric_name].append(val)
        
        category_aggregate = {}
        for cat, metrics in category_metrics.items():
            category_aggregate[cat] = {}
            for metric_name, vals in metrics.items():
                if vals:
                    category_aggregate[cat][metric_name] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'count': len(vals),
                    }
        
        # Memory stats
        total_memory_size = sum(r["memory_construction"]["final_memory_size"] for r in all_results)
        avg_memory_size = total_memory_size / len(all_results) if all_results else 0
        
        return {
            "model": self.answer_llm.model,
            "answer_model": self.answer_model,
            "backbone_model": self.backbone_model,
            "mm_retrieval_mode": self.mm_retrieval_mode,
            "mm_max_turns": self.mm_max_turns,
            "qr_max_turns": self.qr_max_turns,
            "total_samples": len(all_results),
            "total_questions": len(all_qa_results),
            "avg_memory_size": avg_memory_size,
            "aggregate_metrics": aggregate,
            "category_metrics": category_aggregate,
            "sample_results": all_results,
        }


# ==============================================================================
# Metrics
# ==============================================================================

# LLM Judge prompt from LightMem
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(
    question: str,
    gold_answer: str, 
    generated_answer: str,
    client: Optional[Any] = None,
    model_name: str = "gpt-4o-mini",
) -> int:
    """Evaluate the generated answer against the gold answer using an LLM judge.
    
    Args:
        question: the question string
        gold_answer: the ground-truth answer string
        generated_answer: the model's generated answer string
        client: OpenAI client instance
        model_name: model to use for judging (default gpt-4o-mini)
        
    Returns:
        1 if CORRECT, 0 if WRONG
    """
    import re
    
    if client is None:
        if not HAS_OPENAI:
            return 0  # Can't evaluate without OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": ACCURACY_PROMPT.format(
                        question=question, 
                        gold_answer=gold_answer, 
                        generated_answer=generated_answer
                    ),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        # Extract JSON from response
        text = response.choices[0].message.content.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text
        
        result = json.loads(json_str)
        label = result.get("label", "WRONG").upper()
        return 1 if label == "CORRECT" else 0
        
    except Exception as e:
        logging.warning(f"LLM judge failed: {e}")
        return 0


def calculate_simple_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate simple string-based metrics (no LLM judge)."""
    pred_lower = str(prediction).lower().strip() if prediction else ""
    ref_lower = str(reference).lower().strip() if reference else ""
    
    exact_match = 1.0 if pred_lower == ref_lower else 0.0
    containment = 1.0 if (ref_lower in pred_lower or pred_lower in ref_lower) else 0.0
    
    pred_tokens = set(pred_lower.split())
    ref_tokens = set(ref_lower.split())
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        f1 = 0.0
    else:
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "exact_match": exact_match,
        "containment": containment,
        "token_f1": f1,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Memory Cycle Basewe on LoCoMo")
    parser.add_argument("--dataset", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--output_dir", type=str, default="results/multi_agent_baseline", help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Legacy fallback model name")
    parser.add_argument("--backbone_model", type=str, default="", help="Backbone model for memory manager, query rewriter/reasoner, and meta-thinker")
    parser.add_argument("--answer_model", type=str, default="gpt-4o-mini", help="Answer agent and LLM judge model")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI base URL")
    parser.add_argument("--region", type=str, default="us-west-2", help="AWS region for Bedrock Claude models")
    
    # Memory Manager settings
    parser.add_argument("--mm_retrieval_mode", type=str, default="time", choices=["time", "similarity"],
                        help="Memory Manager retrieval mode: 'time' (recent) or 'similarity' (vector)")
    parser.add_argument("--mm_retrieve_k", type=int, default=10, help="Number of memories to retrieve for MM context")
    parser.add_argument("--mm_max_turns", type=int, default=5, help="Maximum Memory Manager turns per chunk")
    
    # Query Rewriter settings
    parser.add_argument("--qr_retrieve_k", type=int, default=10, help="Number of memories to retrieve for QR")
    parser.add_argument("--qr_max_turns", type=int, default=3, help="Maximum Query Rewriter turns per question")
    parser.add_argument("--qr_mode", type=str, default="simple", choices=["simple", "reasoning"],
                        help="Query Rewriter mode: 'simple' (ANSWERABLE/REWRITE) or 'reasoning' (with EXPAND)")
    
    # Dataset settings
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--parser", type=str, default="amem", choices=["simple", "amem"],
                        help="Dataset parser to use")
    parser.add_argument("--ratio", type=float, default=1.0, help="Ratio of samples to evaluate")
    parser.add_argument("--categories", type=str, default="1,2,3,4,5",
                        help="Comma-separated list of categories to evaluate")
    
    # Storage backend settings
    parser.add_argument("--storage_backend", type=str, default="memory", choices=["memory", "qdrant"],
                        help="Storage backend: 'memory' (in-memory numpy) or 'qdrant' (vector DB)")
    parser.add_argument("--qdrant_path", type=str, default=None,
                        help="Path for local Qdrant storage (only used when storage_backend='qdrant')")
    
    # Meta-Thinker settings
    parser.add_argument("--meta_thinker_mode", type=str, default="none", 
                        choices=["none", "vanilla", "reflection"],
                        help="Meta-Thinker mode: 'none' (baseline), 'vanilla' (real-time guidance), 'reflection' (with procedural memory)")
    parser.add_argument("--reflection_test_samples", type=int, default=1,
                        help="Number of samples to use for reflection testing (default: 1). Only used when meta_thinker_mode='reflection'")
    
    args = parser.parse_args()
    
    # Setup output directory and logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, "multi_agent_cycle")
    backbone_model = args.backbone_model or args.model
    answer_model = args.answer_model or args.model
    
    logger.info(f"Running Multi-Agent Memory Cycle Baseline")
    logger.info(f"Backbone Model: {backbone_model}")
    logger.info(f"Answer/Judge Model: {answer_model}")
    logger.info(f"Storage Backend: {args.storage_backend}")
    if args.storage_backend == "qdrant" and args.qdrant_path:
        logger.info(f"Qdrant Path: {args.qdrant_path}")
    logger.info(f"Meta-Thinker Mode: {args.meta_thinker_mode}")
    logger.info(f"MM Retrieval Mode: {args.mm_retrieval_mode}")
    logger.info(f"MM Max Turns: {args.mm_max_turns}")
    logger.info(f"QR Max Turns: {args.qr_max_turns}")
    logger.info(f"QR Mode: {args.qr_mode}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    if args.parser == "amem":
        samples_amem = load_locomo_dataset(args.dataset)
        # Convert to dict format
        samples = []
        for s in samples_amem:
            sample_dict = {
                'sample_id': s.sample_id,
                'conversation': {},
                'qa': []
            }
            sample_dict['conversation']['speaker_a'] = s.conversation.speaker_a
            sample_dict['conversation']['speaker_b'] = s.conversation.speaker_b
            for sess_id, sess in s.conversation.sessions.items():
                sess_key = f'session_{sess_id}'
                sample_dict['conversation'][sess_key] = [
                    {'speaker': t.speaker, 'dia_id': t.dia_id, 'text': t.text}
                    for t in sess.turns
                ]
                sample_dict['conversation'][f'{sess_key}_date_time'] = sess.date_time
            for qa in s.qa:
                sample_dict['qa'].append({
                    'question': qa.question,
                    'answer': qa.final_answer,
                    'category': qa.category,
                    'evidence': qa.evidence,  # Required for turn->QA mapping
                })
            samples.append(sample_dict)
    else:
        samples = parse_locomo_dataset(args.dataset)
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Select subset of samples based on ratio
    if args.ratio < 1.0:
        num_samples = max(1, int(len(samples) * args.ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({args.ratio*100:.1f}% of dataset)")
    
    
    # Parse categories
    allow_categories = [int(c) for c in args.categories.split(',')]
    logger.info(f"Evaluating categories: {allow_categories}")
    
    # Create evaluator
    evaluator = MultiAgentCycleEvaluator(
        model=answer_model,
        backbone_model=backbone_model,
        answer_model=answer_model,
        region=args.region,
        api_key=args.api_key,
        base_url=args.base_url,
        mm_retrieval_mode=args.mm_retrieval_mode,
        mm_retrieve_k=args.mm_retrieve_k,
        mm_max_turns=args.mm_max_turns,
        qr_retrieve_k=args.qr_retrieve_k,
        qr_max_turns=args.qr_max_turns,
        meta_thinker_mode=args.meta_thinker_mode,
        qr_mode=args.qr_mode,
        logger=logger,
    )
    
    # For reflection mode: separate training and test samples
    # training_samples = None
    # if args.meta_thinker_mode == "reflection" and len(samples) > args.reflection_train_samples:
    #     train_count = args.reflection_train_samples
    #     training_samples = samples[:train_count]
    #     samples = samples[train_count:]  # Remaining for inference
    #     logger.info(f"Reflection mode: {train_count} training sample(s), {len(samples)} test sample(s)")
    
    training_samples = None
    if args.meta_thinker_mode == "reflection" and len(samples) > args.reflection_test_samples:
        test_count = args.reflection_test_samples
        train_count = len(samples) - test_count
        training_samples = samples[:train_count]
        samples = samples[train_count:]  # Remaining for inference
        logger.info(f"Reflection mode: {train_count} training sample(s), {test_count} test sample(s)")
    

    # Run evaluation
    results = evaluator.evaluate_dataset(
        samples=samples,
        allow_categories=allow_categories,
        max_samples=args.max_samples,
        storage_backend=args.storage_backend,
        qdrant_path=args.qdrant_path,
        training_samples=training_samples,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Backbone Model: {backbone_model}")
    logger.info(f"Answer/Judge Model: {answer_model}")
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Total questions: {results['total_questions']}")
    logger.info(f"Avg memory size: {results['avg_memory_size']:.1f}")
    
    logger.info("\n--- Aggregate Metrics (All Questions) ---")
    for metric, values in results['aggregate_metrics'].items():
        logger.info(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f} (n={values['count']})")
    
    logger.info("\n--- Per-Category Metrics ---")
    if 'category_metrics' in results:
        for cat in sorted(results['category_metrics'].keys()):
            cat_metrics = results['category_metrics'][cat]
            logger.info(f"  Category {cat}:")
            for metric, values in cat_metrics.items():
                logger.info(f"    {metric}: {values['mean']:.4f} ± {values['std']:.4f} (n={values['count']})")
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
