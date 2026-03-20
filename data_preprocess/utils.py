from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

try:
    import tiktoken
except Exception:
    tiktoken = None
    

@dataclass
class QA:
    """QA pair from LoCoMo dataset."""
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer

@dataclass
class Turn:
    """A single turn in a conversation."""
    speaker: str
    dia_id: str
    text: str

@dataclass
class Session:
    """A session containing multiple turns."""
    session_id: int
    date_time: str
    turns: List[Turn]

@dataclass
class Conversation:
    """Full conversation with multiple sessions."""
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]

@dataclass
class EventSummary:
    """Event summary from LoCoMo."""
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events

@dataclass
class Observation:
    """Observations from LoCoMo."""
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]

@dataclass
class LoCoMoSample:
    """A single sample from the LoComo dataset (A-mem format)."""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]


def parse_locomo_dataset(data_path: str) -> List[Dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = {
            'sample_id': item['sample_id'],
            'conversation': item['conversation'],
            'qa': []
        }

        for qa_item in item.get('qa', []):
            answer = qa_item.get('answer') or qa_item.get('adversarial_answer', '')
            sample['qa'].append({
                'question': qa_item['question'],
                'answer': answer,
                'category': qa_item['category']
            })

        samples.append(sample)

    return samples

def parse_conversation_amem(conv_data: dict) -> Conversation:
    """Parse conversation data (A-mem format)."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time", "")
            if date_time:
                session = parse_session_amem(value, session_id, date_time)
                if session.turns:
                    sessions[session_id] = session
    
    return Conversation(
        speaker_a=conv_data.get("speaker_a", "Speaker_A"),
        speaker_b=conv_data.get("speaker_b", "Speaker_B"),
        sessions=sessions
    )

def parse_session_amem(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
            
        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)



def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file, including image-based content by using captions.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []
    
    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0
            
            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    # Check if QA has image evidence
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break
                    
                    if has_image_evidence:
                        sample_image_qa_count += 1
                        
                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1
                    
                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
            
            # Parse conversation
            conversation = parse_conversation_amem(sample["conversation"])
            
            # Parse event summary
            event_summary = EventSummary(events=sample["event_summary"])
            
            # Parse observation
            observation = Observation(observations=sample["observation"])
            
            # Get session summary
            session_summary = sample.get("session_summary", {})
            
            # Create sample object
            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)
            
            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)
            
            # Print statistics for this sample
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")
    
    return samples

