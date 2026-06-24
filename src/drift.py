"""
Adaptive Persona Engine — Per-Day Mood Drift & Trigger Detection
================================================================
Part 1 of Round 2.

Tracks how user mood/tone changes ACROSS DAYS (conversation_index as day proxy),
detects drift points using adaptive thresholds, and attributes triggers.

Dependencies: vaderSentiment (rule-based, no network calls)
"""

import json
import os
import re
import math
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Mood prototypes — each label maps to an expected mood-vector direction
# ---------------------------------------------------------------------------
MOOD_PROTOTYPES = {
    "curious": {"curiosity": 0.8, "valence": 0.3, "playfulness": 0.2},
    "formal": {"formality": 0.8, "avg_length": 0.6, "curiosity": 0.3},
    "casual": {"formality": -0.8, "playfulness": 0.4, "emoji_rate": 0.4},
    "frustrated": {"frustration": 0.7, "valence": -0.5, "intensity": 0.6},
    "playful": {"playfulness": 0.8, "emoji_rate": 0.5, "valence": 0.4},
    "enthusiastic": {"valence": 0.8, "intensity": 0.7, "playfulness": 0.3},
    "reserved": {"formality": 0.5, "avg_length": -0.4, "intensity": -0.3},
    "anxious": {"frustration": 0.4, "intensity": 0.6, "valence": -0.3, "curiosity": 0.3},
    "warm": {"valence": 0.6, "intensity": 0.4, "playfulness": 0.3, "emoji_rate": 0.3},
    "neutral": {"valence": 0.0, "intensity": -0.3, "formality": 0.1},
}

# Event keywords for trigger detection
EVENT_KEYWORDS = {
    "exam": ["exam", "test", "midterm", "final", "quiz", "assessment"],
    "interview": ["interview", "job interview", "hiring", "recruiter"],
    "trip": ["trip", "travel", "vacation", "holiday", "flight", "visit"],
    "birthday": ["birthday", "bday", "party", "celebration", "celebrate"],
    "deadline": ["deadline", "due date", "due tomorrow", "submission"],
    "breakup": ["breakup", "broke up", "break up", "ended things", "split"],
    "argument": ["argument", "fight", "argued", "disagreement", "heated"],
    "health": ["sick", "doctor", "hospital", "ill", "surgery", "injured"],
    "graduation": ["graduated", "graduation", "degree", "diploma"],
    "moving": ["moving", "moved", "relocating", "new apartment", "new house"],
    "promotion": ["promoted", "promotion", "raise", "new position"],
    "loss": ["lost", "passed away", "died", "funeral", "grief", "mourning"],
}

# Humor markers
HUMOR_MARKERS = [
    "lol", "lmao", "rofl", "haha", "hehe", "😂", "🤣", "😆",
    "lolol", "xd", "hahaha", "lmfao", "😜", "😝", "🤪",
]

# Stress / anger lexicon
STRESS_LEXICON = [
    "stressed", "angry", "furious", "annoyed", "irritated", "fed up",
    "frustrated", "overwhelmed", "exhausted", "burned out", "burnt out",
    "anxious", "worried", "nervous", "freaking out", "can't take",
    "sick of", "tired of", "pissed", "ugh", "argh", "damn",
    "hate", "terrible", "awful", "horrible", "miserable", "struggling",
]

# Slang / abbreviation markers (informality)
SLANG_MARKERS = [
    "gonna", "wanna", "gotta", "kinda", "sorta", "ya", "nah",
    "lol", "omg", "tbh", "imo", "brb", "btw", "idk", "ikr",
    "ngl", "rn", "smh", "nvm", "imho", "af", "irl", "fomo",
    "fyi", "tho", "thx", "u", "ur", "r", "plz", "pls",
]

# Person mention patterns
PERSON_PATTERNS = [
    r'\bmy\s+(sister|brother|mom|mother|dad|father|wife|husband|partner|'
    r'boyfriend|girlfriend|friend|boss|colleague|coworker|neighbor|'
    r'uncle|aunt|cousin|grandma|grandmother|grandpa|grandfather|'
    r'roommate|fiancee?|ex)\b',
    r'\b(sister|brother|mom|mother|dad|father)\s+(?:said|told|called|texted|asked|wants)\b',
]


class DayMoodAnalyzer:
    """Computes a mood vector for each day (conversation_index)."""

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

        # Compile patterns
        self._emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF'
            r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
            r'\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
            r'\U0000200D\U00002600-\U000026FF\U00002700-\U000027BF]'
        )
        self._humor_set = set(HUMOR_MARKERS)
        self._stress_set = set(STRESS_LEXICON)
        self._slang_set = set(SLANG_MARKERS)

    def group_messages_by_day(self, messages: List[Dict]) -> Dict[int, List[Dict]]:
        """Group messages by conversation_index (= day proxy)."""
        days = defaultdict(list)
        for msg in messages:
            days[msg['conversation_index']].append(msg)
        return dict(sorted(days.items()))

    def compute_mood_vector(self, day_messages: List[Dict]) -> Dict[str, float]:
        """Compute a 7-dimensional mood vector for a group of messages."""
        if not day_messages:
            return self._zero_vector()

        # Accumulators
        valences = []
        intensities = []
        frustration_hits = 0
        humor_hits = 0
        question_count = 0
        emoji_count = 0
        total_chars = 0
        total_words = 0
        formal_hits = 0
        informal_hits = 0
        n = len(day_messages)

        for msg in day_messages:
            text = msg.get('text', '')
            text_lower = text.lower()
            words = text_lower.split()
            word_set = set(words)

            # --- Sentiment via VADER ---
            scores = self.vader.polarity_scores(text)
            valences.append(scores['compound'])
            intensities.append(abs(scores['compound']))

            # --- Frustration ---
            frustration_hits += len(word_set & self._stress_set)
            if scores['neg'] > 0.3:
                frustration_hits += 1

            # --- Playfulness / humor ---
            humor_hits += len(word_set & self._humor_set)
            humor_hits += len(self._emoji_pattern.findall(text))

            # --- Formality ---
            formal_hits += sum(1 for w in words if len(w) > 8)  # long words
            formal_hits += text.count('.')  # periods → more formal
            informal_hits += len(word_set & self._slang_set)
            informal_hits += len(self._emoji_pattern.findall(text))
            if text.upper() == text and len(text) > 3:
                informal_hits += 1  # ALL CAPS = informal/shouting

            # --- Curiosity ---
            question_count += text.count('?')

            # --- Emoji rate ---
            emoji_count += len(self._emoji_pattern.findall(text))

            # --- Length ---
            total_chars += len(text)
            total_words += len(words)

        # Normalise to [-1, 1] or [0, 1] ranges
        avg_valence = np.mean(valences) if valences else 0.0
        avg_intensity = np.mean(intensities) if intensities else 0.0

        frustration_score = min(frustration_hits / max(n, 1), 1.0)
        playfulness_score = min(humor_hits / max(n, 1), 1.0)

        # Formality: positive = formal, negative = informal
        formality_raw = (formal_hits - informal_hits) / max(n, 1)
        formality_score = max(min(formality_raw, 1.0), -1.0)

        curiosity_score = min(question_count / max(n, 1), 1.0)
        emoji_rate = min(emoji_count / max(n, 1), 1.0)

        avg_length = total_chars / max(n, 1)
        # Normalise avg_length to [0, 1] (cap at 200 chars)
        avg_length_norm = min(avg_length / 200.0, 1.0)

        return {
            "valence": round(float(avg_valence), 4),
            "intensity": round(float(avg_intensity), 4),
            "frustration": round(float(frustration_score), 4),
            "playfulness": round(float(playfulness_score), 4),
            "formality": round(float(formality_score), 4),
            "curiosity": round(float(curiosity_score), 4),
            "emoji_rate": round(float(emoji_rate), 4),
            "avg_length": round(float(avg_length_norm), 4),
        }

    def labels_from_vector(self, vec: Dict[str, float], top_n: int = 3) -> List[str]:
        """Map a mood vector to 1-3 descriptive labels using prototype matching."""
        scores = {}
        for label, prototype in MOOD_PROTOTYPES.items():
            score = 0.0
            for dim, weight in prototype.items():
                val = vec.get(dim, 0.0)
                score += val * weight
            scores[label] = score

        # Sort by score descending, take labels above a threshold
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        # Adaptive threshold: take labels with score > 0.15
        labels = [lbl for lbl, sc in ranked if sc > 0.15]

        if not labels:
            labels = [ranked[0][0]]  # At least one label

        return labels[:top_n]

    def _zero_vector(self) -> Dict[str, float]:
        return {
            "valence": 0.0, "intensity": 0.0, "frustration": 0.0,
            "playfulness": 0.0, "formality": 0.0, "curiosity": 0.0,
            "emoji_rate": 0.0, "avg_length": 0.0,
        }


class DriftDetector:
    """Detects mood drift between consecutive days using adaptive thresholds."""

    def __init__(self, k: float = 1.5):
        """
        Args:
            k: multiplier for standard deviation in adaptive threshold.
               Higher k = fewer drifts detected. Default 1.5.
        """
        self.k = k

    def detect_drifts(self, timeline: List[Dict]) -> List[Dict]:
        """
        Given a list of day entries (with mood_scores), compute drifts.
        Returns the same timeline with drift_from_prev field set.
        """
        if len(timeline) < 2:
            if timeline:
                timeline[0]['drift_from_prev'] = False
            return timeline

        # Compute all pairwise distances
        distances = []
        dims = ["valence", "intensity", "frustration", "playfulness",
                "formality", "curiosity", "emoji_rate", "avg_length"]

        for i in range(1, len(timeline)):
            vec_prev = timeline[i - 1]['mood_scores']
            vec_curr = timeline[i]['mood_scores']
            dist = self._euclidean_distance(vec_prev, vec_curr, dims)
            distances.append(dist)

        # Adaptive threshold: mean + k * std
        if distances:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = mean_dist + self.k * std_dist
        else:
            threshold = 0.5

        # Mark drifts
        timeline[0]['drift_from_prev'] = False
        timeline[0]['drift_distance'] = 0.0
        timeline[0]['drift_threshold'] = round(float(threshold), 4)

        for i in range(1, len(timeline)):
            dist = distances[i - 1]
            label_changed = (
                timeline[i]['mood_labels'][0] != timeline[i - 1]['mood_labels'][0]
            )
            is_drift = dist > threshold or label_changed

            timeline[i]['drift_from_prev'] = bool(is_drift)
            timeline[i]['drift_distance'] = round(float(dist), 4)
            timeline[i]['drift_threshold'] = round(float(threshold), 4)

        return timeline

    def _euclidean_distance(self, v1: Dict, v2: Dict, dims: List[str]) -> float:
        total = 0.0
        for d in dims:
            diff = v1.get(d, 0.0) - v2.get(d, 0.0)
            total += diff * diff
        return math.sqrt(total)


class TriggerAttributor:
    """Identifies the cause of each mood drift."""

    def __init__(self, topic_checkpoints: List[Dict] = None):
        self.topic_checkpoints = topic_checkpoints or []
        self._person_patterns = [re.compile(p, re.IGNORECASE) for p in PERSON_PATTERNS]

    def attribute_trigger(self, day_idx: int, day_messages: List[Dict],
                          all_days: Dict[int, List[Dict]],
                          prev_day_idx: Optional[int] = None) -> Optional[Dict]:
        """
        For a drift day, identify the trigger.
        Returns: {type: "topic"|"event"|"person", value: str, evidence: [{idx, quote}]}
        """
        if not day_messages:
            return None

        # Collect all text from this day
        texts = [(m.get('global_index', 0), m.get('text', '')) for m in day_messages]

        # 1. Check for EVENT triggers (highest priority)
        event_trigger = self._find_event_trigger(texts)
        if event_trigger:
            return event_trigger

        # 2. Check for PERSON triggers (new person mention)
        person_trigger = self._find_person_trigger(texts, day_idx, all_days, prev_day_idx)
        if person_trigger:
            return person_trigger

        # 3. Check for TOPIC triggers (new topic starting near this day)
        topic_trigger = self._find_topic_trigger(day_idx, day_messages)
        if topic_trigger:
            return topic_trigger

        # 4. Fallback: content-based trigger
        return self._find_content_trigger(texts)

    def _find_event_trigger(self, texts: List[Tuple[int, str]]) -> Optional[Dict]:
        """Check for event keywords in day messages."""
        for event_type, keywords in EVENT_KEYWORDS.items():
            evidence = []
            for idx, text in texts:
                text_lower = text.lower()
                if any(kw in text_lower for kw in keywords):
                    evidence.append({"idx": idx, "quote": text[:200]})
            if evidence:
                return {
                    "type": "event",
                    "value": event_type,
                    "evidence": evidence[:3]
                }
        return None

    def _find_person_trigger(self, texts: List[Tuple[int, str]],
                             day_idx: int, all_days: Dict[int, List[Dict]],
                             prev_day_idx: Optional[int]) -> Optional[Dict]:
        """Check for new person mentions not in previous day."""
        current_persons = set()
        evidence = []

        for idx, text in texts:
            for pattern in self._person_patterns:
                matches = pattern.findall(text)
                for match in matches:
                    person = match.lower().strip()
                    current_persons.add(person)
                    evidence.append({"idx": idx, "quote": text[:200]})

        if not current_persons:
            return None

        # Check if person is NEW (not in previous day)
        prev_persons = set()
        if prev_day_idx is not None and prev_day_idx in all_days:
            for msg in all_days[prev_day_idx]:
                for pattern in self._person_patterns:
                    matches = pattern.findall(msg.get('text', ''))
                    for match in matches:
                        prev_persons.add(match.lower().strip())

        new_persons = current_persons - prev_persons
        if new_persons:
            return {
                "type": "person",
                "value": ", ".join(sorted(new_persons)),
                "evidence": evidence[:3]
            }

        return None

    def _find_topic_trigger(self, day_idx: int, day_messages: List[Dict]) -> Optional[Dict]:
        """Check if a new topic checkpoint starts near this day."""
        if not self.topic_checkpoints or not day_messages:
            return None

        msg_indices = {m.get('global_index', 0) for m in day_messages}
        min_idx = min(msg_indices) if msg_indices else 0
        max_idx = max(msg_indices) if msg_indices else 0

        for tc in self.topic_checkpoints:
            tc_start = tc.get('start_msg_index', 0)
            # Topic starts within or near this day's message range
            if min_idx <= tc_start <= max_idx + 5:
                return {
                    "type": "topic",
                    "value": tc.get('topic_label', 'Unknown Topic'),
                    "evidence": [{
                        "idx": tc_start,
                        "quote": tc.get('summary', '')[:200]
                    }]
                }
        return None

    def _find_content_trigger(self, texts: List[Tuple[int, str]]) -> Optional[Dict]:
        """Fallback: use most distinctive content as trigger."""
        if not texts:
            return None

        # Find the message with highest absolute sentiment
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
            best_msg = None
            best_score = 0
            for idx, text in texts:
                score = abs(vader.polarity_scores(text)['compound'])
                if score > best_score:
                    best_score = score
                    best_msg = (idx, text)

            if best_msg:
                return {
                    "type": "content",
                    "value": "strong emotional expression",
                    "evidence": [{"idx": best_msg[0], "quote": best_msg[1][:200]}]
                }
        except Exception:
            pass

        return {
            "type": "content",
            "value": "general tone shift",
            "evidence": [{"idx": texts[0][0], "quote": texts[0][1][:200]}]
        }


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
def build_mood_timeline(messages: List[Dict],
                        topic_checkpoints: List[Dict] = None,
                        max_days: int = 500,
                        k: float = 1.5) -> List[Dict]:
    """
    Build the complete mood timeline with drift detection and trigger attribution.

    Args:
        messages: list of message dicts (with conversation_index, text, global_index)
        topic_checkpoints: list of topic checkpoint dicts
        max_days: cap on number of days to process (for performance)
        k: drift threshold multiplier

    Returns:
        List of day entries with mood_labels, mood_scores, drift_from_prev, trigger
    """
    print("[Drift] Building mood timeline...")

    analyzer = DayMoodAnalyzer()
    detector = DriftDetector(k=k)
    attributor = TriggerAttributor(topic_checkpoints=topic_checkpoints)

    # Group by day
    days = analyzer.group_messages_by_day(messages)
    day_keys = sorted(days.keys())[:max_days]

    print(f"[Drift] Processing {len(day_keys)} days (capped at {max_days})...")

    # Build timeline
    timeline = []
    for day_idx in day_keys:
        day_msgs = days[day_idx]
        mood_vec = analyzer.compute_mood_vector(day_msgs)
        labels = analyzer.labels_from_vector(mood_vec)

        timeline.append({
            "day": day_idx,
            "message_count": len(day_msgs),
            "mood_labels": labels,
            "mood_scores": mood_vec,
            "drift_from_prev": False,
            "trigger": None,
        })

    # Detect drifts
    timeline = detector.detect_drifts(timeline)

    # Attribute triggers for drift days
    drift_count = 0
    for i, entry in enumerate(timeline):
        if entry['drift_from_prev']:
            day_idx = entry['day']
            prev_day_idx = timeline[i - 1]['day'] if i > 0 else None
            trigger = attributor.attribute_trigger(
                day_idx, days[day_idx], days, prev_day_idx
            )
            entry['trigger'] = trigger
            drift_count += 1

    print(f"[Drift] Timeline complete: {len(timeline)} days, {drift_count} drift points detected")
    return timeline


def save_timeline(timeline: List[Dict], output_path: str):
    """Save timeline to JSON file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    print(f"[Drift] Saved timeline to {output_path}")


def load_timeline(path: str) -> List[Dict]:
    """Load timeline from JSON file."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []
