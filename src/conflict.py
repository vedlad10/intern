"""
Conflict Resolution in RAG — Entity-Aware Retrieval & Contradiction Detection
==============================================================================
Part 3 of Round 2.

Handles queries where an entity appears across multiple topic checkpoints
with contradictory context. Retrieves, ranks, detects contradictions, and
merges into a coherent answer.

Dependencies: vaderSentiment (already required by drift.py)
"""

import json
import os
import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# Negation cues for contradiction detection
NEGATION_CUES = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "can't", "cannot", "couldn't", "won't", "wouldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "don't",
    "doesn't", "didn't", "hasn't", "haven't", "hadn't",
}

# Contradiction verb pairs
CONTRADICTION_VERB_PAIRS = [
    ("coming", "cancelled"),
    ("visit", "cancelled"),
    ("excited", "disappointed"),
    ("happy", "sad"),
    ("love", "hate"),
    ("together", "apart"),
    ("agree", "disagree"),
    ("accept", "reject"),
    ("start", "stop"),
    ("begin", "end"),
    ("open", "close"),
    ("arrive", "leave"),
    ("stay", "go"),
    ("join", "quit"),
    ("win", "lose"),
    ("improve", "worsen"),
    ("like", "dislike"),
    ("support", "oppose"),
    ("friend", "enemy"),
    ("peace", "argument"),
    ("calm", "angry"),
    ("trust", "distrust"),
    ("planning", "cancelled"),
]


class ConflictScenarioLoader:
    """Loads and manages injected conflict scenarios."""

    def __init__(self, scenario_path: str = "data/conflict_scenarios.json"):
        self.scenario_path = scenario_path
        self.scenarios = []
        self.injected_messages = []
        self.injected_topics = []

    def load(self) -> Tuple[List[Dict], List[Dict]]:
        """Load conflict scenarios and return (messages, topic_checkpoints)."""
        if not os.path.exists(self.scenario_path):
            print(f"[Conflict] No scenario file at {self.scenario_path}")
            return [], []

        with open(self.scenario_path, 'r', encoding='utf-8') as f:
            self.scenarios = json.load(f)

        for scenario in self.scenarios:
            self.injected_messages.extend(scenario.get('messages', []))
            self.injected_topics.extend(scenario.get('topic_checkpoints', []))

        print(f"[Conflict] Loaded {len(self.scenarios)} conflict scenarios "
              f"({len(self.injected_messages)} messages, {len(self.injected_topics)} topics)")

        return self.injected_messages, self.injected_topics


class EntityExtractor:
    """Extracts target entity from a user query."""

    # Common entity reference patterns
    ENTITY_PATTERNS = [
        r'(?:about|mention|said about|regarding|related to)\s+(?:my\s+)?(\w+(?:\s+\w+)?)',
        r'(?:my|the)\s+(\w+)',
        r'(?:did i|have i|was there)\s+\w+\s+(?:about|regarding)\s+(?:my\s+)?(\w+)',
    ]

    # Common relationship words to recognize
    ENTITY_WORDS = {
        "sister", "brother", "mom", "mother", "dad", "father",
        "wife", "husband", "partner", "friend", "boss", "colleague",
        "dog", "cat", "pet", "car", "house", "job", "school",
        "doctor", "teacher", "neighbor", "cousin", "uncle", "aunt",
        "grandmother", "grandfather", "boyfriend", "girlfriend",
        "roommate", "coworker", "baby", "child", "children",
    }

    def extract_entity(self, query: str) -> Optional[str]:
        """Extract the main entity from a query string."""
        query_lower = query.lower().strip()

        # Try direct pattern matching
        for pattern in self.ENTITY_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                entity = match.group(1).strip()
                # Check if it's a meaningful entity
                if entity and len(entity) > 1 and entity not in {'it', 'that', 'this', 'i', 'me'}:
                    return entity

        # Try to find known entity words
        words = set(re.findall(r'\b\w+\b', query_lower))
        entity_matches = words & self.ENTITY_WORDS
        if entity_matches:
            return entity_matches.pop()

        # Fallback: find the last noun-like word that's not a stop word
        stop_words = {
            'did', 'i', 'my', 'me', 'you', 'the', 'a', 'an', 'is', 'are',
            'was', 'were', 'have', 'has', 'had', 'do', 'does', 'about',
            'anything', 'something', 'mention', 'mentioned', 'say', 'said',
            'tell', 'told', 'any', 'ever', 'there', 'what', 'how', 'when',
            'where', 'who', 'which', 'that', 'this', 'with', 'for', 'from',
        }
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in reversed(query_words):
            if word not in stop_words and len(word) > 2:
                return word

        return None


class EntityAwareRetriever:
    """
    Retrieves all mentions of an entity, grouped by topic checkpoint and day.
    Ranks by recency + emotional weight.
    """

    def __init__(self, recency_weight: float = 0.6, emotional_weight: float = 0.4):
        """
        Args:
            recency_weight: weight for recency scoring (0-1)
            emotional_weight: weight for emotional intensity scoring (0-1)
        """
        self.recency_weight = recency_weight
        self.emotional_weight = emotional_weight

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            self.vader = None

    def retrieve_entity_mentions(self, entity: str,
                                  messages: List[Dict],
                                  topic_checkpoints: List[Dict]) -> Dict:
        """
        Find all mentions of an entity across messages and topics.

        Returns:
            {
                "entity": str,
                "mentions": [...],  # ranked list of mentions
                "by_topic": {...},  # grouped by topic
                "by_day": {...},    # grouped by day
                "total_mentions": int
            }
        """
        entity_lower = entity.lower()
        entity_pattern = re.compile(r'\b' + re.escape(entity_lower) + r'\b', re.IGNORECASE)

        # Find mentions in messages
        msg_mentions = []
        for msg in messages:
            text = msg.get('text', '') or msg.get('full_text', '')
            if entity_pattern.search(text):
                mention = {
                    "global_index": msg.get('global_index', 0),
                    "conversation_index": msg.get('conversation_index', 0),
                    "speaker": msg.get('speaker', 'Unknown'),
                    "text": text,
                    "day_label": msg.get('day_label', f"Conv {msg.get('conversation_index', '?')}"),
                    "source": "message",
                }
                # Add sentiment
                if self.vader:
                    scores = self.vader.polarity_scores(text)
                    mention["sentiment"] = {
                        "compound": round(scores['compound'], 4),
                        "positive": round(scores['pos'], 4),
                        "negative": round(scores['neg'], 4),
                    }
                    mention["sentiment_magnitude"] = abs(scores['compound'])
                else:
                    mention["sentiment"] = {"compound": 0, "positive": 0, "negative": 0}
                    mention["sentiment_magnitude"] = 0

                msg_mentions.append(mention)

        # Find mentions in topic checkpoints
        topic_mentions = []
        for tc in topic_checkpoints:
            summary = tc.get('summary', '')
            label = tc.get('topic_label', '')
            entities_list = [e.lower() for e in tc.get('key_entities', [])]

            if (entity_pattern.search(summary) or
                entity_pattern.search(label) or
                entity_lower in entities_list):

                # Also check messages within this topic
                tc_msg_texts = []
                for m in tc.get('messages', []):
                    if entity_pattern.search(m.get('text', '')):
                        tc_msg_texts.append(m)

                topic_mention = {
                    "topic_id": tc.get('topic_id'),
                    "topic_label": tc.get('topic_label', ''),
                    "summary": summary,
                    "start_msg_index": tc.get('start_msg_index', 0),
                    "end_msg_index": tc.get('end_msg_index', 0),
                    "day_label": tc.get('day_label', f"Topic {tc.get('topic_id', '?')}"),
                    "relevant_messages": tc_msg_texts[:5],
                    "source": "topic",
                }
                # Sentiment of summary
                if self.vader:
                    scores = self.vader.polarity_scores(summary)
                    topic_mention["sentiment"] = {
                        "compound": round(scores['compound'], 4),
                        "positive": round(scores['pos'], 4),
                        "negative": round(scores['neg'], 4),
                    }
                    topic_mention["sentiment_magnitude"] = abs(scores['compound'])
                else:
                    topic_mention["sentiment"] = {"compound": 0, "positive": 0, "negative": 0}
                    topic_mention["sentiment_magnitude"] = 0

                topic_mentions.append(topic_mention)

        # Combine and rank
        all_mentions = msg_mentions + topic_mentions
        all_mentions = self._rank_mentions(all_mentions)

        # Group by topic
        by_topic = defaultdict(list)
        for m in topic_mentions:
            by_topic[m.get('topic_label', 'Unknown')].append(m)

        # Group by day
        by_day = defaultdict(list)
        for m in msg_mentions:
            by_day[m.get('day_label', 'Unknown')].append(m)

        return {
            "entity": entity,
            "mentions": all_mentions,
            "by_topic": dict(by_topic),
            "by_day": dict(by_day),
            "total_mentions": len(all_mentions),
            "weights": {
                "recency_weight": self.recency_weight,
                "emotional_weight": self.emotional_weight,
            }
        }

    def _rank_mentions(self, mentions: List[Dict]) -> List[Dict]:
        """Rank mentions by combined recency + emotional weight score."""
        if not mentions:
            return mentions

        # Find max index for recency normalization
        max_idx = max(
            m.get('global_index', m.get('end_msg_index', 0))
            for m in mentions
        ) or 1

        for m in mentions:
            idx = m.get('global_index', m.get('end_msg_index', 0))
            recency = idx / max(max_idx, 1)  # 0 = oldest, 1 = newest
            emotional = m.get('sentiment_magnitude', 0)

            combined = (self.recency_weight * recency +
                       self.emotional_weight * emotional)
            m['rank_score'] = round(combined, 4)
            m['recency_score'] = round(recency, 4)

        return sorted(mentions, key=lambda x: x['rank_score'], reverse=True)


class ContradictionDetector:
    """Detects contradictions between entity mentions."""

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            self.vader = None

        self._negation_set = NEGATION_CUES
        self._contradiction_pairs = CONTRADICTION_VERB_PAIRS

    def detect_contradictions(self, mentions: List[Dict]) -> List[Dict]:
        """
        Pairwise compare mentions and flag contradictions.

        Returns list of contradiction entries:
        {
            "mention_a": {...},
            "mention_b": {...},
            "contradiction_type": str,
            "explanation": str,
            "confidence": float
        }
        """
        contradictions = []
        seen_pairs = set()

        for i, a in enumerate(mentions):
            for j, b in enumerate(mentions):
                if i >= j:
                    continue

                pair_key = (
                    a.get('global_index', a.get('topic_id', i)),
                    b.get('global_index', b.get('topic_id', j))
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                contradiction = self._check_pair(a, b)
                if contradiction:
                    contradictions.append(contradiction)

        return contradictions

    def _check_pair(self, a: Dict, b: Dict) -> Optional[Dict]:
        """Check if two mentions contradict each other."""
        text_a = a.get('text', a.get('summary', ''))
        text_b = b.get('text', b.get('summary', ''))

        if not text_a or not text_b:
            return None

        checks = [
            self._check_sentiment_flip(a, b, text_a, text_b),
            self._check_negation_contradiction(text_a, text_b),
            self._check_verb_contradiction(text_a, text_b),
        ]

        # Take the strongest contradiction found
        best = None
        for check in checks:
            if check and (best is None or check['confidence'] > best['confidence']):
                best = check

        if best:
            best['mention_a'] = self._summarize_mention(a)
            best['mention_b'] = self._summarize_mention(b)
            return best

        return None

    def _check_sentiment_flip(self, a: Dict, b: Dict,
                               text_a: str, text_b: str) -> Optional[Dict]:
        """Check for sentiment polarity flip."""
        sent_a = a.get('sentiment', {}).get('compound', 0)
        sent_b = b.get('sentiment', {}).get('compound', 0)

        # If no pre-computed sentiment, compute it
        if sent_a == 0 and self.vader:
            sent_a = self.vader.polarity_scores(text_a)['compound']
        if sent_b == 0 and self.vader:
            sent_b = self.vader.polarity_scores(text_b)['compound']

        # Significant polarity flip
        if (sent_a > 0.3 and sent_b < -0.3) or (sent_a < -0.3 and sent_b > 0.3):
            flip_magnitude = abs(sent_a - sent_b)
            confidence = min(flip_magnitude / 2.0, 1.0)
            direction = "positive → negative" if sent_a > sent_b else "negative → positive"
            return {
                "contradiction_type": "sentiment_flip",
                "explanation": (f"Sentiment flipped from {direction} "
                               f"(scores: {sent_a:.2f} → {sent_b:.2f})"),
                "confidence": round(confidence, 3),
            }
        return None

    def _check_negation_contradiction(self, text_a: str, text_b: str) -> Optional[Dict]:
        """Check for negation-based contradiction."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        negations_in_b = words_b & self._negation_set
        negations_in_a = words_a & self._negation_set

        # One has negation, the other doesn't, with overlapping content
        content_a = words_a - self._negation_set - _STOP_WORDS
        content_b = words_b - self._negation_set - _STOP_WORDS

        shared_content = content_a & content_b

        if shared_content and len(shared_content) >= 2:
            if (negations_in_b and not negations_in_a) or \
               (negations_in_a and not negations_in_b):
                neg_words = negations_in_b or negations_in_a
                return {
                    "contradiction_type": "negation",
                    "explanation": (f"Negation detected ('{', '.join(neg_words)}') "
                                   f"with shared context: {', '.join(list(shared_content)[:5])}"),
                    "confidence": 0.7,
                }
        return None

    def _check_verb_contradiction(self, text_a: str, text_b: str) -> Optional[Dict]:
        """Check for contradicting action verbs / facts."""
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        for word1, word2 in self._contradiction_pairs:
            if ((word1 in text_a_lower and word2 in text_b_lower) or
                (word2 in text_a_lower and word1 in text_b_lower)):
                return {
                    "contradiction_type": "conflicting_facts",
                    "explanation": f"Conflicting terms found: '{word1}' vs '{word2}'",
                    "confidence": 0.8,
                }
        return None

    def _summarize_mention(self, mention: Dict) -> Dict:
        """Create a compact summary of a mention for the contradiction report."""
        return {
            "text": (mention.get('text', mention.get('summary', '')))[:200],
            "day_label": mention.get('day_label', '?'),
            "source": mention.get('source', 'unknown'),
            "sentiment": mention.get('sentiment', {}).get('compound', 0),
            "index": mention.get('global_index',
                                mention.get('topic_id',
                                           mention.get('start_msg_index', 0))),
        }


# Stop words for content comparison
_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into', 'through',
    'during', 'before', 'after', 'and', 'but', 'or', 'so', 'if', 'than',
    'that', 'this', 'these', 'those', 'i', 'me', 'my', 'you', 'your',
    'he', 'she', 'it', 'we', 'they', 'them', 'his', 'her', 'its', 'our',
    'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
}


class ConflictResolver:
    """
    Orchestrates entity-aware retrieval, contradiction detection,
    and merged answer generation.
    """

    def __init__(self, messages: List[Dict] = None,
                 topic_checkpoints: List[Dict] = None,
                 recency_weight: float = 0.6,
                 emotional_weight: float = 0.4):
        self.messages = messages or []
        self.topic_checkpoints = topic_checkpoints or []
        self.recency_weight = recency_weight
        self.emotional_weight = emotional_weight

        self.entity_extractor = EntityExtractor()
        self.retriever = EntityAwareRetriever(
            recency_weight=recency_weight,
            emotional_weight=emotional_weight
        )
        self.contradiction_detector = ContradictionDetector()

        # Load and merge conflict scenarios
        self.scenario_loader = ConflictScenarioLoader()
        self._merge_scenarios()

    def _merge_scenarios(self):
        """Merge injected conflict scenarios into messages and topics."""
        injected_msgs, injected_topics = self.scenario_loader.load()
        if injected_msgs:
            self.messages = self.messages + injected_msgs
            print(f"[Conflict] Merged {len(injected_msgs)} scenario messages")
        if injected_topics:
            self.topic_checkpoints = self.topic_checkpoints + injected_topics
            print(f"[Conflict] Merged {len(injected_topics)} scenario topics")

    def resolve(self, query: str) -> Dict:
        """
        Full conflict resolution pipeline.

        Args:
            query: user query (e.g., "Did I mention anything about my sister?")

        Returns:
            {
                "query": str,
                "entity": str,
                "retrieval": {...},  # entity mentions grouped & ranked
                "contradictions": [...],  # detected contradictions
                "merged_answer": str,  # coherent merged answer
            }
        """
        # 1. Extract entity
        entity = self.entity_extractor.extract_entity(query)
        if not entity:
            return {
                "query": query,
                "entity": None,
                "error": "Could not identify a specific entity in the query.",
                "merged_answer": "I couldn't identify which entity you're asking about. "
                                "Try asking about a specific person, thing, or topic.",
            }

        print(f"[Conflict] Resolving entity: '{entity}'")

        # 2. Retrieve all mentions
        retrieval = self.retriever.retrieve_entity_mentions(
            entity, self.messages, self.topic_checkpoints
        )

        if retrieval['total_mentions'] == 0:
            return {
                "query": query,
                "entity": entity,
                "retrieval": retrieval,
                "contradictions": [],
                "merged_answer": f"I couldn't find any mentions of '{entity}' "
                                f"in the conversation history.",
            }

        # 3. Detect contradictions
        contradictions = self.contradiction_detector.detect_contradictions(
            retrieval['mentions']
        )

        # 4. Generate merged answer
        merged_answer = self._generate_merged_answer(
            entity, retrieval, contradictions
        )

        return {
            "query": query,
            "entity": entity,
            "retrieval": retrieval,
            "contradictions": contradictions,
            "merged_answer": merged_answer,
        }

    def _generate_merged_answer(self, entity: str,
                                 retrieval: Dict,
                                 contradictions: List[Dict]) -> str:
        """Generate a coherent merged answer from retrieved mentions and contradictions."""
        parts = []

        # (a) Confirm entity was mentioned
        n_mentions = retrieval['total_mentions']
        parts.append(f"## '{entity.title()}' — Conflict Resolution\n")
        parts.append(f"Yes, **{entity}** was mentioned **{n_mentions} times** "
                    f"across the conversation history.\n")

        # (b) Most recent / emotionally weighted status
        top_mention = retrieval['mentions'][0] if retrieval['mentions'] else None
        if top_mention:
            parts.append("### Most Relevant Mention (by recency + emotional weight)")
            text = top_mention.get('text', top_mention.get('summary', ''))[:300]
            day = top_mention.get('day_label', '?')
            score = top_mention.get('rank_score', 0)
            parts.append(f"> {text}")
            parts.append(f"— *{day}* (relevance score: {score:.3f})\n")

        # (c) Surface contradictions
        if contradictions:
            parts.append(f"### ⚠️ Contradictions Detected ({len(contradictions)})\n")
            parts.append("The mentions of this entity contain **conflicting information**:\n")

            for i, c in enumerate(contradictions, 1):
                parts.append(f"**Contradiction {i}** — _{c['contradiction_type']}_")
                parts.append(f"  - **{c['explanation']}** (confidence: {c['confidence']:.0%})")

                a = c.get('mention_a', {})
                b = c.get('mention_b', {})
                parts.append(f"  - *{a.get('day_label', '?')}*: \"{a.get('text', '')[:150]}\"")
                parts.append(f"  - *{b.get('day_label', '?')}*: \"{b.get('text', '')[:150]}\"")
                parts.append("")
        else:
            parts.append("### ✅ No Contradictions Detected")
            parts.append("All mentions of this entity are consistent.\n")

        # (d) Evidence timeline
        parts.append("### Evidence Timeline\n")
        # Group by topic for clarity
        by_topic = retrieval.get('by_topic', {})
        if by_topic:
            for topic_label, topic_mentions in by_topic.items():
                parts.append(f"**{topic_label}**")
                for tm in topic_mentions[:3]:
                    sent = tm.get('sentiment', {}).get('compound', 0)
                    sent_emoji = "🟢" if sent > 0.3 else "🔴" if sent < -0.3 else "🟡"
                    summary = tm.get('summary', tm.get('text', ''))[:150]
                    parts.append(f"  {sent_emoji} {summary}")
                parts.append("")

        # Also show message-level evidence
        by_day = retrieval.get('by_day', {})
        if by_day:
            parts.append("**Message-level evidence:**")
            for day_label, day_mentions in list(by_day.items())[:5]:
                for dm in day_mentions[:2]:
                    sent = dm.get('sentiment', {}).get('compound', 0)
                    sent_emoji = "🟢" if sent > 0.3 else "🔴" if sent < -0.3 else "🟡"
                    parts.append(
                        f"  {sent_emoji} [{dm.get('speaker', '?')}, msg #{dm.get('global_index', '?')}] "
                        f"\"{dm.get('text', '')[:120]}\""
                    )

        # Weights used
        parts.append(f"\n*Ranking weights: recency={self.recency_weight}, "
                    f"emotional={self.emotional_weight}*")

        return "\n".join(parts)
