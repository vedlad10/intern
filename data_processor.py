

import csv
import json
import os
import re
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Message:

    global_index: int          # Global index across all conversations
    conversation_index: int    # Which conversation (row) this belongs to
    local_index: int           # Index within the conversation
    speaker: str               # "User 1" or "User 2"
    text: str                  # The message content
    full_text: str             # Speaker + text combined


@dataclass
class TopicCheckpoint:
    
    topic_id: int
    start_msg_index: int
    end_msg_index: int
    message_count: int
    topic_label: str
    summary: str
    key_entities: List[str] = field(default_factory=list)
    messages: List[Dict] = field(default_factory=list)


@dataclass
class HundredMsgCheckpoint:
    
    checkpoint_id: int
    start_msg_index: int
    end_msg_index: int
    summary: str
    key_topics: List[str] = field(default_factory=list)


class ConversationDataProcessor:
    

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.messages: List[Message] = []
        self.conversations: List[List[Message]] = []

    def load_and_parse(self) -> List[Message]:
        
        print("[DataProcessor] Loading CSV...")
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            conv_index = 0
            global_index = 0

            for row in reader:
                if not row or not row[0].strip():
                    continue

                conversation_text = row[0]
                conv_messages = []
                local_index = 0

                # Split by lines and parse User 1/User 2 messages
                lines = conversation_text.split('\n')
                for line in lines:
                    line = line.strip()
                    # Match "User 1:" or "User 2:" at start of line
                    match = re.match(r'^(User [12]):\s*(.*)', line)
                    if match:
                        speaker = match.group(1)
                        text = match.group(2).strip()
                        if text:  # Only add non-empty messages
                            msg = Message(
                                global_index=global_index,
                                conversation_index=conv_index,
                                local_index=local_index,
                                speaker=speaker,
                                text=text,
                                full_text=f"{speaker}: {text}"
                            )
                            self.messages.append(msg)
                            conv_messages.append(msg)
                            global_index += 1
                            local_index += 1

                if conv_messages:
                    self.conversations.append(conv_messages)
                    conv_index += 1

        print(f"[DataProcessor] Loaded {len(self.messages)} messages from {len(self.conversations)} conversations")
        return self.messages

    def get_messages(self) -> List[Message]:
        return self.messages

    def get_conversations(self) -> List[List[Message]]:
        return self.conversations


class TopicDetector:
    """
    Detects topic changes in conversation streams using embedding similarity.

    Strategy:
    1. Encode messages using sentence-transformers (all-MiniLM-L6-v2)
    2. Use a sliding window approach: compare embedding centroid of
       recent N messages vs the next N messages
    3. When cosine similarity drops below threshold, mark topic boundary
    4. Apply conversation boundary detection (new conversation = definite topic change)
    5. Use minimum segment size to avoid micro-topics
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"[TopicDetector] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.window_size = 5        # Messages to look back/forward
        self.similarity_threshold = 0.45  # Below this = topic change
        self.min_segment_size = 3   # Minimum messages per topic
        self.smoothing_window = 3   # Smooth similarity scores

    def detect_topics(self, messages: List[Message],
                      conversations: List[List[Message]]) -> List[TopicCheckpoint]:
        """Detect topic changes and create checkpoints."""
        if not messages:
            return []

        print(f"[TopicDetector] Encoding {len(messages)} messages...")

        # Encode all messages
        texts = [msg.full_text for msg in messages]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=256)

        # Build conversation boundary set
        conv_boundaries = set()
        idx = 0
        for conv in conversations:
            if idx > 0:
                conv_boundaries.add(idx)
            idx += len(conv)

        print("[TopicDetector] Computing topic boundaries...")

        # Compute sliding window similarity
        similarities = []
        for i in range(len(messages) - 1):
            # Get window before and after position i
            start_before = max(0, i - self.window_size + 1)
            end_after = min(len(messages), i + 1 + self.window_size)

            before_embeddings = embeddings[start_before:i + 1]
            after_embeddings = embeddings[i + 1:end_after]

            if len(before_embeddings) == 0 or len(after_embeddings) == 0:
                similarities.append(1.0)
                continue

            # Compute centroid similarity
            before_centroid = np.mean(before_embeddings, axis=0).reshape(1, -1)
            after_centroid = np.mean(after_embeddings, axis=0).reshape(1, -1)
            sim = cosine_similarity(before_centroid, after_centroid)[0][0]
            similarities.append(float(sim))

        # Smooth similarities to reduce noise
        smoothed = self._smooth_similarities(similarities)

        # Find topic boundaries
        boundaries = [0]  # Always start with a boundary at 0

        for i in range(len(smoothed)):
            is_conv_boundary = (i + 1) in conv_boundaries
            is_similarity_drop = smoothed[i] < self.similarity_threshold

            if is_conv_boundary or is_similarity_drop:
                # Check minimum segment size
                if (i + 1) - boundaries[-1] >= self.min_segment_size:
                    boundaries.append(i + 1)

        # Create topic checkpoints
        topic_checkpoints = []
        for idx_b in range(len(boundaries)):
            start = boundaries[idx_b]
            end = boundaries[idx_b + 1] if idx_b + 1 < len(boundaries) else len(messages)

            if start >= end:
                continue

            segment_messages = messages[start:end]
            segment_embeddings = embeddings[start:end]

            # Generate topic label and summary
            topic_label = self._generate_topic_label(segment_messages)
            summary = self._generate_summary(segment_messages)
            key_entities = self._extract_entities(segment_messages)

            checkpoint = TopicCheckpoint(
                topic_id=len(topic_checkpoints) + 1,
                start_msg_index=start,
                end_msg_index=end - 1,
                message_count=len(segment_messages),
                topic_label=topic_label,
                summary=summary,
                key_entities=key_entities,
                messages=[{
                    'global_index': m.global_index,
                    'speaker': m.speaker,
                    'text': m.text
                } for m in segment_messages]
            )
            topic_checkpoints.append(checkpoint)

        print(f"[TopicDetector] Detected {len(topic_checkpoints)} topic segments")
        return topic_checkpoints

    def _smooth_similarities(self, similarities: List[float]) -> List[float]:
        """Apply moving average smoothing to similarity scores."""
        if len(similarities) < self.smoothing_window:
            return similarities

        smoothed = []
        for i in range(len(similarities)):
            start = max(0, i - self.smoothing_window // 2)
            end = min(len(similarities), i + self.smoothing_window // 2 + 1)
            smoothed.append(np.mean(similarities[start:end]))
        return smoothed

    def _generate_topic_label(self, messages: List[Message]) -> str:
        """Generate a descriptive topic label from message content."""
        # Extract key nouns and themes using TF-IDF-like approach
        from collections import Counter

        # Common stop words to filter out
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your',
            'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its',
            'they', 'them', 'their', 'theirs', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
            'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
            'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
            'won', 'wouldn', 'also', 'really', 'like', 'know', 'think', 'would',
            'could', 'get', 'go', 'going', 'got', 'yeah', 'yes', 'okay', 'ok',
            'oh', 'well', 'hi', 'hello', 'hey', 'thanks', 'thank', 'good',
            'great', 'nice', 'love', 'sure', 'right', 'thing', 'things',
            'much', 'lot', 'way', 'time', 'make', 'made', 'one', 'two',
            'something', 'someone', 'anything', 'sounds', 'sound', 'definitely',
            'pretty', 'always', 'never', 'day', 'im', 'ive', 'thats', 'dont',
            'ive', 'cant', 'look', 'see', 'still', 'try', 'come', 'back',
            'keep', 'take', 'want', 'even', 'every', 'let', 'put', 'give'
        }

        words = Counter()
        for msg in messages:
            for word in re.findall(r'\b[a-zA-Z]{3,}\b', msg.text.lower()):
                if word not in stop_words:
                    words[word] += 1

        top_words = [w for w, _ in words.most_common(5)]
        if not top_words:
            return "General Conversation"

        return " & ".join(top_words[:3]).title()

    def _generate_summary(self, messages: List[Message]) -> str:
        """Generate a comprehensive summary of a message segment."""
        if not messages:
            return ""

        # Extract key information
        speakers = set()
        topics_mentioned = []
        facts = []

        for msg in messages:
            speakers.add(msg.speaker)

            # Extract factual statements
            text = msg.text.lower()
            if any(kw in text for kw in ['i am', "i'm", 'i work', 'i have', 'i love',
                                          'i like', 'i enjoy', 'my favorite', 'i live',
                                          'i play', 'i study', 'i used to']):
                facts.append(f"{msg.speaker}: {msg.text}")

        # Build summary
        summary_parts = []

        # Opening context
        first_msg = messages[0].text[:100]
        last_msg = messages[-1].text[:100]
        summary_parts.append(
            f"Conversation segment spanning {len(messages)} messages "
            f"between {' and '.join(speakers)}."
        )

        # Key facts shared
        if facts:
            summary_parts.append("Key information shared:")
            for fact in facts[:8]:  # Limit to 8 facts
                summary_parts.append(f"  - {fact}")

        # General discussion topics
        topic_label = self._generate_topic_label(messages)
        summary_parts.append(f"Main topics discussed: {topic_label}")

        return "\n".join(summary_parts)

    def _extract_entities(self, messages: List[Message]) -> List[str]:
        """Extract named entities and key concepts from messages."""
        entities = set()

        # Patterns for extracting entities
        patterns = [
            r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b',  # Multi-word proper nouns
            r'(?:named?|called?)\s+(\w+)',              # "named X" or "called X"
            r'(?:in|to|from)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',  # Place names
        ]

        for msg in messages:
            for pattern in patterns:
                matches = re.findall(pattern, msg.text)
                for match in matches:
                    if len(match) > 2 and match.lower() not in {'the', 'and', 'for', 'that', 'this', 'user'}:
                        entities.add(match)

        return list(entities)[:10]


class HundredMessageCheckpointer:
    """Creates summaries for every 100 messages."""

    def create_checkpoints(self, messages: List[Message]) -> List[HundredMsgCheckpoint]:
        """Create a checkpoint summary for every 100 messages."""
        checkpoints = []
        total = len(messages)

        for i in range(0, total, 100):
            chunk = messages[i:i + 100]
            if not chunk:
                continue

            summary = self._summarize_chunk(chunk)
            key_topics = self._extract_key_topics(chunk)

            checkpoint = HundredMsgCheckpoint(
                checkpoint_id=len(checkpoints) + 1,
                start_msg_index=chunk[0].global_index,
                end_msg_index=chunk[-1].global_index,
                summary=summary,
                key_topics=key_topics
            )
            checkpoints.append(checkpoint)

        print(f"[100MsgCheckpoint] Created {len(checkpoints)} checkpoints")
        return checkpoints

    def _summarize_chunk(self, messages: List[Message]) -> str:
        """Generate summary for a 100-message chunk."""
        # Group by conversation
        convs = defaultdict(list)
        for msg in messages:
            convs[msg.conversation_index].append(msg)

        summary_parts = [
            f"Messages {messages[0].global_index}-{messages[-1].global_index} "
            f"({len(messages)} messages across {len(convs)} conversation(s))."
        ]

        # Extract key facts from this chunk
        facts = []
        for msg in messages:
            text = msg.text.lower()
            if any(kw in text for kw in ['i am', "i'm", 'i work', 'i have', 'i love',
                                          'i like', 'i enjoy', 'my favorite', 'i live',
                                          'i play', 'i study']):
                facts.append(f"{msg.speaker}: {msg.text}")

        if facts:
            summary_parts.append("Key facts mentioned:")
            for fact in facts[:6]:
                summary_parts.append(f"  - {fact}")

        # Topics discussed
        topics = self._extract_key_topics(messages)
        if topics:
            summary_parts.append(f"Topics covered: {', '.join(topics)}")

        return "\n".join(summary_parts)

    def _extract_key_topics(self, messages: List[Message]) -> List[str]:
        """Extract key topics from messages."""
        from collections import Counter

        stop_words = {
            'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she', 'it', 'they',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'shall', 'to', 'of', 'in', 'for', 'on', 'with',
            'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'but', 'or', 'not', 'so', 'if', 'than', 'too', 'very', 'just',
            'about', 'that', 'this', 'what', 'how', 'when', 'where', 'who', 'why',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'some', 'any',
            'other', 'there', 'here', 'also', 'like', 'know', 'think', 'really',
            'yeah', 'yes', 'okay', 'well', 'good', 'great', 'nice', 'love',
            'sounds', 'thanks', 'thank', 'much', 'lot', 'going', 'get', 'got',
            'make', 'made', 'way', 'time', 'one', 'two', 'thing', 'things',
            'sure', 'right', 'back', 'still', 'never', 'always', 'pretty',
            'something', 'someone', 'anything', 'try', 'come', 'want', 'even'
        }

        word_counts = Counter()
        for msg in messages:
            for word in re.findall(r'\b[a-zA-Z]{3,}\b', msg.text.lower()):
                if word not in stop_words:
                    word_counts[word] += 1

        return [word for word, _ in word_counts.most_common(8)]


def process_data(csv_path: str, output_dir: str = 'processed_data'):
    """Main pipeline: process CSV → topic checkpoints → 100-msg checkpoints."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Parse CSV
    processor = ConversationDataProcessor(csv_path)
    messages = processor.load_and_parse()

    # Step 2: Detect topics and create topic checkpoints
    topic_detector = TopicDetector()
    topic_checkpoints = topic_detector.detect_topics(messages, processor.get_conversations())

    # Step 3: Create 100-message checkpoints
    msg_checkpointer = HundredMessageCheckpointer()
    hundred_msg_checkpoints = msg_checkpointer.create_checkpoints(messages)

    # Step 4: Save everything
    # Save topic checkpoints
    topic_data = [asdict(tc) for tc in topic_checkpoints]
    with open(os.path.join(output_dir, 'topic_checkpoints.json'), 'w', encoding='utf-8') as f:
        json.dump(topic_data, f, indent=2, ensure_ascii=False)

    # Save 100-message checkpoints
    msg_data = [asdict(mc) for mc in hundred_msg_checkpoints]
    with open(os.path.join(output_dir, 'hundred_msg_checkpoints.json'), 'w', encoding='utf-8') as f:
        json.dump(msg_data, f, indent=2, ensure_ascii=False)

    # Save raw messages for indexing
    raw_messages = [{
        'global_index': m.global_index,
        'conversation_index': m.conversation_index,
        'speaker': m.speaker,
        'text': m.text,
        'full_text': m.full_text
    } for m in messages]
    with open(os.path.join(output_dir, 'messages.json'), 'w', encoding='utf-8') as f:
        json.dump(raw_messages, f, ensure_ascii=False)

    print(f"\n[Pipeline] Processing complete!")
    print(f"  - {len(messages)} messages parsed")
    print(f"  - {len(topic_checkpoints)} topic checkpoints created")
    print(f"  - {len(hundred_msg_checkpoints)} 100-message checkpoints created")
    print(f"  - Data saved to '{output_dir}/'")

    return messages, topic_checkpoints, hundred_msg_checkpoints


if __name__ == '__main__':
    process_data('conversations.csv')
