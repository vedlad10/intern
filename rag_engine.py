

import json
import os
import re
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer


class RAGEngine:
    
    def __init__(self, processed_data_dir: str = 'processed_data',
                 model_name: str = 'all-MiniLM-L6-v2'):
        self.data_dir = processed_data_dir
        self.model_name = model_name
        self.model = None

        # Data stores
        self.topic_checkpoints = []
        self.hundred_msg_checkpoints = []
        self.messages = []
        self.personas = {}

        # FAISS indices
        self.topic_index = None
        self.message_index = None
        self.hundred_msg_index = None

        # Embeddings cache
        self.topic_embeddings = None
        self.message_embeddings = None
        self.hundred_msg_embeddings = None

        # Chunk size for message indexing
        self.chunk_size = 5  # Group messages into chunks of 5

    def initialize(self):
        """Load all data and build indices."""
        print("[RAG] Initializing RAG engine...")

        # Load model
        print("[RAG] Loading embedding model...")
        self.model = SentenceTransformer(self.model_name)

        # Load processed data
        self._load_data()

        # Build FAISS indices
        self._build_indices()

        print("[RAG] Engine ready!")

    def _load_data(self):
        """Load all processed data from JSON files."""
        print("[RAG] Loading processed data...")

        # Load topic checkpoints
        topic_path = os.path.join(self.data_dir, 'topic_checkpoints.json')
        if os.path.exists(topic_path):
            with open(topic_path, 'r', encoding='utf-8') as f:
                self.topic_checkpoints = json.load(f)
            print(f"  - Loaded {len(self.topic_checkpoints)} topic checkpoints")

        # Load 100-message checkpoints
        msg_ckpt_path = os.path.join(self.data_dir, 'hundred_msg_checkpoints.json')
        if os.path.exists(msg_ckpt_path):
            with open(msg_ckpt_path, 'r', encoding='utf-8') as f:
                self.hundred_msg_checkpoints = json.load(f)
            print(f"  - Loaded {len(self.hundred_msg_checkpoints)} 100-msg checkpoints")

        # Load messages
        msg_path = os.path.join(self.data_dir, 'messages.json')
        if os.path.exists(msg_path):
            with open(msg_path, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)
            print(f"  - Loaded {len(self.messages)} messages")

        # Load personas
        persona_path = os.path.join(self.data_dir, 'personas.json')
        if os.path.exists(persona_path):
            with open(persona_path, 'r', encoding='utf-8') as f:
                self.personas = json.load(f)
            print(f"  - Loaded personas for {len(self.personas)} users")

    def _build_indices(self):
        """Build or load FAISS vector indices for retrieval."""
        print("[RAG] Building/Loading FAISS indices...")

        # 1. Topic summary index
        topic_index_path = os.path.join(self.data_dir, 'topic_index.bin')
        if os.path.exists(topic_index_path):
            self.topic_index = faiss.read_index(topic_index_path)
            print(f"  - Loaded Topic index from disk: {self.topic_index.ntotal} entries")
        elif self.topic_checkpoints:
            topic_texts = [
                f"{tc['topic_label']}. {tc['summary']}"
                for tc in self.topic_checkpoints
            ]
            
            # Fast fallback: cap at 1000 if building locally to prevent freezing
            if len(topic_texts) > 1000:
                print(f"  - Capping topic index to 1000 entries for fast local startup...")
                topic_texts = topic_texts[:1000]
                self.topic_checkpoints = self.topic_checkpoints[:1000]
            self.topic_embeddings = self.model.encode(topic_texts, show_progress_bar=True, batch_size=128)
            dim = self.topic_embeddings.shape[1]
            self.topic_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.topic_embeddings)
            self.topic_index.add(self.topic_embeddings)
            faiss.write_index(self.topic_index, topic_index_path)
            print(f"  - Built and saved Topic index: {self.topic_index.ntotal} entries")

        # 2. Message chunk index (group messages into chunks)
        message_index_path = os.path.join(self.data_dir, 'message_index.bin')
        if self.messages:
            # We always need message_chunks for retrieval payload, so build it:
            chunk_texts = []
            self.message_chunks = []
            for i in range(0, len(self.messages), self.chunk_size):
                chunk = self.messages[i:i + self.chunk_size]
                chunk_text = " | ".join([m['full_text'] for m in chunk])
                chunk_texts.append(chunk_text)
                self.message_chunks.append({
                    'start_index': chunk[0]['global_index'],
                    'end_index': chunk[-1]['global_index'],
                    'messages': chunk,
                    'text': chunk_text
                })

            if os.path.exists(message_index_path):
                self.message_index = faiss.read_index(message_index_path)
                print(f"  - Loaded Message chunk index from disk: {self.message_index.ntotal} entries")
            else:
                # Fast fallback: cap at 1000 if building locally to prevent freezing
                if len(chunk_texts) > 1000:
                    print(f"  - Capping message chunks to 1000 entries for fast local startup...")
                    chunk_texts = chunk_texts[:1000]
                    self.message_chunks = self.message_chunks[:1000]

                self.message_embeddings = self.model.encode(
                    chunk_texts, show_progress_bar=True, batch_size=256
                )
                dim = self.message_embeddings.shape[1]
                self.message_index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(self.message_embeddings)
                self.message_index.add(self.message_embeddings)
                faiss.write_index(self.message_index, message_index_path)
                print(f"  - Built and saved Message chunk index: {self.message_index.ntotal} entries")

        # 3. 100-message checkpoint index
        hundred_msg_index_path = os.path.join(self.data_dir, 'hundred_msg_index.bin')
        if os.path.exists(hundred_msg_index_path):
            self.hundred_msg_index = faiss.read_index(hundred_msg_index_path)
            print(f"  - Loaded 100-msg checkpoint index from disk: {self.hundred_msg_index.ntotal} entries")
        elif self.hundred_msg_checkpoints:
            ckpt_texts = [mc['summary'] for mc in self.hundred_msg_checkpoints]
            self.hundred_msg_embeddings = self.model.encode(
                ckpt_texts, show_progress_bar=True, batch_size=128
            )
            dim = self.hundred_msg_embeddings.shape[1]
            self.hundred_msg_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.hundred_msg_embeddings)
            self.hundred_msg_index.add(self.hundred_msg_embeddings)
            faiss.write_index(self.hundred_msg_index, hundred_msg_index_path)
            print(f"  - Built and saved 100-msg checkpoint index: {self.hundred_msg_index.ntotal} entries")

    def query(self, question: str, top_k: int = 5) -> Dict:
        
        if not self.model:
            return {'answer': 'RAG engine not initialized', 'sources': []}

        # Check if this is a persona-related query
        is_persona_query = self._is_persona_query(question)

        # Encode query
        query_embedding = self.model.encode([question])
        faiss.normalize_L2(query_embedding)

        # Retrieve from all indices
        retrieved_topics = self._search_topics(query_embedding, top_k=top_k)
        retrieved_messages = self._search_messages(query_embedding, top_k=top_k * 2)
        retrieved_checkpoints = self._search_checkpoints(query_embedding, top_k=top_k)

        # Apply keyword re-ranking
        retrieved_topics = self._keyword_rerank(question, retrieved_topics, 'summary')
        retrieved_messages = self._keyword_rerank(question, retrieved_messages, 'text')

        # Generate answer
        if is_persona_query:
            answer = self._generate_persona_answer(question, retrieved_topics,
                                                   retrieved_messages, retrieved_checkpoints)
        else:
            answer = self._generate_contextual_answer(question, retrieved_topics,
                                                      retrieved_messages, retrieved_checkpoints)

        return {
            'answer': answer,
            'retrieved_topics': retrieved_topics[:top_k],
            'retrieved_messages': retrieved_messages[:top_k],
            'retrieved_checkpoints': retrieved_checkpoints[:3],
            'is_persona_query': is_persona_query,
            'query': question
        }

    def _is_persona_query(self, question: str) -> bool:
        """Detect if the query is about user persona/personality."""
        persona_keywords = [
            'person', 'personality', 'kind of person', 'type of person',
            'habits', 'habit', 'style', 'talk', 'communicate', 'behave',
            'traits', 'trait', 'character', 'who is', 'what is user',
            'how do they', 'how does', 'what are their', 'persona',
            'describe', 'like as a person', 'emoji', 'tone', 'mood',
            'interests', 'hobbies', 'hobby', 'relationship', 'job',
            'occupation', 'work', 'pet', 'animal', 'live', 'from',
            'food', 'eat', 'sleep', 'exercise', 'routine', 'daily',
            'fun', 'free time', 'spare time'
        ]
        question_lower = question.lower()
        return any(kw in question_lower for kw in persona_keywords)

    def _search_topics(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search topic summaries."""
        if self.topic_index is None or self.topic_index.ntotal == 0:
            return []

        scores, indices = self.topic_index.search(query_embedding, min(top_k, self.topic_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.topic_checkpoints):
                tc = self.topic_checkpoints[idx].copy()
                tc['relevance_score'] = float(score)
                results.append(tc)
        return results

    def _search_messages(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search message chunks."""
        if self.message_index is None or self.message_index.ntotal == 0:
            return []

        scores, indices = self.message_index.search(query_embedding, min(top_k, self.message_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.message_chunks):
                chunk = self.message_chunks[idx].copy()
                chunk['relevance_score'] = float(score)
                results.append(chunk)
        return results

    def _search_checkpoints(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Search 100-message checkpoints."""
        if self.hundred_msg_index is None or self.hundred_msg_index.ntotal == 0:
            return []

        scores, indices = self.hundred_msg_index.search(
            query_embedding, min(top_k, self.hundred_msg_index.ntotal)
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.hundred_msg_checkpoints):
                mc = self.hundred_msg_checkpoints[idx].copy()
                mc['relevance_score'] = float(score)
                results.append(mc)
        return results

    def _keyword_rerank(self, query: str, results: List[Dict], text_field: str) -> List[Dict]:
        """Re-rank results using keyword matching (BM25-like)."""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        stop_words = {'the', 'what', 'how', 'does', 'are', 'they', 'their', 'this',
                      'that', 'about', 'who', 'which', 'when', 'where', 'can', 'will',
                      'has', 'have', 'had', 'was', 'were', 'been', 'being', 'with'}
        query_words -= stop_words

        if not query_words:
            return results

        for result in results:
            text = result.get(text_field, '').lower()
            keyword_score = sum(1 for w in query_words if w in text)
            # Boost relevance score with keyword matching
            original_score = result.get('relevance_score', 0)
            result['relevance_score'] = original_score + (keyword_score * 0.1)

        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)

    def _generate_persona_answer(self, question: str, topics: List[Dict],
                                  messages: List[Dict], checkpoints: List[Dict]) -> str:
        """Generate an answer about user persona."""
        question_lower = question.lower()

        # Determine which user is being asked about
        target_user = None
        if 'user 1' in question_lower or 'user1' in question_lower:
            target_user = 'User 1'
        elif 'user 2' in question_lower or 'user2' in question_lower:
            target_user = 'User 2'

        answer_parts = []

        if target_user and target_user in self.personas:
            persona = self.personas[target_user]
            answer_parts.append(f"## {target_user} Persona Profile\n")

            # Determine what aspect is being asked about
            if any(kw in question_lower for kw in ['habit', 'routine', 'daily', 'sleep', 'eat', 'food', 'exercise']):
                answer_parts.append("### Habits & Routines")
                if persona.get('habits'):
                    for habit in persona['habits'][:10]:
                        answer_parts.append(f"- **{habit['category'].replace('_', ' ').title()}**: {habit['detail']}")
                        answer_parts.append(f"  _Evidence: \"{habit['evidence'][:150]}\"_")
                else:
                    answer_parts.append("No specific habits detected in conversations.")

            elif any(kw in question_lower for kw in ['talk', 'communicate', 'style', 'tone', 'emoji', 'how do they']):
                answer_parts.append("### Communication Style")
                style = persona.get('communication_style', {})
                if style:
                    answer_parts.append(f"- **Message Style**: {style.get('message_length_style', 'N/A')}")
                    answer_parts.append(f"- **Avg Words/Message**: {style.get('avg_message_length_words', 'N/A')}")
                    answer_parts.append(f"- **Formality**: {style.get('formality', {}).get('overall_formality', 'N/A')}")
                    answer_parts.append(f"- **Enthusiasm**: {style.get('punctuation', {}).get('enthusiasm_level', 'N/A')}")
                    emoji_info = style.get('emoji_usage', {})
                    answer_parts.append(f"- **Emoji Usage**: {emoji_info.get('usage_level', 'N/A')}")

            elif any(kw in question_lower for kw in ['personality', 'trait', 'character', 'kind of person', 'type of person', 'describe']):
                answer_parts.append("### Personality Traits")
                if persona.get('personality_traits'):
                    for trait in persona['personality_traits'][:8]:
                        score_bar = "█" * int(trait['score'] * 10) + "░" * (10 - int(trait['score'] * 10))
                        answer_parts.append(f"- **{trait['trait'].replace('_', ' ').title()}**: [{score_bar}] {trait['score']:.1%}")
                        if trait.get('evidence_samples'):
                            answer_parts.append(f"  _Example: \"{trait['evidence_samples'][0][:120]}\"_")

            else:
                # Full persona overview
                answer_parts.extend(self._full_persona_summary(persona, target_user))

        else:
            # No specific user mentioned — show both
            answer_parts.append("## User Persona Profiles\n")
            for user_id, persona in self.personas.items():
                answer_parts.extend(self._full_persona_summary(persona, user_id))
                answer_parts.append("---")

        # Add relevant context from RAG
        if topics:
            answer_parts.append("\n### Relevant Conversation Context")
            for t in topics[:3]:
                answer_parts.append(f"- **Topic**: {t.get('topic_label', 'N/A')} "
                                  f"(msgs {t.get('start_msg_index', '?')}-{t.get('end_msg_index', '?')})")

        return "\n".join(answer_parts)

    def _full_persona_summary(self, persona: Dict, user_id: str) -> List[str]:
        """Generate a full persona summary."""
        parts = [f"### {user_id}\n"]

        # Personal facts
        if persona.get('personal_facts'):
            parts.append("**Personal Facts:**")
            facts_by_cat = {}
            for fact in persona['personal_facts']:
                cat = fact['category'].replace('_', ' ').title()
                if cat not in facts_by_cat:
                    facts_by_cat[cat] = []
                facts_by_cat[cat].append(fact['detail'])

            for cat, details in facts_by_cat.items():
                parts.append(f"- **{cat}**: {'; '.join(details[:3])}")

        # Personality traits
        if persona.get('personality_traits'):
            parts.append("\n**Personality Traits:**")
            for trait in persona['personality_traits'][:5]:
                parts.append(f"- {trait['trait'].replace('_', ' ').title()}: {trait['score']:.0%}")

        # Habits
        if persona.get('habits'):
            parts.append("\n**Habits:**")
            for habit in persona['habits'][:5]:
                parts.append(f"- {habit['category'].replace('_', ' ').title()}: {habit['detail'][:100]}")

        # Communication style
        style = persona.get('communication_style', {})
        if style:
            parts.append("\n**Communication Style:**")
            parts.append(f"- {style.get('message_length_style', 'N/A')}")
            parts.append(f"- Formality: {style.get('formality', {}).get('overall_formality', 'N/A')}")
            parts.append(f"- {style.get('punctuation', {}).get('enthusiasm_level', 'N/A')}")

        return parts

    def _generate_contextual_answer(self, question: str, topics: List[Dict],
                                     messages: List[Dict], checkpoints: List[Dict]) -> str:
        """Generate a contextual answer for non-persona queries."""
        answer_parts = []

        # Synthesize from topic summaries
        if topics:
            answer_parts.append("## Relevant Topic Segments\n")
            for i, topic in enumerate(topics[:5], 1):
                answer_parts.append(f"### {i}. {topic.get('topic_label', 'Topic')}")
                answer_parts.append(f"**Messages**: {topic.get('start_msg_index', '?')}-{topic.get('end_msg_index', '?')} "
                                  f"({topic.get('message_count', '?')} messages)")
                answer_parts.append(f"**Relevance Score**: {topic.get('relevance_score', 0):.3f}")
                summary = topic.get('summary', '')
                if summary:
                    # Truncate long summaries
                    if len(summary) > 500:
                        summary = summary[:500] + "..."
                    answer_parts.append(f"\n{summary}\n")

                # Show key entities
                entities = topic.get('key_entities', [])
                if entities:
                    answer_parts.append(f"**Key Entities**: {', '.join(entities)}")
                answer_parts.append("")

        # Show relevant message excerpts
        if messages:
            answer_parts.append("## Relevant Message Excerpts\n")
            for i, chunk in enumerate(messages[:5], 1):
                answer_parts.append(f"**Chunk {i}** (msgs {chunk.get('start_index', '?')}-{chunk.get('end_index', '?')}, "
                                  f"relevance: {chunk.get('relevance_score', 0):.3f}):")
                for msg in chunk.get('messages', [])[:5]:
                    answer_parts.append(f"> {msg.get('full_text', '')}")
                answer_parts.append("")

        # Show checkpoint context
        if checkpoints:
            answer_parts.append("## 100-Message Checkpoint Context\n")
            for ckpt in checkpoints[:2]:
                answer_parts.append(f"**Checkpoint {ckpt.get('checkpoint_id', '?')}** "
                                  f"(msgs {ckpt.get('start_msg_index', '?')}-{ckpt.get('end_msg_index', '?')}):")
                answer_parts.append(ckpt.get('summary', '')[:300])
                topics_list = ckpt.get('key_topics', [])
                if topics_list:
                    answer_parts.append(f"**Key Topics**: {', '.join(topics_list)}")
                answer_parts.append("")

        if not answer_parts:
            return "I couldn't find relevant information for your query. Please try rephrasing."

        return "\n".join(answer_parts)

    def get_system_stats(self) -> Dict:
        """Return system statistics."""
        return {
            'total_messages': len(self.messages),
            'total_topic_checkpoints': len(self.topic_checkpoints),
            'total_100msg_checkpoints': len(self.hundred_msg_checkpoints),
            'total_message_chunks': len(getattr(self, 'message_chunks', [])),
            'personas_loaded': list(self.personas.keys()),
            'topic_index_size': self.topic_index.ntotal if self.topic_index else 0,
            'message_index_size': self.message_index.ntotal if self.message_index else 0,
        }


# Singleton instance
_rag_instance = None

def get_rag_engine(data_dir: str = 'processed_data') -> RAGEngine:
    """Get or create RAG engine singleton."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine(processed_data_dir=data_dir)
        _rag_instance.initialize()
    return _rag_instance
