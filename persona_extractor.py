

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Tuple, Optional


@dataclass
class UserPersona:
    
    user_id: str
    message_count: int = 0

    # Habits
    habits: List[Dict[str, str]] = field(default_factory=list)

    # Personal facts
    personal_facts: List[Dict[str, str]] = field(default_factory=list)

    # Personality traits
    personality_traits: List[Dict[str, float]] = field(default_factory=list)

    # Communication style
    communication_style: Dict = field(default_factory=dict)

    # Raw evidence (messages that support each trait)
    evidence: Dict[str, List[str]] = field(default_factory=dict)


class PersonaExtractor:
    

    # Habit detection patterns
    HABIT_PATTERNS = {
        'food_preference': [
            r'(?:i (?:love|like|enjoy) (?:eating|cooking|making))\s+(.*?)(?:\.|!|$)',
            r'my favorite (?:food|meal|dish) is\s+(.*?)(?:\.|!|$)',
            r'i (?:always|usually|often) (?:eat|cook|make)\s+(.*?)(?:\.|!|$)',
            r'i\'m (?:a )?(?:vegan|vegetarian|pescatarian)',
            r'i (?:love|like|enjoy)\s+(?:cooking|baking)',
        ],
        'exercise_habit': [
            r'i (?:love to|like to|enjoy)?\s*(?:run|jog|hike|swim|bike|cycle|workout|work out|exercise|yoga|lift)',
            r'i (?:go|do)\s+(?:running|jogging|hiking|swimming|biking|cycling|yoga|pilates)',
            r'i (?:play|practice)\s+(?:soccer|football|basketball|tennis|golf|baseball)',
            r'i (?:usually|always|often)\s+(?:work out|exercise|run|jog)',
        ],
        'sleep_pattern': [
            r'i (?:usually|always|tend to)\s+(?:stay up|sleep|wake up|get up)\s+(.*?)(?:\.|!|$)',
            r'i\'m (?:a )?(?:night owl|early bird|morning person|late sleeper|early riser)',
            r'i (?:go to bed|fall asleep)\s+(.*?)(?:\.|!|$)',
        ],
        'daily_routine': [
            r'(?:every|each)\s+(?:morning|evening|night|day)\s+i\s+(.*?)(?:\.|!|$)',
            r'i (?:usually|always|often)\s+(?:start|begin|end)\s+my day\s+(.*?)(?:\.|!|$)',
            r'my (?:morning|evening|daily|night)\s+routine\s+(.*?)(?:\.|!|$)',
        ],
        'hobby': [
            r'i (?:love|like|enjoy)\s+(?:to\s+)?(?:read|paint|draw|write|garden|fish|cook|bake|sing|dance|play|collect)',
            r'my hobbies?\s+(?:is|are|include)\s+(.*?)(?:\.|!|$)',
            r'in my (?:spare|free) time\s+i\s+(.*?)(?:\.|!|$)',
            r'i (?:love|like|enjoy)\s+(.*?)(?:\.|!|$)',
        ],
    }

    # Personal fact patterns
    FACT_PATTERNS = {
        'occupation': [
            r'i(?:\'m| am) (?:a |an )?(.*?(?:engineer|doctor|nurse|teacher|student|developer|designer|manager|chef|firefighter|librarian|barista|programmer|artist|musician|writer|photographer|lawyer|accountant|scientist|researcher|analyst|consultant|pilot|driver|mechanic|electrician|plumber|carpenter|farmer|fisherman|soldier|police|officer|detective|agent|clerk|secretary|receptionist|cashier|waiter|waitress|cook|baker|butcher|grocer|pharmacist|dentist|surgeon|therapist|counselor|psychologist|psychiatrist|professor|instructor|tutor|coach|trainer|mentor|intern|apprentice|volunteer|freelancer|entrepreneur|founder|owner|director|president|ceo|cto|cfo|vp|manager|supervisor|coordinator|administrator|assistant|associate|specialist|expert|consultant|advisor|strategist|architect|designer|developer|programmer|coder|tester|analyst|data|software|hardware|network|system|security|cloud|devops|fullstack|frontend|backend|mobile|web|game|ai|ml|data|science|research|ux|ui|product|project|program|sales|marketing|business|finance|accounting|hr|legal|operations|logistics|supply|chain|customer|service|support|technical|quality|compliance|audit|risk|insurance|banking|investment|real estate|hospitality|tourism|travel|retail|wholesale|manufacturing|construction|mining|agriculture|forestry|fishing|energy|utility|telecom|media|entertainment|publishing|education|health|medical|pharmaceutical|biotech|nonprofit|government|military|religious|social|community|environmental))',
            r'i work (?:as|at|for|in)\s+(.*?)(?:\.|!|$)',
            r'my job is\s+(.*?)(?:\.|!|$)',
            r'i\'m (?:a )?(?:fulltime|full-time|part-time|parttime) (?:student|worker|employee)',
            r'i (?:study|studying|studied)\s+(.*?)(?:\.|!|$)',
        ],
        'relationship': [
            r'my (?:husband|wife|partner|boyfriend|girlfriend|spouse|fiance|fiancee)\s+(.*?)(?:\.|!|$)',
            r'i\'m (?:married|engaged|single|divorced|separated|dating|in a relationship)',
            r'i have (?:a |)(?:son|daughter|kids|children|child|baby|twin)',
            r'i\'m (?:a )?(?:single )?(?:mom|dad|parent|mother|father)',
            r'my (?:family|parents|siblings|brothers?|sisters?|mom|dad|mother|father)',
        ],
        'pet': [
            r'i have (?:a |an |)(?:dog|cat|pet|fish|bird|hamster|rabbit|guinea pig|turtle|snake|lizard|parrot|ferret|horse|pony)',
            r'my (?:dog|cat|pet)(?:\'s name)? is\s+(.*?)(?:\.|!|$)',
            r'(?:dog|cat|pet) named\s+(\w+)',
        ],
        'location': [
            r'i (?:live|moved|moving|relocated)\s+(?:in|to|from)\s+(.*?)(?:\.|!|$)',
            r'i\'m (?:from|originally from|based in)\s+(.*?)(?:\.|!|$)',
            r'i (?:grew up|was born|was raised)\s+(?:in|near)\s+(.*?)(?:\.|!|$)',
        ],
        'age_education': [
            r'i\'m\s+(\d+)\s+years?\s+old',
            r'i (?:graduated|went to|attend|go to)\s+(.*?)(?:\.|!|$)',
            r'i\'m in (?:high school|college|university|grad school|graduate school)',
        ],
        'vehicle': [
            r'i (?:have|own|drive)\s+(?:a |an )?(.*?(?:car|truck|motorcycle|bike|impala|mustang|camaro|corvette|tesla|bmw|audi|mercedes|honda|toyota|ford|chevy|dodge|jeep).*?)(?:\.|!|$)',
        ],
    }

    # Personality indicator words
    PERSONALITY_INDICATORS = {
        'enthusiastic': ['awesome', 'amazing', 'incredible', 'fantastic', 'wonderful', 'love', 'excited', 'thrilled', 'wow', 'cool', 'great'],
        'empathetic': ['sorry', 'understand', 'feel', 'hope', 'care', 'miss', 'wish', 'support', 'help', 'glad'],
        'humorous': ['haha', 'lol', 'funny', 'joke', 'laugh', 'hilarious', 'lmao', 'rofl'],
        'friendly': ['friend', 'together', 'fun', 'enjoy', 'share', 'welcome', 'nice to meet', 'pleasure'],
        'curious': ['tell me', 'what about', 'how about', 'interested', 'wondering', 'curious', 'what kind', 'what type'],
        'supportive': ['good for you', 'that\'s great', 'proud', 'happy for', 'good luck', 'you can', 'believe in'],
        'adventurous': ['travel', 'explore', 'adventure', 'new places', 'road trip', 'discover', 'experience'],
        'introverted': ['alone', 'quiet', 'peace', 'calm', 'relax', 'by myself', 'introvert', 'homebody'],
        'family_oriented': ['family', 'kids', 'children', 'parents', 'siblings', 'home', 'together'],
        'creative': ['art', 'create', 'design', 'paint', 'draw', 'write', 'music', 'craft', 'build'],
    }

    def __init__(self):
        self.personas: Dict[str, UserPersona] = {}

    def extract_personas(self, messages: List) -> Dict[str, Dict]:
        """Extract personas for all users from messages."""
        print("[PersonaExtractor] Extracting user personas...")

        # Group messages by speaker
        user_messages = defaultdict(list)
        for msg in messages:
            user_messages[msg.speaker].append(msg)

        results = {}
        for user_id, msgs in user_messages.items():
            persona = self._extract_single_persona(user_id, msgs)
            results[user_id] = asdict(persona)
            print(f"  - {user_id}: {len(persona.habits)} habits, "
                  f"{len(persona.personal_facts)} facts, "
                  f"{len(persona.personality_traits)} traits")

        return results

    def _extract_single_persona(self, user_id: str, messages: List) -> UserPersona:
        """Extract persona for a single user."""
        persona = UserPersona(user_id=user_id, message_count=len(messages))

        # Extract habits
        persona.habits = self._extract_habits(messages)

        # Extract personal facts
        persona.personal_facts = self._extract_facts(messages)

        # Extract personality traits
        persona.personality_traits = self._analyze_personality(messages)

        # Analyze communication style
        persona.communication_style = self._analyze_communication_style(messages)

        # Collect evidence
        persona.evidence = self._collect_evidence(messages)

        return persona

    def _extract_habits(self, messages: List) -> List[Dict[str, str]]:
        """Extract habits from messages using pattern matching."""
        habits = []
        seen = set()

        for msg in messages:
            text = msg.text.lower()
            for habit_type, patterns in self.HABIT_PATTERNS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            detail = match.strip() if isinstance(match, str) else str(match)
                            if detail and len(detail) > 2:
                                key = f"{habit_type}:{detail[:50]}"
                                if key not in seen:
                                    seen.add(key)
                                    habits.append({
                                        'category': habit_type,
                                        'detail': detail[:200],
                                        'evidence': msg.text[:300],
                                        'confidence': 'high' if len(detail) > 5 else 'medium'
                                    })
                    elif re.search(pattern, text, re.IGNORECASE):
                        # Pattern matched but no capture group
                        match_obj = re.search(pattern, text, re.IGNORECASE)
                        if match_obj:
                            detail = match_obj.group(0)
                            key = f"{habit_type}:{detail[:50]}"
                            if key not in seen:
                                seen.add(key)
                                habits.append({
                                    'category': habit_type,
                                    'detail': detail[:200],
                                    'evidence': msg.text[:300],
                                    'confidence': 'medium'
                                })

        return habits

    def _extract_facts(self, messages: List) -> List[Dict[str, str]]:
        """Extract personal facts from messages."""
        facts = []
        seen = set()

        for msg in messages:
            text = msg.text
            text_lower = text.lower()

            for fact_type, patterns in self.FACT_PATTERNS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            detail = match.strip() if isinstance(match, str) else str(match)
                            if detail and len(detail) > 1:
                                key = f"{fact_type}:{detail[:50]}"
                                if key not in seen:
                                    seen.add(key)
                                    facts.append({
                                        'category': fact_type,
                                        'detail': detail[:200],
                                        'evidence': text[:300],
                                        'confidence': 'high'
                                    })
                    elif re.search(pattern, text_lower, re.IGNORECASE):
                        match_obj = re.search(pattern, text_lower, re.IGNORECASE)
                        if match_obj:
                            detail = match_obj.group(0)
                            key = f"{fact_type}:{detail[:50]}"
                            if key not in seen:
                                seen.add(key)
                                facts.append({
                                    'category': fact_type,
                                    'detail': detail[:200],
                                    'evidence': text[:300],
                                    'confidence': 'medium'
                                })

        return facts

    def _analyze_personality(self, messages: List) -> List[Dict[str, float]]:
        """Analyze personality traits based on word usage patterns."""
        trait_scores = defaultdict(int)
        trait_evidence = defaultdict(list)
        total_messages = len(messages)

        for msg in messages:
            text_lower = msg.text.lower()
            words = set(re.findall(r'\b\w+\b', text_lower))

            for trait, indicators in self.PERSONALITY_INDICATORS.items():
                matches = words.intersection(set(indicators))
                if matches:
                    trait_scores[trait] += len(matches)
                    if len(trait_evidence[trait]) < 5:
                        trait_evidence[trait].append(msg.text[:200])

        # Normalize scores
        traits = []
        if total_messages > 0:
            max_score = max(trait_scores.values()) if trait_scores else 1
            for trait, score in sorted(trait_scores.items(), key=lambda x: -x[1]):
                normalized = round(min(score / max(max_score * 0.5, 1), 1.0), 3)
                if normalized > 0.05:  # Only include meaningful traits
                    traits.append({
                        'trait': trait,
                        'score': normalized,
                        'occurrence_count': score,
                        'evidence_samples': trait_evidence[trait][:3]
                    })

        return traits

    def _analyze_communication_style(self, messages: List) -> Dict:
        """Analyze communication style patterns."""
        if not messages:
            return {}

        # Message length analysis
        lengths = [len(msg.text) for msg in messages]
        word_counts = [len(msg.text.split()) for msg in messages]

        # Emoji/emoticon usage
        emoji_pattern = r'[😀-😿🤐-🤿🥀-🥿🦀-🦿🧀-🧿🩰-🩿🪀-🪿🫀-🫿]|[:;]-?[)(DPpOo/\\|]|<3|:\'[\(]|xD|XD|:\)|;\)|:D'
        emoji_count = sum(len(re.findall(emoji_pattern, msg.text)) for msg in messages)

        # Exclamation usage
        exclamation_count = sum(msg.text.count('!') for msg in messages)

        # Question usage
        question_count = sum(msg.text.count('?') for msg in messages)

        # Formality indicators
        formal_words = {'indeed', 'furthermore', 'however', 'therefore', 'nevertheless',
                       'consequently', 'additionally', 'moreover', 'regarding', 'concerning'}
        informal_words = {'yeah', 'yep', 'nah', 'gonna', 'wanna', 'gotta', 'kinda',
                         'sorta', 'lol', 'omg', 'tbh', 'imo', 'brb', 'btw'}

        formal_count = 0
        informal_count = 0
        for msg in messages:
            words = set(msg.text.lower().split())
            formal_count += len(words.intersection(formal_words))
            informal_count += len(words.intersection(informal_words))

        # Determine overall tone
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

        if avg_words < 8:
            length_style = "concise, short messages"
        elif avg_words < 20:
            length_style = "moderate length messages"
        else:
            length_style = "detailed, lengthy messages"

        if informal_count > formal_count * 2:
            formality = "very informal/casual"
        elif informal_count > formal_count:
            formality = "mostly informal"
        elif formal_count > informal_count:
            formality = "somewhat formal"
        else:
            formality = "balanced/neutral"

        exclamation_rate = exclamation_count / len(messages)
        if exclamation_rate > 1.5:
            enthusiasm_level = "very expressive (heavy exclamation use)"
        elif exclamation_rate > 0.5:
            enthusiasm_level = "moderately expressive"
        else:
            enthusiasm_level = "calm/reserved"

        return {
            'avg_message_length_chars': round(avg_length, 1),
            'avg_message_length_words': round(avg_words, 1),
            'message_length_style': length_style,
            'total_messages_analyzed': len(messages),
            'emoji_usage': {
                'total_emojis': emoji_count,
                'emojis_per_message': round(emoji_count / len(messages), 3),
                'usage_level': 'heavy' if emoji_count / len(messages) > 0.5 else
                              'moderate' if emoji_count / len(messages) > 0.1 else 'minimal'
            },
            'punctuation': {
                'exclamation_rate': round(exclamation_rate, 3),
                'question_rate': round(question_count / len(messages), 3),
                'enthusiasm_level': enthusiasm_level
            },
            'formality': {
                'formal_word_count': formal_count,
                'informal_word_count': informal_count,
                'overall_formality': formality
            },
            'engagement_patterns': {
                'asks_questions': question_count > len(messages) * 0.2,
                'uses_greetings': any(msg.text.lower().startswith(('hi', 'hello', 'hey', 'good morning'))
                                    for msg in messages[:50]),
                'responsive_style': 'engaged' if avg_words > 10 else 'brief'
            }
        }

    def _collect_evidence(self, messages: List) -> Dict[str, List[str]]:
        """Collect evidence messages for each category."""
        evidence = defaultdict(list)

        for msg in messages:
            text_lower = msg.text.lower()

            # Personal info evidence
            if any(kw in text_lower for kw in ['i am', "i'm", 'i work', 'i have', 'i live']):
                if len(evidence['personal_info']) < 20:
                    evidence['personal_info'].append(msg.text[:300])

            # Hobby evidence
            if any(kw in text_lower for kw in ['i love', 'i like', 'i enjoy', 'hobby', 'fun', 'spare time', 'free time']):
                if len(evidence['hobbies_interests']) < 20:
                    evidence['hobbies_interests'].append(msg.text[:300])

            # Emotional expression
            if any(kw in text_lower for kw in ['feel', 'happy', 'sad', 'excited', 'worried', 'scared', 'miss', 'love']):
                if len(evidence['emotional_expression']) < 15:
                    evidence['emotional_expression'].append(msg.text[:300])

        return dict(evidence)


def extract_and_save_personas(messages: List, output_dir: str = 'processed_data') -> Dict:
    """Extract personas and save to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    extractor = PersonaExtractor()
    personas = extractor.extract_personas(messages)

    output_path = os.path.join(output_dir, 'personas.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)

    print(f"[PersonaExtractor] Personas saved to {output_path}")
    return personas


if __name__ == '__main__':
    from data_processor import ConversationDataProcessor
    processor = ConversationDataProcessor('conversations.csv')
    messages = processor.load_and_parse()
    extract_and_save_personas(messages)
