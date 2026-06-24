"""
Flask Application - Conversation RAG Chatbot
=============================================
Serves the chatbot UI and API endpoints for:
- Querying the RAG system
- Retrieving persona information
- System statistics
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Global RAG engine instance — preload at startup to avoid 502 on first request
rag_engine = None

# Detect if running on Render (low memory) — set LITE_MODE=1 on Render env vars
LITE_MODE = os.environ.get('LITE_MODE', '0') == '1'


def get_engine():
    """Get the RAG engine (preloaded at startup)."""
    global rag_engine
    if rag_engine is None:
        _preload_engine()
    return rag_engine


def _preload_engine():
    """Initialize the RAG engine. Called at startup."""
    global rag_engine
    try:
        # Reduce PyTorch memory usage (only when running locally with torch)
        if not LITE_MODE:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        print(f"[STARTUP] Preloading RAG engine (LITE_MODE={LITE_MODE})...")
        from rag_engine import RAGEngine
        rag_engine = RAGEngine(processed_data_dir='processed_data', lite_mode=LITE_MODE)
        rag_engine.initialize()
        print("[STARTUP] RAG engine ready!")
    except Exception as e:
        print(f"[STARTUP] WARNING: Failed to preload RAG engine: {e}")
        import traceback
        traceback.print_exc()


# Preload engine at import time so gunicorn --preload picks it up
_preload_engine()


# ==================== UI Routes ====================

@app.route('/')
def index():
    """Serve the main chatbot UI."""
    return send_file('static/index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


# ==================== API Routes ====================

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint - processes user queries."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        engine = get_engine()
        result = engine.query(question, top_k=5)

        return jsonify({
            'success': True,
            'answer': result['answer'],
            'is_persona_query': result.get('is_persona_query', False),
            'sources': {
                'topics_retrieved': len(result.get('retrieved_topics', [])),
                'messages_retrieved': len(result.get('retrieved_messages', [])),
                'checkpoints_retrieved': len(result.get('retrieved_checkpoints', [])),
            },
            'retrieved_topics': [
                {
                    'topic_id': t.get('topic_id'),
                    'topic_label': t.get('topic_label'),
                    'message_range': f"{t.get('start_msg_index')}-{t.get('end_msg_index')}",
                    'relevance': round(t.get('relevance_score', 0), 3)
                }
                for t in result.get('retrieved_topics', [])[:5]
            ],
            'retrieved_messages': [
                {
                    'message_range': f"{m.get('start_index')}-{m.get('end_index')}",
                    'relevance': round(m.get('relevance_score', 0), 3),
                    'preview': m.get('text', '')[:200]
                }
                for m in result.get('retrieved_messages', [])[:5]
            ]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/persona', methods=['GET'])
def get_persona():
    """Get persona data for a specific user or all users."""
    try:
        engine = get_engine()
        user = request.args.get('user', None)

        if user and user in engine.personas:
            return jsonify({
                'success': True,
                'persona': engine.personas[user]
            })
        else:
            return jsonify({
                'success': True,
                'personas': engine.personas
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/persona/summary', methods=['GET'])
def get_persona_summary():
    """Get a human-readable persona summary."""
    try:
        engine = get_engine()
        summaries = {}

        for user_id, persona in engine.personas.items():
            summary = {
                'user_id': user_id,
                'message_count': persona.get('message_count', 0),
                'top_traits': [
                    {'trait': t['trait'].replace('_', ' ').title(), 'score': t['score']}
                    for t in persona.get('personality_traits', [])[:5]
                ],
                'habits_count': len(persona.get('habits', [])),
                'facts_count': len(persona.get('personal_facts', [])),
                'communication_style': persona.get('communication_style', {}),
                'top_habits': [
                    {'category': h['category'].replace('_', ' ').title(), 'detail': h['detail'][:100]}
                    for h in persona.get('habits', [])[:5]
                ],
                'top_facts': [
                    {'category': f['category'].replace('_', ' ').title(), 'detail': f['detail'][:100]}
                    for f in persona.get('personal_facts', [])[:5]
                ]
            }
            summaries[user_id] = summary

        return jsonify({'success': True, 'summaries': summaries})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get topic checkpoints with pagination."""
    try:
        engine = get_engine()
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))

        start = (page - 1) * per_page
        end = start + per_page

        topics = engine.topic_checkpoints[start:end]

        # Strip messages from response for smaller payload
        lightweight_topics = []
        for t in topics:
            lt = {k: v for k, v in t.items() if k != 'messages'}
            lt['message_count'] = t.get('message_count', len(t.get('messages', [])))
            lightweight_topics.append(lt)

        return jsonify({
            'success': True,
            'topics': lightweight_topics,
            'total': len(engine.topic_checkpoints),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(engine.topic_checkpoints) + per_page - 1) // per_page
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoints', methods=['GET'])
def get_checkpoints():
    """Get 100-message checkpoints."""
    try:
        engine = get_engine()
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))

        start = (page - 1) * per_page
        end = start + per_page

        checkpoints = engine.hundred_msg_checkpoints[start:end]

        return jsonify({
            'success': True,
            'checkpoints': checkpoints,
            'total': len(engine.hundred_msg_checkpoints),
            'page': page,
            'per_page': per_page
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        engine = get_engine()
        stats = engine.get_system_stats()
        return jsonify({'success': True, 'stats': stats})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'conversation-rag-chatbot'})


# ==================== Suggested Queries ====================

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested queries for the user."""
    suggestions = [
        {
            'category': 'Persona',
            'queries': [
                "What kind of person is User 1?",
                "What kind of person is User 2?",
                "What are User 1's habits?",
                "What are User 2's habits?",
                "How does User 1 talk?",
                "How does User 2 communicate?",
                "What are User 1's personality traits?",
                "Describe User 2's personality",
            ]
        },
        {
            'category': 'Facts & Interests',
            'queries': [
                "What do users talk about most?",
                "What hobbies are mentioned?",
                "What pets do users have?",
                "What jobs are discussed?",
                "Tell me about relationships mentioned",
                "What places are discussed?",
            ]
        },
        {
            'category': 'Conversation Topics',
            'queries': [
                "What are the main topics discussed?",
                "Tell me about cooking discussions",
                "What music do they discuss?",
                "Any discussions about travel?",
                "What sports are mentioned?",
                "Tell me about family discussions",
            ]
        }
    ]
    return jsonify({'success': True, 'suggestions': suggestions})


# ==================== Round 2 — Drift Timeline ====================

@app.route('/api/drift/timeline', methods=['GET'])
def get_drift_timeline():
    """Get mood drift timeline data."""
    try:
        timeline_path = os.path.join('processed_data', 'mood_timeline.json')
        if not os.path.exists(timeline_path):
            return jsonify({
                'success': False,
                'error': 'Mood timeline not generated. Run: python generate_drift.py'
            }), 404

        with open(timeline_path, 'r', encoding='utf-8') as f:
            timeline = json.load(f)

        # Optional pagination
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 100))
        start = (page - 1) * per_page
        end = start + per_page

        # Summary stats
        total_days = len(timeline)
        drift_count = sum(1 for d in timeline if d.get('drift_from_prev'))

        return jsonify({
            'success': True,
            'timeline': timeline[start:end],
            'total_days': total_days,
            'drift_count': drift_count,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_days + per_page - 1) // per_page,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/drift/summary', methods=['GET'])
def get_drift_summary():
    """Get drift summary statistics."""
    try:
        timeline_path = os.path.join('processed_data', 'mood_timeline.json')
        if not os.path.exists(timeline_path):
            return jsonify({'success': False, 'error': 'Timeline not generated'}), 404

        with open(timeline_path, 'r', encoding='utf-8') as f:
            timeline = json.load(f)

        drift_days = [d for d in timeline if d.get('drift_from_prev')]
        trigger_types = {}
        for d in drift_days:
            t = d.get('trigger', {})
            if t:
                ttype = t.get('type', 'unknown')
                trigger_types[ttype] = trigger_types.get(ttype, 0) + 1

        return jsonify({
            'success': True,
            'total_days': len(timeline),
            'drift_count': len(drift_days),
            'drift_rate': round(len(drift_days) / max(len(timeline), 1), 4),
            'trigger_breakdown': trigger_types,
            'drift_days': drift_days[:20],  # First 20 drift days
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Round 2 — Intent Classifier ====================

# Lazy-loaded intent classifier
_intent_classifier = None

def _get_intent_classifier():
    global _intent_classifier
    if _intent_classifier is None:
        try:
            from src.intent import IntentClassifier
            _intent_classifier = IntentClassifier(model_path="models/intent_model.joblib")
            _intent_classifier.load()
        except Exception as e:
            print(f"[Intent] Failed to load: {e}")
            return None
    return _intent_classifier


@app.route('/api/intent', methods=['POST'])
def classify_intent():
    """Classify a message's intent."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        classifier = _get_intent_classifier()
        if classifier is None:
            return jsonify({
                'success': False,
                'error': 'Intent model not loaded. Run: python train_intent.py'
            }), 503

        result = classifier.predict(text)
        return jsonify({'success': True, **result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/intent/metrics', methods=['GET'])
def get_intent_metrics():
    """Get intent classifier training metrics."""
    try:
        metrics_path = os.path.join('models', 'intent_metrics.json')
        if not os.path.exists(metrics_path):
            return jsonify({'success': False, 'error': 'Metrics not found'}), 404

        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Round 2 — Conflict Resolution ====================

# Lazy-loaded conflict resolver
_conflict_resolver = None

def _get_conflict_resolver():
    global _conflict_resolver
    if _conflict_resolver is None:
        try:
            engine = get_engine()
            from src.conflict import ConflictResolver
            _conflict_resolver = ConflictResolver(
                messages=engine.messages if engine else [],
                topic_checkpoints=engine.topic_checkpoints if engine else [],
            )
        except Exception as e:
            print(f"[Conflict] Failed to init: {e}")
            return None
    return _conflict_resolver


@app.route('/api/conflict/resolve', methods=['POST'])
def resolve_conflict():
    """Resolve entity conflicts across topic checkpoints."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        resolver = _get_conflict_resolver()
        if resolver is None:
            return jsonify({
                'success': False,
                'error': 'Conflict resolver not initialized.'
            }), 503

        result = resolver.resolve(query)

        # Serialize for JSON (trim large fields)
        response = {
            'success': True,
            'query': result['query'],
            'entity': result.get('entity'),
            'merged_answer': result.get('merged_answer', ''),
            'total_mentions': result.get('retrieval', {}).get('total_mentions', 0),
            'contradictions': [],
            'weights': result.get('retrieval', {}).get('weights', {}),
        }

        # Add contradiction summaries
        for c in result.get('contradictions', []):
            response['contradictions'].append({
                'type': c.get('contradiction_type', ''),
                'explanation': c.get('explanation', ''),
                'confidence': c.get('confidence', 0),
                'mention_a': c.get('mention_a', {}),
                'mention_b': c.get('mention_b', {}),
            })

        # Add top mentions (limit payload size)
        mentions = result.get('retrieval', {}).get('mentions', [])
        response['top_mentions'] = []
        for m in mentions[:10]:
            response['top_mentions'].append({
                'text': (m.get('text', m.get('summary', '')))[:200],
                'day_label': m.get('day_label', '?'),
                'source': m.get('source', ''),
                'rank_score': m.get('rank_score', 0),
                'sentiment': m.get('sentiment', {}),
            })

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/conflict/demo', methods=['GET'])
def get_conflict_demo():
    """Get the conflict demo scenario data."""
    try:
        scenario_path = os.path.join('data', 'conflict_scenarios.json')
        if not os.path.exists(scenario_path):
            return jsonify({'success': False, 'error': 'No scenario data'}), 404

        with open(scenario_path, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)

        return jsonify({'success': True, 'scenarios': scenarios})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
