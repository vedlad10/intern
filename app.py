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

# Global RAG engine instance
rag_engine = None


def get_engine():
    """Lazy-load the RAG engine."""
    global rag_engine
    if rag_engine is None:
        from rag_engine import RAGEngine
        rag_engine = RAGEngine(processed_data_dir='processed_data')
        rag_engine.initialize()
    return rag_engine


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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
