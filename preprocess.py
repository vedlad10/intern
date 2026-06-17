"""
Preprocessing Pipeline
======================
Run this ONCE before starting the server.
Processes the CSV data and creates all checkpoints, personas, and indices.
"""

import os
import sys
import time


def main():
    start_time = time.time()

    print("=" * 60)
    print("  ConvoRAG — Data Preprocessing Pipeline")
    print("=" * 60)
    print()

    csv_path = 'conversations.csv'
    output_dir = 'processed_data'

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        print("Please place the conversations.csv file in the project root.")
        sys.exit(1)

    # Step 1: Parse and process data
    print("STEP 1/3: Parsing CSV and detecting topic changes...")
    print("-" * 50)
    from data_processor import process_data
    messages, topic_checkpoints, hundred_msg_checkpoints = process_data(csv_path, output_dir)

    # Step 2: Extract personas
    print()
    print("STEP 2/3: Extracting user personas...")
    print("-" * 50)
    from persona_extractor import extract_and_save_personas
    from data_processor import ConversationDataProcessor
    processor = ConversationDataProcessor(csv_path)
    msgs = processor.load_and_parse()
    personas = extract_and_save_personas(msgs, output_dir)

    # Step 3: Build RAG indices (test query)
    print()
    print("STEP 3/3: Building RAG indices and testing...")
    print("-" * 50)
    from rag_engine import RAGEngine
    engine = RAGEngine(processed_data_dir=output_dir)
    engine.initialize()

    # Run test queries
    test_queries = [
        "What kind of person is User 1?",
        "What are User 2's habits?",
        "What hobbies are mentioned in conversations?",
    ]

    print()
    print("Running test queries:")
    for q in test_queries:
        print(f"\n  Q: {q}")
        result = engine.query(q)
        answer_preview = result['answer'][:200].replace('\n', ' ')
        print(f"  A: {answer_preview}...")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  Preprocessing complete in {elapsed:.1f}s")
    print(f"  Messages: {len(messages)}")
    print(f"  Topic Checkpoints: {len(topic_checkpoints)}")
    print(f"  100-Msg Checkpoints: {len(hundred_msg_checkpoints)}")
    print(f"  Personas: {len(personas)}")
    print("=" * 60)
    print()
    print("You can now run the server with:")
    print("  python app.py")
    print()


if __name__ == '__main__':
    main()
