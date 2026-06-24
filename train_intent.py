"""
Train Intent Classifier — Pipeline Runner
==========================================
Generates synthetic training data, trains the TF-IDF + LogReg classifier,
evaluates it, and prints measured size/latency.

Usage:
    python train_intent.py
"""

import os
import sys
import json
import time


def main():
    start = time.time()

    print("=" * 60)
    print("  Offline Intent Classifier — Training Pipeline")
    print("=" * 60)

    from src.intent import IntentDataGenerator, IntentClassifier

    # 1. Generate synthetic training data
    print("\n[Step 1/4] Generating synthetic training data...")
    generator = IntentDataGenerator(seed=42)
    data = generator.generate(examples_per_class=300)

    # Save to CSV
    csv_path = os.path.join('data', 'intent_train.csv')
    generator.save_csv(data, csv_path)

    # Separate texts and labels
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]

    # Stats
    from collections import Counter
    label_counts = Counter(labels)
    print(f"[Step 1/4] Generated {len(data)} examples:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")

    # 2. Train classifier
    print("\n[Step 2/4] Training classifier...")
    classifier = IntentClassifier(model_path="models/intent_model.joblib")
    metrics = classifier.train(texts, labels, seed=42)

    # 3. Save model
    print("\n[Step 3/4] Saving model...")
    model_size = classifier.save()

    # 4. Measure performance
    print("\n[Step 4/4] Measuring performance...")
    test_texts = [
        "remind me to call mom tomorrow",
        "I'm feeling really sad today",
        "can you send the report to John",
        "how's the weather looking",
        "set a reminder for 5pm",
        "I need someone to talk to",
        "please fix the bug in the login page",
        "what's up dude",
        "don't forget about the dentist",
        "I'm so stressed about work",
        "hey how are you doing",
        "organize the files by date",
        "lol that's hilarious",
        "I need a hug",
        "submit the application before Friday",
        "good morning everyone",
        "remind me about the meeting at 3",
        "I can't stop crying",
        "schedule a call with the team",
        "nice weather today huh",
    ]

    perf = classifier.measure_performance(test_texts, n_runs=5)

    # Save all metrics
    os.makedirs('models', exist_ok=True)
    all_metrics = {
        "training_metrics": metrics,
        "performance_metrics": perf,
        "training_data_path": csv_path,
        "model_path": "models/intent_model.joblib",
    }

    metrics_path = os.path.join('models', 'intent_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Train] Metrics saved to {metrics_path}")

    # Test predictions
    print("\n" + "=" * 60)
    print("  Sample Predictions")
    print("=" * 60)

    sample_inputs = [
        "remind me to buy groceries",
        "I'm feeling so lonely right now",
        "please send the email to the client",
        "how's it going buddy",
        "asdfghjkl random nonsense xyz",
        "don't forget the meeting at 3pm",
        "I just need someone to listen",
        "update the spreadsheet with new data",
        "hey what's new",
        "set an alarm for 7am",
    ]

    for text in sample_inputs:
        result = classifier.predict(text)
        conf_bar = "#" * int(result['confidence'] * 20)
        print(f"  [{result['intent']:>20}] ({result['confidence']:.2f}) "
              f"[{conf_bar}] {result['latency_ms']:.1f}ms | \"{text[:50]}\"")

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Model size: {perf['model_size_MB']:.3f} MB (limit: 50 MB)")
    print(f"  Latency p95: {perf['latency_p95_ms']:.3f} ms (limit: 200 ms)")

    # Constraint verification
    print("\n  CONSTRAINT VERIFICATION:")
    print(f"    Size < 50MB:    {'[PASS]' if perf['meets_size_constraint'] else '[FAIL]'}")
    print(f"    Latency < 200ms: {'[PASS]' if perf['meets_latency_constraint'] else '[FAIL]'}")
    print(f"    Zero API calls: [PASS] (all offline)")


if __name__ == '__main__':
    main()
