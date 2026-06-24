"""
Generate Mood Timeline — Pipeline Runner
=========================================
Run this to build the mood timeline with drift detection.
Uses Round 1's processed messages and topic checkpoints.

Usage:
    python generate_drift.py
"""

import os
import sys
import json
import time


def main():
    start = time.time()

    print("=" * 60)
    print("  Adaptive Persona Engine — Mood Drift Pipeline")
    print("=" * 60)

    data_dir = 'processed_data'

    # Load messages
    msg_path = os.path.join(data_dir, 'messages.json')
    if not os.path.exists(msg_path):
        print(f"ERROR: {msg_path} not found. Run preprocess.py first.")
        sys.exit(1)

    print("[Pipeline] Loading messages...")
    with open(msg_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    print(f"[Pipeline] Loaded {len(messages)} messages")

    # Load topic checkpoints
    tc_path = os.path.join(data_dir, 'topic_checkpoints.json')
    topic_checkpoints = []
    if os.path.exists(tc_path):
        with open(tc_path, 'r', encoding='utf-8') as f:
            topic_checkpoints = json.load(f)
        print(f"[Pipeline] Loaded {len(topic_checkpoints)} topic checkpoints")

    # Build timeline
    from src.drift import build_mood_timeline, save_timeline

    timeline = build_mood_timeline(
        messages=messages,
        topic_checkpoints=topic_checkpoints,
        max_days=500,  # Cap for performance; increase if you want more
        k=1.5,         # Drift sensitivity: lower = more drifts detected
    )

    # Save
    output_path = os.path.join(data_dir, 'mood_timeline.json')
    save_timeline(timeline, output_path)

    # Summary
    drift_count = sum(1 for d in timeline if d['drift_from_prev'])
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print(f"  Timeline complete in {elapsed:.1f}s")
    print(f"  Days analyzed: {len(timeline)}")
    print(f"  Drift points:  {drift_count}")
    print(f"  Output:        {output_path}")
    print("=" * 60)

    # Show sample
    print("\n  Sample entries:")
    for entry in timeline[:5]:
        drift_marker = " ← DRIFT" if entry['drift_from_prev'] else ""
        trigger_info = ""
        if entry.get('trigger'):
            trigger_info = f" [Trigger: {entry['trigger']['type']}={entry['trigger']['value']}]"
        print(f"    Day {entry['day']:>5}: {', '.join(entry['mood_labels'])}"
              f"{drift_marker}{trigger_info}")

    if drift_count > 0:
        print("\n  Drift points:")
        for entry in timeline:
            if entry['drift_from_prev']:
                trigger = entry.get('trigger', {})
                t_type = trigger.get('type', '?') if trigger else '?'
                t_val = trigger.get('value', '?') if trigger else '?'
                print(f"    Day {entry['day']:>5}: {', '.join(entry['mood_labels'])} "
                      f"(distance={entry.get('drift_distance', 0):.3f}) "
                      f"[{t_type}: {t_val}]")
                # Only show first 10
                if sum(1 for e in timeline if e['drift_from_prev'] and
                       e['day'] <= entry['day']) >= 10:
                    print("    ... (showing first 10)")
                    break


if __name__ == '__main__':
    main()
