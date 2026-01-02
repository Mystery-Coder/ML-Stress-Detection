import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import json
import os
import matplotlib

# Use non-interactive backend
matplotlib.use('Agg')

file_path = "predictions.json"
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

def analyze_emotions():
    if "emotion" not in data or not isinstance(data["emotion"], list):
        raise ValueError("The JSON data does not contain a valid 'emotion' list.")

    emotion_counts = Counter(data["emotion"])
    output_dir = os.path.join("C:/EL-3rdsem/STRESS/ThirdAttempt/webapp/static", "graphs")
    os.makedirs(output_dir, exist_ok=True)

    # Bar Chart
    bar_chart_path = os.path.join(output_dir, "emotion_bar_chart.png")
    fig = plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.title('Emotion Frequencies', fontsize=16)
    plt.xlabel('Emotions', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    try:
        fig.savefig(bar_chart_path)
        print(f"Bar chart saved at {bar_chart_path}")
    except Exception as e:
        print(f"Error saving bar chart: {e}")
    plt.close(fig)

    # Line Chart
    line_chart_path = os.path.join(output_dir, "emotion_line_chart.png")
    time_in_seconds = np.arange(0, len(data["emotion"]))
    emotion_freq = [Counter(data["emotion"][:i + 1]).get(emotion, 0) for i, emotion in enumerate(data["emotion"])]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(time_in_seconds, emotion_freq, color='orange', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.title('Emotion Frequency Over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Emotion Frequency', fontsize=14)
    plt.grid(True)

    try:
        fig.savefig(line_chart_path)
        print(f"Line chart saved at {line_chart_path}")
    except Exception as e:
        print(f"Error saving line chart: {e}")
    plt.close(fig)

    return {
        "bar_chart_path": bar_chart_path,
        "line_chart_path": line_chart_path,
        "counts": dict(emotion_counts)
    }

# Run the analysis
chart_info = analyze_emotions()

# Output the results
print("Emotion counts:", chart_info["counts"])
