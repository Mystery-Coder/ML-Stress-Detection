import matplotlib.pyplot as plt
import json

# Load the JSON data
file_path = "predictions.json"
with open(file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Define the unique emotions and their mapping to y-axis values
unique_depression = ["low","mid","high"]
depression_to_y = {depression: idx for idx, depression in enumerate(unique_depression)}

# Extract emotion data and time intervals
depressions = json_data["depression"]
time_intervals = range(0, len(depressions) * 2*60, 2*60)  # Each data point is 3 seconds apart

# Map emotions to y-axis values
y_values = [depression_to_y[depression] for depression in depressions]

# Plot the line graph
plt.figure(figsize=(12, 6))
plt.plot(time_intervals, y_values, marker='o', linestyle='-', color='b', label="Depression Path")

# Customize the y-axis with emotion labels
plt.yticks(ticks=list(depression_to_y.values()), labels=unique_depression)
plt.xlabel("Time (seconds)")
plt.ylabel("Depression")
plt.title("Depreession Transition Over Time")
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
