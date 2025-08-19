import pickle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your saved predictions
preds_custom_path = "final_predictions"
preds_filename = "final_session_ep70_predictions_with_ai.pickle"

with open(os.path.join(preds_custom_path, preds_filename), "rb") as f:
    data = pickle.load(f)

# You have two options:

# OPTION 1: Use the converted predictions (0-9) - RECOMMENDED
preds = data["predictions_ai"]  # Already global classes 0-9
labels = data["true_labels_ai"]  # Already global classes 0-9

# OPTION 2: Use raw predictions (0 or 1) - if you want to see task-level patterns
# preds_raw = data["predictions"]  # Local predictions (0 or 1)
# labels_raw = data["true_labels"]  # Local true labels (0 or 1)

print(f"Number of predictions: {len(preds)}")
print(f"Number of labels: {len(labels)}")

print(f"Unique predicted classes: {sorted(set(preds))}") # Should be 0 to 9
print(f"Unique true classes: {sorted(set(labels))}") # Should be 0 to 9

# Sample check
print("Sample predictions:", preds[5000:5010])
print("Sample true labels:", labels[5000:5010])

# Generate confusion matrix (using the converted 0-9 values)
cm = confusion_matrix(labels, preds)

# Print classification metrics
print(classification_report(labels, preds, digits=4))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Class (0-9)")
plt.ylabel("True Class (0-9)")
plt.title("Confusion Matrix - All Classes (Using +ai conversion)")
plt.tight_layout()
plt.show()
