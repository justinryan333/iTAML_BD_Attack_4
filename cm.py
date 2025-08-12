import pickle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved predictions
with open("./final_predictions/final_preds_sess_4.pickle", "rb") as f:
    data = pickle.load(f)

preds = data["preds"]
labels = data["labels"]
tasks = data["tasks"]

print(f"Number of predictions: {len(preds)}") # expect 10000
print(f"Number of labels: {len(labels)}") # expect 10000
print(f"Number of tasks: {len(tasks)}")

print(f"Unique predicted classes: {sorted(set(preds))}") # expect 0 to 9
print(f"Unique true classes: {sorted(set(labels))}") # expect 0 to 9
print(f"Unique predicted tasks: {sorted(set(tasks))}") # expect 0 to 9


print(preds[5000:5010])
print(labels[5000:5010])
print(tasks[5000:5010])


# Generate confusion matrix
cm = confusion_matrix(labels, preds)

# print classification metrics
print(classification_report(labels, preds, digits=4))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Final Task")
plt.show()
