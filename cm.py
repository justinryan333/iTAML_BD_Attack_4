import pickle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved predictions
with open("final_predictions/final_preds_sess_4.pickle", "rb") as f:
    data = pickle.load(f)

preds = data["preds"]
labels = data["labels"]

# Generate confusion matrix
cm = confusion_matrix(labels, preds)

# Optional: print classification metrics
print(classification_report(labels, preds, digits=4))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Final Task")
plt.show()
