import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# ========================
# Load Data
# ========================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")

y_test = np.load(os.path.join(data_dir, "y_test.npy"))
y_pred_rf = np.load(os.path.join(data_dir, "y_pred_rf.npy"))
y_pred_svm = np.load(os.path.join(data_dir, "y_pred_svm.npy"))
y_pred_dt = np.load(os.path.join(data_dir, "y_pred_dt.npy"))

# ========================
# Confusion Matrix
# ========================
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Random  Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# Model Comparison (Accuracy)
# ===============================
models = ["Random Forest", "SVM", "Decision Tree"]
accuracies = [
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_svm),
    accuracy_score(y_test, y_pred_dt)
]
plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

# =========================
# 3. ROC CURVE
# =========================
# For ROC, we need probabilities → use RF only
# load probabilities if saved OR simulate
# If not saved, skip SVM & DT ROC (common practice)
fpr, tpr, _ = roc_curve(y_test, y_pred_rf)
roc_curve = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="Random Forest (AUC = %0.2f)" % roc_curve)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
