import numpy as np
import os
from sklearn.model_selection import train_test_split

# =========================
# LOAD PROCESSED DATA
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")

X_scaled = np.load(os.path.join(data_dir, "X_scaled.npy"))
y = np.load(os.path.join(data_dir, "y.npy"))

print("Data Loaded:", X_scaled.shape, y.shape)

# =========================
# TRAIN & TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training Data:", X_train.shape, y_train.shape)
print("Testing Data:", X_test.shape, y_test.shape)

# =========================
# RANDOM FOREST MODEL
# =========================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=90, random_state=62, n_jobs=-4)
print("\nTraining Random Forest...")
rf.fit(X_train, y_train)
print("Random Forest Training Completed!")

# ====================
# SVM Model
# ====================
from sklearn.svm import LinearSVC

print("\nTraining SVM...")

# use linear kernel for speed
svm = LinearSVC(max_iter=2000)
svm.fit(X_train[:50000], y_train[:50000])  # use subset for speed
print("SVM Training Completed!")

# ======================
# DECISION TREE MODEL
# ======================
from sklearn.tree import DecisionTreeClassifier

print("\nTraining Decsion Tree...")
dt = DecisionTreeClassifier(random_state=92)
dt.fit(X_train, y_train)
print("Decision Tree Training Completed!")

# ======================
# Model Evaluation
# ======================
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

# Predictions
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_dt = dt.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_test, y_pred):
    print(f"\n {name} performance:")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test,y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    print("Accuracy:", acc)
    print("Precission:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("False Positive Rate:", fpr)

# Run Evaluation
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("Decision Tree", y_test, y_pred_dt)

# Save Models
np.save(os.path.join(data_dir, "y_test.npy"), y_test)
np.save(os.path.join(data_dir, "y_pred_rf.npy"), y_pred_rf)
np.save(os.path.join(data_dir, "y_pred_svm.npy"), y_pred_svm)
np.save(os.path.join(data_dir, "y_pred_dt.npy"), y_pred_dt)