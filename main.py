import os

print("Starting Cybersecurity ML Pipeline...\n")

# get current project directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# build paths
feature_path = os.path.join(base_dir, "src", "feature_engineering.py")
model_path = os.path.join(base_dir, "src", "model_training.py")
visual_path = os.path.join(base_dir, "src", "data_visualization.py")

# =========================
# STEP 1
# =========================
print("Step 1: Feature Engineering...")
os.system(f'python "{feature_path}"')

# =========================
# STEP 2
# =========================
print("\nStep 2: Model Training...")
os.system(f'python "{model_path}"')

# =========================
# STEP 3
# =========================
print("\nStep 3: Data Visualization...")
os.system(f'python "{visual_path}"')

print("\nPipeline Completed Successfully!")