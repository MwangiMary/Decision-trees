import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# --- 1. Data Loading ---
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names
print("Dataset Loaded. Features (X) shape:", X.shape, "Target (y) shape:", y.shape)

# --- 2. Data Preprocessing ---

# Handling Missing Values (Mentioned but Skipped for clean Iris)
# If data had missing values:
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)

# Feature/Label Separation: Already done in loading (X, y)

# Train-Test Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train/Test split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

# Label Encoding (Mentioned but Skipped as target is already integer-encoded)
# If target were strings like ['setosa', 'versicolor', 'virginica']:
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# --- 3. Model Training ---
model = DecisionTreeClassifier(max_depth=4, random_state=42) # Instantiate the model
model.fit(X_train, y_train) # Train the model
print("Decision Tree Model Trained.")

# --- 4. Evaluation ---
y_pred = model.predict(X_test) # Generate predictions

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
# Use 'macro' average for multi-class to treat all classes equally
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')

print("\n--- Evaluation Metrics (Macro Average) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall (macro): {recall_macro:.4f}")

# --- 5. Pro Tip: Plot the Decision Tree Structure ---
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    filled=True,
    feature_names=feature_names,
    class_names=target_names,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Structure for Iris Classification")
# plt.show() # Uncomment to display the plot
print("Decision Tree plot generated.")