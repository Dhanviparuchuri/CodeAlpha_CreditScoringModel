import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load the dataset
data = pd.read_csv("credit_risk_dataset.csv")

print("Dataset size:", data.shape)
print("\nFirst five rows:")
print(data.head())

# 2. Remove rows with missing values
data = data.dropna()

# 3. Separate features and target
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

# 4. Convert text categories into numeric columns
X = pd.get_dummies(X, drop_first=True)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# 6. Create and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No default", "Default"],
        zero_division=0
    )
)

# 9. Create and save the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No default", "Default"]
)

display.plot(cmap="viridis")
plt.title("Credit Scoring - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)

# 10. Save the trained model and feature column names
joblib.dump(model, "credit_scoring_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nModel saved successfully!")
print("Feature columns saved successfully!")
print("Confusion matrix saved as confusion_matrix.png")

# 11. Show the confusion matrix
plt.show()