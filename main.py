# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Step 2: Load dataset
df = pd.read_csv("loan_approval/loan_approval_dataset.csv")
print(df.columns)


# Step 3: Drop loan_id (not useful for ML)
df.drop("loan_id", axis=1, inplace=True)

# Step 4: Handle missing values
num_cols = df.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0])

# Step 5: Encode categorical features (education, self_employed)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Step 6: Split dataset
df.columns = df.columns.str.strip()

# Now drop
X = df.drop(columns=['loan_status', 'cibil_score'])
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 8: Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_res, y_train_res)
y_pred_log = log_reg.predict(X_test)

print("🔹 Logistic Regression Performance:")
print(classification_report(y_test, y_pred_log))

# Step 9: Train Decision Tree
dt = DecisionTreeClassifier(
    max_depth=6,              # limit depth of the tree
    min_samples_split=20,     # minimum samples to split a node
    min_samples_leaf=10,      # minimum samples in a leaf
    random_state=42
)

dt.fit(X_train_res, y_train_res)
y_pred_dt = dt.predict(X_test)

print("🔹 Decision Tree Performance:")
print(classification_report(y_test, y_pred_dt))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ✅ Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,   # number of trees in the forest
    max_depth=8,        # limit tree depth (prevents overfitting)
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

rf.fit(X_train_res, y_train_res)

# ✅ Predictions
y_pred_rf = rf.predict(X_test)

# ✅ Performance Report
print("🔹 Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))

