import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def clean_numeric(value):
    """Helper function to clean numeric values"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Remove currency symbols, commas, and any other non-numeric characters
        cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else np.nan
    return float(value)

# Load data
df = pd.read_csv('../data-files/loan_forcast.csv')

# Clean numeric columns - convert all to proper float values
numeric_cols = ['customer_income', 'loan_amnt', 'loan_int_rate', 
                'employment_duration', 'term_years', 'cred_hist_length']

for col in numeric_cols:
    df[col] = df[col].apply(clean_numeric)

# Handle missing values - use proper assignment to avoid FutureWarnings
df = df.assign(
    historical_default=df['historical_default'].fillna('N'),
    loan_int_rate=df['loan_int_rate'].fillna(df['loan_int_rate'].median())
)

# Remove rows where target variable (loan_amnt) is NaN
print(f"Initial rows: {len(df)}")
df = df.dropna(subset=['loan_amnt'])
print(f"Rows after dropping NaN in loan_amnt: {len(df)}")

# Verify no NaN in target
if df['loan_amnt'].isna().any():
    raise ValueError("Target variable still contains NaN after cleaning!")

# Define features
num_features = ['customer_age', 'customer_income', 'employment_duration', 
                'loan_int_rate', 'term_years', 'cred_hist_length']
cat_features = ['home_ownership', 'loan_intent', 'loan_grade', 'historical_default']

# Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full pipeline
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Prepare data - drop non-feature columns
X = df.drop(columns=['customer_id', 'loan_amnt', 'Current_loan_status'])
y = df['loan_amnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Save artifacts
with open('loan-forcast-model/loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('loan-forcast-model/loan_preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)