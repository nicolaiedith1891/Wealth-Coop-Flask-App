import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('../data-files/loan_repayment.csv')

# Select relevant features (adjust based on your actual columns)
df = df[['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 
         'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 
         'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid']]

# Remove duplicates
df.drop_duplicates(inplace=True)

# Create target variable (percentage repaid)
# Assuming 'not.fully.paid' is inverse of repayment (0 = fully paid, 1 = not fully paid)
df['percentage_repaid'] = 1 - df['not.fully.paid']  # Basic transformation - adjust as needed

# Define numerical & categorical features
num_features = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 
               'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths']
cat_features = ['credit.policy', 'purpose', 'delinq.2yrs', 'pub.rec']  # Treating these as categorical

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())  
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full transformation pipeline
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Prepare features and target
X = df.drop(['percentage_repaid', 'not.fully.paid'], axis=1)  # Exclude target and its source
y = df['percentage_repaid']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformations
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.transform(X_test)

# Define regressors
regressors = {
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

# Train, evaluate, and store results
results = {}

for name, model in regressors.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"RMSE": rmse, "R2": r2, "model": model}
    print(f"{name} → RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Select the best model based on R² score (or RMSE if preferred)
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = results[best_model_name]["model"]

print(f"\nBest Model: {best_model_name} → R²: {results[best_model_name]['R2']:.4f}")

# Save the best model and preprocessor
with open("repayment-model/model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("repayment-model/preprocessor.pkl", "wb") as file:
    pickle.dump(full_pipeline, file)