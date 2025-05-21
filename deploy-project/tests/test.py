import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("../data-files/loan_repayment.csv")

# Simulate a target: percent_repaid (0.0 to 1.0)
df['percent_repaid'] = (
    0.3 * (df['fico'] - df['fico'].min()) / (df['fico'].max() - df['fico'].min()) +
    0.2 * (1 - df['int.rate']) +
    0.2 * (df['log.annual.inc'] - df['log.annual.inc'].min()) / (df['log.annual.inc'].max() - df['log.annual.inc'].min()) +
    0.2 * (1 - df['dti']) +
    0.1 * df['credit.policy']
)

df['percent_repaid'] = (df['percent_repaid'] - df['percent_repaid'].min()) / (df['percent_repaid'].max() - df['percent_repaid'].min())

# Define features and target
X = df.drop('percent_repaid', axis=1)
y = df['percent_repaid']

# Identify numeric and categorical features
num_features = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                'days.with.cr.line', 'revol.bal', 'revol.util']
cat_features = ['purpose', 'credit.policy']  # treat 'credit.policy' as categorical if desired

# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

print(df.head())