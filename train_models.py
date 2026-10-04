import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Loading Titanic dataset...")
df_raw = sns.load_dataset('titanic')

# Save clean copy for exploration and batch prediction
df_raw.to_csv('data/titanic_raw.csv', index=False)

# Replicating notebook preprocessing faithfully
df = df_raw.copy()
df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1, inplace=True)
df['age'] = df['age'].fillna(df['age'].mean())
df.dropna(subset=['embarked'], inplace=True)

le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])  # female: 0, male: 1

le_embarked = LabelEncoder()
df['embarked'] = le_embarked.fit_transform(df['embarked'])  # C: 0, Q: 1, S: 2

df['alone'] = df['alone'].astype(int)
df = df.astype(int)

# Save processed dataset
df.to_csv('data/titanic_processed.csv', index=False)

# Save encoders
joblib.dump(le_sex, 'models/le_sex.joblib')
joblib.dump(le_embarked, 'models/le_embarked.joblib')

# Feature matrix & target
# Features: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']
X = df.drop('survived', axis=1)
y = df['survived']

# Train test split identical to notebook (test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.joblib')

# Model definitions
model_defs = {
    'svc': {
        'name': 'Support Vector Machine (SVC RBF)',
        'description': 'Best performing kernel model from the notebook with optimal decision boundary in high-dimensional space.',
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'scaled': True,
        'recommended': True
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting Classifier',
        'description': 'Advanced sequential ensemble boosting weak decision trees into a robust predictor.',
        'model': GradientBoostingClassifier(random_state=42, n_estimators=120, learning_rate=0.08, max_depth=3),
        'scaled': False,
        'recommended': False
    },
    'random_forest': {
        'name': 'Random Forest Classifier',
        'description': 'Ensemble of 100 randomized decision trees providing strong variance reduction.',
        'model': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6),
        'scaled': False,
        'recommended': False
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'description': 'Classic linear classification modeling log-odds of survival with high interpretability.',
        'model': LogisticRegression(max_iter=500, random_state=42),
        'scaled': True,
        'recommended': False
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'description': 'Hierarchical rule-based tree partitioner matching the notebook implementation.',
        'model': DecisionTreeClassifier(random_state=42, max_depth=5),
        'scaled': True,
        'recommended': False
    },
    'knn': {
        'name': 'K-Nearest Neighbors (k=5)',
        'description': 'Instance-based classification based on Euclidean distance across feature space.',
        'model': KNeighborsClassifier(n_neighbors=5),
        'scaled': True,
        'recommended': False
    },
    'naive_bayes': {
        'name': 'Gaussian Naive Bayes',
        'description': 'Probabilistic classifier assuming feature conditional independence given survival class.',
        'model': GaussianNB(),
        'scaled': False,
        'recommended': False
    }
}

metrics_summary = {}

for key, item in model_defs.items():
    m = item['model']
    xtr = X_train_scaled if item['scaled'] else X_train
    xte = X_test_scaled if item['scaled'] else X_test
    
    m.fit(xtr, y_train)
    joblib.dump(m, f'models/{key}.joblib')
    
    preds = m.predict(xte)
    probs = m.predict_proba(xte)[:, 1] if hasattr(m, 'predict_proba') else None
    
    cm = confusion_matrix(y_test, preds).tolist()
    
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    auc = float(roc_auc_score(y_test, probs)) if probs is not None else 0.0
    
    # Calculate death count accuracy on the entire dataset (889 passengers)
    full_X = scaler.transform(X) if item['scaled'] else X
    full_preds = m.predict(full_X)
    actual_deaths = int((y == 0).sum())
    predicted_deaths = int((full_preds == 0).sum())
    actual_survivors = int((y == 1).sum())
    predicted_survivors = int((full_preds == 1).sum())
    death_error = abs(predicted_deaths - actual_deaths)
    
    metrics_summary[key] = {
        'key': key,
        'name': item['name'],
        'description': item['description'],
        'scaled': item['scaled'],
        'recommended': item['recommended'],
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'roc_auc': round(auc * 100, 2),
        'confusion_matrix': cm,
        'manifest_stats': {
            'total_passengers': len(df),
            'actual_deaths': actual_deaths,
            'predicted_deaths': predicted_deaths,
            'actual_survivors': actual_survivors,
            'predicted_survivors': predicted_survivors,
            'death_error': death_error,
            'death_count_accuracy': round((1 - (death_error / actual_deaths)) * 100, 2)
        }
    }

with open('models/model_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

# Save feature metadata for frontend
feature_meta = {
    'features': ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone'],
    'sex_mapping': {'female': 0, 'male': 1},
    'embarked_mapping': {'C': 0, 'Q': 1, 'S': 2},
    'age_mean': float(df_raw['age'].mean()),
    'fare_mean': float(df_raw['fare'].mean())
}
with open('models/feature_meta.json', 'w') as f:
    json.dump(feature_meta, f, indent=2)

print("Models trained and exported successfully!")
for k, v in metrics_summary.items():
    print(f"{v['name']:35s} | Acc: {v['accuracy']}% | Pred Deaths: {v['manifest_stats']['predicted_deaths']} (Actual: {v['manifest_stats']['actual_deaths']})")
