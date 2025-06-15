import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib

df = pd.read_csv("fake_job_postings.csv")
df = df.dropna(subset=['fraudulent'])

X = df[['description', 'employment_type', 'required_experience', 'required_education']].fillna("Unknown")
y = df['fraudulent']

preprocessor = ColumnTransformer([
    ('desc_tfidf', TfidfVectorizer(max_features=300, stop_words='english'), 'description'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['employment_type', 'required_experience', 'required_education'])
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")
