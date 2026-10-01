import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv("email.csv")
data['spam'] = data['Category'].map({'ham': 0, 'spam': 1})

X = data['Message']
y = data['spam']
y.fillna(1, inplace=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with TF-IDF and balanced Logistic Regression
model = Pipeline([
    ('v', TfidfVectorizer(stop_words='english')),
    ('logistic_regression', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", model.score(X_test, y_test))
print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

dict = {
    'message' : "Congraulation you won $50000 Just give your credit card number"
}
df_test = pd.DataFrame([dict])
result = model.predict(df_test)
if result[0]==1:
    print('Spam')
elif  result[0]==0:
    print("Ham")
    