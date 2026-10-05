# 📧 Spam Email Classifier

A machine learning project that classifies emails as **Spam** or **Ham (Not Spam)** using **TF-IDF** text vectorization and **Logistic Regression**.

## 🚀 Features

- Loads email data from `email.csv`
- Converts `ham` → 0 and `spam` → 1
- Uses TF-IDF for text feature extraction
- Uses balanced Logistic Regression for classification
- Evaluates the model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Tests the model on a sample email

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib

## 📂 Project Structure

```text
spam-email-classifier/
├── email.csv
├── spam_classifier.py
└── README.md
```

## ▶️ How to Run

Install dependencies:

```bash
pip install pandas scikit-learn joblib
```

Run the classifier:

```bash
python spam_classifier.py
```

The program trains the model, displays evaluation results, and predicts whether the sample email is **Spam** or **Ham**.

## 🧠 Model

The pipeline uses:

```text
Email Message
      ↓
TF-IDF Vectorizer
      ↓
Logistic Regression
      ↓
Spam / Ham
```

The model uses an 80/20 train-test split and balanced class weights.
