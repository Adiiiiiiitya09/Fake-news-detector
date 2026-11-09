import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
import joblib
import os

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data():
    print("Loading datasets...")
    # The dataset requires a 'data' folder containing True.csv and Fake.csv
    if not os.path.exists('data/True.csv') or not os.path.exists('data/Fake.csv'):
        print("Error: data/True.csv or data/Fake.csv not found.")
        print("Please download the Fake News dataset (e.g., ISOT from Kaggle) and place them in a 'data' directory.")
        return None

    df_true = pd.read_csv('data/True.csv')
    df_fake = pd.read_csv('data/Fake.csv')

    # Add class labels: 1 for Real, 0 for Fake
    df_true['class'] = 1
    df_fake['class'] = 0

    # Merge datasets
    df_marge = pd.concat([df_fake, df_true], axis=0)
    
    # Store subjects for visualization before dropping columns
    subjects_df = df_marge[['subject', 'class']].copy()

    # Drop columns not needed for text classification
    df = df_marge.drop(["title", "subject", "date"], axis=1)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    print("Cleaning text...")
    def wordopt(text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r"\\W", " ", text) 
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)    
        return text

    df["text"] = df["text"].apply(wordopt)
    return df, subjects_df, df_marge

# --- 2. Visualization ---
def plot_data(subjects_df, df_marge):
    print("Generating dataset plots...")
    
    # Plot 1: News Categories distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=subjects_df, x='subject', hue='class')
    plt.title('News Categories Distribution (Fake vs True)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('news_categories.png')
    plt.close()
    
    # Plot 2: Real & Fake News Document Count
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_marge, x='class')
    plt.title('Real & Fake News Document Count')
    plt.xticks([0, 1], ['Fake', 'True'])
    plt.tight_layout()
    plt.savefig('real_fake_count.png')
    plt.close()
    
    print("Plots saved as news_categories.png and real_fake_count.png")

# --- 3. Model Training and Evaluation ---
def train_and_evaluate(df):
    x = df["text"]
    y = df["class"]

    # Splitting into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    print("Vectorizing text using TF-IDF...")
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Dictionary of all models mentioned in the report
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42) # using linear kernel for text data efficiency
    }

    accuracies = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(xv_train, y_train)
        pred = model.predict(xv_test)
        
        acc = accuracy_score(y_test, pred)
        accuracies[name] = acc
        
        print(f"{name} Accuracy: {acc * 100:.2f}%")
        print("-" * 50)
        print(classification_report(y_test, pred))
        
        # Save confusion matrix plot for each model
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()

    # Plot Model Accuracies Comparison
    print("\nGenerating model accuracies comparison plot...")
    plt.figure(figsize=(10, 6))
    names = list(accuracies.keys())
    scores = [acc * 100 for acc in accuracies.values()]
    sns.barplot(x=names, y=scores)
    plt.title('Classification Models Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(90, 100) # Setting limit from 90% to 100% since models typically perform >95%
    for i, score in enumerate(scores):
        plt.text(i, score + 0.1, f'{score:.2f}%', ha='center')
    plt.tight_layout()
    plt.savefig('model_accuracies.png')
    plt.close()
    print("Model accuracies plot saved as model_accuracies.png")

    # Save the best performing model and the vectorizer
    best_model_name = max(accuracies, key=accuracies.get)
    print(f"\nBest Model: {best_model_name} with Accuracy: {accuracies[best_model_name] * 100:.2f}%")
    
    print("Saving model and vectorizer...")
    joblib.dump(models[best_model_name], 'model.joblib')
    joblib.dump(vectorization, 'vectorizer.joblib')
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    result = load_and_preprocess_data()
    if result is not None:
        df, subjects_df, df_marge = result
        plot_data(subjects_df, df_marge)
        train_and_evaluate(df)
