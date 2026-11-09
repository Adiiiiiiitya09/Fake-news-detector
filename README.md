# Fake News Detection Using Machine Learning

This project implements the methodology outlined in the `REPORT.fake.news.detection.app.pdf`. It uses machine learning algorithms to classify news articles as either "True" or "Fake" based on their textual content.

## Project Structure
- `train.py`: The main script that loads datasets, preprocesses textual data, visualizes categories, vectorizes texts, and trains machine learning models.
- `requirements.txt`: Necessary Python libraries listed in the report.

## Features & Methodology
1. **Data Preprocessing**: Cleans the text data by removing punctuation, digits, HTML tags, and links using Python's `re` and `string` libraries.
2. **Feature Extraction**: Uses TF-IDF (`TfidfVectorizer`) to convert text into numeric arrays.
3. **Machine Learning Models Evaluated**:
    - Naive Bayes
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
4. **Data Visualization**: Generates bar charts and confusion matrices utilizing `Matplotlib` and `Seaborn`.

## Setup Instructions

### 1. Install Requirements
Ensure you have Python 3 installed. Then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
The project relies on the ISOT Fake News dataset, which contains separate files for real and fake news. Download it from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

The dataset files are not included in this repository. After downloading them:
- Create a folder named `data` inside this directory:
  ```bash
  mkdir data
  ```
- Download the datasets and name them `True.csv` and `Fake.csv`.
- Place both files into the `data/` folder.

### 3. Run the Project
Execute the training script:
```bash
python train.py
```

### 4. Run the App
After training (or with the included model files), launch the Streamlit app:
```bash
streamlit run app.py
```

## Outputs
When you run the script, it will display the training progress, accuracy scores, and a detailed classification report in the terminal.

It will also output several visualization files locally:
- `news_categories.png`: Distribution of news categories (politics, US_news, worldnews, etc.).
- `real_fake_count.png`: Document count for True and Fake datasets.
- `model_accuracies.png`: Bar chart comparing the final accuracies across the 5 models.
- `confusion_matrix_[Model_Name].png`: Detailed matrix of false positives vs. false negatives.