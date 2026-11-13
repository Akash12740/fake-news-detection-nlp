# fake-news-detection-nlp

# Fake News Detection Pipeline (NLP & Text Classification)

An end-to-end Natural Language Processing (NLP) classification pipeline built to detect misinformation in news articles. By combining linguistic text normalization (NLTK), n-gram TF-IDF feature engineering, and hyperparameter-tuned classifiers, the model achieves **95% classification accuracy** on held-out test data.

## Pipeline Architecture
```text
Raw News Corpus ──▶ Regex & HTML Cleaning ──▶ Tokenization & Stopword Removal (NLTK)
                                                               │
                                                               ▼
Evaluation (95% Acc) ◀── GridSearchCV Tuning ◀── TF-IDF Vectorizer (1,2-grams)
