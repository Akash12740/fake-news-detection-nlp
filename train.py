import argparse
import re
import string
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already cached."""
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


class TextPreprocessor:
    """Clean, tokenize, remove stopwords, and lemmatize raw news text."""

    def __init__(self) -> None:
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)
        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in text.split()
            if tok not in self.stop_words and len(tok) > 2
        ]
        return " ".join(tokens)


def load_sample_data() -> pd.DataFrame:
    """Fallback sample corpus for immediate end-to-end verification."""
    samples = [
        ("Central bank announces quarterly interest rate decision after policy review", 0),
        ("Researchers publish peer reviewed climate study in international journal", 0),
        ("Ministry releases annual renewable energy capacity report for fiscal year", 0),
        ("Local transit authority expands metro rail network with three new stations", 0),
        ("University engineering team develops low cost water filtration membrane", 0),
        ("SHOCKING secret cure big pharma is hiding from every citizen click now", 1),
        ("Miracle overnight pill melts fifty pounds without diet or exercise guaranteed", 1),
        ("Aliens secretly control global weather satellites leaked insider claims", 1),
        ("Government to ban all drinking water starting tomorrow viral post warns", 1),
        ("Share this message immediately or your bank account will be frozen tonight", 1),
    ] * 20
    return pd.DataFrame(samples, columns=["text", "label"])


def train_and_evaluate(data_path: str | None, model_out: str) -> None:
    df = pd.read_csv(data_path) if data_path else load_sample_data()
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        df["content"] = df["text"].fillna("")

    preprocessor = TextPreprocessor()
    df["clean_text"] = df["content"].apply(preprocessor.clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    candidates = {
        "MultinomialNB": (
            MultinomialNB(),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {"clf__C": [0.5, 1.0, 5.0]},
        ),
        "PassiveAggressive": (
            PassiveAggressiveClassifier(max_iter=1000, random_state=42),
            {"clf__C": [0.1, 0.5, 1.0]},
        ),
    }

    best_model = None
    best_acc = 0.0
    best_name = ""

    for name, (estimator, param_grid) in candidates.items():
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
                ("clf", estimator),
            ]
        )
        search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)

        preds = search.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n[{name}] Best Params: {search.best_params_} | Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

        if acc >= best_acc:
            best_acc = acc
            best_model = search.best_estimator_
            best_name = name

    joblib.dump(best_model, model_out)
    print(f"Saved best model ({best_name}, accuracy={best_acc:.4f}) to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection NLP Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--out", type=str, default="fake_news_pipeline.joblib", help="Output model path")
    args = parser.parse_args()
    train_and_evaluate(args.data, args.out)
