"""Part A: TF-IDF + Logistic Regression baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import SEED, set_seed
from data import load_splits
from utils import compute_metrics, plot_confusion_matrix, save_approach_metrics, timer


def main():
    set_seed()
    train_df, val_df, test_df = load_splits()

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)

    with timer() as t:
        X_train = vectorizer.fit_transform(train_df["full_text"])
        clf.fit(X_train, train_df["label"])
    train_time = t["seconds"]

    X_test = vectorizer.transform(test_df["full_text"])
    y_pred = clf.predict(X_test)
    y_true = test_df["label"]

    metrics = compute_metrics(y_true, y_pred)
    metrics["train_time_seconds"] = train_time
    metrics["approach"] = "A - TF-IDF + Logistic Regression"

    plot_confusion_matrix(y_true, y_pred, "A_baseline_tfidf_lr")
    save_approach_metrics("A_baseline", metrics)


if __name__ == "__main__":
    main()
