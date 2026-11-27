"""Data mining functions for the classroom facility issue dashboard."""
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

MODELS = {
    "Naive Bayes": lambda: MultinomialNB(),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Linear SVM": lambda: LinearSVC(),
}


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date_reported", "date_resolved"])
    df["month"] = df["date_reported"].dt.to_period("M").astype(str)
    return df


def train_classifiers(df, test_size=0.2, seed=42):
    """80/20 stratified split; returns metrics table, fitted pipelines, test data."""
    X, y = df["issue_description"], df["issue_category"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    rows, fitted = [], {}
    for name, make in MODELS.items():
        pipe = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2)), make())
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                     "Precision": p, "Recall": r, "F1-score": f})
        fitted[name] = (pipe, pred)
    metrics = pd.DataFrame(rows).round(3)
    return metrics, fitted, yte


def confusion(yte, pred):
    labels = sorted(set(yte) | set(pred))
    return pd.DataFrame(confusion_matrix(yte, pred, labels=labels), index=labels, columns=labels)


def build_transactions(df):
    """One transaction = the set of issue categories reported in one room in one month."""
    return df.groupby(["room", "month"])["issue_category"].apply(lambda s: sorted(set(s)))


def mine_rules(df, min_support=0.02, min_confidence=0.2):
    """Association rules (A -> B). Uses mlxtend Apriori if installed, else a pairwise fallback."""
    trans = build_transactions(df)
    n = len(trans)
    items = sorted({i for t in trans for i in t})
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        onehot = pd.DataFrame([{i: (i in t) for i in items} for t in trans])
        freq = apriori(onehot, min_support=min_support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)]
        out = pd.DataFrame({
            "If issue": rules["antecedents"].apply(lambda s: next(iter(s))),
            "Then also": rules["consequents"].apply(lambda s: next(iter(s))),
            "Support": rules["support"], "Confidence": rules["confidence"], "Lift": rules["lift"],
        })
    except ImportError:
        sup = {i: sum(i in t for t in trans) / n for i in items}
        rows = []
        for a, b in itertools.permutations(items, 2):
            s_ab = sum(a in t and b in t for t in trans) / n
            if s_ab >= min_support and sup[a] > 0:
                conf = s_ab / sup[a]
                if conf >= min_confidence:
                    rows.append({"If issue": a, "Then also": b, "Support": s_ab,
                                 "Confidence": conf, "Lift": conf / sup[b]})
        out = pd.DataFrame(rows, columns=["If issue", "Then also", "Support", "Confidence", "Lift"])
    return out.sort_values("Lift", ascending=False).round(3).reset_index(drop=True)


def priority_scores(df):
    """Transparent room priority score: volume + high-priority share + recurrence + unresolved."""
    g = df.groupby(["building", "room"])
    out = g.agg(total_issues=("report_id", "count"),
                high_priority=("priority", lambda s: (s == "High").sum()),
                months_with_issues=("month", "nunique"),
                open_issues=("status", lambda s: (s != "Resolved").sum()),
                top_issue=("issue_category", lambda s: s.mode().iat[0])).reset_index()
    for c in ["total_issues", "high_priority", "months_with_issues", "open_issues"]:
        out[c + "_n"] = out[c] / max(out[c].max(), 1)
    out["priority_score"] = (0.35 * out["total_issues_n"] + 0.30 * out["high_priority_n"]
                             + 0.20 * out["months_with_issues_n"] + 0.15 * out["open_issues_n"]).round(3)
    out["action"] = pd.cut(out["priority_score"], [-1, 0.4, 0.65, 2],
                           labels=["Routine monitoring", "Schedule inspection", "Immediate repair / inspection"])
    return out.drop(columns=[c for c in out if c.endswith("_n")]).sort_values(
        "priority_score", ascending=False).reset_index(drop=True)
