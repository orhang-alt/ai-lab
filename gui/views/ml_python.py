"""M8 — ML in Python (ML module).

The practical toolkit: how classical ML is actually done in Python (NumPy / pandas /
scikit-learn), with **live, runnable** worked examples — a real classifier and a
regression model-comparison that train in-page on scikit-learn's built-in datasets.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import pandas as pd
import streamlit as st

from sklearn.datasets import (load_breast_cancer, load_diabetes, load_iris,
                              load_wine)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import lessons
import ml_lessons2

LESSON = ml_lessons2.PYTHON_ML

# name -> (loader symbol, loader fn)
DATASETS = {
    "Wine (3 classes, 13 features)": ("load_wine", load_wine),
    "Iris (3 classes, 4 features)": ("load_iris", load_iris),
    "Breast cancer (2 classes, 30 features)": ("load_breast_cancer", load_breast_cancer),
}
# name -> (import line, constructor source, factory)
CLASSIFIERS = {
    "Logistic Regression": ("from sklearn.linear_model import LogisticRegression",
                            "LogisticRegression(max_iter=2000)",
                            lambda: LogisticRegression(max_iter=2000)),
    "Random Forest": ("from sklearn.ensemble import RandomForestClassifier",
                      "RandomForestClassifier(n_estimators=200, random_state=0)",
                      lambda: RandomForestClassifier(n_estimators=200, random_state=0)),
    "SVM (RBF kernel)": ("from sklearn.svm import SVC",
                         "SVC(kernel='rbf', random_state=0)",
                         lambda: SVC(kernel="rbf", random_state=0)),
    "k-Nearest Neighbors": ("from sklearn.neighbors import KNeighborsClassifier",
                            "KNeighborsClassifier()",
                            lambda: KNeighborsClassifier()),
}


@st.cache_data(show_spinner=False)
def run_classifier(ds_name: str, model_name: str) -> dict:
    bunch = DATASETS[ds_name][1]()
    X, y = bunch.data, bunch.target
    names = [str(n) for n in bunch.target_names]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    model = make_pipeline(StandardScaler(), CLASSIFIERS[model_name][2]())
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    cv = cross_val_score(model, X, y, cv=5)
    rep = classification_report(y_te, preds, target_names=names,
                                output_dict=True, zero_division=0)
    return {
        "acc": accuracy_score(y_te, preds),
        "cv_mean": cv.mean(), "cv_std": cv.std(),
        "cm": confusion_matrix(y_te, preds), "names": names,
        "rep": rep, "n": len(y), "feat": X.shape[1],
        "n_test": len(y_te),
    }


@st.cache_data(show_spinner=False)
def run_regression() -> pd.DataFrame:
    X, y = load_diabetes(return_X_y=True)
    models = [
        ("LinearRegression", LinearRegression()),
        ("Ridge (alpha=1)", Ridge(alpha=1.0)),
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=0)),
    ]
    rows = []
    for name, m in models:
        s = cross_val_score(m, X, y, cv=5, scoring="r2")
        rows.append({"model": name, "CV R² (mean)": round(float(s.mean()), 3),
                     "± std": round(float(s.std()), 3)})
    return pd.DataFrame(rows)


st.title("ML in Python — the practical toolkit")
st.caption("M8 · How classical ML is actually done in Python (NumPy · pandas · scikit-learn). "
           "The examples below really run — scikit-learn trains them in-page.")

lessons.predict(
    'Across every scikit-learn model — linear, tree, SVM — what three method names do you always call? What does that mean for swapping models?',
    '**`fit` / `predict` / `score`** — the same API for every estimator. So swapping models is a *one-line* change (just the constructor); the whole train / evaluate scaffold stays identical. That uniformity is exactly why pipelines and grid-search work across all models.',
)

tab_demo, tab_theory, tab_quiz, tab_tasks, tab_ref = st.tabs(
    ["🧪 Live examples", "📖 Theory", "❓ Self-check", "🛠 Tasks", "📚 References"]
)

with tab_demo:
    st.subheader("1 · Classification, end to end")
    st.caption("Pick a built-in dataset and a model — it splits, scales, trains and scores live.")
    c1, c2 = st.columns(2)
    ds_name = c1.selectbox("Dataset", list(DATASETS))
    model_name = c2.selectbox("Model", list(CLASSIFIERS))

    r = run_classifier(ds_name, model_name)

    m = st.columns(3)
    m[0].metric("Test accuracy", f"{r['acc']:.1%}")
    m[1].metric("5-fold CV", f"{r['cv_mean']:.1%}", help=f"± {r['cv_std']:.1%} across folds")
    m[2].metric("Data", f"{r['n']}×{r['feat']}", help="samples × features")

    left, right = st.columns(2)
    with left:
        st.caption("Confusion matrix (rows = true, cols = predicted)")
        cm_df = pd.DataFrame(r["cm"],
                             index=[f"true {c}" for c in r["names"]],
                             columns=[f"pred {c}" for c in r["names"]])
        st.dataframe(cm_df, width="stretch")
    with right:
        st.caption("Classification report (per class)")
        rep_df = pd.DataFrame(r["rep"]).T.round(3)
        st.dataframe(rep_df, width="stretch")

    loader_sym = DATASETS[ds_name][0]
    imp, ctor, _ = CLASSIFIERS[model_name]
    st.caption("The exact code that just ran:")
    st.code(
        f"""from sklearn.datasets import {loader_sym}
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
{imp}
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X, y = {loader_sym}(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

model = make_pipeline(StandardScaler(), {ctor})
model.fit(X_tr, y_tr)                       # learn
preds = model.predict(X_te)                 # use

print(accuracy_score(y_te, preds))          # -> {r['acc']:.3f}
print(classification_report(y_te, preds))
print(cross_val_score(model, X, y, cv=5).mean())   # -> {r['cv_mean']:.3f}
""",
        language="python",
    )
    st.info("Swap the **Model** dropdown and only the constructor line changes — that's the "
            "scikit-learn API: `fit` / `predict` / `score` for every model.", icon=":material/sync:")

    st.divider()
    st.subheader("2 · Regression — comparing models by cross-validation")
    st.caption("Three regressors on the diabetes dataset, scored by 5-fold CV R² (higher is better).")
    reg = run_regression()
    cc1, cc2 = st.columns([0.5, 0.5])
    cc1.dataframe(reg, hide_index=True, width="stretch")
    cc2.bar_chart(reg.set_index("model")["CV R² (mean)"], horizontal=True)
    st.code(
        """from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

X, y = load_diabetes(return_X_y=True)
for name, model in [("Linear", LinearRegression()),
                    ("Ridge", Ridge(alpha=1.0)),
                    ("Forest", RandomForestRegressor(n_estimators=200, random_state=0))]:
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(name, scores.mean())
""",
        language="python",
    )
    st.caption("Same `cross_val_score` call, three models — swapping estimators is one line each.")

with tab_theory:
    st.markdown(LESSON.theory, unsafe_allow_html=True)

with tab_quiz:
    st.subheader("Self-check")
    st.caption("Instant feedback, no grading.")
    lessons.render_quiz(LESSON.quiz, prefix=LESSON.key)

with tab_tasks:
    st.subheader("Tasks")
    st.markdown(LESSON.tasks)
    st.divider()
    st.markdown("#### ✅ Worked solutions")
    st.caption("Attempt each first, then check.")
    lessons.solution(
        r"""**Scaling.** SVM and logistic regression are distance/gradient-based, so they **need** feature scaling; tree-based models (random forest, gradient boosting) split one feature at a time and are **scale-invariant**.

**Swapping models.** Every estimator shares `fit` / `predict` / `score`, so `LogisticRegression` → `RandomForestClassifier` is a **one-line** change — the rest of the scaffold is identical.

**Tuning & pipelines.** Wrap `GridSearchCV` around the estimator and read `best_params_`. Put scaling + one-hot in a `ColumnTransformer` **inside** a `Pipeline` so every CV fold transforms on *training* data only — that's what prevents leakage.

**Save/load.** `joblib.dump` the fitted pipeline, reload it, and `predict` returns identical results — the same object you'd ship to production.""",
        label="Worked notes",
    )

with tab_ref:
    st.subheader("Reading & references")
    st.markdown(LESSON.references)
