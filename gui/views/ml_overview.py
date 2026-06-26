import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

st.title("🧮 Machine Learning")
st.caption("Classical ML — same approach as the ANN module: theory + interactive playground + "
           "self-check + tasks + references, one topic at a time.")

st.info("The full M0–M7 curriculum is built — pick any module in the sidebar. "
        "M0–M6 all include interactive playgrounds (M7 is reference).",
        icon=":material/rocket_launch:")

st.subheader("Roadmap")
st.markdown(r"""
- **M0 — Foundations** ✅ *(Foundations lesson built)* — what ML is; supervised vs.
  unsupervised; the train / validation / test split; bias–variance; over/underfitting.
- **M1 — Regression** ✅ *(Regression playground built)* — linear regression
  (least squares + gradient descent), polynomial features, ridge / lasso, R².
- **M2 — Classification** ✅ *(Classification playground built)* — logistic regression
  (the sigmoid neuron), decision boundaries, the threshold, and metrics (accuracy,
  precision / recall, F1, ROC–AUC, confusion matrix). Next: k-NN, naive Bayes.
- **M3 — Trees & ensembles** ✅ — decision trees, random forests, gradient boosting
  (with an interactive depth → overfitting playground).
- **M4 — SVM & kernels** ✅ — max-margin classifiers, support vectors, the kernel trick
  (interactive linear-vs-RBF boundary on XOR).
- **M5 — Unsupervised** ✅ — k-means (interactive), hierarchical clustering, PCA, t-SNE / UMAP.
- **M6 — Model selection** ✅ — cross-validation, hyperparameter search, regularization,
  pipelines, data leakage (interactive k-fold validation curve).
- **M7 — Practical ML** ✅ — feature engineering, scaling, missing data, imbalance,
  leakage, scikit-learn end-to-end.
""")

st.subheader("Where ML meets ANN")
st.markdown(r"""
The two modules are not separate worlds — they meet at the single neuron:

| single neuron | + loss | = classical model |
|---|---|---|
| linear activation | MSE | **linear regression** (M1) |
| sigmoid activation | cross-entropy | **logistic regression** (M2) |

So everything you learned about weights, bias, and activations in the **ANN**
module pays off directly here. Switch modules anytime with the selector at the top
of the sidebar.
""")
