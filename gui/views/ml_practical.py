import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lessons
import ml_lessons2

st.title("M7 · Practical ML")
st.caption("The craft that decides whether a project works: features, scaling, leakage, pipelines.")

lessons.predict(
    'In a real ML project, where does most effort actually go — picking the model, or something else? And what single mistake silently inflates your test score?',
    'Most effort goes into **data and features** (cleaning, scaling, encoding, missing values), not model choice. The silent killer is **data leakage** — fitting a transform (scaling, feature selection) on the *whole* dataset before the split, so test info bleeds in. Fit inside a **pipeline**, on the training fold only.',
)
lessons.render_lesson_content(ml_lessons2.PRACTICAL, solutions=[
    ("Hands-on 1–3",
     r"""A `Pipeline` (impute → one-hot → scale → logistic) cross-validates *without leakage* because every fold refits the transforms on its own training rows. Good engineered features raise the CV score. On an **imbalanced** set, accuracy looks great while **F1 / PR-AUC** expose the failure — class weights or SMOTE, applied *inside* the CV folds, fix it."""),
    ("Spot-the-leak 4",
     r"""Three leaks: (a) scaling/encoding fit on **all** rows before the split; (b) a feature built from the future or the target (e.g. "total future spend"); (c) duplicate or time-overlapping rows across train/test. The test for each: *"would this value be available at prediction time, computed only from training data?"* — and fit every transform inside a `Pipeline`."""),
    ("Concept 5–6",
     r"""**5.** Need scaling (distance / gradient based): **SVM, k-NN, k-means, logistic regression**. Don't (split-based, scale-invariant): **random forest, gradient boosting**.

**6.** Target encoding uses the label, so computing it on the full dataset leaks test labels into the features; done inside each CV fold (training rows only) it stays honest."""),
    ("Bridge 7",
     r"""Standardization here is the same idea as **norms** in the Math module, and it's why ANN training wants **normalized inputs** and good **initialization**: comparable scales make the loss surface rounder so gradient descent behaves (the ravine story from X4 / Optimizers)."""),
])
