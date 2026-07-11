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
lessons.render_lesson_content(ml_lessons2.PRACTICAL)
