import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # gui/

import streamlit as st

import lessons
import ml_lessons2

st.title("M7 · Practical ML")
st.caption("The craft that decides whether a project works: features, scaling, leakage, pipelines.")
lessons.render_lesson_content(ml_lessons2.PRACTICAL)
