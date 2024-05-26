import streamlit as st
import subprocess
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import nltk
import re
def run_toxic():
    subprocess.run(["python", "toxic.py"])

def main():
    st.title("Comment Toxicity Classifier")
    if st.button("Run Toxicity Classifier"):
        run_toxic()

if __name__ == "__main__":
    main()
