import streamlit as st
import subprocess

def run_toxic():
    subprocess.run(["python", "toxic.py"])

def main():
    st.title("Comment Toxicity Classifier")
    if st.button("Run Toxicity Classifier"):
        run_toxic()

if __name__ == "__main__":
    main()
