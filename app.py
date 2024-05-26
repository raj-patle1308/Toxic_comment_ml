import streamlit as st
import subprocess

def run_toxic():
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import roc_auc_score, roc_curve
    import pickle
    import nltk
    import re
    
    # Download NLTK resources
    nltk.download("punkt")
    nltk.download("omw-1.4")
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    
    # Define a function to prepare text data
    def prepare_text(text):
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return nltk.corpus.wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return nltk.corpus.wordnet.VERB
            elif treebank_tag.startswith('N'):
                return nltk.corpus.wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return nltk.corpus.wordnet.ADV
            else:
                return nltk.corpus.wordnet.NOUN
    
        text = re.sub(r'[^a-zA-Z\']', ' ', text)
        text = nltk.word_tokenize(text)
        text = nltk.pos_tag(text)
        lemma = []
        for word, pos in text:
            lemma.append(nltk.WordNetLemmatizer().lemmatize(word, pos=get_wordnet_pos(pos)))
        return ' '.join(lemma)
    
    # Load data
    df = pd.read_csv('FinalBalancedDataset.csv')
    data = df.drop("Unnamed: 0", axis=1)
    
    # Prepare text data
    data["clean_tweets"] = data["tweet"].apply(prepare_text)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['clean_tweets'], data['Toxicity'], test_size=0.8, random_state=42)
    
    # Initialize and fit TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train Multinomial Naive Bayes model
    model_bayes = MultinomialNB()
    model_bayes.fit(X_train_tfidf, y_train)
    
    # Predict probabilities on the test set
    y_pred_proba = model_bayes.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate ROC AUC score
    final_roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("Final ROC AUC Score:", final_roc_auc)
    
    # Save the trained TfidfVectorizer
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    
    def classify_comment():
        # Get input comment from entry widget
        comment = entry_comment.get()
        # Preprocess the comment
        preprocessed_comment = prepare_text(comment)
        # Transform the preprocessed comment using the TfidfVectorizer
        tfidf_comment = tfidf_vectorizer.transform([preprocessed_comment])
        # Predict toxicity using the trained model
        toxicity_prob = model_bayes.predict_proba(tfidf_comment)[0][1]
        # Display the result
        if toxicity_prob >= 0.5:
            result = "Toxic"
        else:
            result = "Non-Toxic"
        messagebox.showinfo("Classification Result", f"The comment is {result}.")
    
        # Text to predict
    text_to_predict = "it was a bullshit"
    
    # Preprocess the text
    preprocessed_text = prepare_text(text_to_predict)
    
    # Transform the preprocessed text using the TfidfVectorizer
    tfidf_text = tfidf_vectorizer.transform([preprocessed_text])
    
    # Use the trained Multinomial Naive Bayes model to predict probabilities
    predicted_probability = model_bayes.predict_proba(tfidf_text)[0][1]
    
    # Determine if the text is toxic based on the predicted probability
    if predicted_probability >= 0.5:
        print("The text is toxic.")
    else:
        print("The text is not toxic.")
    
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import roc_auc_score, roc_curve
    import pickle
    import nltk
    import re
    import tk
    from tk import messagebox
    
    # Download NLTK resources
    nltk.download("punkt")
    nltk.download("omw-1.4")
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    
    # Define a function to prepare text data
    def prepare_text(text):
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return nltk.corpus.wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return nltk.corpus.wordnet.VERB
            elif treebank_tag.startswith('N'):
                return nltk.corpus.wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return nltk.corpus.wordnet.ADV
            else:
                return nltk.corpus.wordnet.NOUN
    
        text = re.sub(r'[^a-zA-Z\']', ' ', text)
        text = nltk.word_tokenize(text)
        text = nltk.pos_tag(text)
        lemma = []
        for word, pos in text:
            lemma.append(nltk.WordNetLemmatizer().lemmatize(word, pos=get_wordnet_pos(pos)))
        return ' '.join(lemma)
    
    # Load data
    df = pd.read_csv(r"FinalBalancedDataset.csv")
    data = df.drop("Unnamed: 0", axis=1)
    
    # Prepare text data
    data["clean_tweets"] = data["tweet"].apply(prepare_text)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['clean_tweets'], data['Toxicity'], test_size=0.8, random_state=42)
    
    # Initialize and fit TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train Multinomial Naive Bayes model
    model_bayes = MultinomialNB()
    model_bayes.fit(X_train_tfidf, y_train)
    
    # Save the trained TfidfVectorizer
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    
    def classify_comment():
        # Get input comment from entry widget
        comment = entry_comment.get()
        # Preprocess the comment
        preprocessed_comment = prepare_text(comment)
        # Transform the preprocessed comment using the TfidfVectorizer
        tfidf_comment = tfidf_vectorizer.transform([preprocessed_comment])
        # Predict toxicity using the trained model
        toxicity_prob = model_bayes.predict_proba(tfidf_comment)[0][1]
        # Display the result
        if toxicity_prob >= 0.5:
            result = "Toxic"
        else:
            result = "Non-Toxic"
        messagebox.showinfo("Classification Result", f"The comment is {result}.")
    
    root = Tk()
    root.title("Login")
    root.geometry('925x500+300+200')
    root.configure(bg="#fff")
    root.resizable(False,False)
    
    def signin():
        username = user.get()
        password = code.get()
    
        if username == 'admin' and password =='1234':
            screen = Toplevel(root)
            screen.title("App")
            screen.geometry("925x500+300+200")
            screen.config(bg='White')
            Label(screen, text="Hello World", bg='#fff', font=('Calibri(Body)')).pack(expand = True)
    
            # GUI for comment classification
            comment_frame = Frame(screen, bg="#fff")
            comment_frame.place(x=50, y=200)
    
            Label(comment_frame, text="Enter Comment:", bg="#fff", font=("Calibri", 12)).grid(row=0, column=0, padx=10, pady=10)
    
            global entry_comment
            entry_comment = Entry(comment_frame, width=50)
            entry_comment.grid(row=0, column=1, padx=10, pady=10)
    
            classify_button = Button(comment_frame, text="Classify", command=classify_comment)
            classify_button.grid(row=1, columnspan=2, pady=10)
    
            screen.mainloop()
        elif username!= 'admin' and password != '1234':
            messagebox.showerror("Invalid", "invalid username and password")
        elif password != '1234':
            messagebox.showerror("Invalid", "invalid password")
        elif username != 'admin':
            messagebox.showerror("Invalid", "invalid username")
    
    img = PhotoImage(file=r"login.png")
    Label(root,image=img, bg="White").place(x=50, y=50)
    
    frame = Frame(root, width=350, height=350, bg="White" )
    frame.place(x=480, y=70)
    
    heading = Label(frame, text="Sign in", fg="#57a1f8",bg='White', font=('Microsoft YaHei UI Light', 23, 'bold'))
    heading.place(x=100,y=5)
    
    def on_enter(e):
        user.delete(0, 'end')
    
    def on_leve(e):
        name = user.get()
        if name == '':
            user.insert(0, 'Username')
    
    user = Entry(frame, width=25, fg='black', border=2, bg="White", font=('Microsoft YaHei UI Light', 11))
    user.place(x=30, y=80)
    user.insert(0,'Username')
    user.bind('<FocusIn>', on_enter)
    user.bind('<FocusOut>',on_leve)
    Frame(frame,width=295, height=2, bg='black').place(x=25, y=107)
    
    def on_enter(e):
        code.delete(0, 'end')
    
    def on_leve(e):
        name = code.get()
        if name == '':
            code.insert(0, 'Password')
    
    code = Entry(frame, width=25, fg='black', border=2, bg="White", font=('Microsoft YaHei UI Light', 11))
    code.place(x=30, y=150)
    code.insert(0,'Password')
    code.bind('<FocusIn>', on_enter)
    code.bind('<FocusOut>',on_leve)
    Frame(frame,width=295, height=2, bg='black').place(x=25, y=177)
    
    Button(frame, width=39, pady=7, text="Sign in", bg='#57a1f8', fg='White', border=0, command= signin).place(x=35, y=204)
    label = Label(frame,text="Dont Have any account?", fg='black', bg='White', font=('Microsoft YaHei UI Light',9))
    label.place(x=75, y=270)
    
    sign_up = Button(frame, width=6, text="Sign up", border=0, bg='White', cursor='hand2', fg='#57a1f8')
    sign_up.place(x=215,y=270)
    
    root.mainloop()

def main():
    st.title("Comment Toxicity Classifier")
    if st.button("Run Toxicity Classifier"):
        run_toxic()

if __name__ == "__main__":
    main()
