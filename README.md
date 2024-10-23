
---

# 🌟 Toxic_comment
![3 1](https://github.com/user-attachments/assets/f2ecc0d4-83a6-4108-834c-5bd7db9b8fa5)

## 📖 Overview
**Toxic_comment_ml** is an interactive machine learning application that classifies comments as **toxic** or **non-toxic**. Utilizing **Streamlit** for a sleek front-end experience and **NLTK** for robust natural language processing, this project aims to empower users with the ability to quickly assess comment toxicity in real time!

## 🚀 Features
- **⚡ Real-time Toxicity Classification**: Input any comment and receive instant feedback on its toxicity level.
- **🔒 User Authentication**: A simple and secure login system ensures safe access to the application.
- **📊 Machine Learning Model**: Built on a trained **Multinomial Naive Bayes** model for accurate comment classification.
- **🛠️ Data Preprocessing**: Advanced techniques like lemmatization and stopword removal enhance model performance.
- **🌈 User-Friendly Interface**: Developed using Streamlit, ensuring a smooth and intuitive user experience.

## 🛠️ Installation
To set up this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    cd Toxic_comment_ml
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - For Windows:
      ```bash
      venv\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application**:
    ```bash
    streamlit run app.py
    python toxic.py
    ```

## 🖥️ Usage
### 🔑 Login Credentials:
- **Username**: `admin`
- **Password**: `1234`
  
### 📝 Sign-up Password: 
- `1234`

After logging in, users can enter comments in the text field, click the **"Classify"** button, and instantly see whether the comment is toxic or non-toxic.

## 📦 Dependencies
This project requires several Python libraries. Make sure to install the following:
- **Streamlit**: For building the web application.
- **NLTK**: For natural language processing tasks.
- **scikit-learn**: For machine learning capabilities.
- **Pandas** and **NumPy**: For efficient data manipulation.

## 📸 Screenshots
![2](https://github.com/user-attachments/assets/75a91428-e9da-42b9-86ba-f9050dc20f5e)
![3](https://github.com/user-attachments/assets/73a9063d-1af3-438f-806e-beecba603b89)
![1](https://github.com/user-attachments/assets/93fa3687-9cbb-405d-a4ff-c42f4f61d35a)

## 💡 Future Improvements
- **🔍 Model Enhancements**: Integrate additional algorithms and perform hyperparameter tuning for better accuracy.
- **👤 User Management**: Develop a more robust user management system.
- **📊 Result Visualization**: Provide users with visual analytics on comment toxicity.

## 🙏 Acknowledgments
- **NLTK**: For their fantastic natural language processing toolkit.
- **Streamlit**: For simplifying the creation of interactive web applications.
- **Scikit-learn**: For offering powerful machine learning tools.

## 🎉 Contribute
Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.

---
