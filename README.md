# 🧠 Naive Bayes Interactive Learning System (NB-Mastery Lab)

An interactive web-based application built with **Streamlit** to learn, visualize, and experiment with the **Naive Bayes algorithm** step by step.

---

## 📌 Features

### 🔍 Exploratory Data Analysis (EDA)
- Preview dataset
- Visualize feature distributions using histograms
- Select target and feature columns interactively

### 🛠 Data Preprocessing
- Handles missing values automatically
- Encodes categorical data
- Applies **Min-Max Scaling** for Multinomial NB compatibility

### ⚙️ Learning Module
- Train:
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
- Adjustable parameters:
  - Train-test split
  - Laplace smoothing (α)
- View:
  - Prior probabilities
  - Step-by-step model trace
- Try custom input predictions

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix visualization
- K-Fold Cross Validation

### 🎓 Educational Quiz
- 5 interactive questions
- Reinforces core Naive Bayes concepts

---

## 🧮 Core Concept

Naive Bayes is based on Bayes’ Theorem:

P(C|x) = (P(x|C) * P(C)) / P(x)

Where:
- P(C|x) → Posterior probability  
- P(x|C) → Likelihood  
- P(C) → Prior probability  
- P(x) → Evidence  

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

---

## 📂 Project Structure

NB-Mastery-Lab/
│── app.py          # Main Streamlit application  
│── README.md       # Project documentation  
│── dataset.csv     # Sample dataset (optional)  

---

## ▶️ How to Run the Project

### 1. Install Dependencies
pip install streamlit pandas numpy scikit-learn plotly

### 2. Run the App
streamlit run app.py

### 3. Open in Browser
http://localhost:8501

---

## 📊 How to Use

1. Upload a CSV dataset  
2. Select target variable  
3. Explore data (EDA tab)  
4. Preprocess data  
5. Train model  
6. Evaluate performance  
7. Take the quiz  

---

## ⚠️ Important Notes

- Multinomial NB requires non-negative values  
- Scaling is automatically applied  
- Dataset should be clean and structured  

---

## 🎯 Learning Outcomes

- Understand Naive Bayes mathematically  
- Learn preprocessing techniques  
- Visualize model performance  
- Gain hands-on ML experience  

---

## 🚀 Future Improvements

- Add Bernoulli Naive Bayes  
- Save trained models  
- Deploy online (Streamlit Cloud)  

---

##

Sumana Paul
