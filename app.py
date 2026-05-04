import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# --- 1. UI SETTINGS & ADVANCED CSS ---
st.set_page_config(page_title="NB-Mastery Lab", layout="wide")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; padding: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: rgba(128, 128, 128, 0.05);
        border-radius: 8px; padding: 10px 25px; border: 1px solid rgba(128, 128, 128, 0.1);
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important; color: white !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    div[data-testid="stTable"] table { margin-left: auto; margin-right: auto; text-align: center; }
    th, td { text-align: center !important; }
    [data-testid="stMetricValue"] { color: #ff4b4b !important; font-weight: bold !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PIPELINE CONTROL ---
with st.sidebar:
    st.header("🚀 Pipeline Control")
    uploaded_file = st.file_uploader("1. Upload CSV Dataset", type="csv")
    if uploaded_file:
        st.divider()
        st.subheader("Algorithm Settings")
        nb_variant = st.selectbox("Algorithm Variant", ["Gaussian NB", "Multinomial NB"], 
                                 help="Multinomial requires non-negative features. Scaling will be applied.")
        laplace = st.slider("Laplace Smoothing (α)", 0.0, 5.0, 1.0)
        test_ratio = st.slider("Train-Test Split %", 10, 50, 20) / 100
        k_folds = st.number_input("Cross-Validation (K)", 2, 10, 5)

# --- 3. MAIN HEADER ---
st.title("🧠 Naive Bayes: Interactive Learning System")
st.caption("Design and Development of an Interactive Web-Based Learning System")

if not uploaded_file:
    st.info("👋 Please upload a CSV dataset in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --- 4. THE PIPELINE TABS ---
tab_eda, tab_pre, tab_train, tab_eval, tab_edu = st.tabs([
    "🔍 1. Exploratory Data", "🛠 2. Preprocessing", "⚙️ 3. Learning Module", 
    "📊 4. Evaluation Metrics", "🎓 5. Educational Lab"
])

# --- TAB 1: EDA ---
with tab_eda:
    st.header("Exploratory Data Analysis (EDA)")
    target_col = st.selectbox("Select Target Variable", df.columns, index=len(df.columns)-1)
    col1, col2 = st.columns(2)
    with col1: st.subheader("Preview"); st.table(df.head(5))
    with col2:
        feat_col = st.selectbox("Feature to Visualize", [c for c in df.columns if c != target_col])
        st.plotly_chart(px.histogram(df, x=feat_col, color=target_col, barmode='group', template="plotly_white"), use_container_width=True)

# --- TAB 2: PREPROCESSING (SCALING ADDED) ---
with tab_pre:
    st.header("Preprocessing Interface")
    st.info("**What & Why:** Multinomial NB cannot handle negative values. We apply Min-Max Scaling to transform data to [0, 1] for mathematical compatibility.")
    
    df_clean = df.copy()
    
    # 1. Handle Missing Values
    if df_clean.isnull().values.any():
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    
    # 2. Force Encoding on non-numeric columns
    le = LabelEncoder()
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    
    # 3. Apply Min-Max Scaling (Crucial for Multinomial NB compatibility)
    scaler = MinMaxScaler()
    X_raw = df_clean.drop(columns=[target_col])
    X_scaled = scaler.fit_transform(X_raw)
    df_ready = pd.DataFrame(X_scaled, columns=X_raw.columns)
    df_ready[target_col] = df_clean[target_col].values
            
    c1, c2 = st.columns(2)
    with c1: st.subheader("Before (Raw Data)"); st.table(df.head(5))
    with c2: st.subheader("After (Encoded & Scaled)"); st.table(df_ready.head(5))
    st.session_state['df_ready'] = df_ready

# --- TAB 3: LEARNING MODULE ---
with tab_train:
    st.header("⚙️ Naive Bayes Learning & Trace")
    if 'df_ready' not in st.session_state:
        st.warning("Please visit the Preprocessing tab first.")
    else:
        df_final = st.session_state['df_ready']
        X = df_final.drop(columns=[target_col])
        y = df_final[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        
        st.latex(r"P(C|x) = \frac{P(x|C) \cdot P(C)}{P(x)}")
        
        if st.button("🚀 TRAIN YOUR MODEL"):
            try:
                model = GaussianNB() if "Gaussian" in nb_variant else MultinomialNB(alpha=laplace)
                model.fit(X_train, y_train)
                st.session_state['trained_model'] = model
                st.session_state['eval_data'] = (X, y, X_test, y_test)
                st.success(f"Model ({nb_variant}) Trained Successfully!")
            except Exception as e:
                st.error(f"Training Error: {e}")

        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            priors = model.class_prior_ if hasattr(model, 'class_prior_') else (np.bincount(y_train)/len(y_train))
            
            st.subheader("Step-by-Step Parameter Trace")
            st.table(pd.DataFrame({'Class': model.classes_, 'Prior Probability P(C)': priors}))
            
            st.divider()
            st.write("#### 🔮 Interactive Sample Inference")
            input_vals = [st.number_input(f"{col}", value=float(X[col].mean()), key=f"t_{col}") for col in X.columns]
            
            if st.button("Calculate Posterior"):
                probs = model.predict_proba([input_vals])[0]
                st.table(pd.DataFrame({'Class': model.classes_, 'Posterior P(C|x)': probs}))

# --- TAB 4: EVALUATION METRICS ---
with tab_eval:
    if 'trained_model' in st.session_state:
        model = st.session_state['trained_model']
        X, y, X_test, y_test = st.session_state['eval_data']
        y_pred = model.predict(X_test)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        m2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
        m3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
        m4.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")
        
        st.divider()
        st.subheader("Confusion Matrix ")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Reds',
                           x=model.classes_, y=model.classes_, labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        cv_res = cross_val_score(model, X, y, cv=k_folds)
        st.subheader(f"{k_folds}-Fold CV Results")
        st.table(pd.DataFrame({"Fold": range(1, k_folds+1), "Accuracy": cv_res}))
    else:
        st.warning("Please train the model in the Learning Module first.")

# --- TAB 5: EDUCATIONAL LAB (QUIZ) ---
with tab_edu:
    st.header("🎓 NB-Mastery Knowledge Check")
    questions = [
        {"q": "What does the 'Naive' in Naive Bayes signify?", "o": ["Simple code", "Feature independence assumption", "Low accuracy"], "a": "Feature independence assumption"},
        {"q": "Which theorem is the foundation of this algorithm?", "o": ["Pythagorean Theorem", "Bayes' Theorem", "Central Limit Theorem"], "a": "Bayes' Theorem"},
        {"q": "What is the purpose of Laplace Smoothing (Alpha)?", "o": ["Speed up training", "Prevent zero probabilities", "Reduce overfitting"], "a": "Prevent zero probabilities"},
        {"q": "Which variant is best for continuous numerical data?", "o": ["Multinomial NB", "Gaussian NB", "Bernoulli NB"], "a": "Gaussian NB"},
        {"q": "In the formula P(C|x), what is P(C) called?", "o": ["Likelihood", "Posterior", "Prior Probability"], "a": "Prior Probability"}
    ]
    
    if 'curr_q' not in st.session_state: 
        st.session_state.curr_q, st.session_state.score, st.session_state.quiz_done = 0, 0, False

    if not st.session_state.quiz_done:
        q_idx = st.session_state.curr_q
        st.subheader(f"Question {q_idx + 1} of 5")
        st.write(questions[q_idx]["q"])
        user_choice = st.radio("Select answer:", questions[q_idx]["o"], key=f"q_{q_idx}")
        
        if st.button("Next Question ➡️"):
            if user_choice == questions[q_idx]["a"]: st.session_state.score += 1
            if st.session_state.curr_q < 4: 
                st.session_state.curr_q += 1
                st.rerun()
            else: 
                st.session_state.quiz_done = True
                st.rerun()
    else:
        st.balloons(); st.subheader("Quiz Complete!"); st.metric("Final Score", f"{st.session_state.score} / 5")
        if st.button("Restart Quiz"): 
            st.session_state.curr_q, st.session_state.score, st.session_state.quiz_done = 0, 0, False
            st.rerun()