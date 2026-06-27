# 🔬 Breast Cancer Diagnostic Assistant

An interactive, production-ready machine learning workspace designed to assist medical professionals in predicting breast mass malignancy. Powered by clinical cytology features, this application transitions complex high-dimensional statistical boundaries into an intuitive, real-time diagnostic dashboard.

st.markdown(
    '📊 **Data Source:** [UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | '
    '🌐 **Live Application:** [Streamlit Cloud App](https://cancer-predict-app-poweredby-logregression.streamlit.app/) | '
    '💻 **Repository:** [GitHub Source](https://github.com/mafaiziyas/streamlit_cancer_predict_app.git)'
)
st.markdown("---")

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Plotly (Interactive Radar Plots)
* **Machine Learning:** Scikit-Learn (`StandardScaler`, `LogisticRegression`, `SVC`)
* **Data Processing:** Pandas, NumPy
* **Serialization:** Pickle

---

## 🚀 Key Features & UI Experience

* **Interactive Sidebar Control:** Finely tune 30 separate cellular measurements (mean, standard error, and worst-case dimensions) derived from Fine Needle Aspirates (FNA).
* **High-Dimensional Radar Spectrum:** Real-time Plotly charts contrast custom user inputs simultaneously against dataset means, standard errors, and worst-case boundaries.
* **Immediate Diagnostic Inferences:** Generates deterministic classification outputs along with explicit class probability distributions.

---

## 📊 Dataset & EDA Insights

The application is built on the historic public **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset (Donor: Nick Street, 1995), comprising 569 instances with 30 real-valued cell nuclei features.

<br>

Exploratory Data Analysis (EDA) confirms that features like `concave_points_worst` and `radius_mean` exhibit stark, highly separable distinct distributions between benign and malignant classes. This distinct spatial separation allows linear classifiers to map high-accuracy decision boundaries.

---

## 🧠 Modeling & Clinical Performance

While both models show exceptional diagnostic capabilities, **clinical risk evaluation dictates our final deployment choice.**

### Logistic Regression
* **Training Accuracy:** 98.6%  
* **Validation Accuracy:** 94.7%  
* **Testing Accuracy:** 98.2%  
* **Class 1 (Malignant) Recall:** **1.00**

<br>

### Support Vector Machine (SVM)
* **Training Accuracy:** 99.6%  
* **Validation Accuracy:** 96.5%  
* **Testing Accuracy:** 98.2%  
* **Class 1 (Malignant) Recall:** **0.95**

<br>

### 🩺 The Critical Medical Pivot: Why Logistic Regression Wins
In clinical oncology diagnostics, **a False Negative is a critical failure.** A missing diagnosis means life-threatening delays in patient treatment. 

* Although the SVM yielded a marginally higher training/validation profile, **Logistic Regression achieved a perfect 1.00 Recall for Malignant cases (Class 1)** on the test partition.
* By prioritizing safety over raw aggregate accuracy, Logistic Regression is chosen to ensure no malignant tumor goes unnoticed.

---

## 🗺️ Next Steps & Future Roadmap

* **Automated REST API Pipeline:** Transition the backend out of manual sidebar sliders and into an automated API endpoint capable of ingestion directly from cytology laboratory imaging software.
* **Dynamic Decision Threshold Optimization:** Implement adjustable classification thresholds. Lowering the cut-off boundary below the standard 0.5 mark will aggressively minimize false-negative tolerances for stricter medical environments.

---
