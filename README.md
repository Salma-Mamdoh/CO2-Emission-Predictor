# Machine Learning Assignment 1

## 📖 Project Overview
This project involves analyzing the `co2_emissions_data.csv` dataset using Python, performing data preprocessing, implementing machine learning models, and evaluating their performance.

### 🎯 Objectives:
- Load and explore the dataset.
- Handle missing values and feature scaling.
- Visualize data relationships (pairplot & heatmap).
- Preprocess data: encoding, scaling, and splitting.
- Implement **Linear Regression using Gradient Descent**.
- Train and evaluate a **Logistic Regression Model**.

---

## 📊 Dataset Analysis
### 🔍 1. Missing Values Check
✔ No missing values found (`.isnull().sum()`).

### 📏 2. Feature Scaling Analysis
✖ Features have different scales and require scaling.

### 📈 3. Pairplot Insights
- **Engine Size, Cylinders & CO₂ Emissions:** Strong positive correlation.
- **Fuel Consumption Metrics:** Linear relationships.
- **CO₂ Emissions & Fuel Efficiency:** Inverse correlation.
- **Histograms:** Right-skewed distribution for engine size.

### 🔥 4. Correlation Heatmap Insights
- **Strong Positive Correlations:** Fuel consumption and CO₂ emissions.
- **Strong Negative Correlations:** Higher mpg → Lower CO₂ emissions.
- **Moderate Correlations:** Engine size vs. fuel consumption.

---

## 🛠 Data Preprocessing
### ✅ Steps:
1️⃣ **Feature & Target Separation**
2️⃣ **Data Splitting** (Training & Testing)
3️⃣ **Encoding Categorical Variables**
4️⃣ **Scaling Numerical Features**

---

## 🤖 Machine Learning Models
### 📌 1. **Linear Regression (Gradient Descent)**
✔ Selected features based on correlation.
✔ Achieved **R² score = 0.8675** (Good performance).

### 📌 2. **Logistic Regression**
✔ Implemented using `SGDClassifier` (loss=`log_loss`, max_iter=2000).
✔ **Performance Analysis:**
  - **High Class:** Excellent prediction.
  - **Low Class:** Poor due to class imbalance.
  - **Moderate Class:** Good performance with minor misclassifications.

---

## 🚀 Key Takeaways
✅ Preprocessing (scaling & encoding) is essential.
✅ Feature selection improves model accuracy.
✅ Linear regression performed well.
✅ Class imbalance affects classification models.

---

## 🛠 Technologies & Libraries Used
🔹 **Python**
🔹 **Pandas, NumPy** (Data manipulation)
🔹 **Matplotlib, Seaborn** (Data visualization)
🔹 **Scikit-learn** (ML models & evaluation)

---

## 🔮 Future Enhancements
🔹 Handle class imbalance with oversampling or weighting.
🔹 Experiment with polynomial regression.
🔹 Test other classification models.

---

### 📌 **Authors: Team Members (Cairo University)**

