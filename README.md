# Heart-Disease-Prediction


This is a complete machine learning project that predicts the risk of heart disease using patient health indicators. It includes data preprocessing, dimensionality reduction, model training and evaluation, a web interface with Streamlit, and deployment via Ngrok.

## Project Overview:-

The project uses the **UCI Heart Disease Dataset** to predict the likelihood of heart disease in patients based on clinical features like:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression, slope
- Number of major vessels
- Thalassemia

The goal is to build an accurate, interpretable, and easy-to-use model that assists in early heart disease detection.

---

## Features:-

- 🔍 Data cleaning & encoding  
- 📉 Dimensionality reduction using **PCA**  
- 🧪 Multiple classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- 🎯 Model evaluation using Accuracy, Precision, Recall, F1 Score, ROC AUC  
- 🛠 Hyperparameter tuning with GridSearchCV  
- 🌐 Web interface using Streamlit  
---

## 🖥️ How to Run the App

1. **Install Dependencies**

```bash
pip install -r requirements.txt
Run the Streamlit App

streamlit run ui/app.py
Expose Public URL with Ngrok


📊 Model Evaluation Summary
Stored in: results/evaluation_metrics.txt

Model	Accuracy	Precision	Recall	F1 Score	ROC AUC
Logistic Regression	0.6333	0.5779	0.6333	0.6023	0.8830
Decision Tree	0.4333	0.4998	0.4333	0.4568	0.6340
Random Forest	0.5500	0.4565	0.5500	0.4952	0.8223
SVM	0.5333	0.2844	0.5333	0.3710	0.7659

Best Performing Model: Logistic Regression

📦 requirements.txt:-

Sample dependencies (already included):
streamlit
pyngrok
ucimlrepo
scikit-learn
pandas
numpy
matplotlib
seaborn

🌐 Deployment Steps:-

File: deployment/ngrok_setup.txt

1. Install Ngrok and set auth token:
   ngrok authtoken YOUR_TOKEN

2. Run Streamlit App:
   streamlit run ui/app.py

3. Open public access:
   ngrok http 8501

4. Share the public URL

📌Deliverables:-

✔️ Cleaned dataset
✔️ PCA results
✔️ Supervised & unsupervised models
✔️ Model evaluation metrics
✔️ Hyperparameter-tuned model
✔️ Trained model (.pkl)
✔️ Streamlit interface
✔️ Ngrok live app
