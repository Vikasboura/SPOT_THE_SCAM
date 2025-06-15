import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("🕵️‍♂️ Fake Job Postings Detector")

model = joblib.load("model.pkl")

uploaded_file = st.file_uploader("Upload a CSV file with job postings")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview", data.head())

    # Predict
    predictions = model.predict(data.fillna(""))
    probabilities = model.predict_proba(data.fillna(""))[:, 1]

    data['Prediction'] = predictions
    data['Fraud_Probability'] = probabilities

    # Display results
    st.write("🔍 Prediction Results", data[['Prediction', 'Fraud_Probability']])

    # Summary counts
    fraud_count = sum(data['Prediction'] == 1)
    genuine_count = sum(data['Prediction'] == 0)

    st.write(f"✅ Genuine postings: {genuine_count}")
    st.write(f"⚠️ Fraudulent postings: {fraud_count}")

    # Pie chart
    fig, ax = plt.subplots()
    labels = ['Genuine', 'Fraudulent']
    sizes = [genuine_count, fraud_count]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Download results
    st.download_button("📥 Download Results", data.to_csv(index=False), "predictions.csv")
    st.success("✅ Predictions completed successfully!")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")   