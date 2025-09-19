import streamlit as st
import joblib
import numpy as np

# Load model & encoder (hasil training Dataset 2)
model = joblib.load("dt_model_dataset2.pkl")
label_encoder = joblib.load("label_encoder2.pkl")

st.title("🔎 Prediksi Korosi Stainless Steel (Dataset 2 - 3 Fitur)")
st.write("Masukkan konsentrasi ion H⁺, S²⁻, dan kadar Fe untuk memprediksi perilaku korosi.")

# --- Input 3 fitur ---
H = st.number_input("Konsentrasi ion H⁺", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
S2 = st.number_input("Konsentrasi ion S²⁻", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
Fe = st.number_input("Fe (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)

# --- Susun input sesuai urutan training ---
X_input = np.array([[H, S2, Fe]])

# --- Prediksi ---
if st.button("Prediksi"):
    pred = model.predict(X_input)[0]
    hasil = label_encoder.inverse_transform([pred])[0]
    st.success(f"📊 Prediksi Tingkat Korosi: **{hasil}**")

    # Tambahan: probabilitas tiap kelas
    probs = model.predict_proba(X_input)[0]
    st.write("### 🔎 Probabilitas per kelas:")
    for cls, p in zip(label_encoder.classes_, probs):
        st.write(f"- {cls}: {p*100:.2f}%")
