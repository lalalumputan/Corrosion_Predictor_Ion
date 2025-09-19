import streamlit as st
import joblib
import numpy as np

# Load model & encoder
model = joblib.load("bagging_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🔎 Prediksi Korosi Stainless Steel")
st.write("Masukkan parameter stainless steel dan kondisi lingkungan untuk memprediksi tingkat korosi.")

# --- Input Komposisi Kimia ---
st.header("⚙️ Komposisi Kimia (%)")
C = st.slider("Carbon (C)", 0.0, 2.0, 0.08)
Mn = st.slider("Manganese (Mn)", 0.0, 3.0, 1.0)
Si = st.slider("Silicon (Si)", 0.0, 2.0, 1.0)
P = st.slider("Phosphorus (P)", 0.0, 0.2, 0.04)
S = st.slider("Sulfur (S)", 0.0, 0.2, 0.03)
Cr = st.slider("Chromium (Cr)", 0.0, 30.0, 18.0)
Mo = st.slider("Molybdenum (Mo)", 0.0, 5.0, 2.0)
Ni = st.slider("Nickel (Ni)", 0.0, 20.0, 8.0)
N = st.slider("Nitrogen (N)", 0.0, 1.0, 0.1)
Ti = st.slider("Titanium (Ti)", 0.0, 2.0, 0.5)
Nb = st.slider("Niobium (Nb)", 0.0, 2.0, 0.5)
Al = st.slider("Aluminium (Al)", 0.0, 5.0, 1.0)
Fe = st.slider("Fe balance (%)", 40.0, 90.0, 70.0)

# --- Input Lingkungan (Electrolytes) ---
st.header("🌍 Kondisi Lingkungan (Electrolytes)")
electrolytes = [
    "Formic acid (HCOOH)", "Ammonium chloride(NH4Cl)", "Acetic acid(CH3COOH)",
    "Potasium Hydroxide (KOH)", "Lactic acid ", "oxalic acid (COOH)2.2H2O",
    "Phosphric acid (H3PO4)", "Sulfuric acid (H2SO4)", "Nitric acid (HNO3)",
    "Hydrocloric acid (HCl)", "Citric acid(HOC(CH2COOH)2COOH.H2O", 
    "KHSO4", "KNO3", "MgCl2.6H2O"
]
electrolyte = st.selectbox("Pilih Electrolyte", electrolytes)

# Buat dummy vector untuk environment (14 kolom)
env_vector = [1 if e == electrolyte else 0 for e in electrolytes]

# --- Input Temperature ---
temp = st.slider("Suhu (°C)", 20, 350, 25)

# --- Susun input sesuai urutan training ---
X_input = np.array([[C, Mn, Si, P, S, Cr, Mo, Ni, N, Ti, Nb, Al, Fe] + env_vector + [temp]])

# --- Prediksi ---
if st.button("Prediksi"):
    pred = model.predict(X_input)[0]
    hasil = label_encoder.inverse_transform([pred])[0]
    st.success(f"📊 Prediksi Tingkat Korosi: **{hasil}**")

    probs = model.predict_proba(X_input)[0]
    st.write("### 🔎 Probabilitas per kelas:")
    for cls, p in zip(label_encoder.classes_, probs):
        st.write(f"- {cls}: {p*100:.2f}%")

