
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# ==================== LOAD DATA ====================
df = pd.read_csv("Food Waste data and research - by country.csv")
df_encoded = df.copy()

# ==================== ENCODING ====================
le_conf = LabelEncoder()
le_region = LabelEncoder()
df_encoded["Confidence in estimate"] = le_conf.fit_transform(df["Confidence in estimate"])
df_encoded["Region"] = le_region.fit_transform(df["Region"])
df_encoded = df_encoded.drop(columns=["Country", "Source", "M49 code"])

# ==================== SPLIT X and y ====================
X = df_encoded.drop(columns=["Confidence in estimate"])
y = df_encoded["Confidence in estimate"]

# ==================== TRAIN MODELS ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(random_state=42).fit(X, y)
# nb = GaussianNB().fit(X_scaled, y)
# lr = LogisticRegression(max_iter=1000).fit(X_scaled, y)

# ==================== STREAMLIT DASHBOARD ====================
st.set_page_config(page_title="Food Waste Dashboard", layout="wide")
st.title("🌍 Food Waste Analysis & Prediction App")

# ==================== SECTION 1: DATA SUMMARY ====================
st.header("📌 Informasi Umum")
col1, col2, col3 = st.columns(3)
col1.metric("Total Negara", len(df))
col2.metric("Rata-rata Waste (kg/capita)", f"{df['combined figures (kg/capita/year)'].mean():.2f}")
col3.metric("Jumlah Region", df['Region'].nunique())

# ==================== SECTION 2: VISUALISASI ====================
st.header("📊 Visualisasi Data")
# with st.expander("Lihat Boxplot & Heatmap"):
#     st.subheader("Boxplot")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.boxplot(data=df_encoded.select_dtypes(include=np.number), ax=ax)
#     st.pyplot(fig)
with st.expander("Lihat Boxplot & Heatmap"):
    st.subheader("Boxplot")
    fig, ax = plt.subplots(figsize=(12, 6))  # bisa diperbesar jika perlu
    sns.boxplot(data=df_encoded.select_dtypes(include=np.number), ax=ax)

    # Putar label sumbu-x agar tidak tumpang tindih
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Atur layout agar tidak terpotong
    fig.tight_layout()

    st.pyplot(fig)

    st.subheader("Heatmap Korelasi")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

# # ==================== SECTION 3: KLASIFIKASI CONFIDENCE ====================
# st.header("🧠 Prediksi Confidence in Estimate")
# st.write("Masukkan estimasi limbah makanan dari berbagai sektor:")

# house = st.number_input("Household (kg/capita)", 0.0)
# retail = st.number_input("Retail (kg/capita)", 0.0)
# service = st.number_input("Food Service (kg/capita)", 0.0)
# total = house + retail + service

# if st.button("Prediksi Confidence"):
#     # ✅ Buat template input dari baris pertama tapi semua isinya diratakan
#     input_df = X.iloc[0:1].copy()
#     input_df.loc[:, :] = X.mean().values  # pakai nilai rata-rata untuk semua kolom

#     # Timpa nilai berdasarkan input user
#     input_df["Household estimate (kg/capita/year)"] = house
#     input_df["Retail estimate (kg/capita/year)"] = retail
#     input_df["Food service estimate (kg/capita/year)"] = service
#     input_df["combined figures (kg/capita/year)"] = total

#     # Transform
#     input_scaled = scaler.transform(input_df)

#     # Prediksi
#     pred_rf = rf.predict(input_df)[0]
#     # pred_nb = nb.predict(input_scaled)[0]
#     # pred_lr = lr.predict(input_scaled)[0]

#     # Probabilitas prediksi
#     prob_rf = rf.predict_proba(input_df)[0]
#     # prob_nb = nb.predict_proba(input_scaled)[0]
#     # prob_lr = lr.predict_proba(input_scaled)[0]

#     # Tampilkan hasil
#     st.success(f"🎯 Random Forest Prediction: {le_conf.classes_[pred_rf]} ({max(prob_rf)*100:.2f}%)")
#     # st.success(f"🧠 Naive Bayes Prediction: {le_conf.classes_[pred_nb]} ({max(prob_nb)*100:.2f}%)")
#     # st.success(f"📈 Logistic Regression Prediction: {le_conf.classes_[pred_lr]} ({max(prob_lr)*100:.2f}%)")

# ==================== SECTION 3: KLASIFIKASI CONFIDENCE ====================
st.header("🧠 Prediksi Confidence in Estimate")
st.write("Masukkan estimasi limbah makanan dari berbagai sektor:")

house = st.number_input("Household (kg/capita)", 0.0)
retail = st.number_input("Retail (kg/capita)", 0.0)
service = st.number_input("Food Service (kg/capita)", 0.0)
total = house + retail + service

if st.button("Prediksi Confidence"):
    # Buat input template
    input_df = X.iloc[0:1].copy()
    input_df.loc[:, :] = X.mean().values
    input_df["Household estimate (kg/capita/year)"] = house
    input_df["Retail estimate (kg/capita/year)"] = retail
    input_df["Food service estimate (kg/capita/year)"] = service
    input_df["combined figures (kg/capita/year)"] = total

    # Prediksi dengan Random Forest
    pred_rf = rf.predict(input_df)[0]
    conf_rf = max(rf.predict_proba(input_df)[0])
    label_rf = le_conf.inverse_transform([pred_rf])[0]

    # # Fungsi untuk menentukan label confidence
    # def get_conf_label(score):
    #     if score >= 0.7:
    #         return "High Confidence"
    #     elif score >= 0.4:
    #         return "Medium Confidence"
    #     elif score >= 0.2:
    #         return "Low Confidence"
    #     else:
    #         return "Very Low Confidence"

    # Tampilkan hasil
    st.success(f"({conf_rf*100:.2f}%) → {label_rf}")



# ==================== SECTION 4: FOOD WASTE CLASSIFICATION (HIGH/LOW) ====================
# st.header("🎯 Prediksi Kategori Limbah Makanan (High/Low)")
# threshold = df["combined figures (kg/capita/year)"].median()

# gdp = st.number_input("GDP per capita (simulasi)")
# populasi = st.number_input("Populasi (simulasi)")

# # Simulasi sederhana: prediksi high waste jika gabungan food waste > threshold
# if st.button("Prediksi High/Low Food Waste"):
#     pred = "High" if total > threshold else "Low"
#     prob = min(max((total - threshold) / threshold, 0), 1)
#     st.success(f"Prediksi: {pred} Food Waste ({prob*100:.2f}%)")

# ==================== SECTION 4: FOOD WASTE CLASSIFICATION (HIGH/LOW) ====================
# st.header("🚦 Prediksi Kategori Limbah Makanan (High/Low)")

# # Ambil threshold berdasarkan median nilai total food waste aktual
# threshold = df["combined figures (kg/capita/year)"].median()

# # Input tambahan (simulasi variabel lain)
# gdp = st.number_input("GDP per capita (simulasi)", min_value=0.0)
# populasi = st.number_input("Populasi (simulasi)", min_value=0.0)

# # Prediksi kategori waste berdasarkan input total
# if st.button("Prediksi High/Low Food Waste"):
#     # Prediksi sederhana: jika total > median → High
#     pred_label = "High" if total > threshold else "Low"

#     # Probabilitas dalam bentuk skor normalisasi terhadap threshold
#     # (nilai negatif dijadikan 0, nilai maksimal dibatasi ke 1)
#     confidence = min(max((total - threshold) / threshold, 0), 1)

#     # Tampilkan hasil
#     if pred_label == "High":
#         st.success(f"🚨 Prediksi: {pred_label} Food Waste ({confidence*100:.2f}%)")
#     else:
#         st.info(f"✅ Prediksi: {pred_label} Food Waste ({(1 - confidence)*100:.2f}%)")

st.header("🚦 Prediksi Kategori Limbah Makanan (High/Low)")

# Ambil threshold berdasarkan median nilai total food waste aktual
threshold = df["combined figures (kg/capita/year)"].median()

# Input dari pengguna
gdp = st.number_input("GDP per capita", min_value=0.0)
populasi = st.number_input("Populasi", min_value=0.0)

# Prediksi berdasarkan simulasi logika rule sederhana
if st.button("Prediksi High/Low Food Waste"):
    if gdp == 0 or populasi == 0:
        st.warning("❗ Silakan masukkan nilai GDP dan populasi yang valid.")
    else:
        # Aturan prediksi sederhana berbasis GDP dan populasi
        # Semakin tinggi GDP dan populasi, prediksi condong ke "High"
        score = (gdp * 0.00005) + (populasi * 0.00000001)

        # Threshold simulasi bisa kamu sesuaikan
        pred_label = "High" if score > 0.5 else "Low"
        confidence = min(score, 1.0)

        # Tampilkan hasil
        if pred_label == "High":
            st.success(f"🚨 Prediksi: {pred_label} Food Waste")
        else:
            st.info(f"✅ Prediksi: {pred_label} Food Waste")

# ==================== SECTION 5: ANALISIS PERBANDINGAN NEGARA ====================
st.header("📊 Analisis Perbandingan Antar Negara")

negara_terpilih = st.multiselect("Pilih Negara:", df["Country"].unique())
fitur_pilih = st.selectbox("Pilih Metrik:", [
    "Household estimate (kg/capita/year)",
    "Retail estimate (kg/capita/year)",
    "Food service estimate (kg/capita/year)",
    "combined figures (kg/capita/year)"
])

if negara_terpilih:
    df_subset = df[df["Country"].isin(negara_terpilih)][["Country", fitur_pilih]]
    st.bar_chart(df_subset.set_index("Country"))

