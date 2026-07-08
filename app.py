import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score  # <-- Import baru untuk Evaluasi Model

# 1. Judul Aplikasi
st.title("Aplikasi Data Mining: Hierarchical Clustering (HAC)")
st.write("Aplikasi ini digunakan untuk memetakan karakteristik wilayah atau objek berdasarkan kemiripannya.")

# Data Sampel Bawaan
data_sampel = {
    'Provinsi': ['Aceh', 'Sumut', 'Sumbar', 'Riau', 'Jambi', 'Jakarta', 'Jabar', 'Jateng', 'DIY', 'Jatim'],
    'Tingkat_Kemiskinan': [14.64, 8.15, 6.04, 6.68, 7.58, 4.44, 7.62, 10.77, 11.49, 10.14],
    'Tingkat_Pengangguran': [5.75, 5.47, 5.94, 4.23, 4.70, 7.57, 7.44, 5.13, 3.69, 4.88]
}
df_default = pd.DataFrame(data_sampel)

# 2. Pilihan Sumber Data di Sidebar
st.sidebar.header("Pilihan Sumber Data")
sumber_data = st.sidebar.radio(
    "Pilih Cara Memasukkan Data:", 
    ("1. Gunakan Data Sampel (Demo Instan)", 
     "2. Input Data Manual (Ketik Langsung)", 
     "3. Unggah File CSV")
)

df = None

# Logika berdasarkan pilihan sumber data
if sumber_data == "1. Gunakan Data Sampel (Demo Instan)":
    df = df_default.copy()
    st.write("### Preview Data Sampel:")
    st.dataframe(df, use_container_width=True)

elif sumber_data == "2. Input Data Manual (Ketik Langsung)":
    st.write("### Input Data Manual")
    st.info("💡 **Petunjuk Pengisian:**\n- **Edit Data:** Klik dua kali pada sel tabel di bawah untuk mengetik/mengubah angka.\n- **Tambah Baris:** Arahkan kursor atau scroll ke baris paling bawah, baris kosong akan otomatis muncul.\n- **Hapus Baris:** Klik kotak kecil di sebelah kiri baris, lalu tekan tombol `Delete` atau `Backspace` di keyboard.")
    
    template_data = pd.DataFrame({
        'Nama_Objek': ['Objek A', 'Objek B', 'Objek C', 'Objek D', 'Objek E'],
        'Fitur_1': [10.5, 12.0, 8.5, 15.0, 9.0],
        'Fitur_2': [2.1, 2.5, 1.8, 3.0, 2.0]
    })
    
    df = st.data_editor(template_data, num_rows="dynamic", use_container_width=True)

elif sumber_data == "3. Unggah File CSV":
    uploaded_file = st.file_uploader("Pilih file CSV data Anda", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview Data CSV:")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Silakan unggah file CSV Anda.")
        st.stop()

# Memastikan df tidak kosong
if df is not None and not df.empty:
    kolom_numerik = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(kolom_numerik) >= 2:
        st.sidebar.header("Pengaturan Algoritma HAC")
        
        with st.sidebar.expander("ℹ️ Panduan Pengaturan (Klik di sini)"):
            st.markdown("""
            **1. Fitur X & Y:** Pilih dua kolom angka yang akan dianalisis. Fitur X untuk sumbu datar, Fitur Y untuk sumbu tegak pada grafik.
            
            **2. Metode Linkage (Jarak):** - **Ward:** Paling direkomendasikan. Menghasilkan kelompok yang rapi dan seimbang.
            - **Complete:** Mengukur jarak terjauh antar kelompok.
            - **Average:** Mengukur jarak rata-rata.
            - **Single:** Mengukur jarak terdekat, cocok untuk mencari data yang menyimpang (outlier).
            
            **3. Jumlah Klaster (K):** Mau dibagi jadi berapa kelompok? (Misal: Pilih 3 jika ingin membagi data menjadi kategori Tinggi, Sedang, Rendah).
            """)

        x_axis = st.sidebar.selectbox("Pilih Fitur X (Horizontal)", kolom_numerik, index=0)
        y_axis = st.sidebar.selectbox("Pilih Fitur Y (Vertikal)", kolom_numerik, index=1 if len(kolom_numerik) > 1 else 0)
        
        linkage_method = st.sidebar.selectbox("Metode Linkage", ["ward", "complete", "average", "single"])
        k_value = st.sidebar.slider("Tentukan jumlah Klaster (K)", min_value=2, max_value=5, value=3)

        if st.button("Jalankan Hierarchical Clustering (HAC)"):
            X = df[[x_axis, y_axis]]
            
            # --- 1. DENDROGRAM ---
            st.write("---")
            st.write("### 1. Visualisasi Dendrogram (Pohon Hierarki)")
            fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
            
            label_kolom = df.iloc[:, 0].astype(str).values if df.shape[1] > 0 else None
            
            linkage_matrix = sch.linkage(X, method=linkage_method)
            sch.dendrogram(linkage_matrix, labels=label_kolom, ax=ax_dendro, leaf_rotation=45)
            plt.title("Dendrogram HAC")
            plt.ylabel("Jarak Euclidean")
            st.pyplot(fig_dendro)
            
            # --- 2. HASIL TABEL ---
            hac = AgglomerativeClustering(n_clusters=k_value, metric='euclidean', linkage=linkage_method)
            df['Cluster'] = hac.fit_predict(X)
            
            st.write("---")
            st.write(f"### 2. Hasil Pengelompokan Data ({k_value} Klaster):")
            st.dataframe(df, use_container_width=True)
            
            # --- 3. SCATTER PLOT ---
            st.write("---")
            st.write("### 3. Visualisasi Scatter Plot")
            fig_scatter, ax_scatter = plt.subplots()
            ax_scatter.scatter(df[x_axis], df[y_axis], c=df['Cluster'], cmap='rainbow', edgecolors='k', s=100)
            
            if df.shape[1] > 0:
                for i, txt in enumerate(df.iloc[:, 0]):
                    ax_scatter.annotate(str(txt), (df[x_axis].iloc[i], df[y_axis].iloc[i]), fontsize=9, alpha=0.8)
                    
            ax_scatter.set_xlabel(x_axis)
            ax_scatter.set_ylabel(y_axis)
            plt.title("Persebaran Klaster")
            st.pyplot(fig_scatter)

            # ================= FITUR BARU =================
            
            # --- 4. EVALUASI MODEL (SILHOUETTE SCORE) ---
            st.write("---")
            st.write("### 4. Evaluasi Kualitas Klaster")
            if k_value > 1 and k_value < len(X):
                score = silhouette_score(X, df['Cluster'], metric='euclidean')
                st.success(f"**Nilai Silhouette Score: {score:.3f}**")
                st.caption("💡 *Info: Nilai berkisar antara -1 hingga 1. Semakin mendekati angka 1, berarti pengelompokan data semakin akurat dan terpisah dengan baik.*")

            # --- 5. PROFILING KLASTER ---
            st.write("---")
            st.write("### 5. Ringkasan Karakteristik (Profiling)")
            st.write("Tabel ini menunjukkan **nilai rata-rata** dari tiap fitur untuk masing-masing kelompok, sehingga kita tahu ciri khas dari tiap klaster.")
            profiling = df.groupby('Cluster')[[x_axis, y_axis]].mean().reset_index()
            st.dataframe(profiling, use_container_width=True)

            # --- 6. TOMBOL UNDUH CSV ---
            st.write("---")
            csv_export = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil Klasterisasi (Format CSV)",
                data=csv_export,
                file_name="hasil_klaster_hac.csv",
                mime="text/csv",
            )
            
    else:
        st.error("Data yang diinputkan harus memiliki minimal 2 kolom yang berisi angka!")