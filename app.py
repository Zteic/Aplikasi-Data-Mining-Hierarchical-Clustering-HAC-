import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Klasifikasi HAC", page_icon="📊", layout="wide")

# Menambahkan CSS custom agar tabel di HP bisa di-scroll horizontal dengan mulus
st.markdown("""
    <style>
    div[data-testid="stDataFrame"] > div {
        overflow-x: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Data Mining: Hierarchical Clustering (HAC)")
st.write("Aplikasi cerdas untuk memetakan dan mengelompokkan karakteristik objek secara otomatis.")

# --- FITUR CACHE (OPTIMASI KECEPATAN) ---
@st.cache_data
def load_default_data():
    data_sampel = {
        'Provinsi': ['Aceh', 'Sumut', 'Sumbar', 'Riau', 'Jambi', 'Jakarta', 'Jabar', 'Jateng', 'DIY', 'Jatim'],
        'Tingkat_Kemiskinan': [14.64, 8.15, 6.04, 6.68, 7.58, 4.44, 7.62, 10.77, 11.49, 10.14],
        'Tingkat_Pengangguran': [5.75, 5.47, 5.94, 4.23, 4.70, 7.57, 7.44, 5.13, 3.69, 4.88]
    }
    return pd.DataFrame(data_sampel)

df_default = load_default_data()

# --- SIDEBAR & INPUT DATA ---
st.sidebar.header("📁 Pengaturan Data")
sumber_data = st.sidebar.radio(
    "Pilih Sumber Data:", 
    ("1. Data Sampel (Demo)", "2. Input Manual", "3. Upload CSV")
)

df = None

if sumber_data == "1. Data Sampel (Demo)":
    df = df_default.copy()
    with st.expander("👀 Lihat Data Sampel"):
        st.dataframe(df, use_container_width=True)

elif sumber_data == "2. Input Manual":
    st.write("### Input Data Manual")
    st.info("💡 **Petunjuk:** Klik ganda pada sel untuk mengedit. Scroll ke baris terbawah untuk menambah data baru.")
    template_data = pd.DataFrame({
        'Nama_Objek': ['Objek A', 'Objek B', 'Objek C', 'Objek D', 'Objek E'],
        'Fitur_1': [10.5, 12.0, 8.5, 15.0, 9.0],
        'Fitur_2': [2.1, 2.5, 1.8, 3.0, 2.0]
    })
    df = st.data_editor(template_data, num_rows="dynamic", use_container_width=True)

elif sumber_data == "3. Upload CSV":
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        with st.expander("👀 Lihat Data CSV"):
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("Silakan unggah file CSV untuk memulai.")
        st.stop()

# --- PROSES ALGORITMA ---
if df is not None and not df.empty:
    kolom_numerik = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(kolom_numerik) >= 2:
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Pengaturan HAC")
        
        with st.sidebar.expander("ℹ️ Panduan Pengaturan (Klik di sini)"):
            st.markdown("""
            **1. Fitur X & Y:** Pilih dua kolom angka.
            **2. Metode Linkage:** Ward (Direkomendasikan), Complete, Average, Single.
            **3. Jumlah Klaster (K):** Jumlah pembagian kelompok data.
            """)
        
        x_axis = st.sidebar.selectbox("Pilih Sumbu X", kolom_numerik, index=0)
        y_axis = st.sidebar.selectbox("Pilih Sumbu Y", kolom_numerik, index=1 if len(kolom_numerik) > 1 else 0)
        
        linkage_method = st.sidebar.selectbox("Metode Linkage", ["ward", "complete", "average", "single"])
        k_value = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=5, value=3)

        if st.sidebar.button("🚀 Jalankan Klasterisasi"):
            X = df[[x_axis, y_axis]]
            
            # Memproses Model HAC
            hac = AgglomerativeClustering(n_clusters=k_value, metric='euclidean', linkage=linkage_method)
            df['Cluster'] = hac.fit_predict(X).astype(str)
            
            st.markdown("### 📈 Hasil Analisis Klasterisasi")
            tab1, tab2, tab3 = st.tabs(["🌳 Dendrogram", "🎯 Scatter Plot (Interaktif)", "📝 Tabel & Profiling"])
            
            # TAB 1: DENDROGRAM
            with tab1:
                st.caption("Pohon hierarki menunjukkan proses penggabungan data.")
                fig_dendro, ax_dendro = plt.subplots(figsize=(8, 5))
                label_kolom = df.iloc[:, 0].astype(str).values if df.shape[1] > 0 else None
                linkage_matrix = sch.linkage(X, method=linkage_method)
                sch.dendrogram(linkage_matrix, labels=label_kolom, ax=ax_dendro, leaf_rotation=90)
                plt.ylabel("Jarak Euclidean")
                plt.tight_layout()
                st.pyplot(fig_dendro)
            
            # TAB 2: SCATTER PLOT INTERAKTIF
            with tab2:
                st.caption("Arahkan kursor atau ketuk titik untuk melihat detail objek.")
                nama_objek = df.columns[0] 
                fig_scatter = px.scatter(
                    df, 
                    x=x_axis, 
                    y=y_axis, 
                    color="Cluster",
                    hover_name=nama_objek,
                    title=f"Persebaran {k_value} Klaster",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_scatter.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
                fig_scatter.update_layout(
                    autosize=True,
                    height=500,
                    margin=dict(l=40, r=40, b=40, t=40)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # TAB 3: TABEL, EVALUASI & DOWNLOAD
            with tab3:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Hasil Evaluasi Model**")
                    if k_value > 1 and k_value < len(X):
                        score = silhouette_score(X, df['Cluster'], metric='euclidean')
                        st.success(f"Silhouette Score: **{score:.3f}**")
                
                with col2:
                    st.write("**Unduh Laporan**")
                    csv_export = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil (CSV)",
                        data=csv_export,
                        file_name="hasil_klaster_hac.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.write("---")
                st.write("**Ringkasan Karakteristik (Rata-rata per Klaster)**")
                df['Cluster_Int'] = df['Cluster'].astype(int)
                profiling = df.groupby('Cluster_Int')[[x_axis, y_axis]].mean().reset_index()
                profiling.rename(columns={'Cluster_Int': 'Cluster'}, inplace=True)
                st.dataframe(profiling, use_container_width=True)
                
                st.write("**Tabel Data Lengkap**")
                st.dataframe(df.drop(columns=['Cluster_Int']), use_container_width=True)

    else:
        st.error("Data yang diinputkan harus memiliki minimal 2 kolom yang berisi angka!")
