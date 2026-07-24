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

# Reset session state jika sumber data berubah
if 'last_sumber_data' not in st.session_state or st.session_state['last_sumber_data'] != sumber_data:
    st.session_state['last_sumber_data'] = sumber_data
    if 'df_result' in st.session_state:
        del st.session_state['df_result']

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

        btn_jalankan = st.sidebar.button("🚀 Jalankan Klasterisasi")

        # Jika tombol diklik, proses klasterisasi dan simpan hasilnya ke session_state
        if btn_jalankan:
            X = df[[x_axis, y_axis]]
            hac = AgglomerativeClustering(n_clusters=k_value, metric='euclidean', linkage=linkage_method)
            
            # Membuat copy dataframe agar data asli tidak tertimpa
            df_result = df.copy()
            df_result['Cluster'] = hac.fit_predict(X).astype(str)
            
            # Menyimpan ke session_state
            st.session_state['df_result'] = df_result
            st.session_state['x_axis'] = x_axis
            st.session_state['y_axis'] = y_axis
            st.session_state['linkage_method'] = linkage_method
            st.session_state['k_value'] = k_value

        # Menampilkan hasil jika analisis sudah pernah dijalankan
        if 'df_result' in st.session_state:
            df_res = st.session_state['df_result']
            curr_x = st.session_state['x_axis']
            curr_y = st.session_state['y_axis']
            curr_linkage = st.session_state['linkage_method']
            curr_k = st.session_state['k_value']
            X_curr = df_res[[curr_x, curr_y]]

            st.markdown("### 📈 Hasil Analisis Klasterisasi")
            tab1, tab2, tab3 = st.tabs(["🌳 Dendrogram", "🎯 Scatter Plot (Interaktif)", "📝 Tabel & Profiling"])
            
            # TAB 1: DENDROGRAM
            with tab1:
                st.caption("Pohon hierarki menunjukkan proses penggabungan data.")
                fig_dendro, ax_dendro = plt.subplots(figsize=(8, 5))
                label_kolom = df_res.iloc[:, 0].astype(str).values if df_res.shape[1] > 0 else None
                linkage_matrix = sch.linkage(X_curr, method=curr_linkage)
                sch.dendrogram(linkage_matrix, labels=label_kolom, ax=ax_dendro, leaf_rotation=90)
                plt.ylabel("Jarak Euclidean")
                plt.tight_layout()
                st.pyplot(fig_dendro)
            
            # TAB 2: SCATTER PLOT INTERAKTIF
            with tab2:
                st.caption("Arahkan kursor atau ketuk titik untuk melihat detail objek.")
                nama_objek = df_res.columns[0] 
                fig_scatter = px.scatter(
                    df_res, 
                    x=curr_x, 
                    y=curr_y, 
                    color="Cluster",
                    hover_name=nama_objek,
                    title=f"Persebaran {curr_k} Klaster",
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
                    if curr_k > 1 and curr_k < len(X_curr):
                        score = silhouette_score(X_curr, df_res['Cluster'], metric='euclidean')
                        st.success(f"Silhouette Score: **{score:.3f}**")
                
                with col2:
                    st.write("**Unduh Laporan**")
                    # Mengunduh data hasil klasterisasi (yang sudah berisi kolom Cluster)
                    csv_export = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Klasterisasi (CSV)",
                        data=csv_export,
                        file_name="hasil_analisis_klaster_hac.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.write("---")
                st.write("**Ringkasan Karakteristik (Rata-rata per Klaster)**")
                df_res['Cluster_Int'] = df_res['Cluster'].astype(int)
                profiling = df_res.groupby('Cluster_Int')[[curr_x, curr_y]].mean().reset_index()
                profiling.rename(columns={'Cluster_Int': 'Cluster'}, inplace=True)
                st.dataframe(profiling, use_container_width=True)
                
                st.write("**Tabel Data Lengkap Hasil Klasterisasi**")
                st.dataframe(df_res.drop(columns=['Cluster_Int']), use_container_width=True)

    else:
        st.error("Data yang diinputkan harus memiliki minimal 2 kolom yang berisi angka!")
