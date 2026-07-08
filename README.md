📊 Aplikasi Data Mining: Hierarchical Clustering (HAC)

Aplikasi interaktif berbasis antarmuka web (Streamlit) yang dikembangkan untuk memetakan dan mengelompokkan karakteristik data secara hierarki menggunakan algoritma Hierarchical Agglomerative Clustering (HAC). Proyek ini disusun untuk memenuhi tugas mata kuliah Penambangan Data.

✨ Fitur Utama:

Fleksibilitas Input Data: Mendukung 3 mode pengisian data, yaitu menggunakan data sampel (bawaan), input manual secara langsung pada antarmuka tabel interaktif, maupun unggah file .csv.

Kustomisasi Algoritma: Pengguna dapat memilih parameter secara mandiri, seperti Fitur X/Y, penentuan Metode Linkage (Ward, Complete, Average, Single), dan jumlah klaster (K).

Visualisasi Komprehensif: Menampilkan grafik pohon hierarki (Dendrogram) dan Scatter Plot secara instan.

Evaluasi Kualitas Model: Dilengkapi dengan perhitungan Silhouette Score otomatis untuk mengukur tingkat akurasi pemisahan klaster.

Profiling Data & Ekspor: Menampilkan tabel ringkasan karakteristik tiap klaster berdasarkan nilai rata-rata, lengkap dengan fitur unduh (download) hasil akhir ke dalam format CSV.

🛠️ Teknologi yang Digunakan:

Bahasa Pemrograman: Python

Antarmuka Web: Streamlit

Pemrosesan & Visualisasi Data: Pandas, Matplotlib

Machine Learning: Scikit-Learn, SciPy
