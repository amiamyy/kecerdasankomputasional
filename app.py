import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score

def show_data():
    st.header('Menu Data')
    st.write('Ini adalah halaman untuk menampilkan data dari Excel.')

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        st.session_state.upload_count = 0

    if st.session_state.upload_count < 3:
        uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"], key=f"file_uploader_{st.session_state.upload_count}")

        if uploaded_file is not None:
            st.session_state.uploaded_files.append(uploaded_file)
            st.session_state.upload_count += 1

            df = pd.read_excel(uploaded_file, sheet_name='Daftar Peserta Didik')
            st.session_state[f'data_{st.session_state.upload_count}'] = df

    else:
        st.write("Anda telah mengunggah maksimal 3 file.")

    for i in range(1, st.session_state.upload_count + 1):
        df = st.session_state.get(f'data_{i}')
        if df is not None:
            st.write(f"Data dari Upload {i}:")
            st.dataframe(df)
            st.write("Informasi statistik ringkas:")
            st.write(df.describe())

def show_kategorial():
    st.header('Menu Kategorial')
    st.write('Ini adalah halaman untuk menampilkan data kategorial dari Excel.')

    if st.session_state.uploaded_files:
        uploaded_file = st.session_state.uploaded_files[0]
        df_kategorial = pd.read_excel(uploaded_file, sheet_name='kategorial')
        st.write("Data Kategorial dari Excel:")
        st.dataframe(df_kategorial)
    else:
        st.write("Silakan unggah file Excel terlebih dahulu di menu Data.")

def show_perhitungan():
    st.header('Menu Perhitungan')
    st.write('Ini adalah halaman untuk menghitung data menggunakan metode K-Medoids.')

    if st.session_state.uploaded_files:
        uploaded_file = st.session_state.uploaded_files[0]
        df_perhitungan = pd.read_excel(uploaded_file, sheet_name='number')
        st.write("Data Perhitungan dari Excel:")
        st.dataframe(df_perhitungan)

        columns_to_drop = []
        df_perhitungan = df_perhitungan.drop(columns=columns_to_drop)

        if df_perhitungan.isnull().values.any():
            st.warning("Data mengandung nilai NaN. Melakukan penggantian NaN dengan nilai rata-rata kolom.")
            df_perhitungan = df_perhitungan.fillna(df_perhitungan.mean())

        for column in df_perhitungan.columns:
            if df_perhitungan[column].dtype == 'object':
                df_perhitungan[column] = pd.to_numeric(df_perhitungan[column], errors='coerce')

        n_clusters = st.slider('Pilih jumlah cluster', min_value=2, max_value=min(10, len(df_perhitungan) - 1), value=3)

        if st.button('Lakukan Perhitungan K-Medoids'):
            try:
                kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
                labels = kmedoids.fit_predict(df_perhitungan)

                df_perhitungan['Cluster'] = labels
                st.write("Hasil Clustering:")
                st.dataframe(df_perhitungan)

                plot_clusters(df_perhitungan, kmedoids)

                if len(set(labels)) > 1:  # Pastikan jumlah cluster lebih dari 1 untuk menghitung Silhouette Score
                    silhouette_avg = silhouette_score(df_perhitungan.drop(columns=['Cluster']), labels)
                    st.write(f"Silhouette Coefficient: {silhouette_avg:.2f}")
                else:
                    st.warning("Tidak dapat menghitung Silhouette Coefficient karena jumlah cluster kurang dari 2.")

            except ValueError as e:
                st.error(f"Terjadi kesalahan saat melakukan clustering: {str(e)}")

        if st.button('Cari Jumlah Cluster Terbaik'):
            silhouette_scores = []
            K = range(2, min(11, len(df_perhitungan)))  # Batasi K agar tidak melebihi jumlah sampel
            for k in K:
                kmedoids = KMedoids(n_clusters=k, random_state=0)
                labels = kmedoids.fit_predict(df_perhitungan)
                if len(set(labels)) > 1:  # Pastikan jumlah cluster lebih dari 1
                    silhouette_avg = silhouette_score(df_perhitungan, labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(-1)  # Nilai negatif untuk jumlah cluster yang tidak valid

            best_k = K[np.argmax(silhouette_scores)]
            st.write(f"Jumlah cluster terbaik adalah: {best_k} dengan Silhouette Coefficient: {max(silhouette_scores):.2f}")

            fig, ax = plt.subplots()
            ax.plot(K, silhouette_scores, 'bo-', markersize=8)
            ax.set_xlabel('Jumlah cluster (k)')
            ax.set_ylabel('Silhouette Coefficient')
            ax.set_title('Silhouette Coefficient untuk berbagai jumlah cluster')
            st.pyplot(fig)

    else:
        st.write("Silakan unggah file Excel terlebih dahulu di menu Data.")

def plot_clusters(df, kmedoids):
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    unique_labels = np.unique(kmedoids.labels_)
    colors = sns.color_palette('husl', n_colors=len(unique_labels))

    for label, color in zip(unique_labels, colors):
        subset = df[df['Cluster'] == label]
        plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=f'Cluster {label}', color=color, alpha=0.7)

    plt.title('Clustering menggunakan K-Medoids')
    plt.xlabel('Fitur 1')
    plt.ylabel('Fitur 2')
    plt.legend()
    
    st.pyplot(plt)

def main():
    st.title('Clustering Data Siswa Menggunakan Algoritma K-Medoid')
    menu = ['Data', 'Kategorial', 'Perhitungan']
    choice = st.sidebar.selectbox('Pilih Menu', menu)

    if choice == 'Data':
        show_data()
    elif choice == 'Kategorial':
        show_kategorial()
    elif choice == 'Perhitungan':
        show_perhitungan()

if __name__ == "__main__":
    main()
