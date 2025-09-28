import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Visualisasi Outlier ---
def tampilkan_boxplot_outliers(df, columns, prefix=''):
    """Fungsi untuk menampilkan boxplot secara interaktif guna mendeteksi outlier."""
    print(f"--- 4. Menampilkan Visualisasi Outlier ({prefix}) ---")
    for kolom in columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[kolom])
        plt.title(f"Boxplot Kolom '{kolom}' ({prefix})", fontsize=15)
        plt.xlabel(kolom, fontsize=12)
        
        print(f"[INFO] Menampilkan visualisasi untuk '{kolom}'. Tutup jendela plot untuk melanjutkan...")
        plt.show() 
        
    print("\n" + "="*50 + "\n")

try:
    df = pd.read_csv('train.csv')
    print("="*50)
    print("               PROSES PREPROCESSING DATA              ")
    print("="*50)
    print("\n[INFO] Dataset 'train.csv' berhasil dimuat.")
    print(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom.\n")
    print("--- Informasi Awal Dataset ---")
    df.info()
    print("\n" + "="*50 + "\n")

except FileNotFoundError:
    print("[ERROR] File 'train.csv' tidak ditemukan. Pastikan file berada di folder yang sama.")
    exit()

# --- MENANGANI MISSING VALUE ---
print("--- Langkah 1: Penanganan Missing Values ---")
print("Jumlah missing value sebelum ditangani:")
print(df.isnull().sum())
print("-" * 30)

# - Age diisi dengan nilai tengah (median)
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"[OK] Missing values pada kolom 'Age' diisi dengan median: {median_age}")

# - Embarked diisi dengan nilai yang paling sering muncul
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
print(f"[OK] Missing values pada kolom 'Embarked' diisi dengan modus: '{mode_embarked}'")

# - Cabin memiliki banyak missing value (lebih dari 75%) sehingga lebih baik dihapus
# Membuat Has_Cabin (kolom baru) yang menunjukkan apakah penumpang memiliki kabin atau tidak
df['Has_Cabin'] = df['Cabin'].notna().astype(int)

# Menghapus kolom Cabin yang asli
df.drop('Cabin', axis=1, inplace=True) 
print("[OK] Kolom 'Cabin' dihapus karena terlalu banyak missing values.")

print("\nJumlah missing value setelah ditangani:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")


# --- MENANGANI OUTLIER ---
print("--- Langkah 2: Penanganan Outlier ---")
kolom_numerik = ['Age', 'Fare']
df_sebelum_handler = df.copy()

# Tampilkan visualisasi SEBELUM handler
tampilkan_boxplot_outliers(df_sebelum_handler, kolom_numerik, prefix="Sebelum_Handler")

for kolom in kolom_numerik:
    print(f"Mendeteksi outlier pada kolom '{kolom}':")
    
    # Menghitung IQR (Interquartile Range)
    Q1 = df[kolom].quantile(0.25)
    Q3 = df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    
    # Menentukan batas atas dan batas bawah
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    
    print(f"  - Batas Bawah (IQR): {batas_bawah:.2f}")
    print(f"  - Batas Atas (IQR): {batas_atas:.2f}")
    
    # Mencari jumlah outlier
    outliers = df[(df[kolom] < batas_bawah) | (df[kolom] > batas_atas)]
    jumlah_outlier = len(outliers)
    
    if jumlah_outlier > 0:
        print(f"  - Ditemukan {jumlah_outlier} outlier.")
        
        # Handler: Mengganti nilai outlier dengan batas atas/bawah
        df[kolom] = np.where(
            df[kolom] > batas_atas, 
            batas_atas, 
            np.where(
                df[kolom] < batas_bawah,
                batas_bawah,
                df[kolom]
            )
        )
        print(f"[OK] Outlier pada '{kolom}' telah ditangani dengan metode Mengganti Batas Maksimum & Minimum IQR.\n")
    else:
        print(f"  - Tidak ditemukan outlier pada kolom '{kolom}'.\n")

# --- Tampilkan visualisasi SETELAH handler ---
tampilkan_boxplot_outliers(df, kolom_numerik, prefix="Setelah_Handler")


print("\n--- Informasi Akhir Dataset Setelah Dibersihkan ---")
df.info()

print("\n--- 5 Baris Pertama Data yang Sudah Dibersihkan ---")
print(df.head())

# Menyimpan data yang sudah bersih ke file CSV baru
df.to_csv('train_cleaned.csv', index=False)
print("\n[INFO] Data yang sudah bersih telah disimpan ke 'train_cleaned.csv'")