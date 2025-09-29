import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# ================================================================
# --- Langkah 1: Preprocessing Data ---
# ================================================================
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
    print("="*100)
    print("               Preprocessing Data dengan Penanganan Missing Value & Outlier              ")
    print("="*100)
    print("\n[INFO] Dataset 'train.csv' berhasil dimuat.")
    print(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom.\n")
    print("--- Informasi Awal Dataset ---")
    df.info()
    print("\n" + "="*50 + "\n")

except FileNotFoundError:
    print("[ERROR] File 'train.csv' tidak ditemukan. Pastikan file berada di folder yang sama.")
    exit()

# --- MENANGANI MISSING VALUE ---
print("--- Penanganan Missing Values ---")
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

# Membuat Has_Cabin (kolom baru) yang menunjukkan apakah penumpang memiliki kabin atau tidak
df['Has_Cabin'] = df['Cabin'].notna().astype(int)
# Menghapus kolom Cabin yang asli
df.drop('Cabin', axis=1, inplace=True) 
print("[OK] Kolom 'Cabin' dihapus karena terlalu banyak missing values.")

print("\nJumlah missing value setelah ditangani:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

# --- MENANGANI OUTLIER ---
print("--- Penanganan Outlier ---")
kolom_numerik = ['Age', 'Fare']
df_sebelum_handler = df.copy()
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

tampilkan_boxplot_outliers(df, kolom_numerik, prefix="Setelah_Handler")

print("\n--- Informasi Akhir Dataset Setelah Dibersihkan ---")
df.info()

print("\n--- 5 Baris Pertama Data yang Sudah Dibersihkan ---")
print(df.head())

# Menyimpan data yang sudah bersih ke file CSV baru
df.to_csv('train_cleaned.csv', index=False)
print("\n[INFO] Data yang sudah bersih telah disimpan ke 'train_cleaned.csv'")
print("\n" + "\n")



# ================================================================
# --- Langkah 2: Transformasi dengan MinMaxScaler ---
# ================================================================
print("="*65)
print("               Transformasi dengan MinMaxScaler              ")
print("="*65)
# Pilih kolom numerik yang akan di-scale (Age, Fare, SibSp, Parch)
kolom_numerik_scale = ['Age', 'Fare', 'SibSp', 'Parch']

# Simpan copy sebelum scaling untuk visualisasi perbandingan
df_before_scaling = df.copy()

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Terapkan scaling pada kolom numerik
df[kolom_numerik_scale] = scaler.fit_transform(df[kolom_numerik_scale])

print("[OK] Transformasi MinMaxScaler telah diterapkan pada kolom numerik.")
print("\n--- 5 Baris Pertama Data Setelah Scaling ---")
print(df.head())

# Simpan data setelah scaling
df.to_csv('train_scaled.csv', index=False)
print("\n[INFO] Data setelah scaling disimpan ke 'train_scaled.csv'")

# --- Visualisasi distribusi sebelum & sesudah scaling ---
print("\n[INFO] Menampilkan distribusi sebelum & sesudah MinMaxScaler...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, kolom in enumerate(kolom_numerik_scale):
    axes[i//2, i%2].hist(df_before_scaling[kolom], bins=20, alpha=0.5, label='Sebelum', color='red')
    axes[i//2, i%2].hist(df[kolom], bins=20, alpha=0.5, label='Sesudah', color='blue')
    axes[i//2, i%2].set_title(f"Distribusi {kolom}: Sebelum vs Sesudah Scaling")
    axes[i//2, i%2].legend()

plt.tight_layout()
plt.show()
print("\n" + "\n")



# ================================================================
# --- Langkah 3: Ekstraksi Fitur (LDA) ---
# ================================================================
print("="*50)
print("               Ekstraksi Fitur (LDA)              ")
print("="*50)
# Pisahkan fitur & label
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)

# One-hot encoding kategorikal
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
y = df['Survived']
print("\n--- Langkah 3: Transformasi & Ekstraksi Fitur (LDA) ---")
print(f"Jumlah fitur sebelum LDA: {X.shape[1]}")

# Scaling ke range [0,1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# LDA: jumlah komponen maksimal = jumlah kelas - 1 (di Titanic = 1)
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)
print(f"Bentuk data setelah LDA: {X_lda.shape}")

# Konversi hasil LDA ke DataFrame
X_lda_df = pd.DataFrame(X_lda, columns=['LDA_Feature'])
y_df = pd.DataFrame(y, columns=['Survived'])

# Gabungkan kembali
df_lda = pd.concat([X_lda_df, y_df.reset_index(drop=True)], axis=1)

print("\n--- 5 Baris Pertama Dataset Setelah LDA ---")
print(df_lda.head())

# Simpan hasil ekstraksi LDA
df_lda.to_csv('train_lda.csv', index=False)
print("\n[INFO] Dataset hasil ekstraksi LDA disimpan ke 'train_lda.csv'")

# Visualisasi distribusi hasil LDA
print("\n[INFO] Menampilkan distribusi hasil LDA berdasarkan kelas Survived...")

plt.figure(figsize=(8, 6))
sns.stripplot(
    x="Survived",
    y="LDA_Feature",
    data=df_lda,
    hue="Survived",
    palette="muted",
    dodge=False,
    jitter=True,   
    alpha=0.7
)

plt.title("Distribusi Fitur LDA berdasarkan Label Survived")
plt.xlabel("Survived")
plt.ylabel("LDA Feature")
plt.legend(title="Survived")
plt.show()
print("\n" + "\n")






# ================================================================
# --- Langkah 4: Menangani Imbalanced Data dengan SMOTE-ENN ---
# imbalanced data aryo
# ================================================================
print("="*70)
print("               Menangani Imbalanced Data dengan SMOTE-ENN              ")
print("="*70)
# Drop kolom yang tidak relevan untuk fitur
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)

# Encoding kolom kategorikal (Sex, Embarked) ke numerik
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Label target
y = df['Survived']

print("\n--- Langkah 4: Imbalanced Data (SMOTE-ENN) ---")
print(f"Distribusi sebelum SMOTE-ENN: {Counter(y)}")

# Terapkan SMOTE-ENN
smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X, y)

print(f"Distribusi sesudah SMOTE-ENN: {Counter(y_res)}")

# Konversi kembali ke DataFrame
X_resampled = pd.DataFrame(X_res, columns=X.columns)
y_resampled = pd.Series(y_res, name='Survived')

# Gabungkan kembali jadi satu DataFrame
df_balanced = pd.concat([X_resampled, y_resampled], axis=1)

print("\n--- 5 Baris Pertama Dataset Setelah SMOTE-ENN ---")
print(df_balanced.head())

# Simpan hasil balancing
df_balanced.to_csv('train_balanced.csv', index=False)
print("\n[INFO] Dataset hasil balancing disimpan ke 'train_balanced.csv'")

# --- Visualisasi Distribusi Sebelum & Sesudah Balancing ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x=y, ax=axes[0])
axes[0].set_title("Distribusi Sebelum SMOTE-ENN")

sns.countplot(x=y_res, ax=axes[1])
axes[1].set_title("Distribusi Sesudah SMOTE-ENN")

plt.show()