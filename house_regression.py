# %% [markdown]
# # **1. Import Library**

# %% [markdown]
# Pada bagian ini, kita mengimpor berbagai pustaka yang diperlukan untuk analisis dan pemodelan data. Beberapa pustaka ini digunakan untuk manipulasi data, visualisasi, dan penerapan model pembelajaran mesin.

# %%
# Library Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Library Scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# # **2. 📊Load Data & Data Understanding**

# %% [markdown]
# Pada bagian ini, kita memuat dataset dari file CSV yang bernama AmesHousing.csv ke dalam sebuah DataFrame menggunakan pustaka pandas. Dataset ini berisi informasi tentang properti rumah di Ames, Iowa, yang digunakan untuk analisis prediksi harga rumah.

# %%
df = pd.read_csv('AmesHousing.csv')
print("\n📂 Dataset Shape:", df.shape)
df.head()

# %% [markdown]
# Dataset yang digunakan memiliki bentuk **2930 baris dan 82 kolom**, yang merepresentasikan informasi lengkap mengenai properti rumah yang akan dianalisis, termasuk fitur-fitur seperti ukuran lahan, kualitas material, jumlah kamar, hingga informasi penjualan. Kolom `SalePrice` menjadi variabel target (label) yang ingin diprediksi, sedangkan sisanya merupakan fitur penjelas baik numerik maupun kategorikal. Setiap baris mewakili satu rumah, dan data mencakup berbagai aspek penting seperti lokasi `Neighborhood`, kondisi fisik (`Overall Qual`, `Year Built`, `Garage Cars`, dll.), serta elemen tambahan seperti kolam, pagar, dan fitur lainnya. Dataset ini cukup kaya dan kompleks, memberikan dasar yang kuat untuk membangun model prediksi harga rumah yang akurat.

# %% [markdown]
# #### **Informasi Dataset**
# Pada bagian ini, kita menggunakan fungsi info() dari pandas untuk mendapatkan informasi umum tentang dataset, seperti jumlah entri, tipe data setiap kolom, dan jumlah nilai yang tidak kosong (non-null) pada setiap kolom.

# %%
print("\n📋 Info Dataset:")
print(df.info())

# %% [markdown]
# Berdasarkan informasi dataset, terdapat **2930 entri** dan **82 kolom**, yang terdiri dari beragam tipe data: 28 kolom bertipe integer, 11 bertipe float, dan 43 bertipe object (kategorikal). Meskipun sebagian besar kolom memiliki data yang lengkap, beberapa kolom menunjukkan adanya missing values cukup signifikan, seperti `Alley`, `Pool QC`, `Fence`, dan `Misc Feature`, yang hanya terisi sebagian kecil dari total data. Ini menunjukkan bahwa sebelum membangun model, perlu dilakukan data preprocessing seperti penanganan nilai hilang dan encoding variabel kategorikal. Struktur dataset ini cukup komprehensif dan merepresentasikan berbagai aspek properti rumah yang berpengaruh terhadap harga, menjadikannya sangat cocok untuk proyek prediksi harga rumah berbasis machine learning.

# %% [markdown]
# #### **Deksripsi Statistik**
# Pada bagian ini, kita menggunakan fungsi describe() dari pandas untuk menghasilkan deskripsi statistik dari dataset. Deskripsi ini mencakup berbagai metrik statistik untuk setiap kolom numerik, seperti jumlah data yang ada, rata-rata, standar deviasi, nilai minimum dan maksimum, serta kuartil (25%, 50%, dan 75%).

# %%
print("\n🔍 Deskripsi Statistik:")
print(df.describe().T)

# %% [markdown]
# Hasil deskripsi statistik menunjukkan bahwa data terdiri dari **2.930 observasi** dengan berbagai variabel numerik terkait properti perumahan. Harga jual rumah (SalePrice) memiliki rata-rata sekitar $180.796 dengan deviasi standar $79.887, mencerminkan variasi yang signifikan, dari harga serendah $12.789 hingga setinggi $755.000. Ukuran total bangunan seperti `Gr Liv Area` dan `Total Bsmt SF` juga bervariasi luas, dengan nilai maksimum masing-masing mencapai lebih dari 5.600 kaki persegi dan 6.100 kaki persegi. Sebagian besar rumah memiliki 2 hingga 3 kamar tidur, 1 hingga 2 kamar mandi penuh, dan rata-rata kualitas keseluruhan (`Overall Qual`) adalah sekitar 6 (dengan skala 1–10). Beberapa fitur seperti `Pool Area`, `3Ssn Porch`, dan `Low Qual Fin SF` memiliki median nol, menandakan bahwa sebagian besar rumah tidak memiliki fitur-fitur ini. Terdapat pula variabel dengan nilai hilang, seperti `Lot Frontage` dan `Garage Yr Blt`, yang perlu ditangani dalam tahap preprocessing. Secara keseluruhan, statistik ini memberikan gambaran umum mengenai distribusi, sebaran, dan keberadaan outlier dalam dataset.

# %% [markdown]
# #### **Periksa Nilai Hilang**
# Pada bagian ini, kita menggunakan metode isnull() untuk memeriksa apakah ada nilai yang hilang (missing) pada setiap kolom dalam dataset. Kemudian, fungsi sum() digunakan untuk menghitung jumlah nilai yang hilang untuk setiap kolom. Hasilnya diurutkan berdasarkan jumlah missing values secara menurun dengan sort_values(ascending=False).

# %%
missing_values = df.isnull().sum()
print("\n🧼 Missing Values:")
print(missing_values[missing_values > 0].sort_values(ascending=False))

# %% [markdown]
# Dari hasil analisis missing values, terlihat bahwa beberapa fitur dalam dataset memiliki jumlah nilai kosong yang cukup signifikan, seperti `Pool QC` (2917 nilai hilang), `Misc Feature` (2824), `Alley` (2732), dan `Fence` (2358), yang kemungkinan besar disebabkan oleh ketidakhadiran fitur tersebut pada sebagian besar properti, bukan karena kesalahan pencatatan. Beberapa fitur terkait basement (`Bsmt Qual`, `Bsmt Cond`, dll.) dan garasi (`Garage Type`, `Garage Yr Blt`, dll.) juga memiliki nilai hilang, menunjukkan bahwa sebagian rumah mungkin tidak memiliki basement atau garasi. Fitur `Lot Frontage` juga memiliki jumlah missing yang cukup besar (490), yang bisa mempengaruhi analisis terkait ukuran lahan. Terdapat pula fitur dengan hanya satu atau dua nilai hilang, seperti `Electrical` dan `Garage Area`, yang bisa dengan mudah diimputasi. Identifikasi dan penanganan missing values ini sangat penting untuk memastikan integritas dan kualitas data sebelum melakukan analisis atau pelatihan model prediktif.

# %% [markdown]
# # 🔍 **3. Exploratory Data Analysis (EDA)**

# %% [markdown]
# Pada bagian ini, kita menganalisis korelasi antar fitur dalam dataset dengan harga jual rumah (SalePrice) dan melakukan beberapa visualisasi untuk membantu memahami hubungan tersebut.

# %% [markdown]
# ##### **Korelasi Tertinggi dengan `SalePrice`**

# %%
corr = df.corr(numeric_only=True)
top_corr = corr['SalePrice'].sort_values(ascending=False)[1:11]
print("\n📌 Korelasi Tertinggi dengan SalePrice:")
print(top_corr)

# %% [markdown]
# Korelasi antar fitur dengan `SalePrice` memberikan wawasan penting mengenai faktor-faktor yang memengaruhi harga jual rumah. Dari hasil yang diperoleh, fitur-fitur seperti `Overall Qual` (kualitas keseluruhan) dan `Gr Liv Area` (luas lantai ruang tamu) memiliki korelasi yang sangat kuat dengan harga rumah. Korelasi tinggi menunjukkan bahwa semakin baik kualitas rumah dan semakin besar ukuran ruang tamu, maka semakin tinggi pula harga jual rumah. Fitur-fitur ini adalah indikator utama yang perlu dipertimbangkan dalam model prediksi harga rumah.

# %% [markdown]
# #### **Heatmap Korelasi Fitur terhadap SalePrice**

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df[top_corr.index.tolist() + ['SalePrice']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi Fitur Top dengan SalePrice")
plt.tight_layout()
plt.show()

# %% [markdown]
# Heatmap ini menunjukkan hubungan korelasi antara fitur numerik dengan harga rumah (SalePrice), di mana warna merah menandakan korelasi positif yang kuat, dan biru menunjukkan korelasi yang lemah atau negatif. Dari heatmap, terlihat bahwa fitur seperti `OverallQual`, `GrLivArea`, `GarageCars`, dan `TotalBsmtSF` memiliki korelasi paling tinggi terhadap harga rumah, yang berarti fitur-fitur tersebut sangat berpengaruh dalam menentukan nilai jual rumah. Sementara fitur seperti `YrSold` atau `BsmtHalfBath` memiliki korelasi rendah, menandakan pengaruhnya terhadap harga cukup kecil.

# %% [markdown]
# #### **Distribusi Harga Rumah (SalePrice)**

# %%
# Distribusi SalePrice
plt.figure(figsize=(8, 6))
sns.histplot(df['SalePrice'], kde=True, color='green')
plt.title("Distribusi Harga Rumah (SalePrice)")
plt.xlabel("SalePrice")
plt.tight_layout()
plt.show()

# %% [markdown]
# Histogram ini menggambarkan distribusi frekuensi harga rumah yang ada dalam dataset. Distribusi terlihat miring ke kanan (right-skewed), artinya sebagian besar rumah dijual pada kisaran harga yang relatif rendah (antara 100.000 hingga 200.000), sementara hanya sebagian kecil yang dijual dengan harga sangat tinggi. Hal ini menunjukkan bahwa harga rumah tidak terdistribusi normal, dan keberadaan outlier perlu diperhatikan dalam proses modeling atau analisis statistik lebih lanjut.

# %% [markdown]
# #### **Scatter Plot Gr Liv Area vs SalePrice**

# %%
# Scatter plot Gr Liv Area vs SalePrice
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice', hue='Overall Qual', palette='viridis', alpha=0.7)
plt.title("Gr Liv Area vs SalePrice")
plt.tight_layout()
plt.show()

# %% [markdown]
# Scatter plot ini menampilkan hubungan antara luas area tinggal `Gr Liv Area` dan harga rumah `SalePrice`, yang menunjukkan pola hubungan linear positif—semakin luas rumah, umumnya semakin mahal harganya. Warna titik mencerminkan kualitas keseluruhan rumah `OverallQual`, dan terlihat bahwa rumah dengan kualitas lebih tinggi (warna lebih terang) cenderung memiliki harga yang lebih tinggi meskipun pada area tinggal yang sama. Ini menunjukkan bahwa baik luas rumah maupun kualitas bangunan berperan besar dalam menentukan harga rumah.

# %% [markdown]
# # ✨ **3. Feature Selection & Preprocessing**

# %% [markdown]
# Pada tahap Feature Selection & Preprocessing, dilakukan pemilihan fitur numerik dan kategorikal yang dianggap relevan untuk memprediksi harga rumah `SalePrice`. Fitur numerik yang dipilih seperti `Overall Qual`, `Gr Liv Area`, dan `Total Bsmt SF` memiliki korelasi tinggi terhadap harga rumah, sedangkan fitur kategorikal seperti `Neighborhood` dan `Exter Qual` dinilai mewakili kualitas lingkungan dan eksterior rumah. Selain itu, ditambahkan fitur buatan bernama `House Age` yang dihitung dari selisih antara tahun penjualan dan tahun pembangunan rumah, sebagai indikator usia rumah. Data kemudian difilter agar hanya berisi fitur yang relevan dan menghapus baris yang mengandung nilai kosong. Untuk preprocessing, dibuat pipeline yang menstandarisasi fitur numerik menggunakan `StandardScaler` dan mengubah fitur kategorikal menjadi bentuk numerik menggunakan `OneHotEncoder`, lalu semua transformasi tersebut digabungkan menggunakan `ColumnTransformer`. Tahapan ini penting untuk memastikan semua fitur dalam format yang sesuai sebelum dimasukkan ke model prediksi.

# %%
# Fitur numerik dan kategorikal
numeric_features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF',
                    'Year Built', '1st Flr SF', 'Full Bath', 'TotRms AbvGrd']
categorical_features = ['Neighborhood', 'Exter Qual']

# Fitur buatan
df['House Age'] = df['Yr Sold'] - df['Year Built']

# Pilih data dan drop NaN
features = numeric_features + categorical_features + ['House Age']
df_model = df[features + ['SalePrice']].dropna()

X = df_model.drop(columns='SalePrice')
y = df_model['SalePrice']

# Preprocessing pipeline
numeric_transformer = Pipeline([ 
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([ 
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[ 
    ('num', numeric_transformer, numeric_features + ['House Age']),
    ('cat', categorical_transformer, categorical_features)
])

# %% [markdown]
# # 🔄 **4. Train-Test Split**

# %% [markdown]
# Pada tahap Train-Test Split, data dibagi menjadi dua bagian utama yaitu data latih `X_train`, `y_train` dan data uji `X_test`, `y_test` dengan proporsi 80:20. Artinya, 80% dari data digunakan untuk melatih model, sementara 20% sisanya digunakan untuk menguji performa model setelah dilatih. Pembagian ini bertujuan untuk mengevaluasi kemampuan model dalam memprediksi data yang belum pernah dilihat sebelumnya. Parameter `random_state=42` digunakan agar pembagian data dilakukan secara acak namun tetap bisa direproduksi (reproducible), sehingga hasil pembagian akan konsisten setiap kali kode dijalankan. Tahap ini penting untuk mencegah overfitting dan memastikan model memiliki generalisasi yang baik.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# # 🤖 **5. Modeling & Tuning**

# %% [markdown]
# Pada tahap Modeling & Tuning, dilakukan pelatihan model dengan berbagai algoritma regresi serta penyetelan (tuning) untuk meningkatkan performanya. Tiga model dasar yang digunakan adalah **Linear Regression**, **Random Forest Regressor**, dan **XGBoost Regressor**.

# %%
# Base models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)

# Tuning Random Forest
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_tuned = RandomizedSearchCV(rf, param_dist_rf, n_iter=10, cv=3,
                              scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
rf_pipeline = Pipeline([ 
    ('preprocessor', preprocessor),
    ('regressor', rf_tuned)
])
rf_pipeline.fit(X_train, y_train)
best_rf = rf_tuned.best_estimator_

# XGBoost pipeline
xgb_pipeline = Pipeline([ 
    ('preprocessor', preprocessor),
    ('regressor', xgb)
])
xgb_pipeline.fit(X_train, y_train)

# Stacking
stacked_model = StackingRegressor(
    estimators=[ 
        ('lr', Pipeline([('pre', preprocessor), ('lr', lr)])),
        ('rf', Pipeline([('pre', preprocessor), ('rf', best_rf)])),
        ('xgb', Pipeline([('pre', preprocessor), ('xgb', xgb)])),
    ],
    final_estimator=LinearRegression()
)
stacked_model.fit(X_train, y_train)

# %% [markdown]
# Untuk model Random Forest, dilakukan tuning hyperparameter menggunakan `RandomizedSearchCV` dengan kombinasi parameter seperti jumlah pohon`(n_estimators`, kedalaman maksimum `max_depth`, dan jumlah minimum sampel untuk split `min_samples_split` untuk mencari kombinasi terbaik berdasarkan skor `neg_root_mean_squared_error`. Setelah itu, model Random Forest terbaik digabungkan dalam pipeline bersama preprocessing. Model XGBoost juga dilatih menggunakan pipeline serupa. Terakhir, dilakukan **teknik Stacking**, yaitu menggabungkan ketiga model (Linear Regression, Random Forest, dan XGBoost) sebagai model base, kemudian hasil prediksi mereka digabung dan dipelajari oleh model akhir berupa Linear Regression. Pendekatan stacking ini bertujuan untuk memanfaatkan kekuatan masing-masing model dan menghasilkan prediksi yang lebih akurat.

# %% [markdown]
# # 📈 **6. Evaluation**

# %% [markdown]
# Pada tahap Evaluation, dilakukan pengujian performa model stacking menggunakan metrik **RMSE Root Mean Squared Error** untuk mengukur seberapa jauh hasil prediksi dari nilai sebenarnya dalam satuan yang sama dengan target `SalePrice`. 

# %%
# Evaluate on Training Data
y_train_pred = stacked_model.predict(X_train)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

# RMSE untuk Test Data
y_pred = stacked_model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)

# Plotting RMSE for Training and Testing
rmse_values = {'Train': train_rmse, 'Test': test_rmse}

# Plot RMSE comparison
plt.figure(figsize=(8, 6))
sns.barplot(x=list(rmse_values.keys()), y=list(rmse_values.values()), palette='viridis')
plt.title('RMSE Comparison: Train vs Test')
plt.xlabel('Dataset')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()

print("🔎 RMSE Stacking Regressor:")
print(f"  - Train: {train_rmse:.4f}")
print(f"  - Test : {test_rmse:.4f}")
stacked_r2 = r2_score(y_test, y_pred)
print(f"📈 Stacking Regressor R² Score: {stacked_r2:.4f}")

# %% [markdown]
# Hasil evaluasi menunjukkan bahwa RMSE pada data train adalah sekitar **14.902**, sedangkan pada data test meningkat menjadi **26.559**. Grafik batang memperjelas perbedaan ini, menunjukkan bahwa model memiliki performa lebih baik pada data pelatihan dibanding data pengujian. Hal ini bisa menjadi indikasi **overfitting ringan**, di mana model terlalu menyesuaikan diri pada data train. Meskipun begitu, nilai **R² Score sebesar 0.9142** menunjukkan bahwa model masih mampu menjelaskan lebih dari **91% variansi** harga rumah pada data test, yang merupakan indikasi performa yang sangat baik dalam regresi. Evaluasi ini penting untuk memahami keseimbangan antara akurasi dan generalisasi model terhadap data baru.

# %% [markdown]
# #### **Evaluation Every Base Model**
# Bagian ini bertujuan untuk mengevaluasi kontribusi masing-masing model dasar yang digunakan dalam ensemble Stacking Regressor. 
# 
# Pertama, dibuat dictionary `model_keys` untuk memetakan nama model (seperti Linear Regression, Random Forest, dan XGBoost) ke label yang digunakan dalam objek `stacked_model`. Kemudian, dilakukan iterasi untuk menghitung **R² Score** dari setiap model terhadap data uji `X_test, y_test`. 

# %%
# Mapping label ke nama estimator pada stacking
model_keys = {
    'Linear Regression': 'lr',
    'Random Forest': 'rf',
    'XGBoost': 'xgb'
}

# Menampilkan R² Score per model
for label, key in model_keys.items():
    score = stacked_model.named_estimators_[key].score(X_test, y_test)
    print(f"{label} R² Score: {score:.4f}")

# %% [markdown]
# Hasil evaluasi menunjukkan bahwa **Random Forest memiliki performa terbaik** dengan R² sebesar **0.9174**, diikuti oleh **XGBoost (0.9121)**, dan **Linear Regression (0.8427)**. Ini menandakan bahwa meskipun semua model berkontribusi terhadap model ensemble, Random Forest dan XGBoost memiliki peran dominan karena kemampuannya yang lebih tinggi dalam menjelaskan variansi data target pada data uji.

# %% [markdown]
# # 📊 **7. Visualisasi Hasil**

# %% [markdown]
# Pada tahap Visualisasi Hasil, dilakukan tiga jenis visualisasi untuk memperkuat evaluasi model. 

# %% [markdown]
# #### 📌 **Prediksi vs Aktual Harga Rumah**

# %%
y_pred = stacked_model.predict(X_test)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Prediksi vs Aktual Harga Rumah')
plt.xlabel('Harga Aktual')
plt.ylabel('Harga Prediksi')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Visualisasi ini berupa scatter plot yang membandingkan harga rumah aktual (sumbu X) dengan hasil prediksi model (sumbu Y). Garis merah putus-putus menunjukkan garis ideal di mana nilai prediksi sama persis dengan nilai aktual. Sebagian besar titik mendekati garis ini, artinya model menghasilkan prediksi yang cukup akurat. Namun, terlihat juga beberapa outlier, terutama untuk harga rumah yang sangat tinggi, yang menunjukkan bahwa model agak kesulitan memprediksi nilai ekstrem.

# %% [markdown]
# #### 📌 **Distribusi Error (Aktual - Prediksi)**

# %%
# Distribusi Error
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, color='purple')
plt.title('Distribusi Error (Aktual - Prediksi)')
plt.xlabel('Error')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Grafik ini menunjukkan sebaran selisih antara harga aktual dan hasil prediksi. Bentuk histogramnya mendekati distribusi normal, berpusat di sekitar nol, yang menandakan bahwa error model cukup seimbang—jumlah prediksi yang terlalu tinggi dan terlalu rendah hampir sama. Ini adalah indikasi bahwa model tidak mengalami bias signifikan ke satu arah (over atau under prediction). Namun, masih ada ekor distribusi yang cukup panjang ke kiri dan kanan, menandakan ada beberapa prediksi yang jauh dari nilai sebenarnya.

# %% [markdown]
# #### 📌 **RMSE per Model in Stacking Regressor**

# %%
# RMSE per Model dalam Stacking Regressor
model_rmse = {
    'Linear Regression': mean_squared_error(y_test, stacked_model.named_estimators_['lr'].predict(X_test), squared=False),
    'Random Forest': mean_squared_error(y_test, stacked_model.named_estimators_['rf'].predict(X_test), squared=False),
    'XGBoost': mean_squared_error(y_test, stacked_model.named_estimators_['xgb'].predict(X_test), squared=False)
}

plt.figure(figsize=(8, 6))
sns.barplot(x=list(model_rmse.keys()), y=list(model_rmse.values()), palette='coolwarm')
plt.title('RMSE per Model in Stacking Regressor')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()

# %% [markdown]
# Visualisasi ini membandingkan nilai Root Mean Squared Error (RMSE) dari masing-masing model yang digunakan dalam ensemble Stacking: Linear Regression, Random Forest, dan XGBoost. Hasilnya, Linear Regression memiliki RMSE tertinggi, yang berarti performanya paling lemah dalam hal akurasi prediksi. Sementara itu, Random Forest dan XGBoost menunjukkan RMSE yang lebih rendah dan hampir seimbang, menandakan bahwa keduanya memiliki kontribusi yang lebih kuat dan stabil dalam meningkatkan performa keseluruhan dari model stacking.


