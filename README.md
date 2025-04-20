# Laporan Proyek Machine Learning - Defrizal Yahdiyan Risyad

## **Domain Proyek**

Proyek ini berada pada domain **Real Estate** atau **Properti**, dengan fokus utama pada prediksi harga rumah. Penentuan harga properti yang akurat sangat penting bagi berbagai pemangku kepentingan seperti agen real estate, pembeli, penjual, serta bank dan lembaga keuangan untuk tujuan penilaian jaminan.

Prediksi harga rumah secara otomatis dapat mempercepat proses penilaian properti, mengurangi bias manusia, dan membantu dalam pengambilan keputusan berbasis data.

**Referensi**:
- [House Price Prediction using Machine Learning Algorithms](https://www.academia.edu/download/98410960/I7849078919.pdf)

## **Business Understanding**

### **1. Problem Statements**
- Bagaimana memprediksi harga rumah berdasarkan fitur-fitur seperti ukuran bangunan, jumlah garasi, dan kualitas keseluruhan?
- Algoritma machine learning apa yang paling baik dalam memberikan prediksi harga rumah yang akurat?

### **2. Goals**
- Mengembangkan model machine learning untuk memprediksi harga rumah dengan akurasi tinggi.
- Membandingkan performa beberapa algoritma regresi untuk menemukan model terbaik.

### **3. Solution Statements**
- Membangun beberapa model regresi: Linear Regression, Random Forest Regressor, dan XGBoost Regressor.
- Melakukan penggabungan model (stacking ensemble) untuk meningkatkan akurasi prediksi.
- Evaluasi model dilakukan menggunakan metrik RMSE dan R².

## **Data Understanding**

Data Understanding adalah tahap kedua dalam proses data science setelah business understanding. Pada tahap ini, kita mengenal lebih dalam struktur, tipe, distribusi, dan kualitas data yang kita miliki sebelum melanjutkan ke tahap preprocessing dan modeling.

### **1. Sumber Dataset**

Dataset yang digunakan adalah **Ames Housing Dataset**, sebuah dataset populer dalam prediksi harga rumah yang dikembangkan oleh Dean De Cock.

**Sumber** : [Kaggle - Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

### **2. Dimensi Dataset**

Dataset ini berisi 82 variabel mengenai properti rumah yang dijual di Ames, Iowa. Namun, pada eksperimen ini hanya beberapa fitur yang digunakan untuk mempermudah dan menjaga performa model

- **Jumlah data:** 2930 baris

- **Jumlah fitur:** 82 fitur

### **3. Struktur Dataset**

Dari hasil `df.info()`, kita dapat menyimpulkan:

- Ada tipe data campuran:

   - **28 kolom bertipe** `int64`

   - **11 kolom bertipe** `float64`

   - **43 kolom bertipe** `object` (biasanya kategori/teks)

- Beberapa kolom memiliki missing values (nilai kosong/NaN):

   - Contoh: Alley hanya memiliki 198 entri dari 2930 → missing >90%

   - Mas Vnr Type, Fireplace Qu, Pool QC, Fence, dll juga memiliki missing values yang signifikan

   - Artinya, perlu dilakukan penanganan data kosong (data cleaning/imputation)

### **4. Deskripsi Statistik Fitur Numerik**

Beberapa statistik deskriptif dari fitur numerik yang relevan

| Fitur             | Count  | Mean       | Std Dev    | Min     | 25%     | Median   | 75%     | Max     | Penjelasan                                               |
|------------------|--------|------------|------------|---------|---------|----------|---------|---------|----------------------------------------------------------------|
| **SalePrice**        | 2930   | 180,796    | 79,887     | 12,789  | 129,500 | 160,000  | 213,500 | 755,000 | Ini adalah harga rumah yang ingin diprediksi. |
| **Gr Liv Area**      | 2930   | 1,499.7    | 505.5      | 334     | 1,126   | 1,442    | 1,743   | 5,642   | Semakin besar area lantai utama, semakin tinggi harga rumahnya. |
| **Overall Qual**     | 2930   | 6.09       | 1.41       | 1       | 5       | 6        | 7       | 10      | Penilaian umum kualitas rumah dari 1 (terburuk) hingga 10 (terbaik), sangat berpengaruh terhadap harga rumah. |
| **Year Built**       | 2930   | 1971.4     | 30.2       | 1872    | 1954    | 1973     | 2001    | 2010    | Usia rumah, yang bisa mempengaruhi kondisi dan harga rumah. Rumah yang lebih baru mungkin lebih mahal. |
| **Total Bsmt SF**    | 2929   | 1,051.6    | 440.6      | 0       | 793     | 990      | 1,302   | 6,110   | Area total basement. Banyak rumah yang tidak memiliki basement, sehingga nilai 0 mungkin menandakan absennya basement. |
| **1st Flr SF**       | 2930   | 1,159.6    | 391.9      | 334     | 876     | 1,084    | 1,384   | 5,095   | Ukuran lantai pertama, yang umumnya memiliki hubungan langsung dengan harga rumah. |
| **Garage Cars**      | 2929   | 1.77       | 0.76       | 0       | 1       | 2        | 2       | 5       | Banyaknya mobil yang dapat ditampung di garasi rumah. Semakin besar jumlah mobil yang dapat ditampung, semakin tinggi harga rumah. |
| **Garage Area**      | 2929   | 472.8      | 215.0      | 0       | 320     | 480      | 576     | 1,488   | Ukuran garasi (dalam sqft), biasanya berhubungan erat dengan jumlah mobil yang dapat ditampung di garasi. |
| **Lot Area**         | 2930   | 10,148     | 7,880      | 1,300   | 7,440   | 9,436.5  | 11,555  | 215,245 | Luas tanah rumah, distribusinya sangat skewed dengan beberapa rumah yang memiliki luas tanah yang sangat besar. |
| **Full Bath**        | 2930   | 1.57       | 0.55       | 0       | 1       | 2        | 2       | 4       | Jumlah kamar mandi penuh, fitur penting yang seringkali berhubungan dengan ukuran dan harga rumah. |
| **Fireplaces**       | 2930   | 0.60       | 0.65       | 0       | 0       | 1        | 1       | 4       | Banyak rumah yang tidak memiliki perapian, namun jika ada, bisa meningkatkan harga rumah. |



## **Exploratory Data Analysis (EDA)**

### **1. Distribusi Harga Rumah (`SalePrice`)**
Distribusi `SalePrice` menunjukkan pola skewed ke kanan, yang mengindikasikan bahwa sebagian besar rumah memiliki harga di bawah rata-rata, sementara hanya sedikit yang memiliki harga sangat tinggi. Hal ini dapat mempengaruhi performa model regresi linier, sehingga teknik transformasi seperti log transformation dapat dipertimbangkan.


### **2. Korelasi Fitur dengan `SalePrice`**
Berdasarkan analisis korelasi antara fitur numerik dengan `SalePrice`, ditemukan bahwa beberapa fitur memiliki hubungan linier yang cukup kuat. Heatmap di bawah ini menunjukkan korelasi antar fitur, dengan fokus terhadap korelasi tertinggi terhadap `SalePrice`.

📌 **10 Fitur dengan Korelasi Tertinggi terhadap `SalePrice`:**
| Fitur              | Korelasi |
|--------------------|----------|
| Overall Qual       | 0.799    |
| Gr Liv Area        | 0.707    |
| Garage Cars        | 0.648    |
| Garage Area        | 0.640    |
| Total Bsmt SF      | 0.632    |
| 1st Flr SF         | 0.622    |
| Year Built         | 0.558    |
| Full Bath          | 0.546    |
| Year Remod/Add     | 0.533    |
| Garage Yr Blt      | 0.527    |

### **3. Hubungan `Gr Liv Area` dan `SalePrice`**
Scatter plot di bawah memperlihatkan hubungan antara luas bangunan di atas tanah (`Gr Liv Area`) dengan harga rumah (`SalePrice`). Titik-titik berwarna berdasarkan nilai `Overall Qual` menunjukkan bahwa semakin besar dan semakin tinggi kualitas rumah, cenderung memiliki harga jual yang lebih tinggi.


### **4. Insight Utama:**
- `Overall Qual` adalah indikator kualitas rumah yang paling kuat hubungannya dengan `SalePrice`.
- Fitur-fitur terkait ukuran seperti `Gr Liv Area`, `Total Bsmt SF`, dan `Garage Area` juga menunjukkan korelasi yang cukup tinggi.
- Distribusi harga rumah tidak normal, sehingga normalisasi mungkin diperlukan sebelum modeling.


## **Data Preparation**

Tahap persiapan data adalah salah satu bagian penting dalam proyek machine learning karena kualitas dan bentuk data sangat mempengaruhi performa model. Pada tahap ini, dilakukan serangkaian langkah preprocessing dan feature selection untuk memastikan bahwa data dalam format yang sesuai untuk modeling. Berikut adalah tahapan secara rinci:

### **1. Feature Selection**

   Pertama-tama, dilakukan pemilihan fitur (feature selection) berdasarkan domain knowledge dan hasil eksplorasi data sebelumnya. Fitur yang dipilih dibagi menjadi dua jenis utama:

- **Fitur Numerik**

   - `Overall Qual`: Kualitas keseluruhan material dan finishing rumah.
   
   - `Gr Liv Area`: Luas area tinggal di atas tanah.

   - `Garage Cars`: Kapasitas mobil di garasi.

   - `Total Bsmt SF`: Total luas basement.

   - `Year Built`: Tahun pembangunan rumah.

   - `1st Flr SF`: Luas lantai satu.

   - `Full Bath`: Jumlah kamar mandi penuh.

   - `TotRms AbvGrd`: Total jumlah ruangan di atas tanah.

- **Fitur Kategorikal**

   - `Neighborhood`: Lingkungan perumahan tempat rumah berada.

   - `Exter Qual`: Kualitas eksterior rumah berdasarkan penilaian.

Fitur-fitur ini dipilih karena memiliki korelasi signifikan terhadap harga rumah berdasarkan analisis sebelumnya.

### **2. Fitur Buatan (Feature Engineering)**
Untuk menambah representasi data, dibuat satu fitur buatan baru yaitu House Age, yang merepresentasikan usia rumah pada saat dijual:

`df['House Age'] = df['Yr Sold'] - df['Year Built']`

Fitur ini menambah konteks temporal, karena rumah yang lebih baru cenderung memiliki harga lebih tinggi. Fitur ini kemudian disertakan ke dalam pemodelan.

### **3. Pembersihan Data**
Setelah pemilihan fitur dan pembuatan fitur buatan, dilakukan penggabungan semua fitur yang akan digunakan:

`features = numeric_features + categorical_features + ['House Age']`

Dataset kemudian difilter agar hanya berisi baris-baris dengan data lengkap untuk semua fitur dan target `SalePrice`, dengan menghapus baris yang mengandung nilai `NaN`:

`df_model = df[features + ['SalePrice']].dropna()`

Langkah ini penting agar model tidak gagal saat proses pelatihan akibat adanya nilai kosong yang tidak bisa ditangani secara langsung.

### **4. Pemisahan Data**
Dataset yang sudah bersih dibagi menjadi fitur (X) dan target (y):

```
X = df_model.drop(columns='SalePrice')
y = df_model['SalePrice']
```

Kemudian, data dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split:

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

### **5. Preprocessing Pipeline**
Agar preprocessing dapat dilakukan secara konsisten dan modular, dibuat sebuah pipeline untuk menangani dua jenis fitur secara berbeda:

   **a. Numerik**

   Untuk semua fitur numerik, termasuk fitur buatan House Age, dilakukan standardisasi menggunakan StandardScaler. Teknik ini mengubah nilai fitur menjadi distribusi dengan rata-rata 0 dan standar deviasi 1.

   ```
   numeric_transformer = Pipeline([ 
    ('scaler', StandardScaler())
   ])
   ```
   Alasan pemilihan scaling ini adalah agar semua fitur numerik memiliki skala yang seragam, yang penting terutama untuk algoritma seperti Linear Regression dan XGBoost.

   **b. Kategorikal**

   Fitur kategorikal diproses menggunakan OneHotEncoder untuk mengubah nilai kategori menjadi vektor biner (0 dan 1). Encoder ini juga diset agar dapat mengabaikan kategori tak dikenal di data uji (handle_unknown='ignore'), agar model tidak error saat menerima kategori baru.

   ```
   categorical_transformer = Pipeline([ 
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
   ])
   ```

   **c. Gabungan Pipeline**

   Kedua jenis transformasi tersebut digabungkan dalam satu `ColumnTransformer` agar masing-masing diterapkan ke kolom yang relevan:
   ```
   preprocessor = ColumnTransformer(transformers=[ 
    ('num', numeric_transformer, numeric_features + ['House Age']),
    ('cat', categorical_transformer, categorical_features)
   ])
   ```
   Pipeline ini kemudian digunakan dalam semua model untuk memastikan bahwa preprocessing dilakukan secara otomatis dalam proses pelatihan dan prediksi.


## **Modeling**

### **1. Linear Regression**:
- **Cara Kerja**

   Linear Regression adalah algoritma statistik dasar yang digunakan untuk memodelkan hubungan linier antara satu atau lebih fitur independen dengan target numerik. Model ini bekerja dengan mencari parameter (koefisien) terbaik yang meminimalkan residual sum of squares antara nilai aktual dan nilai prediksi.

- **Parameter**

   Model `LinearRegression()` digunakan tanpa parameter tambahan, sehingga semua parameter berada pada kondisi default:

   - `fit_intercept=True` : model akan menghitung intercept secara otomatis.

   - `normalize='deprecated'` : normalisasi dilakukan di pipeline preprocessing, bukan di model ini.

   - `n_jobs=None` : komputasi berjalan secara default (tanpa paralelisme).

- **Kelebihan**
   - Mudah diimplementasikan dan sangat cepat dilatih.

   - Interpretasinya jelas (koefisien bisa menunjukkan pengaruh masing-masing fitur).

   - Sangat baik untuk baseline model.

- **Kekurangan**
   - Tidak dapat menangkap hubungan non-linier.

   - Sangat rentan terhadap outlier.

   - Multikolinearitas antar variabel independen bisa mempengaruhi hasil.

### **2. Random Forest Regressor**:

- **Cara Kerja**

   Random Forest adalah algoritma ensemble yang terdiri dari banyak pohon keputusan (decision tree). Setiap pohon dilatih pada subset acak dari data, dan prediksi akhir diambil dari rata-rata prediksi semua pohon. Ini membantu mengurangi overfitting yang biasa terjadi pada decision tree tunggal.

   - Model berbasis ensemble decision tree.
   - Dapat menangani non-linearitas dengan baik.
   - Hyperparameter seperti `n_estimators=200`, `max_depth=10` disesuaikan.

- **Parameter**

   Model dituning menggunakan `RandomizedSearchCV` dengan ruang pencarian berikut:

   - `n_estimators`: [100, 200, 300] — jumlah pohon yang digunakan dalam hutan.

   - `max_depth`: [10, 20, None] — kedalaman maksimum setiap pohon.

   - `min_samples_split`: [2, 5, 10] — jumlah minimum sampel untuk melakukan split internal.

   Tuning dilakukan menggunakan:

   - `cv=3`: 3-fold cross-validation.

   - `scoring='neg_root_mean_squared_error'`: metrik evaluasi.

   - `n_iter=10`: jumlah kombinasi parameter yang dicoba secara acak.

- **Kelebihan**

   - Mampu menangkap hubungan non-linear.

   - Tahan terhadap overfitting berkat averaging dari banyak pohon.

   - Dapat menangani fitur numerik dan kategorikal dengan baik.

- **Kekurangan**

   - Memakan waktu pelatihan lebih lama.

   - Kurang transparan (sulit diinterpretasi dibanding Linear Regression).

### **3. XGBoost Regressor**:

   XGBoost (Extreme Gradient Boosting) adalah teknik boosting yang membangun model secara bertahap dengan menambahkan pohon yang memperbaiki kesalahan dari model sebelumnya. Model ini sangat efisien dan memiliki regularisasi bawaan untuk menghindari overfitting.

- **Parameter**

   Model digunakan tanpa tuning eksplisit, sehingga parameter utamanya adalah default:

   - `n_estimators=100` : jumlah boosting round.

   - `learning_rate=0.1`: step size pada setiap boosting step.

   - `max_depth=3` : kedalaman maksimum pohon.

- **Kelebihan**  
   - Sangat akurat dan efisien.

   - Dapat menangani missing value secara internal.

   - Cocok untuk data besar dan kompleks.

- **Kekurangan**
   - Overfitting pada Dataset Kecil

   - Pengaturan Hyperparameter yang Kompleks

   - Waktu Pelatihan yang Lebih Lama untuk Dataset Besar

   - Sensitif terhadap Data yang Tidak Bersih


### **4. Stacking Regressor**:
Stacking adalah teknik ensemble yang menggabungkan beberapa model (disebut base learners) dan menggunakan model lain (meta-learner) untuk menggabungkan hasil prediksi dari model-model tersebut. Dalam eksperimen ini, base learners adalah Linear Regression, Random Forest (tuned), dan XGBoost, sedangkan meta-learner yang digunakan adalah Linear Regression.

- **Parameter**

   - `estimators`: daftar model dasar yang masing-masing dibungkus dengan pipeline preprocessing.

   - `final_estimator=LinearRegression()`: model akhir yang dilatih berdasarkan prediksi dari base learners.

- **Kelebihan**

   - Menggabungkan kelebihan berbagai jenis model.

   - Bias dan varians lebih seimbang karena memanfaatkan keragaman model.

- **Kekurangan**
   - Kompleksitas meningkat.

   - Waktu pelatihan lebih lama.

   - Interpretasi lebih sulit karena banyaknya lapisan model.

## **Evaluation**

Model dievaluasi menggunakan dua metrik utama:

- **RMSE (Root Mean Squared Error)**: Mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan target. Semakin rendah nilai RMSE, semakin baik performa model.
- **R² Score (Coefficient of Determination)**: Mengukur proporsi variasi data yang dijelaskan oleh model. Nilai maksimum adalah 1.0.

### **1. Hasil Evaluasi Model**

| Model             | RMSE     | R² Score |
|------------------|----------|----------|
| Linear Regression| ~35,000  | ~0.8427  |
| Random Forest    | ~27,000  | ~0.9174  |
| XGBoost          | ~26,500  | ~0.9121  |
| **Stacking**     | **26,559.52** | **0.9142** |

📌 **RMSE Stacking Regressor:**
- Train: 14,902.88
- Test : 26,559.52

📈 **R² Score Stacking Regressor**: 0.9142

### **2. Visualisasi Evaluasi**
- **RMSE Train vs Test**

   Menunjukkan perbedaan error pada data pelatihan dan data pengujian. Selisih RMSE cukup wajar, menandakan model tidak mengalami overfitting.

- **Distribusi Error**

   Distribusi error simetris dan terpusat di sekitar nol, menandakan bahwa model tidak bias secara sistematis.

-  **RMSE per Model in Stacking Regressor**

   Memvisualisasikan performa masing-masing base model pada stacking.


Model **Stacking Regressor** terbukti menjadi model terbaik dengan performa yang paling stabil dan akurat pada data test (RMSE rendah dan R² tinggi). Visualisasi menunjukkan prediksi model mendekati nilai aktual, dan distribusi error yang normal.


---

**Catatan**:
- Dapat dikembangkan dengan lebih banyak fitur (total 70+) dan preprocessing lanjutan (seperti encoding dan feature engineering).
- Model juga dapat ditingkatkan dengan hyperparameter tuning berbasis grid/random search atau Optuna.