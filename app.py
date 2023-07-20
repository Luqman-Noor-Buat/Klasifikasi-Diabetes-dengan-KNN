from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump
from pickle import load
from sklearn.metrics import r2_score
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/dataset/", methods=["GET","POST"])
def dataset():
    if request.method == "POST":
        global diabetes
        cek = 1
        link = request.form['link']
        # Membaca dataset
        diabetes = pd.read_csv(link)
        return render_template("dataset.html", cek=cek, a=link, b=diabetes.shape, 
            dataset_Loan=[diabetes.to_html(justify='center',index=False)], 
            dataset_Loan_isi=[''], coba=diabetes.to_numpy())
    else:
        cek=0
        link=""
        tampilan=""
        return render_template("dataset.html", a=link, b=tampilan, cek=cek)

@app.route("/preprocessing/", methods=["GET","POST"])
def preprocessing():
    if request.method == "POST":
        global normalisasi_request
        normalisasi_request = request.form['normalisasi']
        cek = 1
        # Mengambil kolom selain kode_kontrak dan risk_rating dan melakukan normalisasi data
        normalisasi = diabetes.drop(["Diagnosa"], axis=1)
        if normalisasi_request == '1':
            global diabetes_Zscore
            # Melakukan normalisasi dengan Z Score atau StandardScaler
            scaler = StandardScaler()
            scaler.fit(normalisasi)
            scale_data = (scaler.transform(normalisasi))
            # Menampilkan data normalisasi dari Z score
            dataZScale = pd.DataFrame(scale_data, columns=normalisasi.columns.values)
            dataZScale1 = dataZScale.head()
            # Mengambil class diagnosa
            Diagnosa = pd.DataFrame(diabetes, columns=["Diagnosa"])
            # Menggabungkan kolom yang sudah dinormalisasi Min Max dan data Loan_Status
            diabetes_Zscore = pd.concat([dataZScale, Diagnosa], axis=1)
            diabetes_Zscore1 = diabetes_Zscore.head()
            # save the scaler
            dump(scaler, open('scaler_ZScore.pkl', 'wb'))
            return render_template("preprocesing.html", cek=cek, b=normalisasi_request,
                                dataZScale1=[dataZScale1.to_html(justify='center',index=False)], dataZScale1_isi=[''],
                                diabetes_Zscore1=[diabetes_Zscore1.to_html(justify='center',index=False)], diabetes_Zscore1_isi=[''])
        else:
            global diabetes_min_max
            # melakukan skala fitur
            scaler = MinMaxScaler()
            model = scaler.fit(normalisasi)
            scaled_data=model.transform(normalisasi)
            # menampilkan data normalisasi dari Min Max
            namakolom = normalisasi.columns.values
            dataMinMax = pd.DataFrame(scaled_data, columns=namakolom)
            dataMinMax1 = dataMinMax.head()
            # Mengambil class diagnosa
            Diagnosa = pd.DataFrame(diabetes, columns=["Diagnosa"])
            # Menggabungkan kolom yang sudah dinormalisasi Min Max dan data Loan_Status
            diabetes_min_max = pd.concat([dataMinMax, Diagnosa], axis=1)
            diabetes_min_max1 = diabetes_min_max.head()
            # save the scaler
            dump(scaler, open('scaler_MinMax.pkl', 'wb'))
            return render_template("preprocesing.html", cek=cek, b=normalisasi_request,
                                dataMinMax1=[dataMinMax1.to_html(justify='center',index=False)], dataMinMax1_isi=[''],
                                diabetes_min_max1=[diabetes_min_max1.to_html(justify='center',index=False)], diabetes_min_max1_isi=[''])
    else:
        cek = 0
        normalisasi_request = ""
        return render_template("preprocesing.html", cek=cek, b=normalisasi_request)

@app.route("/modelling/", methods=["GET","POST"])
def modelling():
    if request.method == "POST":
        global modeling
        cek = 1
        modeling = request.form['modelling']
        nilai_k = request.form['nilai_k']
        if modeling == "1":
            if normalisasi_request == '1':
                # Mengambil kelas dan fitur dari dataset pada Z score
                # fiturnya
                X_Zscore = diabetes_Zscore.iloc[:,0:11].values
                # classnya
                y_Zscore = diabetes_Zscore.iloc[:,11].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_Zscore, X_testn_Zscore, y_trainn_Zscore, y_testn_Zscore = train_test_split(X_Zscore, y_Zscore, test_size=0.30, random_state=0, stratify=y_Zscore)
                # Menghitung akurasi dari KNN dengan normalisasi Z score
                neigh = KNeighborsClassifier(int(nilai_k))
                neigh.fit(X_trainn_Zscore, y_trainn_Zscore)
                acc_knn = round(neigh.score(X_trainn_Zscore, y_trainn_Zscore) * 100, 2)
                dump(neigh, open('model_ZScore_KNN.pkl', 'wb'))
                hasil_akurasi = 'Akurasi KNN dengan normalisasi Z Score : %.3f'%acc_knn
                return render_template("modeling.html", cek=cek, a=modeling, nilai_k=nilai_k, akurasi=hasil_akurasi)
            else:
                # Mengambil kelas dan fitur dari dataset pada Min Max
                # fiturnya
                X_min_max = diabetes_min_max.iloc[:,0:11].values
                # classnya
                y_min_max = diabetes_min_max.iloc[:,11].values
                # Membagi data menjadi data training dan data uji dengan data uji berjumlah 30%
                X_trainn_min_max, X_testn_min_max, y_trainn_min_max, y_testn_min_max = train_test_split(X_min_max, y_min_max, test_size=0.30, random_state=0, stratify=y_min_max)
                # Menghitung akurasi dari KNN dengan normalisasi min max
                neigh = KNeighborsClassifier(int(nilai_k))
                neigh.fit(X_trainn_min_max, y_trainn_min_max)
                acc_knn = round(neigh.score(X_trainn_min_max, y_trainn_min_max) * 100, 2)
                dump(neigh, open('model_MinMax_KNN.pkl', 'wb'))
                hasil_akurasi = 'Akurasi KNN dengan normalisasi Min Max : %.3f'%acc_knn
                return render_template("modeling.html", cek=cek, a=modeling, nilai_k=nilai_k, akurasi=hasil_akurasi)
    else:
        cek=0
        modelling = ""
        nilai_k = ""
    return render_template("modeling.html", cek=cek, a=modelling, nilai_k=nilai_k)

@app.route("/prediksi/", methods=["GET","POST"])
def prediksi():
    if request.method == "POST":
        Jenis_Kelamin = request.form['Jenis_Kelamin']
        Umur_Tahun = request.form['Umur_Tahun']
        Lama_Sakit = request.form['Lama_Sakit']
        Tinggi = request.form['Tinggi']
        Berat_Badan = request.form['Berat_Badan']
        Lingkar_Perut = request.form['Lingkar_Perut']
        Sistole = request.form['Sistole']
        Diastole = request.form['Diastole']
        Nafas = request.form['Nafas']
        Detak_Nadi = request.form['Detak_Nadi']
        Suhu = request.form['Suhu']
        cek = 1
        # load the scaler
        scaler_ZScore = load(open('scaler_ZScore.pkl', 'rb'))
        scaler_MinMax = load(open('scaler_MinMax.pkl', 'rb'))
        # load the model
        model_ZScore_KNN = load(open('model_ZScore_KNN.pkl', 'rb'))
        model_MinMax_KNN = load(open('model_MinMax_KNN.pkl', 'rb'))
        # Data Ditampung Dalam Bentuk Array
        dataArray = [[int(Jenis_Kelamin), int(Umur_Tahun), int(Lama_Sakit), int(Tinggi), int(Berat_Badan), int(Lingkar_Perut), int(Sistole), int(Diastole), int(Nafas), int(Detak_Nadi), int(Suhu)]]
        if (normalisasi_request == '1'):
            # Data Dinormalisasi
            hasil_scale_ZScore = (scaler_ZScore.transform(dataArray))
            # Data dimodelkan
            hasil = model_ZScore_KNN.predict(hasil_scale_ZScore)
            pesan = "Anda menggunakan normalisasi Z Score dan pemodelan KNN."
            if (hasil == 1):
                prediksine = "DITERIMA"
            elif (hasil == 0):
                prediksine = "DITOLAK"
            return render_template("prediksi.html", cek=cek, a=Jenis_Kelamin, b=Umur_Tahun, e=Lama_Sakit,
                            i=Tinggi, j=Berat_Badan, k=Lingkar_Perut, l=Sistole, m=Diastole, n=Nafas, o=Detak_Nadi, p=Suhu, hasil=hasil, pesan=pesan)
        elif (normalisasi_request == '2'):
            # Data Dinormalisasi
            hasil_scale_MinMax = (scaler_MinMax.transform(dataArray))
            # Data dimodelkan
            hasil = model_MinMax_KNN.predict(hasil_scale_MinMax)
            pesan = "Anda menggunakan normalisasi Min Max dan pemodelan KNN."
            if (hasil == 1):
                prediksine = "DITERIMA"
            elif (hasil == 0):
                prediksine = "DITOLAK"
            return render_template("prediksi.html", cek=cek, a=Jenis_Kelamin, b=Umur_Tahun, e=Lama_Sakit,
                            i=Tinggi, j=Berat_Badan, k=Lingkar_Perut, l=Sistole, m=Diastole, n=Nafas, o=Detak_Nadi, p=Suhu, hasil=hasil, pesan=pesan)
    else:
        Jenis_Kelamin = ''
        Umur_Tahun = ''
        Lama_Sakit = ''
        Tinggi = ''
        Berat_Badan = ''
        Lingkar_Perut = ''
        Sistole = ''
        Diastole = ''
        Nafas = ''
        Detak_Nadi = ''
        Suhu = ''
        hasil = ''
        pesan = ''
        cek = 0
        return render_template("prediksi.html", cek=cek, a=Jenis_Kelamin, b=Umur_Tahun, e=Lama_Sakit, i=Tinggi, 
                j=Berat_Badan, k=Lingkar_Perut, l=Sistole, m=Diastole, n=Nafas, o=Detak_Nadi, p=Suhu, hasil=hasil, pesan=pesan)
