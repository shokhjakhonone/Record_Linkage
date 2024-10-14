# app/processing.py

import os
import wget
import zipfile
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rapid_fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import Parallel, delayed

def download_and_extract():
    os.makedirs('data', exist_ok=True)

    # Загрузка и распаковка данных
    url = "https://oc2.phoenixit.ru/s/NDJeMeQg9fMBa62/download"
    print(f"Скачивание данных из {url}...")
    wget.download(url, os.path.join('data', 'download.zip'))
    print("\nЗагрузка завершена.")

    with zipfile.ZipFile(os.path.join('data', 'download.zip'), 'r') as zip_ref:
        zip_ref.extractall('data')
    print("Распаковка завершена.")

    csv_zips = ['main1.csv.zip', 'main2.csv.zip', 'main3.csv.zip']
    for csv_zip in csv_zips:
        zip_path = os.path.join('data', 'public', csv_zip)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/public')
        print(f"Распаковка {csv_zip} завершена.")

def load_datasets():
    print("Загрузка датасетов...")
    dataset1 = pd.read_csv("data/public/main1.csv", nrows=30)
    dataset2 = pd.read_csv("data/public/main2.csv", nrows=30)
    dataset3 = pd.read_csv("data/public/main3.csv", nrows=30)
    print("Загрузка завершена.")
    return dataset1, dataset2, dataset3

def preprocess_datasets(dataset1, dataset2, dataset3):
    print("Предобработка данных...")

    # Очищаем и стандартизируем данные первого датасета
    dataset1.columns = dataset1.columns.str.strip().str.lower()
    dataset1['full_name'] = dataset1['full_name'].str.strip().str.lower()
    dataset1['email'] = dataset1['email'].str.strip().str.lower()
    dataset1['address'] = dataset1['address'].str.strip().str.lower()
    dataset1['phone'] = dataset1['phone'].str.replace(r'\D', '', regex=True)
    dataset1['phone'] = dataset1['phone'].apply(
        lambda x: f"+7 (0{x[1:4]}) {x[4:7]}-{x[7:9]}-{x[9:11]}" if len(x) >= 11 else x
    )

    # Добавление ключей блокировки по дате рождения
    dataset1['blocking_key'] = dataset1.get('birthdate', None)

    # Очищаем и стандартизируем данные второго датасета
    dataset2.columns = dataset2.columns.str.strip().str.lower()
    dataset2['first_name'] = dataset2['first_name'].str.strip().str.lower()
    dataset2['middle_name'] = dataset2['middle_name'].str.strip().str.lower()
    dataset2['last_name'] = dataset2['last_name'].str.strip().str.lower()
    dataset2['address'] = dataset2['address'].str.strip().str.lower()
    dataset2['phone'] = dataset2['phone'].str.replace(r'\D', '', regex=True)
    dataset2['phone'] = dataset2['phone'].apply(
        lambda x: f"+7 (0{x[1:4]}) {x[4:7]}-{x[7:9]}-{x[9:11]}" if len(x) >= 11 else x
    )

    # Добавление ключей блокировки по дате рождения
    dataset2['blocking_key'] = dataset2.get('birthdate', None)

    # Очищаем и стандартизируем данные третьего датасета
    dataset3.columns = dataset3.columns.str.strip().str.lower()
    dataset3['name'] = dataset3['name'].str.strip().str.lower()
    dataset3['email'] = dataset3['email'].str.strip().str.lower()
    dataset3['birthdate'] = pd.to_datetime(dataset3['birthdate'], errors='coerce').dt.date
    dataset3['sex'] = dataset3['sex'].str.strip().str.lower()

    # Добавление ключей блокировки по дате рождения
    dataset3['blocking_key'] = dataset3.get('birthdate', None)

    print("Предобработка завершена.")
    return dataset1, dataset2, dataset3

def create_features(row1, row2, row3, index1, index2, index3):
    features = {}
    features['name_similarity'] = fuzz.token_sort_ratio(
        row1['full_name'],
        " ".join([row2['first_name'], row2['middle_name'], row2['last_name']])
    )
    features['email_similarity'] = fuzz.ratio(row1['email'], row3['email'])
    features['name_similarity_3'] = fuzz.ratio(row1['full_name'], row3['name'])
    features['email_similarity_3'] = fuzz.ratio(row1['email'], row3['email'])
    features['index1'] = index1
    features['index2'] = index2
    features['index3'] = index3
    return features

def compute_features(dataset1, dataset2, dataset3):
    print("Создание признаков...")
    features_list = Parallel(n_jobs=-1)(
        delayed(create_features)(row1, row2, row3, i, j, k)
        for i, row1 in dataset1.iterrows()
        for j, row2 in dataset2.iterrows()
        for k, row3 in dataset3.iterrows()
    )
    features_df = pd.DataFrame(features_list)
    print("Создание признаков завершено.")
    return features_df

def find_matches(features_df, similarity_threshold=85):
    print(f"Поиск совпадений с порогом {similarity_threshold}...")
    matches = features_df[
        (features_df['name_similarity'] >= similarity_threshold) |
        (features_df['email_similarity'] >= similarity_threshold) |
        (features_df['name_similarity_3'] >= similarity_threshold) |
        (features_df['email_similarity_3'] >= similarity_threshold)
    ]
    print(f"Найдено совпадений: {len(matches)}")
    return matches

def merge_records(row1, row2, row3):
    merged = {
        'full_name': row1['full_name'],
        'email': row1['email'] if row1['email'] else (row2['email'] if row2['email'] else row3['email']),
        'address': row1['address'] if row1['address'] else row2['address'],
        'phone': row1['phone'] if row1['phone'] else row2['phone'],
        'birthdate': row1['birthdate'] if row1['birthdate'] else (row2['birthdate'] if row2['birthdate'] else row3['birthdate']),
    }
    return merged

def process_matches(matches, dataset1, dataset2, dataset3):
    print("Объединение совпавших записей...")
    final_records = []
    for _, match in matches.iterrows():
        row1 = dataset1.iloc[int(match['index1'])]
        row2 = dataset2.iloc[int(match['index2'])]
        row3 = dataset3.iloc[int(match['index3'])]
        final_records.append(merge_records(row1, row2, row3))
    final_df = pd.DataFrame(final_records)
    print("Объединение завершено.")
    return final_df

def evaluate_model(features_df):
    print("Оценка модели...")
    # Генерация случайных истинных меток
    true_labels = np.random.randint(0, 2, size=len(features_df))
    features_df['true_labels'] = true_labels

    # Подготовка данных для оценки
    X = features_df[['name_similarity', 'email_similarity', 'name_similarity_3', 'email_similarity_3']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.5, min_samples=2)
    features_df['cluster'] = dbscan.fit_predict(X_scaled)

    # Фильтрация и объединение по кластерам
    final_records = []
    clusters = features_df[features_df['cluster'] != -1]

    for cluster_id in clusters['cluster'].unique():
        cluster_data = clusters[clusters['cluster'] == cluster_id]
        indices1 = cluster_data['index1'].unique()
        indices2 = cluster_data['index2'].unique()
        indices3 = cluster_data['index3'].unique()

        row1 = dataset1.iloc[indices1[0]] if len(indices1) > 0 else None
        row2 = dataset2.iloc[indices2[0]] if len(indices2) > 0 else None
        row3 = dataset3.iloc[indices3[0]] if len(indices3) > 0 else None

        final_record = merge_records(row1, row2, row3)
        final_records.append(final_record)

    final_df = pd.DataFrame(final_records).drop_duplicates().reset_index(drop=True)

    # Вычисление метрик
    y_true = features_df['true_labels']
    y_pred = features_df['cluster']
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    print("Оценка модели завершена.")
    return final_df

#def run_all():
download_and_extract()
dataset1, dataset2, dataset3 = load_datasets()
dataset1, dataset2, dataset3 = preprocess_datasets(dataset1, dataset2, dataset3)
features_df = compute_features(dataset1, dataset2, dataset3)
matches = find_matches(features_df)
final_df = process_matches(matches, dataset1, dataset2, dataset3)
evaluated_df = evaluate_model(features_df)

# Сохранение результатов
final_df.to_csv('final_records.csv', index=False)
evaluated_df.to_csv('evaluated_features.csv', index=False)
print("Результаты сохранены.")
    
