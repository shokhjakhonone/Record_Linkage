import pandas as pd
import recordlinkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def preprocess_data(dataset):
    column_mapping = {
        'first_name': 'first_name',
        'last_name': 'last_name',
        'name': 'full_name',
        'email': 'email',
        'phone_number': 'phone_number',
        'contact': 'contact_info',
        'other_info': 'other_info'
    }
    dataset.rename(columns={col: column_mapping.get(col, col) for col in dataset.columns}, inplace=True)

    if 'first_name' in dataset and 'last_name' in dataset:
        dataset['full_name'] = dataset['first_name'] + ' ' + dataset['last_name']
    elif 'full_name' in dataset:
        dataset['full_name'] = dataset['full_name']
    
    if 'email' in dataset and 'phone_number' in dataset:
        dataset['contact_info'] = dataset['email'] + ' ' + dataset['phone_number']
    elif 'contact' in dataset:
        dataset['contact_info'] = dataset['contact']
    
    return dataset

def extract_features(dataset1, dataset2, candidate_links):
    compare = recordlinkage.Compare()
    compare.string('full_name', 'full_name', method='jarowinkler', threshold=0.8)
    compare.exact('contact_info', 'contact_info')
    compare.exact('other_info', 'other_info')
    
    features = compare.compute(candidate_links, dataset1, dataset2)
    return features

def auto_label_data(features, threshold=2):
    labels = (features.sum(axis=1) >= threshold).astype(int)
    return labels

def train_ml_models(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    print("RandomForest Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))
    
    return rf_model, svm_model

def merge_records_with_ml(record_pairs, dataset1, dataset2, model):
    merged_data = pd.DataFrame(columns=['full_name', 'contact_info', 'other_info'])
    
    for index_pair in record_pairs:
        dataset1_index, dataset2_index = index_pair[0], index_pair[1]
        record1 = dataset1.loc[dataset1_index]
        record2 = dataset2.loc[dataset2_index]
        
        merged_record = {
            'full_name': record1['full_name'] if not pd.isnull(record1['full_name']) else record2['full_name'],
            'contact_info': record1['contact_info'] if not pd.isnull(record1['contact_info']) else record2['contact_info'],
            'other_info': record1.get('other_info', '') if not pd.isnull(record1.get('other_info', '')) else record2.get('other_info', '')
        }

        merged_data = pd.concat([merged_data, pd.DataFrame([merged_record])], ignore_index=True)
    
    merged_data = merged_data.drop_duplicates()
    
    return merged_data

def create_central_repository(dataset1, dataset2):
    dataset1 = preprocess_data(dataset1)
    dataset2 = preprocess_data(dataset2)
    
    indexer = recordlinkage.Index()
    indexer.full()
    candidate_links = indexer.index(dataset1, dataset2)

    features = extract_features(dataset1, dataset2, candidate_links)
    
    labels = auto_label_data(features)
    
    rf_model, svm_model = train_ml_models(features, labels)
    
    merged_data = merge_records_with_ml(candidate_links, dataset1, dataset2, rf_model)
    
    return merged_data
