import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple
import numpy as np
import constantes

def delete_columns(df, columns_to_delete):
    if columns_to_delete:
        df = df.drop(columns=columns_to_delete)
    return df
def check_columns_after_drop(data, columns_to_drop):
    # Check if any of the specified columns still exist in the DataFrame
    columns_exist = any(col in data.columns for col in columns_to_drop)
    
    # If any of the columns still exist, return False, otherwise return True
    return not columns_exist


def data_preprocessing(multiclass=constantes.MULTICLASS,multiclass_target=constantes.MULTICLASS_TARGET_COL,binary_target=constantes.BINARY_TARGET):
        # Load the UNSW-NB15 training and testing data from CSV
    train_data = pd.read_csv(constantes.TRAINING_DATA)  
    test_data = pd.read_csv(constantes.TESTING_DATA) 
    
    if(multiclass):
        train_labels = train_data[multiclass_target]
        test_labels = test_data[multiclass_target]
    else:
        train_labels = train_data[binary_target]
        test_labels = test_data[binary_target]
      
     
    
    train_data = train_data.drop(columns=[multiclass_target,binary_target])
    test_data = test_data.drop(columns=[multiclass_target,binary_target])   
    
    # Combine the training and testing datasets for preprocessing
    combined_data = pd.concat([train_data, test_data], axis=0)
    combined_data=combined_data.drop(columns=constantes.DELETE_LIST)
    #combined_data ['saddr'] = combined_data  ['saddr'].apply(ip_to_int)
    #combined_data ['daddr'] = combined_data  ['daddr'].apply(ip_to_int)
    combined_data['sport'] =  combined_data['sport'].apply(lambda x: int(x, 16) if 'x' in x else int(x))
    combined_data['dport'] =  combined_data['dport'].apply(lambda x: int(x, 16) if 'x' in x else int(x))
    # Identify numerical columns
    numerical_columns = combined_data.select_dtypes(include=[np.number]).columns

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the numerical columns (excluding "label")
    combined_data[numerical_columns] = scaler.fit_transform(
        combined_data[numerical_columns]
    )

    # Perform one-hot encoding for categorical columns
    categorical_columns = ['proto']
    combined_data = pd.get_dummies(
        combined_data, columns=categorical_columns, dtype=int
    )

    # Split the combined dataset back into training and testing datasets
    train_data = combined_data[: len(train_data)]
    test_data = combined_data[len(train_data) :]


    # Convert the data to float64
    train_data = train_data.astype(np.float64)
    test_data = test_data.astype(np.float64)
    
    print("train:",train_data.shape)
    print("test:",test_data.shape)
    print("trainlabels:",train_labels.shape)
    print("testlabels:",test_labels.shape)
    return train_data,test_data ,train_labels, test_labels,combined_data


def one_hot_encode_column(column):
    # Check if the column contains numeric values

    column = column.astype(str)

    # One-hot encode
    one_hot_encoded = pd.get_dummies(column, prefix='encoded')

    return one_hot_encoded
def one_hot_encode_columns(df, column_names):
    for column_name in column_names:
        # Check if the column contains numeric values
        if pd.api.types.is_numeric_dtype(df[column_name]):
            # Convert numeric values to strings
            df[column_name] = df[column_name].astype(str)
        
        # One-hot encode
        one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
        
        # Concatenate one-hot encoded columns with the original DataFrame
        df = pd.concat([df, one_hot_encoded], axis=1)
        
        # Drop the original column
        df.drop(column_name, axis=1, inplace=True)
    return df

def split_balanced_datasets_clients(train_data, labels,multiclass=constantes.MULTICLASS,nbrClients=10):
    
    balancedDatasets = []  # Create a list to store balanced datasets
    FakeLabels = []  # Create a list to store fake labels
    if(multiclass):
        target=constantes.MULTICLASS_TARGET_COL
    else:
        target=constantes.BINARY_TARGET_COL
    train_data[target] = labels
    unique_classes = sorted(train_data[target].unique())
    class_partition_sizes = {cls: len(train_data[train_data[target] == cls]) // nbrClients for cls in unique_classes}
    
    for i in range(nbrClients):
        client_data = pd.DataFrame()
        fake_labels = []
        for cls in unique_classes:
            partition_size = class_partition_sizes[cls]
            start_idx = i * partition_size
            end_idx = start_idx + partition_size
            client_data_cls = train_data[train_data[target] == cls].iloc[start_idx:end_idx]
            client_data = pd.concat([client_data, client_data_cls], axis=0)
            fake_labels.extend([cls] * len(client_data_cls))
        balancedDatasets.append(client_data.drop(columns=[target]))  # Remove the 'normality' column here
        FakeLabels.append(pd.Series(fake_labels, name=target))
    train_data.drop(columns=[target], inplace=True)  # Remove the 'normality' column from the original DataFrame
    return balancedDatasets, FakeLabels

def get_client_dataset(client_id: int,datasets:list,fake_labels:list, test,iid=True):
    client_data = datasets[client_id]
    client_fake_labels = fake_labels[client_id]
    return client_data, client_fake_labels, test

from sklearn.model_selection import train_test_split

def split_train_server_clients(ratioLabel=constantes.RATIO_LABEL):
    train_data, test_data, train_labels, test_labels ,combined_data= data_preprocessing()
    # Split the data into training and sampled training sets
    train_data_sampled, client_data, train_labels_sampled, client_labels = train_test_split(train_data, train_labels, train_size=ratioLabel, stratify=train_labels, random_state=42)
    train_labels_sampled = one_hot_encode_column(train_labels_sampled)
    combined_labels= pd.concat([train_labels, test_labels], axis=0)
    print("Sampled train data shape:", train_data_sampled.shape)
    print("Sampled train labels shape:", train_labels_sampled.shape)
    print("Client data shape:", client_data.shape)
    print("Client labels shape:", client_labels.shape)

    return train_data_sampled, train_labels_sampled, test_data, test_labels, client_data, client_labels,combined_data,combined_labels


