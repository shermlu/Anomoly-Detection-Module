import json
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow.math import count_nonzero
from keras.models import model_from_yaml
from sklearn.metrics import confusion_matrix

def get_df(path):
    """
    Load data
    """
    with open(path) as jsonfile:
        labels = json.load(jsonfile)
    cpc = []
    cpm = []
    for file in labels:
        data = pd.read_csv(file, parse_dates=['timestamp'])
        data['anomaly'] = 0
        for anomaly in labels[file]:
            data.loc[data['timestamp'] == anomaly, 'anomaly'] = 1
        if 'cpc' in file:
            cpc.append(data)
        else:
            cpm.append(data)
    return cpc, cpm

def standardize(data_list):
    for data in data_list:
        mean = data['value'].mean()
        std = data['value'].std()
        data['train'] = (data['value']-mean)/std
        data['train'] = (data['train']-data['train'].min())/(data['train'].max()-data['train'].min())
    return data_list


def remove_anomaly(data_list):
    no_anomaly = []
    for data in data_list:
        no_anomaly.append(data[data['anomaly'] == 0])
    return no_anomaly


def read_model(path, x_train):
    """
    Create autoencoder model
    """
    with open(path, 'r') as file:
        yaml_model=file.read()

    model= keras.models.model_from_yaml(yaml_model)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")]
    )
    return model


def create_sequences(data_list):
    output = []
    for data in data_list:
        output = output + create_sequence(data)
    return np.stack(output)

def create_sequence(data):
    output = []
    values = data.train.to_frame()
    for i in range(len(values) - 24):
        output.append(values[i : (i + 24)])
    return output

    
def fit_model(model, x_train):
    model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")]
    )
    
def predict_vals(model, data):
    x_train_pred = model.predict(data)
    train_mse_loss = np.mean((x_train_pred - data)**2, axis=1)

def get_threshold(mse_loss):
    loss = [mse for mse in mse_loss.tolist()]
    loss.sort(reverse=True)

    return np.mean(loss[:int(len(loss)*0.01)])

def anom_preds(data, anomalies):
    data['anom_pred'] = 0
    for indices in np.where(anomalies):
        for i in indices:
            data.at[i, 'anom_pred'] = 1
    return data

def get_preds(model, data, counter):
    seq = np.stack(create_sequence(data))
    pred = model.predict(seq)
    mse = np.mean((pred - seq)**2, axis=1)
    mse = mse.reshape((-1))
    anomalies = mse > get_threshold(mse)

    fulldf = anom_preds(data, anomalies)
    print('Confusion matrix: {}'.format(counter))
    print(confusion_matrix(fulldf['anomaly'], fulldf['anom_pred']))
    
def main():
    cpc, cpm = get_df('labels.json')
    cpc = standardize(cpc)
    cpm = standardize(cpm)

    cpc_noanomaly = remove_anomaly(cpc)
    cpm_noanomaly = remove_anomaly(cpm)
    
    cpc_train = create_sequences(cpc)
    cpc_full = create_sequences(cpc_noanomaly)
    
    cpm_train = create_sequences(cpc)
    cpm_full = create_sequences(cpc_noanomaly)
    
    model_cpc = read_model('cpc.yaml', cpc_full)
    model_cpm = read_model('cpm.yaml', cpc_full)
    
    print('cpc results: ')
    for data in cpc:
        counter = 2
        get_preds(model_cpc, data, counter)
        counter += 1
       
    print('cpm results: ')
    for data in cpm:
        counter = 2
        get_preds(model_cpm, data, counter)
        counter += 1


if __name__ == '__main__':
    main()
    