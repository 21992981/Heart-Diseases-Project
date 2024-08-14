import tensorflow as tf
import logging
import time
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow.keras as keras
from tensorflow.keras import layers


if tf.config.list_physical_devices('GPU'):
  print("GPU is available")
  print("CUDA Available: ", tf.config.list_physical_devices('GPU'))
  
else:
  print("GPU is not available")


# Read timeseries data
cwd = os.getcwd()
timeseries_df = pd.read_csv(cwd + "/datasets/main.csv")
timeseries_df = timeseries_df.drop('Unnamed: 0', axis=1)
timeseries_df = timeseries_df.drop(columns=['datetime'])

# Read result data and merge with timeseries data
result_df = pd.read_csv(cwd + "/datasets/patient.csv")
result_df = result_df.drop('Unnamed: 0', axis=1)
filtered_df = result_df[(result_df['result'] == 'LA') | (result_df['result'] == 'LA DİLATASYONU')]
filtered_df.loc[:, 'result'] = filtered_df['result'].apply(lambda x: 0 if x == 'LA' else 1)
timeseries_df = pd.merge(timeseries_df, filtered_df, on='patient')

# Read patient info data
personal_df = pd.read_csv(cwd + "/datasets/patient_info.csv")
personal_df = personal_df.drop('Unnamed: 0', axis=1)
personal_df


# Fill null values
mean_weight = personal_df[personal_df.sex == 'K']['weight'].mean()
mean_height = personal_df[personal_df.sex == 'K']['height'].mean()
mean_sys_w = personal_df[personal_df.sex == 'K']['systolic_pressure'].mean()
mean_dia_W = personal_df[personal_df.sex == 'K']['diastolic_pressure'].mean()
mean_sys_m = personal_df[personal_df.sex == 'E']['systolic_pressure'].mean()
mean_dia_m = personal_df[personal_df.sex == 'E']['diastolic_pressure'].mean()

personal_df.loc[personal_df.id == 4, 'height'] = mean_height
personal_df.loc[personal_df.id == 30, 'weight'] = mean_weight
personal_df.loc[personal_df.id == 30, 'height'] = mean_height
personal_df.loc[personal_df.id == 33, 'systolic_pressure'] = mean_sys_w
personal_df.loc[personal_df.id == 33, 'diastolic_pressure'] = mean_dia_W
personal_df.loc[personal_df.id == 36, 'systolic_pressure'] = mean_sys_m
personal_df.loc[personal_df.id == 36, 'diastolic_pressure'] = mean_dia_m

personal_df['HT'] = personal_df['HT'].fillna(int(personal_df['HT'].mean())) 
personal_df['DM'] = personal_df['DM'].fillna(int(personal_df['DM'].mean())) 
personal_df['HL'] = personal_df['HL'].fillna(int(personal_df['HL'].mean())) 
personal_df['KAH'] = personal_df['KAH'].fillna(int(personal_df['KAH'].mean())) 
personal_df['KOAH'] = personal_df['KOAH'].fillna(int(personal_df['KOAH'].mean())) 
personal_df['KBY'] = personal_df['KBY'].fillna(int(personal_df['KBY'].mean())) 
personal_df['KC HST'] = personal_df['KC HST'].fillna(int(personal_df['KC HST'].mean()))

# Merge all data
df = pd.merge(timeseries_df, personal_df.drop(columns=['name_surname', 'date', 'NOT', 'complaint', 'identification_number']), left_on='patient', right_on='id')
df = df.drop('id', axis=1)

# Last preprocessing for model adaptation
df.result = df.result.astype('int')
df.sex = df.sex.astype('category')
df.sex = df.sex.cat.codes


# Standard normalization
features = ['ECG', 'IR', 'RED', 'rate_avg', 'rate_std',
       'rate_min', 'rate_max', 'HT', 'DM', 'HL', 'KAH',
       'KOAH', 'KBY', 'KC HST', 'height', 'weight', 'systolic_pressure',
       'diastolic_pressure']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

n_classes = len(np.unique(df.result))


# Transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# Lists for Report
y_true_list = []
y_pred_list = []
y_pred_bool_list = []
model_list = []


from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import classification_report


patients = df['patient'].unique()
np.random.shuffle(patients)
kf = KFold(n_splits=len(patients) // 8)

for train_index, test_index in kf.split(patients):
    train_patients = patients[train_index]
    test_patients = patients[test_index]

    train_data = df[df['patient'].isin(train_patients)]
    test_data = df[df['patient'].isin(test_patients)]

    train_X = train_data.drop('result', axis=1)
    train_y = train_data.result
    test_X = test_data.drop('result', axis=1)
    test_y = test_data.result
    train_X = train_X.values.reshape((train_X.values.shape[0], train_X.values.shape[1], 1))
    test_X = test_X.values.reshape((test_X.values.shape[0], test_X.values.shape[1], 1))
    n_classes = len(np.unique(train_y))


    input_shape = train_X.shape[1:]
    model = build_model(
        input_shape,
        head_size=512,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[512, 256, 256, 128],
        mlp_dropout=0.4,
        dropout=0.1,
    )
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(
        train_X,
        train_y,
        epochs=100,
        batch_size=2048
    )


    y_pred = model.predict(test_X, batch_size=4096, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    y_true_list.append(test_y)
    y_pred_list.append(y_pred)
    y_pred_bool_list.append(y_pred_bool)
    model_list.append(model)

for i in range(5):
    print("----------------------",i, "----------------------")
    print(classification_report(y_true_list[i], y_pred_bool_list[i]))



