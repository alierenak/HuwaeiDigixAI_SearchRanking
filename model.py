#!/usr/bin/env python

# Author: Ali Eren Ak // akali@sabanciuniv.edu
# Description of this script
# Inputs: Huawei Digix AI Challenge Search Ranking Data
# Outputs: Keras Classification model

import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import class_weight
import argparse


parser = argparse.ArgumentParser(description='Training script for Huawei Digix AI Challenge Search Ranking Data')
parser.add_argument('--data', help='Huawei Digix AI Challenge Search Ranking Data')
parser.add_argument('--test', help='Path to test data')
parser.add_argument('--csv', help='Path to submission file')
args = parser.parse_args()


def create_submission(model, scaler):
    test_data = pd.read_csv(args.test, sep='\t', header=None)

    ids = test_data.iloc[:, 0:2]
    ids[0] = ids[0].astype(int)
    ids[0] = ids[0].astype(str)
    ids[1] = ids[1].astype(int)
    ids[1] = ids[1].astype(str)

    test_data = np.array(test_data)

    test_data = scaler.transform(test_data)
    test_data = model.predict(test_data, batch_size=10)

    pred_lables = np.argmax(test_data, axis=1)
    preds = pd.DataFrame(pred_lables)

    preds[0] = preds[0].astype(int)
    preds[0] = preds[0].astype(str)

    result = pd.concat([ids, preds], axis=1, join="outer", ignore_index=True)
    result.to_csv(args.csv, index=False)


def splitData(data):

    encoder = LabelEncoder()    # Initialize encoder to encode non-integer ids.
    print("Ids are encoding...")
    data.iloc[:, 1] = encoder.fit_transform(data.iloc[:, 1])
    data.iloc[:, 2] = encoder.fit_transform(data.iloc[:, 2])
    print("Encoded!")

    train_df, val_df = train_test_split(data, test_size=0.2, random_state=27, shuffle=True)

    train_labels = train_df[0]
    labels = train_labels                                   # Before change labels to one-hot encoding take a copy for class_weight
    train_df = train_df.drop(train_df.columns[0], axis=1)   # Drop labels

    train_labels = pd.get_dummies(train_labels)
    train_labels = train_labels.values
    train_df = train_df.values

    val_labels = val_df[0]
    val_df = val_df.drop(val_df.columns[0], axis=1) # Drop labels
    val_labels = pd.get_dummies(val_labels)
    val_df = val_df.values
    val_labels = val_labels.values

    return train_df, train_labels, val_df, val_labels, labels


def pre_process(train_df, val_df, test_df, labels):

    scaler = StandardScaler()   # Data is not scaled, I only try Standard Normalization due to time limit,
                                # but you could try MinMax also...

    # Scaling all parts of data...
    print("Scaling the data...")
    scaler.fit_transform(train_df)
    scaler.transform(val_df)
    scaler.transform(test_df)
    print("Data is scaled!")

    print("Calculating weights for the labels...")
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    class_weights = dict(enumerate(class_weights))
    print("Weights are calculated!")

    return class_weights, train_df, val_df, test_df, scaler


def make_model(INPUT_DIM): # X is for get shape

    # Softmax and Relu is preffered, simple size layer added.
    model = Sequential()
    model.add(Dense(15, input_dim=INPUT_DIM, activation='relu'))  # Rectified Linear Unit Activation Function
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Softmax for multi-class classification


    # Compile model, AUC is not proper metrics however I used becasue AUC is commonly used for classifation
    # Huawei using ERR Metrics to evaluate
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics= [
            tf.keras.metrics.AUC()
        ],
    )
    return model


def train_model(train_df, val_df, train_labels, val_labels, class_weights):

    model = make_model(train_df.shape[1])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        train_df,
        train_labels,
        batch_size=1024,
        epochs=100,
        callbacks=[early_stopping],
        validation_data=(val_df, val_labels),
        class_weight=class_weights
    )

    return model


def pipeline():

    print("Data is loading...")
    data = pd.read_csv(args.input, sep="\t", header=None)
    print("Data is loaded!")

    # Test Result could be obtain by Huawei System
    train_df, train_labels, val_df, val_labels, labels = splitData(data)

    # Pre_process is needed. Standardization, Encoding, Class weighting is used!
    class_weights, train_df, val_df, test_df, scaler = pre_process(train_df, val_df, labels)

    # Train Keras Classification model
    model = train_model(train_df, val_df, train_labels, val_labels, class_weights)

    # Creating submission file in format that Huawei wants
    create_submission(model, scaler)


pipeline() # Do not forget to trigger code :)
