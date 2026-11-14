from constants import *

import numpy as np
import re
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense, AveragePooling2D
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import pandas as pd

TF_ENABLE_ONEDNN_OPTS=0

def load_data():
    """
    Using Pandas dataframe, load in the CSV data from https://www.kaggle.com/datasets/nih-chest-xrays/data?select=Data_Entry_2017.csv
    """
    df = pd.read_csv(DATA_CSV_PATH)
    # There is an error in the reading where a column "Unnamed: 11" appears. Needs to be removed because all values are null
    df = df.drop(columns='Unnamed: 11')

    return df

def df_analysis(df):
    """
    Basic overview of a dataframe
    """
    # overview
    print(list(df.columns))
    print(df.info())
    print(f'\nNull counts:\n{df.isnull().sum()}')

    num_cols = df.select_dtypes(include='number')
    if not num_cols.empty:
        print(f'\n{df.describe(include='number')}')

    obj_cols = df.select_dtypes(include='object')
    if not obj_cols.empty:
        print(f'\n{df.describe(include='object')}')

    # get all unique disease combinations and their frequency
    with pd.option_context('display.max_rows', None):
        print(f"Number of unique combinations: {df['Finding Labels'].nunique()}")
        print(df['Finding Labels'].value_counts().head(10).to_dict())
        # The top ten most common labels (including combinations): 'No Finding': 60361, 'Infiltration': 9547, 'Atelectasis': 4215, 'Effusion': 3955, 'Nodule': 2705, 'Pneumothorax': 2194, 'Mass': 2139, 'Effusion|Infiltration': 1603, 'Atelectasis|Infiltration': 1350, 'Consolidation': 1310

def filter_classes(df):
    """
    There are many possible label combinations. Remove rows which have a label that is not exactly listed in CLASSES_LIST
    """
    to_keep = CLASSES_LIST

    # Extract the matched substring, or None if not found
    df = df[df['Finding Labels'].isin(to_keep)]
    
    return df 


def clean_dataframe(df):
    # drop duplicates, drop nulls
    num_r1 = df.shape[0]
    df = df.drop_duplicates() 
    df = df.dropna()
    num_r2 =  df.shape[0]
    print(f'{num_r1 - num_r2} duplicate and na rows removed')


    features_to_use = FEATURES_TO_USE
    df = df[features_to_use]

    return df


# def make_model():
#     # making the model
#     model = Sequential()
#     model.add(Conv2D(16, kernel_size=(2, 2), input_shape=(28,28,1,), activation="sigmoid"))
#     model.add(MaxPooling2D(pool_size=(1, 1)))
#     model.add(Conv2D(8, kernel_size=(2, 2), input_shape=(28,28,1,), activation="sigmoid"))
#     model.add(MaxPooling2D(pool_size=(1, 1)))

#     model.add(Flatten())

#     # adding dropout
#     model.add(Dropout(0.05))
#     model.add(Dense(3000,  activation='relu'))
#     model.add(Dense(NUM_CLASSES,  activation='softmax')) # total_num_classes, activation

#     # visualize the model design
#     model.summary()

#     return model

# def get_train_test_split(df):
#     pass
#     # getting training and testing data
   
#     # (x_train, y_train), (x_test, y_test) 

#     # x_train = np.expand_dims(x_train, -1)
#     # x_test = np.expand_dims(x_test, -1)

#     # print (x_train.shape)
#     # print (y_train.shape)

#     # # make a list of lists
#     # y_train = [[y] for y in y_train]

#     # encoder = OneHotEncoder()
#     # encoder.fit(y_train)
#     # y_train = encoder.transform(y_train).toarray()

# def train_model():
#     # training the model
#     optim = Adam(learning_rate=0.001) #learning rate is the "eta" in our equation
#     loss_fn = CategoricalCrossentropy()

#     model.compile(loss=loss_fn,optimizer=optim,metrics=["accuracy"])

#     batch_size = 128
#     epochs = 15

#     model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# def predict_labels():
#     # predictions
#     predictions = model.predict(x_test)
#     print (predictions)
#     predicted_class_indices = predictions.argmax(axis=1)
#     print (predicted_class_indices)
#     acc = accuracy_score(y_test,predicted_class_indices)
#     print (f"Testing accuracy = {acc}")