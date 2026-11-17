import constants as c

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd

TF_ENABLE_ONEDNN_OPTS=0

def load_data():
    """
    Using Pandas DataFrame, load in the CSV data from https://www.kaggle.com/datasets/nih-chest-xrays/data?select=Data_Entry_2017.csv.

    Returns:
        df : The loaded Pandas DataFrame.
    """
    df = pd.read_csv(c.DATA_CSV_PATH)
    # There is an error in the reading where a column "Unnamed: 11" appears. Needs to be removed because all values are null
    df = df.drop(columns='Unnamed: 11')

    return df

def df_analysis(df: pd.DataFrame):
    """
    Presents a basic overview of a dataframe.

    Args:
        df : Dataframe to be summarized.
    """
    # overview
    print(f'\n{list(df.columns)}')
    print(f'\n{df.info()}')
    print(f'\nNull counts:\n{df.isnull().sum()}')

    num_cols = df.select_dtypes(include='number')
    if not num_cols.empty:
        print(f'\n{df.describe(include='number')}')

    obj_cols = df.select_dtypes(include='object')
    if not obj_cols.empty:
        print(f'\n{df.describe(include='object')}')

    # get all unique disease combinations and their frequency
    with pd.option_context('display.max_rows', None):
        print(f"\nNumber of unique combinations: {df['Finding Labels'].nunique()}\n")
        print(df['Finding Labels'].value_counts().head(10).to_dict())
        # The top ten most common labels (including combinations): 'No Finding': 60361, 'Infiltration': 9547, 'Atelectasis': 4215, 'Effusion': 3955, 'Nodule': 2705, 'Pneumothorax': 2194, 'Mass': 2139, 'Effusion|Infiltration': 1603, 'Atelectasis|Infiltration': 1350, 'Consolidation': 1310

def filter_classes(df: pd.DataFrame):
    """
    There are many possible label combinations. Remove rows which have a label that is not exactly listed in CLASSES_LIST
    
    Args:
        df : The DataFrame to filter.

    Returns:
        df : The filtered DataFrame.
    """
    to_keep = c.CLASSES_LIST

    # Extract the matched substring, or None if not found
    df = df[df['Finding Labels'].isin(to_keep)]
    
    return df 


def clean_dataframe(df: pd.DataFrame):
    """
    Clean the DataFrame of duplicate entries and null values

    Args:
        df : The DataFrame to clean.

    Returns:
        df : The cleaned DataFrame.
    """
    # drop duplicates, drop nulls
    num_r1 = df.shape[0]
    df = df.drop_duplicates() 
    df = df.dropna()
    num_r2 =  df.shape[0]
    print(f'\n{num_r1 - num_r2} duplicate and na rows removed')


    features_to_use = c.FEATURES_TO_USE
    df = df[features_to_use]

    return df


def get_train_test_split(df: pd.DataFrame, train_count: int, image_size: tuple[int, int]=c.IMAGE_SIZE):
    """
    Obtain the train and test data for training and evaluating the model.

    Args:
        df : DataFrame from which to pull the data
        train_count : Number of examples from each relevant class to pull training data from.
        image_size : Size of the image in the final x train and test data. Default is 1024x1024, the image size from our dataset.

    Returns:
        x_train : Training features.
        x_test : Testing features.
        y_train : Correct labels for the training features.
        y_test : Correct labels for the test features.
    """
    # getting training and testing data
    train_df = pd.DataFrame()

    # Take train_count number of examples from each class for training
    for cl in c.CLASSES_LIST:
        train_df = pd.concat([train_df, df.loc[df['Finding Labels'] == cl].head(train_count)], axis=0)


    images = []
    for x in train_df['Image Index']:
        image = cv2.imread('RAE---NIH-Chest-X-rays-Classifier-/data/images/' + x, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        images.append(image)

    x_train, x_test, y_train, y_test = train_test_split(images, train_df['Finding Labels'], test_size=0.10, random_state=42)

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # make a list of lists
    y_train = np.array([[y] for y in y_train])

    print(f'\nx_train shape: {x_train.shape}\n')
    print(f'\ny_train shape: {y_train.shape}\n')

    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train)

    return x_train, x_test, y_train, y_test