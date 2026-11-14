from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent # current directory
ROOT_DIR = CURRENT_DIR.parent # one level up from the current directory

DATA_CSV_PATH = CURRENT_DIR / "data" / "Data_Entry_2017.csv"

NUM_CLASSES = 5 # number of classes and the number of neurons in the output layer 
CLASSES_LIST = ['No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule', 'Effusion|Infiltration']
FEATURES_TO_USE = ['Image Index', 'Finding Labels', 'View Position']