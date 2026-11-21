from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent # current directory
ROOT_DIR = CURRENT_DIR.parent # one level up from the current directory

IMAGES_PATH = CURRENT_DIR / "data" / "images" # the "images" folder contains subfolders "images_001", _002, _003, etc.
DATA_CSV_PATH = CURRENT_DIR / "data" / "Data_Entry_2017.csv"

# old, was used for testing
# CLASSES_LIST = ['No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule', 'Effusion|Infiltration']

# All individual classes, no combinations
ALL_IND_CLASSES_LIST = ['No Finding', 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis' 'Effusion', 'Pneumonia', 'Pleural_thickening', 'Cardiomegaly', 'Nodule', 'Mass' 'Hernia']

# Subset of the possible individual classes. For experimenting purposes
CLASSES_LIST = ['No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule']



FEATURES_TO_USE = ['Image Index', 'Finding Labels', 'View Position']

IMAGE_SIZE = (1024, 1024)

SEED = 42 # input for any random_state parameters. For consistency purposes 
MODEL_BATCH_SIZE = 128
MODEL_EPOCHS = 1