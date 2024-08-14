
NUM_ROUNDS=5

FRACTION_FIT=1  # Sample 10% of available clients for training
FRACTION_EVALUATE=1  # Sample 5% of available clients for evaluation
MIN_FIT_CLIENTS=2 # Never sample less than 10 clients for training
MIN_EVALUATE_CLIENTS=2# Never sample less than 5 clients for evaluation
MIN_AVAILABLE_CLIENTS=2

ENCODER_LAYERS=[14,14,9]
DECODER_LAYERS=[14,14]
VAE=False



EPOCHS_CLIENT=3
EPOCHS_SERVEUR=5


BATCH_SIZE=64
LEARNING_RATE=0.0001

RATIO_LABEL=0.3


MULTICLASS=True

NUM_CLASSES=5
  
TRAINING_DATA="train.csv"
TESTING_DATA="test.csv"


MULTICLASS_TARGET_COL="category"
BINARY_TARGET="attack"

ONE_HOT_ENCODING_LIST=['proto']
DELETE_LIST=["subcategory","saddr" ,"daddr"]
