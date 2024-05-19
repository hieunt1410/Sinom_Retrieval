IMG_PATH = "../input/animals-data/dataset/"
IMG_HEIGHT = 512
IMG_WIDTH = 512

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 30
FULL_BATCH_SIZE = 256

###### Train and Test time #########

DATA_PATH = "../../"
AUTOENCODER_MODEL_PATH = "baseline_autoencoder.pt"
ENCODER_MODEL_PATH = "../pretrained/deep_encoder.pt"
DECODER_MODEL_PATH = "../pretrained/deep_decoder.pt"
EMBEDDING_PATH = "../pretrained/data_embedding_f.npy"
EMBEDDING_SHAPE = (1, 256, 16, 16)
# TEST_RATIO = 0.2

###### Test time #########
NUM_IMAGES = 5
TEST_IMAGE_PATH = "../../queries/5.png"