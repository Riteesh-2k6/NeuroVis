NUM_IMAGES_3D = 64
TRAINING_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
IMAGE_SIZE = 256
N_EPOCHS = 15
do_valid = True
n_workers = 0  # Fixed for Windows

# Hyperparameters for training new models
LR = 1e-4
WEIGHT_DECAY = 1e-6
OPTIM_THRESHOLD = True

# Inference Settings for Existing Weights
MODEL_NAME = "resnet10" 
Z_SCORE_NORM = False # 0-1 scaling matches original weights
TTA_STEPS = 4        # Test Time Augmentation iterations