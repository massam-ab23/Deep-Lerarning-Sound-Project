'''
A code summary with comments explaining its functionality at necessary points:

This `parameters.py` file sets key hyperparameters and configuration values for a sound classification model. 
It defines parameters such as the number of classes, batch size, learning rate, number of epochs, and device type. 
It also specifies the directory paths for saving model checkpoints and accessing images, ensuring these values can be easily imported and referenced across various scripts. 
Additionally, it defines `CLASS_LIST`, which provides class labels used in classification tasks.
'''

# Training configuration

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
START_EPOCH = 0
PRINT_INTERVAL = 5

# Model checkpoints
CHECKPOINT_LOCATION = './checkpoints/'

# Sound classes
CLASS_LIST = [
    'Bark',
    'Burping_or_eructation',
    'Cough',
    'Laughter',
    'Unknown'
]

NUM_CLASSES = len(CLASS_LIST)

# Device configuration
DEVICE = 'xla'  # Use 'cuda' for GPU or 'cpu' for CPU

# Spectrogram image directory
img_dir = './trainIMG/Spectrogram_images/'
