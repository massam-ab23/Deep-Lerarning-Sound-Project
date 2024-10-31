'''
A code summary with comments explaining its functionality at necessary points:

This `parameters.py` file sets key hyperparameters and configuration values for a sound classification model. 
It defines parameters such as the number of classes, batch size, learning rate, number of epochs, and device type. 
It also specifies the directory paths for saving model checkpoints and accessing images, ensuring these values can be easily imported and referenced across various scripts. 
Additionally, it defines `CLASS_LIST`, which provides class labels used in classification tasks.
'''

# Define the number of output classes for the model based on available classes
NUM_CLASSES = 10  # Initial example value

# Setting batch size for training and validation
BATCH_SIZE = 128  # Example batch size

# Setting learning rate for the optimizer
LEARNING_RATE = 0.0001  # Learning rate

# Define the number of training epochs
NUM_EPOCHS = 20  # Number of epochs

# Starting epoch, useful if resuming from a checkpoint
START_EPOCH = 0  # Starting epoch

# How frequently to print training status updates
PRINT_INTERVAL = 5  # Interval to print updates

# Define the path for saving model checkpoints
CHECKPOINT_LOCATION = '/content/drive/MyDrive/SoundClassification/checkpoints/'

# Class list defining the categories for classification; update with actual class names
CLASS_LIST = ['class1', 'class2', 'class3', ..., 'class10']  # Placeholder classes

# Specific list of actual class names used for classification
CLASS_LIST = ['Bark', 'Burping_or_eructation', 'Cough', 'Laughter', 'Unknown']  # Actual class labels
NUM_CLASSES = len(CLASS_LIST)  # Dynamically set NUM_CLASSES based on class list length

# Specifying the device type, useful for TPU usage
DEVICE = 'xla'  # For TPU; use 'cuda' for GPU or 'cpu' if neither available

# Changing to the working directory containing sound classification files in Google Drive
import os
os.chdir('/content/drive/MyDrive/SoundClassification')

# Path to the directory where spectrogram images are stored
img_dir = '/content/drive/MyDrive/SoundClassification/trainIMG/Spectrogram_images/'

# Uncomment below line to list files in the specified image directory
# print(os.listdir(img_dir))
