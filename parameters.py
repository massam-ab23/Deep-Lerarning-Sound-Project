# Create parameters.py

NUM_CLASSES = 10  # Example: number of output classes
BATCH_SIZE = 128  # Example: batch size
LEARNING_RATE = 0.0001  # Learning rate for the optimizer
NUM_EPOCHS = 20  # Number of epochs for training
START_EPOCH = 0  # Starting epoch, if you're resuming training
PRINT_INTERVAL = 5  # How often to print the training status
CHECKPOINT_LOCATION = '/content/drive/MyDrive/SoundClassification/checkpoints/'  # Where to save model checkpoints
CLASS_LIST = ['class1', 'class2', 'class3', ..., 'class10']  # Replace with your actual class names
DEVICE = 'xla'  # If using TPU

CLASS_LIST = ['Bark', 'Burping_or_eructation', 'Cough', 'Laughter', 'Unknown']
NUM_CLASSES = len(CLASS_LIST)


import os
os.chdir('/content/drive/MyDrive/SoundClassification')

img_dir = '/content/drive/MyDrive/SoundClassification/trainIMG/Spectrogram_images/'
#print(os.listdir(img_dir))
