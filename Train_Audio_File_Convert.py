'''
A code summary with comments explaining its functionality at necessary points:

This script processes audio files from a specified directory and generates spectrogram, mel-spectrogram, and MFCC visual representations for each `.wav` file. 
The `librosa` library is used for loading and transforming audio data, while `matplotlib` is used for plotting and saving the visualizations. 
Each type of spectrogram is saved as an image in designated directories, and the script ensures the directories are created if they do not already exist.
'''

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for handling numerical operations

# Updatting the correct path to the audio directory in Google Drive
audio_dir = "./train_audio_wav/"

# Generatting and saving a spectrogram image from an audio file
def save_spectrogram_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)  # Loadded the audio file
    S = librosa.stft(y)  # Computting the Short-Time Fourier Transform
    S_db = librosa.amplitude_to_db(abs(S))  # Convertting amplitude to decibels

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')  # Displaying as a spectrogram
    plt.colorbar(format='%+2.0f dB')  # Added color bar for decibel values
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)  # Saving the spectrogram as an image file
    plt.close()

# Generatting and saving a mel-spectrogram image from an audio file
def save_mel_spectrogram_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)  # Loadded the audio file
    S = librosa.feature.melspectrogram(y=y, sr=sr)  # Computting mel-spectrogram
    S_db = librosa.power_to_db(S, ref=np.max)  # Convertting power to decibels

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')  # Displaying as mel-spectrogram
    plt.colorbar(format='%+2.0f dB')  # Added color bar for decibel values
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)  # Saving the mel-spectrogram as an image file
    plt.close()

# Generatting and saving an MFCC (Mel-Frequency Cepstral Coefficients) image from an audio file
def save_mfcc_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)  # Loadded the audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Computting the first 13 MFCCs

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')  # Displaying MFCCs
    plt.colorbar()  # Added color bar
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(output_image_path)  # Saving the MFCC as an image file
    plt.close()

# Main function to process all .wav files in a directory and save spectrograms, mel-spectrograms, and MFCC images
def process_audio_to_images(audio_dir, spectrogram_dir, mel_spectrogram_dir, mfcc_dir):
    # Create directories if they don't exist
    if not os.path.exists(spectrogram_dir):
        os.makedirs(spectrogram_dir)
    if not os.path.exists(mel_spectrogram_dir):
        os.makedirs(mel_spectrogram_dir)
    if not os.path.exists(mfcc_dir):
        os.makedirs(mfcc_dir)

    # Checking if there are files to process
    files_found = False
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):  # Processing only .wav files
            files_found = True
            audio_path = os.path.join(audio_dir, filename)  # Getting the full path to the audio file
            base_filename = os.path.splitext(filename)[0]  # Removing the file extension for image naming

            print(f"Processing file: {audio_path}")
            # Generate and save each type of image representation
            save_spectrogram_image(audio_path, os.path.join(spectrogram_dir, f"{base_filename}.png"))
            save_mel_spectrogram_image(audio_path, os.path.join(mel_spectrogram_dir, f"{base_filename}.png"))
            save_mfcc_image(audio_path, os.path.join(mfcc_dir, f"{base_filename}.png"))

    if not files_found:
        print("No .wav files found in the directory.")  # Notify if no .wav files were found

# Updatting these paths to point to your correct directories
audio_dir = "./train_audio_wav/"  # Directory for audio files
spectrogram_dir = "./trainIMG/Spectrogram_images/"  # Directory for spectrogram images
mel_spectrogram_dir = "./trainIMG/Mel_Spectrogram_images/"  # Directory for mel-spectrogram images
mfcc_dir = "./trainIMG/MFCC_images/"  # Directory for MFCC images

# Processing the audio files and save the images
process_audio_to_images(audio_dir, spectrogram_dir, mel_spectrogram_dir, mfcc_dir)
