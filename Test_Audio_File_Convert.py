'''
A code summary with comments explaining its functionality at necessary points:

This script processes audio files in a specified directory and generates three types of visual representations for each audio file: spectrogram, mel-spectrogram, and MFCC. 
It uses the `librosa` library to load and analyze audio files, creating the corresponding images for each type of representation and saving them in designated directories. 
The code includes functions for creating each type of image and a main function to process all `.wav` files in a directory.
'''

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Generating and saving a standard spectrogram image from an audio file
def save_spectrogram_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)  # Loadded the audio file
    S = librosa.stft(y)  # Computting the Short-Time Fourier Transform
    S_db = librosa.amplitude_to_db(abs(S))  # Convertting amplitude to decibels

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')  # Displaying as spectrogram
    plt.colorbar(format='%+2.0f dB')  # Add color bar for decibel levels
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)  # Saving the spectrogram as an image file
    plt.close()

# Generatting and saving a mel-spectrogram image from an audio file
def save_mel_spectrogram_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)  # Load the audio file
    S = librosa.feature.melspectrogram(y=y, sr=sr)  # Computting mel-spectrogram
    S_db = librosa.power_to_db(S, ref=np.max)  # Convertting power to decibels

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')  # Displaying mel-spectrogram
    plt.colorbar(format='%+2.0f dB')  # Added color bar for decibel levels
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)  # Saving the mel-spectrogram as an image file
    plt.close()

# Generatting and saving an MFCC (Mel-Frequency Cepstral Coefficients) image from an audio file
def save_mfcc_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)  # Loadded the audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Compute 13 MFCCs

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')  # Displaying MFCC
    plt.colorbar()  # Added color bar
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(output_image_path)  # Saving the MFCC as an image file
    plt.close()

# Main function to process all .wav files in a directory and save the images
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
            audio_path = os.path.join(audio_dir, filename)  # Getting full path to audio file
            base_filename = os.path.splitext(filename)[0]  # Removing file extension for image naming

            print(f"Processing file: {audio_path}")
            # Generatting and save each type of image representation
            save_spectrogram_image(audio_path, os.path.join(spectrogram_dir, f"{base_filename}.png"))
            save_mel_spectrogram_image(audio_path, os.path.join(mel_spectrogram_dir, f"{base_filename}.png"))
            save_mfcc_image(audio_path, os.path.join(mfcc_dir, f"{base_filename}.png"))

    if not files_found:
        print("No .wav files found in the directory.")  # Informing if no .wav files were found

# Updatting these paths to point to your correct directories for test data
audio_dir = "./Mollie_audio_testWAV/"  # Input directory for test audio files
spectrogram_dir = "./testIMG/Spectrogram_images/"  # Directory to save spectrogram images
mel_spectrogram_dir = "./testIMG/Mel_Spectrogram_images/"  # Directory to save mel-spectrogram images
mfcc_dir = "./testIMG/MFCC_images/"  # Directory to save MFCC images

# Processing the test audio files and save the images
process_audio_to_images(audio_dir, spectrogram_dir, mel_spectrogram_dir, mfcc_dir)
