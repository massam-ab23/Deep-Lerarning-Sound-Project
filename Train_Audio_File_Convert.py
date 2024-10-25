
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np  # Import numpy to fix the np error

# Update the correct path to the audio directory in Google Drive
audio_dir = "/content/drive/MyDrive/SoundClassification/Mollie_audio_trainWAV/"

def save_spectrogram_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def save_mel_spectrogram_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def save_mfcc_image(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def process_audio_to_images(audio_dir, spectrogram_dir, mel_spectrogram_dir, mfcc_dir):
    if not os.path.exists(spectrogram_dir):
        os.makedirs(spectrogram_dir)
    if not os.path.exists(mel_spectrogram_dir):
        os.makedirs(mel_spectrogram_dir)
    if not os.path.exists(mfcc_dir):
        os.makedirs(mfcc_dir)

    # Check if there are files to process
    files_found = False
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            files_found = True
            audio_path = os.path.join(audio_dir, filename)
            base_filename = os.path.splitext(filename)[0]

            print(f"Processing file: {audio_path}")
            save_spectrogram_image(audio_path, os.path.join(spectrogram_dir, f"{base_filename}.png"))
            save_mel_spectrogram_image(audio_path, os.path.join(mel_spectrogram_dir, f"{base_filename}.png"))
            save_mfcc_image(audio_path, os.path.join(mfcc_dir, f"{base_filename}.png"))

    if not files_found:
        print("No .wav files found in the directory.")

# Update these paths to point to your correct directories
audio_dir = "/content/drive/MyDrive/SoundClassification/Mollie_audio_trainWAV/"  # Correct directory for audio files
spectrogram_dir = "/content/drive/MyDrive/SoundClassification/trainIMG/Spectrogram_images/"  # Directory to save spectrogram images
mel_spectrogram_dir = "/content/drive/MyDrive/SoundClassification/trainIMG/Mel_Spectrogram_images/"  # Directory to save mel-spectrogram images
mfcc_dir = "/content/drive/MyDrive/SoundClassification/trainIMG/MFCC_images/"  # Directory to save MFCC images

# Process the audio files and save the images
process_audio_to_images(audio_dir, spectrogram_dir, mel_spectrogram_dir, mfcc_dir)
