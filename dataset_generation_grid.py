import os
import numpy as np
import cv2
import torchaudio
import librosa
from tqdm import tqdm
import pyroomacoustics as pra
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configuration
dataset_path = ".../datasets/GRID_Interm"
output_path = "../output_folder/noisy_GRID"


# Function to add reverberation and noise to the audio
def add_reverberation_and_noise(signal, sr=16000, snr_db_mean=30, snr_db_std=1.5, sir_db_min=5, sir_db_max=25, room_dim=[8, 6, 3], absorption=0.5, mic_loc=[2, 3, 2], source_loc=[4, 3, 1.5]):
    room = pra.ShoeBox(room_dim, absorption=absorption, fs=sr, max_order=15)
    room.add_source(source_loc, signal=signal)
    room.add_microphone_array(pra.MicrophoneArray(np.array([mic_loc]).T, room.fs))
    room.simulate()

    mic_signal = room.mic_array.signals[0, :]
    snr_db = np.random.normal(snr_db_mean, snr_db_std)
    sir_db = np.random.uniform(sir_db_min, sir_db_max)
    signal_power = np.mean(mic_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    interference_power = signal_power / (10 ** (sir_db / 10))

    noise = np.sqrt(noise_power) * np.random.randn(len(mic_signal))
    interference = np.sqrt(interference_power) * np.random.randn(len(mic_signal))
    noisy_signal = mic_signal + noise + interference
    return noisy_signal


# Function to add salt-and-pepper noise to images
def add_salt_pepper_noise(image, salt_pepper_ratio=0.02, amount_mean=0.01, amount_std=0.008):
    noisy_image = np.copy(image)
    amount = np.clip(np.random.normal(amount_mean, amount_std), 0, 1)
    num_salt = np.ceil(amount * image.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_pepper_ratio))

    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 1
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    return noisy_image


# Function to apply mild blur to the images
def apply_mild_blur(image, ksize_mean=13, ksize_std=3):
    ksize = int(abs(np.random.normal(ksize_mean, ksize_std)))
    ksize = max(1, ksize)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image


# Function to apply color shift to images
def apply_color_shift(image, intensity_mean=12, intensity_std=3):
    shifted_image = image.copy()
    for channel in range(image.shape[2]):
        intensity = int(np.random.normal(intensity_mean, intensity_std))
        shifted_image[:, :, channel] = np.clip(shifted_image[:, :, channel] + intensity, 0, 255)
    return shifted_image


# Function to preprocess and save the noisy audio and visual data
def process_file(audio_path, image_path, speaker_label):
    if os.path.exists(audio_path) and os.path.exists(image_path):
        # Process Audio
        signal, fs = librosa.load(audio_path, sr=16000)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        noisy_audio = add_reverberation_and_noise(signal, sr=fs)

        # Process Image
        original_image = cv2.imread(image_path)
        noisy_image = add_salt_pepper_noise(original_image)
        blurred_image = apply_mild_blur(noisy_image)
        shifted_image = apply_color_shift(blurred_image)

        # Save the noisy audio
        noisy_audio_path = os.path.join(output_path, speaker_label, os.path.basename(audio_path))
        os.makedirs(os.path.dirname(noisy_audio_path), exist_ok=True)
        torchaudio.save(noisy_audio_path, torch.tensor(noisy_audio).unsqueeze(0), fs)

        # Save the noisy image
        noisy_image_path = os.path.join(output_path, speaker_label, os.path.basename(image_path))
        cv2.imwrite(noisy_image_path, shifted_image)

    return None


# Function to process all files in parallel
def process_files_parallel(dataset_path):
    speaker_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    max_workers = os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for speaker_dir in speaker_dirs:
            files = os.listdir(speaker_dir)
            for file in files:
                if file.endswith('.wav'):
                    base_filename = file[:-4]  # Remove .wav extension
                    audio_path = os.path.join(speaker_dir, file)
                    image_path = os.path.join(speaker_dir, base_filename + '.jpg')
                    speaker_label = speaker_dir.split(os.sep)[-1]  # Extract speaker label from the folder name
                    futures.append(executor.submit(process_file, audio_path, image_path, speaker_label))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            future.result()


# Main script to process the dataset and generate noisy data
if __name__ == "__main__":
    process_files_parallel(dataset_path)
