from pydub import AudioSegment
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mp3_to_wav(target_dir, file_path):
	base_name = os.path.basename(file_path)
	output_dst = f'{target_dir}/{base_name}.wav'

	sound = AudioSegment.from_mp3(file_path)
	sound.export(output_dst, format="wav")

def main():
	root_dir = os.getcwd()
	train_dir = root_dir + "/test"
	train_wav_dir = root_dir + "/test_wav"

	# mp3 to wav conversion
	for file in os.listdir(train_dir):
		file_name = os.fsdecode(file)
		if file_name.endswith(".mp3"):
			file_path = os.path.join(root, file_name)
			mp3toWav(target_dir, file_path)

	# feature extraction: MFCC
	for file in os.listdir(train_wav_dir):
		x, sr = librosa.load(file)
		mfccs = librosa.feature.mfcc(x=y, sr=sr, n_mfcc=40)

	# data labelling: align transcription with audio segments
		# use wav2vec2phoneme to get phoneme of Bengali words
		# forced alignment to match audio with transcription

	# train ASR model (HMM) on dataset

if __name__ == "__main__":
    main()