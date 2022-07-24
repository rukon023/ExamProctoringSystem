from scipy.spatial.distance import cdist
from pyannote.audio import Audio
from pyannote.core import Segment
import torch
import os
import sounddevice as sd
import soundfile as sf
import time
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

AUDIO_DATABASE_DIR = r"F:\ExamProctoringSystem\SpeakerVerificationModule\audio_database"
AUDIO_TEMP_DIR = r"F:\ExamProctoringSystem\SpeakerVerificationModule\temp_data"
test_wav_path = r"F:\ExamProctoringSystem\SpeakerVerificationModule\temp_data\temp.wav"
model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")


def add_user():
    name = input("Name:")
    sample_rate = 16000
    seconds = 5
    print("Please count 1 to 10 when recording starts.")
    for i in range(3, 0, -1):
        print(f"Recording in {i}")
        time.sleep(0.5)
    print(f"Recording...")

    myrecording = sd.rec(int(seconds * sample_rate),
                         samplerate=sample_rate, channels=1)
    # wait until recording is finished
    sd.wait()
    print("Recording Finished")

    save_path = os.path.join(AUDIO_DATABASE_DIR, f"{name}.wav")
    if os.path.isfile(save_path):
        os.remove(save_path)

    # Save as FLAC file at correct sampling rate
    sf.write(save_path, myrecording, sample_rate)


def match_user():
    # Record Audio
    sample_rate = 16000
    seconds = 5
    print("Please count 1 to 10 when recording starts.")
    time.sleep(3)
    for i in range(3, 0, -1):
        print(f"Recording in {i}")
        time.sleep(1)
    print(f"Recording...")
    myrecording = sd.rec(int(seconds * sample_rate),
                         samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording Finished!")

    save_path = os.path.join(AUDIO_TEMP_DIR, "temp.wav")
    if os.path.isfile(save_path):
        os.remove(save_path)
    sf.write(save_path, myrecording, sample_rate)

    # reference_embedding
    audio = Audio(16000, True)
    waveform, sample_rate = audio(save_path)
    ref_embedding = model(waveform[None])

    # Match user
    embedding_dict = {}
    wav_list = os.listdir(AUDIO_DATABASE_DIR)
    # print(wav_list)
    for wav_file in wav_list:
        wav_path = os.path.join(AUDIO_DATABASE_DIR, wav_file)
        waveform, samp_rate = audio(wav_path)
        embedding = model(waveform[None])

        # compare embeddings using "cosine" distance
        distance = cdist(ref_embedding, embedding, metric="cosine")[0, 0]
        # print(distance)
        embedding_dict[wav_file[:-4]] = distance
    # print(embedding_dict)

    # Matched speaker with minimul distance:
    speaker = min(embedding_dict, key=embedding_dict.get)
    print(f"You are {speaker}")
    return speaker


if __name__ == "__main__":
    choice = int(input("Choices:\n1. Add User\n2. Match User:"))
    if choice == 1:
        add_user()
    elif choice == 2:
        speaker = match_user()
        print(speaker)
