import wave
from pyannote.audio import Pipeline
import torchaudio
import sounddevice as sd
import time
import os
import pandas
import torch
import numpy as np

# test_data_path = r"F:\ExamProctoringSystem\VoiceAcitivityDetection\test_data\temp_output_speech.wav"
vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
# output = vad(test_data_path)
# print(dir(vad))


def detect_speech():
    sample_rate = 16000
    seconds = 5
    for i in range(3, 0, -1):
        print(f"Recording in {i}")
        time.sleep(0.5)
    print(f"Recording...")

    myrecording = sd.rec(int(seconds * sample_rate),
                         samplerate=sample_rate, channels=1)
    # wait until recording is finished
    sd.wait()
    print("Recording finished!")
    myrecording = np.reshape(myrecording, (1, 80000))
    # print(myrecording.shape)
    myrecording = torch.from_numpy(myrecording)

    # waveform, sample_rate = torchaudio.load(test_data_path)
    # # print(type(waveform))
    # print(waveform.shape)
    # print(waveform)

    audio_in_memory = {"waveform": myrecording, "sample_rate": sample_rate}
    output = vad(audio_in_memory)
    # print("="*100)
    # print(output.get_timeline().support())
    return output.label_duration("SPEECH")


if __name__ == "__main__":
    speech_duration = detect_speech()
    print(speech_duration)
