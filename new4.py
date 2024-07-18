import streamlit as st
import numpy as np
import pyaudio
import wave
import librosa
from keras.models import load_model
from keras.initializers import Orthogonal
import matplotlib.pyplot as plt
import librosa.display
import time

def record_audio_pyaudio(duration, fs, filename):
    FORMAT = pyaudio.paInt16    
    CHANNELS = 1                
    CHUNK = 1024                

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=fs, input=True,
                        frames_per_buffer=CHUNK)

    st.write("Recording...")

    frames = []

    for i in range(0, int(fs / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(fs)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)

def preprocess_audio(filename):
    y, sr = librosa.load(filename, duration=4, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return np.expand_dims(mfcc, axis=0)

def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

st.title("MACHINE HEALTH MONITORING SYSTEM")

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

if st.session_state.is_recording:
    if st.button("Stop", key="stop_button"):
        st.session_state.is_recording = False
else:
    if st.button("Start", key="start_button"):
        st.session_state.is_recording = True

if st.session_state.is_recording:
    duration = 3
    fs = 44100
    audio_file = 'recorded_audio.wav'
    model_path = 'my_model_fianl4.h5'
    custom_objects = {'Orthogonal': Orthogonal}
    model = load_model(model_path, custom_objects=custom_objects)
    class_labels = ['metal25%', 'metal50%', 'metal75%']

    iteration = 0
    while st.session_state.is_recording:
        audio = record_audio_pyaudio(duration, fs, audio_file)
        
        y, sr = librosa.load(audio_file, sr=fs)
        plot_waveform(y, sr)
        if np.all(y >= 0.6) and np.all(y <= -0.6):
            st.markdown("### YOUR MACHINE WILL BE FAULTY WITHIN 5 Days because I am getting a noise of 75 in it")
         
        elif np.all(y <= 0.1) and np.all(y >= -0.1):
            st.markdown("### Everything is clear and there is no anomaly in the machine.")
       
        else:
            preprocessed_audio = preprocess_audio(audio_file)
            preprocessed_audio = np.reshape(preprocessed_audio, (1, preprocessed_audio.shape[1]))
            
            predictions = model.predict(preprocessed_audio)
            
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            if predicted_class_label == 'metal25%':
                st.markdown("### YOUR MACHINE WILL BE FAULTY WITHIN 20 Days because I am getting a noise of 25 in it")
            elif predicted_class_label == 'metal50%':
                st.markdown("### YOUR MACHINE WILL BE FAULTY WITHIN 10 Days because I am getting a noise of 50 in it")
            elif predicted_class_label == 'metal75%':
                st.markdown("### YOUR MACHINE WILL BE FAULTY WITHIN 5 Days because I am getting a noise of 75 in it")
            else:
                st.markdown(f"### I CANNOT UNDERSTAND THIS LET ME RECORD IT AGAIN")
        
        time.sleep(10)
        
        if st.button("Stop", key=f"stop_button_loop_{iteration}"):
            st.session_state.is_recording = False
            break
        
        iteration += 1
