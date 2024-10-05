import json
import requests
import numpy as np
import librosa
import threading
import time
import gradio as gr
import noisereduce as nr  # Import the noisereduce library
from service.transcript.stt_v5 import ThaiSpeechToText
from service.GTTS.tts_v1 import TextToSpeech
from service.Chatbot.ollama_generator import OllamaGenerator  # Import your OllamaGenerator class

thai_stt = ThaiSpeechToText()
thai_tts = TextToSpeech()
ollama_generator = OllamaGenerator()  # Initialize the OllamaGenerator

# Global variables with a lock for thread safety
transcription_lock = threading.Lock()
last_audio_time = None
current_transcription = ""
HELLO_AUDIO_FILE = "service/GTTS/hello.mp3"
is_playing = False
is_first_play = True
file_audio = None

def normalize_and_resample_audio(data, sr):
    """Normalize and resample the audio if necessary."""
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    return data, sr

def reduce_noise(data, sr):
    """Reduce noise in the audio using noisereduce."""
    reduced_noise_data = nr.reduce_noise(y=data, sr=sr)
    return reduced_noise_data

def process_transcription(text):
    """Process the transcription and trigger appropriate responses."""
    global current_transcription, is_playing, file_audio, is_first_play

    current_transcription += text

    # Handle first play logic
    if is_first_play:
        file_audio = HELLO_AUDIO_FILE
        is_first_play = False
        is_playing = True  # Ensure the audio plays

    # Keyword detection for subsequent plays
    elif "สวัสดี" in current_transcription:
        file_audio = thai_tts.speak(text='สวัสดีค่ะ')
        reset_transcription_state()

    elif "ถาม" in current_transcription:
        current_transcription += text
        print(current_transcription)
        ollama_response = ollama_generator.generate(prompt=current_transcription)
        file_audio = thai_tts.speak(text=ollama_response)
        reset_transcription_state()

    return current_transcription, file_audio

def reset_transcription_state():
    """Reset the transcription state after a response is generated."""
    global current_transcription, is_playing
    current_transcription = ""
    is_playing = True

def transcribe_audio(audio):
    global last_audio_time

    try:
        sr, data = audio
        data, sr = normalize_and_resample_audio(data, sr)

        # Apply noise reduction
        data = reduce_noise(data, sr)

        with transcription_lock:
            last_audio_time = time.time()

        text = thai_stt.transcribe_audio(data, sr)

        with transcription_lock:
            return process_transcription(text)

    except Exception as e:
        return f"An error occurred during transcription: {str(e)}", None

# Define the input audio component with custom waveform options
input_audio = gr.Audio(
    type="numpy",
    label="Record your voice",
    show_label=True,
    streaming=True,
)

# Define the output text component
output_textbox = gr.Textbox(label="ถอดเสียง")

# Create the Gradio interface
app = gr.Interface(
    fn=transcribe_audio,
    inputs=input_audio,
    outputs=[output_textbox, gr.Audio(label="ตอบกลับ", autoplay=True)],
    live=True,
    title="AI KAK v2",
    description="บันทึกเสียงของคุณ และแอปจะคุยกับ ai ชื่อว่า กาก สามารถทดสอบพูด เช่น สวัสดี พร้อมรับคำสั่ง กาก"
)

if __name__ == "__main__":
    app.launch()  # Launch the Gradio app
