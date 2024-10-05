import pyaudio
import audioop
import torch
import struct
import soundfile as sf
import io
import noisereduce as nr
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from service.GTTS.tts_v1 import TextToSpeech
from service.Chatbot.ollama_generator import OllamaChat 
from colorama import Fore, Style, init



class ThaiSpeechToText:
    def __init__(self, max_silence_seconds=3):
        # Check device for computation
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using device: {self.dev}')

        # Load pretrained processor and model
        self.processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
        self.model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
        self.model.to(self.dev)

        # Audio recording settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16_000
        self.CHUNK = 8192
        self.SILENT_THRESHOLD = 10000
        self.MAX_SILENCE_SECONDS = 5

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.FORMAT,
                                       channels=self.CHANNELS,
                                       rate=self.RATE,
                                       input=True,
                                       frames_per_buffer=self.CHUNK)

    def write_header(self, _bytes, _nchannels, _sampwidth, _framerate):
        WAVE_FORMAT_PCM = 0x0001
        initlength = len(_bytes)
        bytes_to_add = b'RIFF'
        
        _nframes = initlength // (_nchannels * _sampwidth)
        _datalength = _nframes * _nchannels * _sampwidth

        bytes_to_add += struct.pack('<L4s4sLHHLLHH4s',
            36 + _datalength, b'WAVE', b'fmt ', 16,
            WAVE_FORMAT_PCM, _nchannels, _framerate,
            _nchannels * _framerate * _sampwidth,
            _nchannels * _sampwidth,
            _sampwidth * 8, b'data')

        bytes_to_add += struct.pack('<L', _datalength)

        return bytes_to_add + _bytes

    def record_audio(self):
        frames = []
        silence_frames = 0

        while True:
            data = self.stream.read(self.CHUNK)
            rms = audioop.rms(data, 2)
            reduced_noise = nr.reduce_noise(y=np.frombuffer(data, np.int16), sr=self.RATE, 
                                             thresh_n_mult_nonstationary=2, stationary=False)
            frames.append(reduced_noise.tobytes())

            if rms < self.SILENT_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0
            
            if silence_frames / (self.RATE / self.CHUNK) > self.MAX_SILENCE_SECONDS:
                break

        audio_data = b''.join(frames)
        wav_data = self.write_header(audio_data, 1, 2, self.RATE)
        return wav_data

    def convert_audio_to_text(self):
        wav_data = self.record_audio()
        raw_audio, _ = sf.read(io.BytesIO(wav_data))
        raw_audio = nr.reduce_noise(y=raw_audio, sr=self.RATE)

        inputs = self.processor(raw_audio, sampling_rate=self.RATE, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.dev)).logits.cpu()

        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)

        return transcriptions[0]

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

if __name__ == "__main__":
    speech_to_text = ThaiSpeechToText()
    thai_tts = TextToSpeech()
    ollama = OllamaChat()

    
    try:
        while True:
            start_input = input(Fore.CYAN + "Press Write 'start' to start: " + Style.RESET_ALL)
            if start_input != "start":
                print("Invalid input. Please try again.")
                continue
            if start_input == "start":
                print("กำลังบันทึกเสียง...")
                text = speech_to_text.convert_audio_to_text()
                print( "Transcribed Text:", text)
                user_messages = [{"role": "user", "content": text.replace(" ", "")}]
                res = ollama.chat(user_messages)
                print(Fore.GREEN + res['content']+ Style.RESET_ALL )
                thai_tts.speak(res['content'])
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        speech_to_text.close()
