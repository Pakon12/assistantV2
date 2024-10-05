import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pythainlp import correct

class ThaiSpeechToText:
    def __init__(self, model_name="airesearch/wav2vec2-large-xlsr-53-th"):
        # self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = "cpu"
        print(f"Running on {'GPU' if self.dev == 'cuda' else 'CPU'}")

        # โหลดโปรเซสเซอร์และโมเดลที่ผ่านการฝึกอบรมล่วงหน้า
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.dev)

    def word_correction(self, sentence):
        newText = ""
        for subword in sentence.split(" "):
            newText += " " + correct(subword) if newText else correct(subword)
        return newText
        
    def transcribe_audio(self, audio_data, sample_rate):
        # ให้แน่ใจว่าความยาวขั้นต่ำสำหรับเสียง
        min_length = int(16000 * 1.0)  # ขั้นต่ำ 1 วินาทีของเสียงที่ 16kHz
        if len(audio_data) < min_length:
            padding = np.zeros(min_length - len(audio_data))
            audio_data = np.concatenate([audio_data, padding])
        
        # ประมวลผลเสียงล่วงหน้าสำหรับโมเดล
        inputs = self.processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.dev)).logits.cpu()

        predicted_ids = torch.argmax(logits, dim=-1)

        # ถอดรหัส ID ที่คาดการณ์ไว้เป็นข้อความ
        transcriptions = self.processor.batch_decode(predicted_ids)
        
        # ใช้การแก้ไขคำ
        correct_word = self.word_correction(transcriptions[0])
        return correct_word
