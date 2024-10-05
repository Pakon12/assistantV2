from playsound import playsound
from gtts import gTTS

class TextToSpeech:
    def __init__(self, lang='th'):
        self.lang = lang

    def speak(self, text, path='service/GTTS/temp_audio.mp3'):
        # สร้างคำพูด
        tts = gTTS(text, lang=self.lang)
        tts.save(path)
        # Play the generated speech
        playsound(path)
        # file_path = path
        
        
        # return file_path
 
# Usage การใช้งาน
# tts = TextToSpeech()
# tts.speak("สวัสดีครับ")
