from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import numpy as np
import torch
import os

# Magyar beszédgenerálás
def speak(text):
    print(f"Shadow: {text}")
    os.system(f'espeak-ng -v hu "{text}"')

# Modell betöltése (első indításnál letölt)
print("📦 Modell betöltése...")
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian")

# Hang rögzítése
def record_audio(duration=5, samplerate=16000):
    print("🎤 Beszélj most...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

# Felismerés
def transcribe(audio, samplerate=16000):
    inputs = processor(audio, sampling_rate=samplerate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# Parancs feldolgozás
def process_command(text):
    if "hogy vagy" in text:
        return "Jól vagyok, főnök."
    elif "mennyi az idő" in text or "hány óra" in text:
        from datetime import datetime
        now = datetime.now().strftime("%H:%M")
        return f"Most {now} van."
    elif "kilépés" in text:
        speak("Viszlát főnök.")
        exit()
    else:
        return "Ezt még nem tudom."

# Fő program
def main():
    speak("Shadow készen áll.")
    while True:
        audio = record_audio()
        text = transcribe(audio)
        print("🧠 Felismert szöveg:", text)
        if text:
            response = process_command(text)
            speak(response)

if __name__ == "__main__":
    main()