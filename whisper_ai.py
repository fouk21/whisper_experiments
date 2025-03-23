import torch
import whisper
import torchaudio
import torchaudio.transforms as T
import re
import os
import json
import tempfile
from pydub import AudioSegment
from transformers import pipeline
import numpy as np

class WhisperAI:
    def __init__(self, model_size='base', bad_words=None, output_json=False, save_transcript=False):
        """Initialize the Whisper AI module with options for output format, bad word filtering, and sentiment analysis."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Whisper model ({model_size}) on {self.device}...")
        self.model = whisper.load_model(model_size).to(self.device)
        self.bad_words = bad_words if bad_words else ["badword1", "badword2"]
        self.output_json = output_json
        self.save_transcript = save_transcript
        
        # Load sentiment analysis models
        print("Loading sentiment analysis models...")
        self.text_sentiment_analyzer = pipeline("sentiment-analysis")
        self.audio_sentiment_analyzer = pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")
        print("All models loaded successfully!")

    def convert_audio_to_wav(self, audio_path):
        """Converts various audio formats to WAV (16kHz) for processing."""
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name

    def preprocess_audio(self, audio_path):
        """Loads an audio file, resamples it to 16kHz, and returns the processed waveform."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
        
        print(f"Loading and processing audio file: {audio_path}")
        converted_audio = self.convert_audio_to_wav(audio_path)
        waveform, sample_rate = torchaudio.load(converted_audio)
        transform = T.Resample(sample_rate, 16000)
        waveform = transform(waveform)
        print("Audio preprocessing complete.")
        return waveform.squeeze().numpy()

    def transcribe_audio(self, audio_path):
        """Transcribes the given audio file using Whisper AI, filters out bad words, and performs sentiment analysis."""
        print("Starting transcription...")
        audio = self.preprocess_audio(audio_path)
        result = self.model.transcribe(audio)
        text = result["text"]
        print("Raw Transcription:", text)
        filtered_text = self.filter_bad_words(text)
        text_sentiment = self.analyze_text_sentiment(filtered_text)
        audio_sentiment = self.analyze_audio_sentiment(audio_path)
        return self.format_output(filtered_text, text_sentiment, audio_sentiment)

    def filter_bad_words(self, text):
        """Replaces occurrences of bad words in the transcript with asterisks."""
        for word in self.bad_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub("*" * len(word), text)
        return text

    def analyze_text_sentiment(self, text):
        """Analyzes sentiment of the transcribed text."""
        sentiment_result = self.text_sentiment_analyzer(text)
        return sentiment_result[0]
    
    def analyze_audio_sentiment(self, audio_path):
        """Performs sentiment analysis on the audio file."""
        audio_waveform = self.preprocess_audio(audio_path)
        audio_input = torch.tensor(audio_waveform).unsqueeze(0)
        sentiment_result = self.audio_sentiment_analyzer(audio_input)
        return sentiment_result[0]

    def format_output(self, text, text_sentiment, audio_sentiment):
        """Formats the output including transcribed text and sentiment analysis results."""
        output_data = {
            "transcription": text,
            "text_sentiment": text_sentiment,
            "audio_sentiment": audio_sentiment
        }
        if self.output_json:
            formatted_output = json.dumps(output_data, indent=4)
        else:
            formatted_output = f"Transcription: {text}\nText Sentiment: {text_sentiment}\nAudio Sentiment: {audio_sentiment}"
        
        if self.save_transcript:
            self.save_to_file(output_data)
        
        return formatted_output

    def save_to_file(self, data):
        """Saves the transcription and sentiment analysis to a file."""
        file_name = "transcription.json" if self.output_json else "transcription.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            if self.output_json:
                json.dump(data, f, indent=4)
            else:
                f.write(f"Transcription: {data['transcription']}\n")
                f.write(f"Text Sentiment: {data['text_sentiment']}\n")
                f.write(f"Audio Sentiment: {data['audio_sentiment']}\n")
        print(f"Transcript saved to {file_name}")

if __name__ == "__main__":
    print("Initializing Whisper AI Transcription System...")
    ai = WhisperAI(bad_words=["examplebadword", "anotherbadword"], output_json=True, save_transcript=True)
    
    audio_file = "sample.mp3"  # Replace with your actual audio file path
    
    try:
        transcript = ai.transcribe_audio(audio_file)
        print("Filtered Transcript:", transcript)
    except Exception as e:
        print("Error:", str(e))