# Whisper AI Transcription and Sentiment Analysis

## Overview

This project is an AI-powered audio transcription system using OpenAI's Whisper model. It enhances transcription with:

- **Bad word filtering**
- **Multi-format audio support**
- **Sentiment analysis for both text and audio**
- **JSON and text output options**
- **Automatic transcript saving**

## Features

- Converts audio to WAV format (supports MP3, WAV, and other formats)
- Uses Whisper AI for accurate speech-to-text transcription
- Filters out predefined bad words from the transcript
- Analyzes sentiment of the transcribed text
- Performs sentiment analysis on audio signals
- Saves output as JSON or plain text

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install torch whisper torchaudio pydub transformers
```

## Usage

### Running the Script

```bash
python whisper_ai.py
```

By default, it processes `sample.mp3`. Modify the script or pass a different audio file for transcription.

### Configuring Parameters

Modify the `WhisperAI` initialization in `whisper_ai.py` to:

- Change bad words filtering
- Enable/disable JSON output
- Enable/disable automatic transcript saving

Example:

```python
ai = WhisperAI(bad_words=["badword1", "badword2"], output_json=True, save_transcript=True)
```

## Output

- The transcription and sentiment analysis results are printed in the console.
- If `save_transcript=True`, the results are saved to `transcription.json` or `transcription.txt`.

## Dependencies

- **PyTorch** for deep learning computations
- **Whisper** for speech recognition
- **Torchaudio** for audio processing
- **Pydub** for handling different audio formats
- **Transformers** for text and audio sentiment analysis

## Acknowledgments

- OpenAI for Whisper
- Hugging Face for sentiment analysis models
