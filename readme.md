# Clarityn

**AI-powered speaking intelligence.**

Clarityn is an AI-powered public speaking coach built with Streamlit, Whisper, and Gemini AI. It analyzes uploaded speech recordings and provides detailed coaching feedback, speaking analytics, filler-word detection, sentence-level review, and progress tracking.

The goal of Clarityn is simple:
> Help people speak more clearly, confidently, and effectively.

---

# Features

## 🎤 Speech Transcription
- Uses OpenAI Whisper for accurate speech-to-text transcription.
- Supports multiple audio formats:
  - MP3
  - WAV
  - M4A

---

## 📊 Speaking Analytics
Clarityn analyzes:
- Total words
- Speaking speed (WPM)
- Filler word usage
- Confidence score
- Speech structure quality

---

## 🗣️ Filler Word Detection
Detects filler words such as:
- um
- uh
- like
- basically
- actually
- matlab
- toh

Detected filler words are highlighted directly inside the transcription.

---

## 🚩 Speech Quality Flags
Automatically detects common speaking issues such as:
- weak openings
- repetitive wording
- poor structure
- overuse of filler phrases

---

## 📌 Sentence-Level Review
Clarityn reviews individual sentences and highlights weak or filler-heavy lines.

---

## 🤖 AI Coaching Feedback
Powered by Gemini AI.

Provides:
- strengths
- weaknesses
- actionable improvement tips

Includes two coaching modes:
- Balanced
- Strict

---

## 📈 Progress Tracking
Tracks speaking progress over time:
- confidence trends
- WPM trends
- filler-word trends

Each user gets their own local progress history.

---

# Tech Stack

- Streamlit
- OpenAI Whisper
- Gemini AI (`google-genai`)
- PyTorch
- Pandas
- Python

---

# Project Structure

```text
clarityn/
│
├── data/
├── assets/
│
├── .env
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/clarityn-ai.git
cd clarityn-ai
```

---

## 2. Create virtual environment

```bash
python -m venv .venv
```

Activate:

### Windows
```bash
.venv\Scripts\activate
```

### Mac/Linux
```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add Gemini API key

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## 5. Run the app

```bash
streamlit run app.py
```

---


# Future Roadmap

Planned improvements:
- 🎙️ Live microphone recording
- 📹 Video-based posture and expression analysis
- ☁️ Cloud deployment
- 📱 Mobile-friendly interface
- 🧠 More advanced speech scoring
- 🌍 Better multilingual analysis

---

# Why Clarityn?

Most people struggle with:
- filler words
- confidence
- pacing
- clarity
- structure

Clarityn helps speakers improve through:
- immediate analysis
- AI coaching
- measurable progress tracking

---


# Built With

- Whisper
- Gemini AI
- Streamlit

---

> Cut the ums. Keep the impact.

