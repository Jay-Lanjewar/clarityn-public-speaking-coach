import streamlit as st
import whisper
import tempfile
import os
import re
import shutil
import google.generativeai as genai

# Added: cached Whisper model loader.
@st.cache_resource
def load_model():
    return whisper.load_model("base")

os.environ["PATH"] += os.pathsep + r"C:\Users\ravil\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"


print("FFMPEG DETECTED:", shutil.which("ffmpeg"))


# Gemini feedback with local fallback.
def generate_gemini_feedback(text, word_count, wpm, filler_count):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        strengths = []
        weaknesses = []
        tips = []

        if 120 <= wpm <= 170:
            strengths.append("Your speaking pace is clear and comfortable.")
        elif wpm > 170:
            weaknesses.append("Your pace is fast, which may make the speech harder to follow.")
            tips.append("Pause briefly after key points and aim for a slightly slower delivery.")
        else:
            weaknesses.append("Your pace is slow, which may reduce energy and audience engagement.")
            tips.append("Practice with a timer and try to add a little more flow between sentences.")

        if filler_count <= 5:
            strengths.append("You used relatively few filler words.")
        else:
            weaknesses.append("You used several filler words, which can weaken clarity.")
            tips.append("Replace filler words with short silent pauses.")

        if word_count > 0:
            strengths.append("You completed a speech sample with enough content to analyze.")
        else:
            weaknesses.append("The transcription appears empty or too short to analyze fully.")
            tips.append("Upload a clearer or longer speech sample for better feedback.")

        if not tips:
            tips.append("Keep practicing with the same structure and focus on stronger vocal variety.")

        fallback_feedback = f"""
1. Strengths
- {' '.join(strengths)}

2. Weaknesses
- {' '.join(weaknesses) if weaknesses else 'No major weakness detected from the basic metrics.'}

3. Improvement tips
- {' '.join(tips)}
"""
        return fallback_feedback, None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""
You are a strict and practical public speaking coach.

Analyze this speech critically:

{text}

Metrics:
- Word count: {word_count}
- WPM: {round(wpm, 2)}
- Filler words: {filler_count}

IMPORTANT RULES:
- Do NOT be overly positive
- Identify real problems in grammar, clarity, and delivery
- If there are mistakes, point them out clearly
- Avoid generic advice

Give output in this format:

1. Strengths
- (real strengths only)

2. Weaknesses
- (must include real issues like grammar errors, weak structure, lack of variation)

3. Improvement tips
- (specific, actionable, not generic)
"""
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)


# ---------------- UI ----------------
st.set_page_config(page_title="AI Speaking Coach", layout="centered")

st.title("🎤 AI Public Speaking Coach")
# Feature 6: Description text improved.
st.write("Upload your speech (MP3, WAV, M4A) and get instant AI coaching feedback.")

# ---------------- Upload ----------------
audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])

if audio_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    st.info("⏳ Transcribing...")

    # Changed: initialize cached Whisper model before transcription.
    model = load_model()

    # Changed: wrap transcription in a loading spinner.
    with st.spinner("Analyzing your speech..."):
        result = model.transcribe(audio_path)

    text = result["text"]
    language = result.get("language", "unknown")

    # ---------------- Output: Transcription ----------------
    st.subheader("📝 Transcription")
    st.write(text)

    st.write(f"🌍 Detected Language: **{language.upper()}**")

    # ---------------- Analysis ----------------
    words = text.split()
    word_count = len(words)

    duration = result["segments"][-1]["end"] if result["segments"] else 1
    wpm = word_count / (duration / 60)

    # Multilingual filler words
    filler_words = [
    "um", "uh", "like", "basically", "actually",
    "so", "you know", "i mean",
    "matlab", "toh", "hmm",
    "मतलब", "तो", "हम्म"
]

    # Feature 1: Regex-based filler word detection added.
    speech_words = re.findall(r'\b\w+\b', text.lower())
    filler_count = sum(1 for word in speech_words if word in filler_words)

    # ---------------- Display Analysis ----------------
    st.subheader("📊 Analysis")
    st.write(f"🧾 Total Words: {word_count}")
    st.write(f"⚡ Speaking Speed: {round(wpm, 2)} WPM")
    st.write(f"🗣️ Filler Words Used: {filler_count}")

    # Feature 2: Filler word highlighting added without changing original transcription.
    st.subheader("🔍 Filler Word Highlight")
    filler_pattern = "|".join(re.escape(word) for word in filler_words)
    highlighted_text = text

    for word in filler_words:
        highlighted_text = re.sub(
            rf"(?i)\b{re.escape(word)}\b",
            lambda m: f"**{m.group(0)}**",
            highlighted_text
        )

    st.markdown(highlighted_text)

    # ---------------- Feedback Engine ----------------
    st.subheader("🧠 Feedback")

    feedback = []

    # Speed feedback
    if wpm > 170:
        feedback.append("⚠️ You are speaking too fast. Try slowing down for clarity.")
    elif wpm < 120:
        feedback.append("⚠️ You are speaking too slow. Add more energy and flow.")
    else:
        feedback.append("✅ Good speaking pace.")

    # Filler feedback
    if filler_count > 5:
        feedback.append("⚠️ Too many filler words. Practice cleaner delivery.")
    else:
        feedback.append("✅ Good fluency.")

    # Language-aware suggestion
    if language != "en":
        feedback.append("🌍 Try mixing a bit more English for wider audience reach.")

    # Confidence estimation (basic logic)
    confidence_score = 10 - (filler_count * 0.5) - abs(wpm - 140)/20
    confidence_score = max(1, min(10, confidence_score))
    feedback.append(f"💪 Confidence Score: {round(confidence_score, 1)}/10")

    # Show feedback
    for f in feedback:
        st.write(f)

    # ---------------- AI Feedback ----------------
    st.subheader("AI Feedback")

    if st.button("Generate AI Feedback"):
        with st.spinner("Generating AI feedback..."):
            ai_feedback, ai_error = generate_gemini_feedback(
                text,
                word_count,
                wpm,
                filler_count
            )

        if ai_error:
            st.error(ai_error)
        else:
            st.write(ai_feedback)
    st.write(f"🧠 Detected filler words: {', '.join([w for w in speech_words if w in filler_words])}")