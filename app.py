import streamlit as st
import whisper
import tempfile
import os

os.environ["PATH"] += os.pathsep + r"C:\Users\ravil\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

import shutil
print("FFMPEG DETECTED:", shutil.which("ffmpeg"))


def generate_gpt_feedback(transcription_text, word_count, wpm, filler_count):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY environment variable is not set."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = f"""
You are an AI public speaking coach. Analyze the speech using these inputs:

Transcription:
{transcription_text}

Word count: {word_count}
WPM: {round(wpm, 2)}
Filler count: {filler_count}

Generate feedback with exactly these sections:
1. Strengths
2. Weaknesses
3. Specific improvement tips
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Give concise, practical public speaking feedback."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# ---------------- UI ----------------
st.set_page_config(page_title="AI Speaking Coach", layout="centered")

st.title("🎤 AI Public Speaking Coach")
st.write("Upload your speech and get smart feedback")

# ---------------- Upload ----------------
audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])

if audio_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    st.info("⏳ Transcribing...")

    # ---------------- Model ----------------
    model = whisper.load_model("base")

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
        "um", "uh", "like", "basically",
        "matlab", "toh", "hmm",
        "मतलब", "तो", "हम्म"
    ]

    filler_count = sum(text.lower().count(word) for word in filler_words)

    # ---------------- Display Analysis ----------------
    st.subheader("📊 Analysis")
    st.write(f"🧾 Total Words: {word_count}")
    st.write(f"⚡ Speaking Speed: {round(wpm, 2)} WPM")
    st.write(f"🗣️ Filler Words Used: {filler_count}")

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
    confidence_score = max(1, 10 - (filler_count / 2))
    feedback.append(f"💪 Confidence Score: {round(confidence_score, 1)}/10")

    # Show feedback
    for f in feedback:
        st.write(f)

    # ---------------- GPT Feedback ----------------
    st.subheader("GPT-Based Feedback")

    if st.button("Generate GPT Feedback"):
        with st.spinner("Generating GPT feedback..."):
            gpt_feedback, gpt_error = generate_gpt_feedback(
                text,
                word_count,
                wpm,
                filler_count
            )

        if gpt_error:
            st.error(gpt_error)
        else:
            st.write(gpt_feedback)
