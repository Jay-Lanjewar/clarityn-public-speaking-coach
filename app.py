import streamlit as st
import whisper
import tempfile
import os
import re
import shutil
import csv
from datetime import datetime
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Added: cached Whisper model loader.
@st.cache_resource
def load_model():
    return whisper.load_model("base")


os.environ["PATH"] += os.pathsep + r"C:\Users\ravil\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

print("FFMPEG DETECTED:", shutil.which("ffmpeg"))


# Feature 1: Lightweight user profile storage folder.
DATA_DIR = "data"
PROGRESS_FIELDS = ["timestamp", "word_count", "wpm", "filler_count", "confidence_score"]


def sanitize_username(username):
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", username.strip().lower())
    return safe_name.strip("_")


# Feature 1 and 2: Build a dynamic per-user progress file inside data/.
def get_progress_file(username):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{username}.csv")


# Feature 2: Progress file helpers now accept a dynamic progress_file parameter.
def ensure_progress_file(progress_file):
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)

    if not os.path.exists(progress_file):
        with open(progress_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=PROGRESS_FIELDS)
            writer.writeheader()


def save_progress(progress_file, word_count, wpm, filler_count, confidence_score):
    ensure_progress_file(progress_file)

    with open(progress_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PROGRESS_FIELDS)
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "word_count": word_count,
            "wpm": round(wpm, 2),
            "filler_count": filler_count,
            "confidence_score": round(confidence_score, 1),
        })


def load_progress(progress_file):
    ensure_progress_file(progress_file)
    return pd.read_csv(progress_file)


# Feature 4: Improvement metric helpers.
def get_balanced_wpm_distance(wpm):
    return abs(wpm - 140)


def calculate_improvement_messages(progress_df):
    if len(progress_df) < 2:
        return ["ℹ️ Complete another analysis to see improvement metrics."]

    previous = progress_df.iloc[-2]
    current = progress_df.iloc[-1]
    messages = []

    filler_diff = int(current["filler_count"] - previous["filler_count"])
    if filler_diff < 0:
        messages.append(f"✅ Filler words reduced by {abs(filler_diff)}")
    elif filler_diff > 0:
        messages.append(f"⚠️ Filler words increased by {filler_diff}")
    else:
        messages.append("➖ Filler word count stayed the same")

    previous_wpm_distance = get_balanced_wpm_distance(float(previous["wpm"]))
    current_wpm_distance = get_balanced_wpm_distance(float(current["wpm"]))
    wpm_diff = round(abs(float(current["wpm"]) - float(previous["wpm"])), 1)

    if current_wpm_distance < previous_wpm_distance:
        messages.append(f"✅ WPM became more balanced by {wpm_diff} WPM")
    elif current_wpm_distance > previous_wpm_distance:
        messages.append(f"⚠️ WPM moved farther from the balanced range by {wpm_diff} WPM")
    else:
        messages.append("➖ WPM balance stayed about the same")

    confidence_diff = round(float(current["confidence_score"]) - float(previous["confidence_score"]), 1)
    if confidence_diff > 0:
        messages.append(f"✅ Confidence improved by {confidence_diff} points")
    elif confidence_diff < 0:
        messages.append(f"⚠️ Confidence decreased by {abs(confidence_diff)} points")
    else:
        messages.append("➖ Confidence score stayed the same")

    return messages


# Feature 4: Overall performance label based on confidence score.
def get_performance_label(confidence_score):
    if confidence_score >= 8:
        return "Excellent"
    if confidence_score >= 6:
        return "Good"
    return "Needs Improvement"


# Gemini feedback with local fallback.
def generate_gemini_feedback(text, word_count, wpm, filler_count, feedback_mode, flags, sentences):
    # Feature 1: Coaching style selector changes the AI coach persona dynamically.
    coach_persona = {
        "Balanced": "You are a supportive but honest public speaking coach.",
        "Strict": "You are a brutally honest and highly critical public speaking coach.",
    }.get(feedback_mode, "You are a supportive but honest public speaking coach.")

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

        style_note = (
            "This is balanced feedback: supportive, honest, and practical."
            if feedback_mode == "Balanced"
            else "This is strict feedback: direct, critical, and demanding."
        )

        fallback_feedback = f"""
Coaching Style: {feedback_mode}
{style_note}

1. Strengths
- {' '.join(strengths)}

2. Weaknesses
- {' '.join(weaknesses) if weaknesses else 'No major weakness detected from the basic metrics.'}

3. Improvement tips
- {' '.join(tips)}
"""
        return fallback_feedback, None

    try:
        client = genai.Client(api_key=api_key)

        grammar_hint = "The speech may contain grammatical inconsistencies or unnatural phrasing."

        prompt = f"""
{coach_persona}
{grammar_hint}

Speech:
{text}

Metrics:
- Word count: {word_count}
- WPM: {round(wpm, 2)}
- Filler words: {filler_count}

Detected Issues:
- Flags: {flags}
- Weak Sentences: {[s.strip() for s in sentences if any(f in s.lower() for f in ["um", "uh", "like", "basically"])]}

RULES:
- You MUST include these detected issues in your analysis
- Do NOT say "no major weakness"
- Be critical and specific
- Match the selected coaching style: {feedback_mode}

Give:
1. Strengths
2. Weaknesses (include detected issues)
3. Improvement tips
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text, None

    except Exception:
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
            tips.append("Practice with a timer and try to add more flow between sentences.")

        if filler_count <= 5:
            strengths.append("You used relatively few filler words.")
        else:
            weaknesses.append("You used several filler words, which can weaken clarity.")
            tips.append("Replace filler words with short silent pauses.")

        if flags:
            weaknesses.extend(flags)

        weak_sentences = [
            s.strip() for s in sentences
            if any(f in s.lower() for f in ["um", "uh", "like", "basically"])
        ]

        if weak_sentences:
            weaknesses.append("Some sentences contain unnecessary filler words.")

        if not tips:
            tips.append("Work on stronger openings and more expressive delivery.")

        style_note = (
            "This is balanced feedback: supportive, honest, and practical."
            if feedback_mode == "Balanced"
            else "This is strict feedback: direct, critical, and demanding."
        )

        fallback_feedback = f"""
    Coaching Style: {feedback_mode}
    {style_note}

    1. Strengths
    - {' '.join(strengths)}

    2. Weaknesses
    - {' '.join(weaknesses)}

    3. Improvement tips
    - {' '.join(tips)}
    """

        return fallback_feedback, None


# ---------------- UI ----------------
st.set_page_config(page_title="Clarityn", layout="centered")

st.title("Clarityn")
st.caption("AI-powered speaking intelligence.")
# Feature 5: Better page organization with clearer top controls.
st.write("Upload your speech and receive detailed AI-powered speaking analysis, coaching feedback, and progress tracking.")
st.divider()

# Feature 1: Lightweight user profile input near the top, without auth or passwords.
username = st.text_input("👤 Enter your name")
safe_username = sanitize_username(username)
progress_file = None

if username and not safe_username:
    st.warning("Please enter at least one letter or number for your name.")
elif safe_username:
    # Feature 1: Automatically create the user's progress file if it does not exist.
    progress_file = get_progress_file(safe_username)
    ensure_progress_file(progress_file)

# Existing feature: Coaching style selector near the top of the UI.
feedback_mode = st.selectbox(
    "🎯 Coaching Style",
    ["Balanced", "Strict"]
)
st.divider()

# ---------------- Upload ----------------
# Feature 5: Better page organization starts with a focused upload section.
st.subheader("🎙️ Upload Speech")
audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])

if audio_file:
    # Feature 1: Username is required before any analysis runs.
    if not safe_username:
        st.warning("👤 Please enter your name before analysis.")
        st.stop()

    # Feature 1 and 2: Each user gets a separate local progress CSV.
    progress_file = progress_file or get_progress_file(safe_username)

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

    # ---------------- Analysis ----------------
    words = text.split()
    word_count = len(words)

    duration = result["segments"][-1]["end"] if result["segments"] else 1
    wpm = word_count / (duration / 60)

    # ---------------- Speech Quality Flags ----------------
    flags = []

    if "today i am going to talk" in text.lower():
        flags.append("⚠️ Weak opening detected (too generic)")

    if text.count(".") < 3:
        flags.append("⚠️ Speech may lack structure")

    if words and len(set(words)) / len(words) < 0.5:
        flags.append("⚠️ Repetitive wording detected")

    # ---------------- Sentence Analysis ----------------
    sentences = re.split(r'[.!?]', text)
    weak_sentences = []

    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue

        if any(filler in s.lower() for filler in ["um", "uh", "like", "basically"]):
            weak_sentences.append(s)

    # Multilingual filler words
    filler_words = [
        "um", "uh", "like", "basically", "actually",
        "so", "you know", "i mean",
        "matlab", "toh", "hmm",
        "मतलब", "तो", "हम्म"
    ]

    # Existing feature: Regex-based filler word detection.
    speech_words = re.findall(r'\b\w+\b', text.lower())
    filler_count = sum(1 for word in speech_words if word in filler_words)

    # Existing feature: Filler word highlighting without changing original transcription.
    highlighted_text = text

    for word in filler_words:
        highlighted_text = re.sub(
            rf"(?i)\b{re.escape(word)}\b",
            lambda m: f"**{m.group(0)}**",
            highlighted_text
        )

    # ---------------- Feedback Engine ----------------
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
    confidence_score = 10 - (filler_count * 0.5) - abs(wpm - 140) / 20
    confidence_score = max(1, min(10, confidence_score))
    feedback.append(f"💪 Confidence Score: {round(confidence_score, 1)}/10")

    # Feature 4: Overall performance label.
    performance_label = get_performance_label(confidence_score)

    # Feature 2: Save user-specific progress after each completed analysis.
    progress_signature = (
        safe_username,
        audio_file.name,
        getattr(audio_file, "size", 0),
        word_count,
        round(wpm, 2),
        filler_count,
        round(confidence_score, 1),
    )

    if st.session_state.get("last_progress_signature") != progress_signature:
        save_progress(progress_file, word_count, wpm, filler_count, confidence_score)
        st.session_state["last_progress_signature"] = progress_signature

    progress_df = load_progress(progress_file)

    # Feature 5: Better page organization with tabs instead of one long results page.
    st.divider()
    analysis_tab, ai_feedback_tab, progress_tab = st.tabs([
        "📊 Analysis",
        "🤖 AI Feedback",
        "📈 Progress"
    ])

    with analysis_tab:
        st.subheader("📝 Transcription")
        st.write(text)
        st.write(f"🌍 Detected Language: **{language.upper()}**")

        st.subheader("📊 Analysis")
        st.write(f"🧾 Total Words: {word_count}")
        st.write(f"⚡ Speaking Speed: {round(wpm, 2)} WPM")
        st.write(f"🗣️ Filler Words Used: {filler_count}")
        # Feature 4: Display overall performance based on the current confidence score.
        st.write(f"🏆 Overall Performance: {performance_label}")

        st.subheader("🚩 Speech Quality Flags")
        if flags:
            for f in flags:
                st.write(f)
        else:
            st.write("✅ No major structural issues detected.")

        st.subheader("📌 Sentence Review")
        if weak_sentences:
            for sentence in weak_sentences:
                st.write(f"⚠️ Weak sentence: {sentence}")
        else:
            st.write("✅ No obvious weak sentences detected.")

        st.subheader("🔍 Filler Word Highlight")
        st.markdown(highlighted_text)

        st.subheader("🧠 Feedback")
        for f in feedback:
            st.write(f)

        detected_fillers = [w for w in speech_words if w in filler_words]
        st.write(f"🧠 Detected filler words: {', '.join(detected_fillers)}")

    with ai_feedback_tab:
        st.subheader("🤖 AI Feedback")

        if st.button("Generate AI Feedback"):
            with st.spinner("Generating AI feedback..."):
                ai_feedback, ai_error = generate_gemini_feedback(
                    text,
                    word_count,
                    wpm,
                    filler_count,
                    feedback_mode,
                    flags,
                    sentences
                )

            if ai_error:
                st.error(ai_error)
            else:
                st.write(ai_feedback)

        if os.getenv("GEMINI_API_KEY"):
            st.write("🟢 Using Gemini API")
        else:
            st.write("🟡 Using Local Fallback")

    with progress_tab:
        st.subheader("📈 Progress History")
        st.caption(f"Showing progress for {username.strip()}")
        st.dataframe(progress_df.tail(5), width="stretch")

        st.subheader("📌 Improvement Metrics")
        for message in calculate_improvement_messages(progress_df):
            st.write(message)

        # Feature 3: Analytics charts using Streamlit native line charts.
        st.subheader("📉 Analytics Trends")
        chart_df = progress_df.tail(10).copy()

        if chart_df.empty:
            st.write("No progress entries yet.")
        else:
            chart_df["timestamp"] = chart_df["timestamp"].astype(str)
            chart_df = chart_df.set_index("timestamp")

            st.write("Confidence Score Trend")
            st.line_chart(chart_df[["confidence_score"]])

            st.write("Filler Word Trend")
            st.line_chart(chart_df[["filler_count"]])

            st.write("WPM Trend")
            st.line_chart(chart_df[["wpm"]])

st.divider()
st.caption("Built with Whisper + Gemini AI")
