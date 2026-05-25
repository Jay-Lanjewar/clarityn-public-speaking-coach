import streamlit as st
import whisper
import tempfile
import os
import re
import shutil
import csv
import hashlib
from datetime import datetime
import pandas as pd
from google import genai
from dotenv import load_dotenv

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

load_dotenv()


def configure_ffmpeg():
    """Make ffmpeg discoverable without relying on a machine-specific path."""
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")

    ffmpeg_path = os.getenv("FFMPEG_PATH") or os.getenv("FFMPEG_BINARY")
    if not ffmpeg_path:
        return None

    if os.path.isdir(ffmpeg_path):
        ffmpeg_dir = ffmpeg_path
    else:
        ffmpeg_dir = os.path.dirname(ffmpeg_path)

    if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    return shutil.which("ffmpeg")


# Added: cached Whisper model loader.
@st.cache_resource
def load_model():
    return whisper.load_model("base")


print("FFMPEG DETECTED:", configure_ffmpeg())


# Feature 1: Lightweight user profile storage folder.
DATA_DIR = "data"
PROGRESS_FIELDS = ["timestamp", "word_count", "wpm", "filler_count", "confidence_score"]
MIN_AUDIO_BYTES = 2048


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
        return ["Complete another analysis to see improvement metrics."]

    previous = progress_df.iloc[-2]
    current = progress_df.iloc[-1]
    messages = []

    filler_diff = int(current["filler_count"] - previous["filler_count"])
    if filler_diff < 0:
        messages.append(f"Filler words reduced by {abs(filler_diff)}")
    elif filler_diff > 0:
        messages.append(f"Filler words increased by {filler_diff}")
    else:
        messages.append("Filler word count stayed the same")

    previous_wpm_distance = get_balanced_wpm_distance(float(previous["wpm"]))
    current_wpm_distance = get_balanced_wpm_distance(float(current["wpm"]))
    wpm_diff = round(abs(float(current["wpm"]) - float(previous["wpm"])), 1)

    if current_wpm_distance < previous_wpm_distance:
        messages.append(f"WPM became more balanced by {wpm_diff} WPM")
    elif current_wpm_distance > previous_wpm_distance:
        messages.append(f"WPM moved farther from the balanced range by {wpm_diff} WPM")
    else:
        messages.append("WPM balance stayed about the same")

    confidence_diff = round(float(current["confidence_score"]) - float(previous["confidence_score"]), 1)
    if confidence_diff > 0:
        messages.append(f"Confidence improved by {confidence_diff} points")
    elif confidence_diff < 0:
        messages.append(f"Confidence decreased by {abs(confidence_diff)} points")
    else:
        messages.append("Confidence score stayed the same")

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
    # Coaching style selector changes the AI coach persona dynamically.
    coach_persona = {
        "Supportive Coach": "You are an encouraging and supportive public speaking coach who focuses on building confidence and providing gentle, constructive feedback.",
        "Strict Coach": "You are a brutally honest and highly critical public speaking coach. You do not sugarcoat anything.",
        "Debate Coach": "You are a debate coach focused on logic, clarity, and confidence. You critique argument structure, reasoning, and persuasive delivery.",
        "Interview Coach": "You are an interview coach focused on conciseness, professionalism, and impact. You evaluate how answers are structured and how clearly key points are communicated.",
        "TED-style Coach": "You are a TED-style speaking coach focused on storytelling, audience engagement, and memorable delivery. You prioritize narrative flow, emotional connection, and impactful messaging.",
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
            tips.append("Upload or record a clearer or longer speech sample for better feedback.")

        if not tips:
            tips.append("Keep practicing with the same structure and focus on stronger vocal variety.")

        style_notes_map = {
            "Supportive Coach": "This is supportive coaching: encouraging, gentle, and focused on building confidence.",
            "Strict Coach": "This is strict coaching: direct, critical, and demanding.",
            "Debate Coach": "This is debate coaching: focused on logic, clarity, and persuasive reasoning.",
            "Interview Coach": "This is interview coaching: focused on conciseness, professionalism, and impact.",
            "TED-style Coach": "This is TED-style coaching: focused on storytelling, engagement, and memorable delivery.",
        }
        style_note = style_notes_map.get(feedback_mode, "This is balanced feedback: supportive, honest, and practical.")

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
        focus_areas = {
            "Supportive Coach": "encouragement and gentle improvement suggestions",
            "Strict Coach": "harsh critique and high standards",
            "Debate Coach": "logical structure, argument clarity, and persuasive confidence",
            "Interview Coach": "conciseness, professionalism, answer structure, and clarity under pressure",
            "TED-style Coach": "storytelling, narrative flow, emotional connection, and audience engagement",
        }.get(feedback_mode, "balanced speaking improvement")

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
- Focus your analysis on: {focus_areas}

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

        style_notes_map = {
            "Supportive Coach": "This is supportive coaching: encouraging, gentle, and focused on building confidence.",
            "Strict Coach": "This is strict coaching: direct, critical, and demanding.",
            "Debate Coach": "This is debate coaching: focused on logic, clarity, and persuasive reasoning.",
            "Interview Coach": "This is interview coaching: focused on conciseness, professionalism, and impact.",
            "TED-style Coach": "This is TED-style coaching: focused on storytelling, engagement, and memorable delivery.",
        }
        style_note = style_notes_map.get(feedback_mode, "This is balanced feedback: supportive, honest, and practical.")

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


# Speech Structure Analysis
def analyze_speech_structure(text, sentences):
    issues = []
    suggestions = []
    text_lower = text.lower()

    weak_opening_patterns = [
        r"^today\b", r"^so\b", r"^hi\b", r"^hello\b", r"^hey\b",
        r"i(?:'m| am) going to (?:talk|speak|discuss|present)",
        r"i will (?:talk|speak|discuss|present)",
        r"my (?:topic|presentation|speech|talk).*(?:is|will be|is about)",
    ]
    opening_match = None
    for p in weak_opening_patterns:
        m = re.search(p, text_lower)
        if m:
            opening_match = m.group()
            break

    if opening_match:
        issues.append(("Weak Opening", f'"{opening_match}" is a generic opening. Start with a hook, question, or bold statement instead.'))
        suggestions.append("Open with a compelling hook: a surprising fact, rhetorical question, or bold claim.")

    conclusion_markers = ["in conclusion", "to sum up", "to summarize", "in summary", "overall", "finally", "let me leave you with", "thank you"]
    has_conclusion = any(marker in text_lower[-300:] for marker in conclusion_markers) if len(text) > 100 else False
    if not has_conclusion:
        issues.append(("Weak Conclusion", "The speech ends without a clear concluding statement."))
        suggestions.append("End with a strong conclusion that reinforces your main message or includes a call to action.")

    transition_words = [
        "however", "moreover", "furthermore", "additionally", "nevertheless",
        "on the other hand", "therefore", "thus", "consequently", "in addition",
        "as a result", "for example", "for instance", "specifically", "in particular",
        "firstly", "secondly", "finally"
    ]
    transition_count = sum(1 for t in transition_words if re.search(rf'\b{re.escape(t)}\b', text_lower))
    if transition_count < 2 and len(sentences) >= 3:
        issues.append(("Abrupt Transitions", "Limited use of transition words makes the speech feel choppy."))
        suggestions.append("Use transition phrases like 'however', 'for example', or 'as a result' to connect ideas smoothly.")

    words_list = text.split()
    if words_list and len(set(words_list)) / len(words_list) < 0.55:
        issues.append(("Repetitive Wording", "Vocabulary is too repetitive — many words are reused."))
        suggestions.append("Use synonyms and vary your word choice to keep the audience engaged.")

    storytelling_markers = [
        "i remember", "for example", "for instance", "let me tell you",
        "picture this", "imagine", "when i", "one time", "a story",
        "let me share", "i recall", "here's a", "consider this"
    ]
    story_count = sum(1 for m in storytelling_markers if m in text_lower)
    if story_count < 2 and len(words_list) >= 20:
        issues.append(("Lack of Storytelling", "No personal stories, examples, or anecdotes found."))
        suggestions.append("Weave in a short personal story, example, or analogy to make your message memorable.")

    engagement_markers = [
        r"\?", r"you might", r"have you ever", r"think about", r"imagine",
        r"consider this", r"here's the thing", r"the truth is",
        r"what if", r"you"
    ]
    engagement_count = sum(1 for m in engagement_markers if re.search(m, text_lower))
    if engagement_count < 3 and len(words_list) >= 30:
        issues.append(("Low Audience Engagement", "The speech doesn't directly engage the audience."))
        suggestions.append("Ask rhetorical questions, use 'you' to address the audience directly, and invite them to imagine scenarios.")

    return issues, suggestions


# AI Rewrite Suggestions
def suggest_improvements(text, sentences):
    suggestions = []

    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped or len(s_stripped) < 10:
            continue
        s_lower = s_stripped.lower()

        if re.search(r"today\s+i(?:'m| am)?\s+(?:going to|will|want to)\s+(?:talk|speak|discuss|present)", s_lower):
            suggestions.append({
                "original": s_stripped[:80],
                "improved": "Start with a bold statement or question that grabs attention immediately."
            })
            break

    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped or len(s_stripped) < 10:
            continue
        s_lower = s_stripped.lower()

        if re.search(r'\bi\s+think\s+(that\s+)?', s_lower):
            improved = re.sub(r'\bi\s+think\s+(that\s+)?', '', s_stripped, count=1, flags=re.IGNORECASE).strip().capitalize()
            if len(improved) > 5:
                suggestions.append({
                    "original": s_stripped[:80],
                    "improved": improved[:80]
                })
                break

    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped or len(s_stripped) < 10:
            continue
        s_lower = s_stripped.lower()

        m = re.search(r'\bi\s+want\s+to\s+(talk|speak|discuss)\s+about', s_lower)
        if m:
            suggestions.append({
                "original": s_stripped[:80],
                "improved": "Make a direct statement about your topic instead of announcing your intention."
            })
            break

    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped or len(s_stripped) < 10:
            continue
        s_lower = s_stripped.lower()

        if re.search(r'\bvery\s+\w+', s_lower):
            suggestions.append({
                "original": s_stripped[:80],
                "improved": re.sub(r'\bvery\s+(\w+)', r'use a stronger word than \1', s_stripped, count=1, flags=re.IGNORECASE)[:80]
            })
            break

    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped or len(s_stripped) < 10:
            continue
        s_lower = s_stripped.lower()

        if re.search(r'\bjust\b', s_lower):
            improved = re.sub(r'\bjust\s+', '', s_stripped, count=1, flags=re.IGNORECASE).strip()
            if len(improved) > 5:
                suggestions.append({
                    "original": s_stripped[:80],
                    "improved": improved[:80]
                })
                break

    for s in sentences:
        s_stripped = s.strip()
        if len(s_stripped.split()) > 30:
            suggestions.append({
                "original": s_stripped[:80],
                "improved": "Break this into shorter sentences for better clarity and impact."
            })
            break

    return suggestions


# Audience Engagement Score
def calculate_engagement_score(text, wpm, word_count, sentences, flags):
    score = 5.0
    reasons = []

    words = text.split()
    if not words:
        return 1.0, ["No speech content to analyze."]

    sentence_starts = set()
    valid_sentences = [s for s in sentences if s.strip()]
    for s in valid_sentences:
        first_word = s.strip().split()[0].lower() if s.strip().split() else ""
        if first_word:
            sentence_starts.add(first_word)

    variety_ratio = len(sentence_starts) / max(len(valid_sentences), 1)
    if variety_ratio > 0.6:
        score += 1.5
        reasons.append("Good sentence variety — you vary how you start sentences.")
    elif variety_ratio > 0.4:
        score += 0.5
        reasons.append("Moderate sentence variety — try more diverse sentence openings.")
    else:
        score -= 1.0
        reasons.append("Low sentence variety — most sentences start the same way.")

    question_count = text.count("?")
    exclamation_count = text.count("!")
    if question_count >= 2:
        score += 1.0
        reasons.append("Good use of rhetorical questions to engage the audience.")
    if exclamation_count >= 2:
        score += 0.5
        reasons.append("Good use of emphasis with exclamatory phrases.")

    if 130 <= wpm <= 150:
        score += 1.5
        reasons.append("Excellent pacing — ideal speaking speed keeps the audience engaged.")
    elif 120 <= wpm <= 170:
        score += 0.5
        reasons.append("Good pacing — within comfortable speaking range.")
    elif wpm > 180:
        score -= 1.0
        reasons.append("Speaking too fast — audience may struggle to keep up.")
    elif wpm < 110:
        score -= 1.0
        reasons.append("Speaking too slow — energy and attention may drop.")

    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio > 0.65:
        score += 1.0
        reasons.append("Rich vocabulary — good use of diverse words.")
    elif unique_ratio < 0.5:
        score -= 1.0
        reasons.append("Too much word repetition — use more varied vocabulary.")

    structure_flags = [f for f in flags if f]
    if len(structure_flags) == 0:
        score += 1.0
        reasons.append("Good speech structure — clear organization and flow.")
    elif len(structure_flags) >= 3:
        score -= 1.0
        reasons.append("Speech structure needs improvement — work on openings, transitions, and conclusions.")

    score = max(1.0, min(10.0, score))
    return round(score, 1), reasons


def get_audio_suffix(audio_format, default="wav"):
    extension = str(audio_format or default).split("/")[-1].lower()
    extension = re.sub(r"[^a-z0-9]+", "", extension) or default
    return f".{extension}"


def write_temp_audio(audio_bytes, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def is_valid_audio_file(audio_path):
    try:
        return os.path.exists(audio_path) and os.path.getsize(audio_path) >= MIN_AUDIO_BYTES
    except OSError:
        return False


def analyze_audio(audio_path, audio_signature, source_label):
    global progress_file

    # Feature 1: Username is required before any analysis runs.
    if not safe_username:
        st.warning("Please enter your name before analysis.")
        return

    # Feature 1 and 2: Each user gets a separate local progress CSV.
    progress_file = progress_file or get_progress_file(safe_username)

    if not is_valid_audio_file(audio_path):
        st.warning("⚠️ Audio recording appears empty or too short. Please record again.")
        return

    stage_progress = st.progress(0, text="Preparing speech analysis...")
    stage_status = st.status("Analyzing speech", expanded=True)

    stage_status.write("\U0001f399\ufe0f Audio received")
    stage_progress.progress(10, text="\U0001f399\ufe0f Audio received")

    stage_status.update(label="\U0001f9e0 Transcribing speech...", state="running")
    stage_status.write("\U0001f9e0 Transcribing speech...")
    stage_progress.progress(30, text="\U0001f9e0 Transcribing speech...")

    # Changed: initialize cached Whisper model before transcription.
    try:
        model = load_model()
        result = model.transcribe(audio_path)
    except Exception:
        stage_status.update(label="⚠️ Speech analysis failed", state="error", expanded=True)
        stage_progress.progress(100, text="⚠️ Speech analysis failed")
        st.warning("⚠️ Speech analysis failed. Please try another recording.")
        return

    text = result.get("text", "").strip()
    language = result.get("language", "unknown")

    if not text or not result.get("segments"):
        stage_status.update(label="⚠️ Audio too short", state="error", expanded=True)
        stage_progress.progress(100, text="⚠️ Audio recording appears empty or too short")
        st.warning("⚠️ Audio recording appears empty or too short. Please record again.")
        return

    stage_status.update(label="\U0001f4ca Analyzing speaking patterns...", state="running")
    stage_status.write("\U0001f4ca Analyzing speaking patterns...")
    stage_progress.progress(50, text="\U0001f4ca Analyzing speaking patterns...")

    # ---------------- Analysis ----------------
    words = text.split()
    word_count = len(words)

    duration = result["segments"][-1]["end"] if result["segments"] else 1
    wpm = word_count / (duration / 60)

    # ---------------- Speech Quality Flags ----------------
    flags = []

    if "today i am going to talk" in text.lower():
        flags.append("Weak opening detected (too generic)")

    if text.count(".") < 3:
        flags.append("Speech may lack structure")

    if words and len(set(words)) / len(words) < 0.5:
        flags.append("Repetitive wording detected")

    # ---------------- Sentence Analysis ----------------
    sentences = re.split(r'[.!?]', text)
    weak_sentences = []

    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue

        if any(filler in s.lower() for filler in ["um", "uh", "like", "basically"]):
            weak_sentences.append(s)

    stage_status.update(label="\U0001f5e3\ufe0f Detecting filler words...", state="running")
    stage_status.write("\U0001f5e3\ufe0f Detecting filler words...")
    stage_progress.progress(70, text="\U0001f5e3\ufe0f Detecting filler words...")

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

    # Speech structure analysis
    structure_issues, structure_suggestions = analyze_speech_structure(text, sentences)

    # Rewrite suggestions
    rewrite_suggestions = suggest_improvements(text, sentences)

    # Engagement score
    engagement_score, engagement_reasons = calculate_engagement_score(text, wpm, word_count, sentences, flags)

    stage_status.update(label="\U0001f916 Generating AI feedback...", state="running")
    stage_status.write("\U0001f916 Generating AI feedback...")
    stage_progress.progress(85, text="\U0001f916 Generating AI feedback...")

    # ---------------- Feedback Engine ----------------
    feedback = []

    # Speed feedback
    if wpm > 170:
        feedback.append("You are speaking too fast. Try slowing down for clarity.")
    elif wpm < 120:
        feedback.append("You are speaking too slow. Add more energy and flow.")
    else:
        feedback.append("Good speaking pace.")

    # Filler feedback
    if filler_count > 5:
        feedback.append("Too many filler words. Practice cleaner delivery.")
    else:
        feedback.append("Good fluency.")

    # Language-aware suggestion
    if language != "en":
        feedback.append("Try mixing a bit more English for wider audience reach.")

    # Confidence estimation (basic logic)
    confidence_score = 10 - (filler_count * 0.5) - abs(wpm - 140) / 20
    confidence_score = max(1, min(10, confidence_score))
    feedback.append(f"Confidence Score: {round(confidence_score, 1)}/10")

    # Feature 4: Overall performance label.
    performance_label = get_performance_label(confidence_score)

    # Feature 2: Save user-specific progress after each completed analysis.
    progress_signature = (
        safe_username,
        source_label,
        audio_signature,
        word_count,
        round(wpm, 2),
        filler_count,
        round(confidence_score, 1),
    )

    if st.session_state.get("last_progress_signature") != progress_signature:
        save_progress(progress_file, word_count, wpm, filler_count, confidence_score)
        st.session_state["last_progress_signature"] = progress_signature

    progress_df = load_progress(progress_file)

    stage_status.write("\u2705 Analysis complete")
    stage_status.update(label="\u2705 Analysis complete", state="complete", expanded=False)
    stage_progress.progress(100, text="\u2705 Analysis complete")

    # Feature 5: Better page organization with tabs instead of one long results page.
    st.divider()
    analysis_tab, ai_feedback_tab, progress_tab = st.tabs([
        "Analysis",
        "AI Feedback",
        "Progress"
    ])

    widget_key = hashlib.sha1(str(progress_signature).encode("utf-8")).hexdigest()[:12]

    with analysis_tab:
        st.subheader("Transcription")
        st.write(text)
        st.write(f"Detected Language: **{language.upper()}**")

        st.subheader("Analysis")
        st.write(f"Total Words: {word_count}")
        st.write(f"Speaking Speed: {round(wpm, 2)} WPM")
        st.write(f"Filler Words Used: {filler_count}")
        # Feature 4: Display overall performance based on the current confidence score.
        st.write(f"Overall Performance: {performance_label}")

        # Audience Engagement Score
        st.subheader("🔥 Audience Engagement Score")
        st.write(f"**{engagement_score}/10**")
        for reason in engagement_reasons:
            st.write(f"- {reason}")

        # Speech Structure Review
        st.subheader("🎯 Speech Structure Review")
        if structure_issues:
            for issue_type, issue_desc in structure_issues:
                st.write(f"**{issue_type}:** {issue_desc}")
            st.markdown("**Suggestions:**")
            for sug in structure_suggestions:
                st.write(f"- {sug}")
        else:
            st.write("No major structural issues detected. Your speech has good structure and flow.")

        # Suggested Improvements
        if rewrite_suggestions:
            st.subheader("✍️ Suggested Improvements")
            for r in rewrite_suggestions:
                st.write(f"**Weak:** \"{r['original']}\"")
                st.write(f"**Improved:** {r['improved']}")
                st.markdown("---")

        st.subheader("Speech Quality Flags")
        if flags:
            for f in flags:
                st.write(f)
        else:
            st.write("No major structural issues detected.")

        st.subheader("Sentence Review")
        if weak_sentences:
            for sentence in weak_sentences:
                st.write(f"Weak sentence: {sentence}")
        else:
            st.write("No obvious weak sentences detected.")

        st.subheader("Filler Word Highlight")
        st.markdown(highlighted_text)

        st.subheader("Feedback")
        for f in feedback:
            st.write(f)

        detected_fillers = [w for w in speech_words if w in filler_words]
        st.write(f"Detected filler words: {', '.join(detected_fillers)}")

    with ai_feedback_tab:
        st.subheader("AI Feedback")

        if st.button("Generate AI Feedback", key=f"ai_feedback_{widget_key}"):
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

    with progress_tab:
        st.subheader("Progress History")
        st.caption(f"Showing progress for {username.strip()}")
        st.dataframe(progress_df.tail(5), width="stretch")

        st.subheader("Improvement Metrics")
        for message in calculate_improvement_messages(progress_df):
            st.write(message)

        # Feature 3: Analytics charts using Streamlit native line charts.
        st.subheader("Analytics Trends")
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


# ---------------- UI ----------------
st.set_page_config(page_title="Clarityn", layout="centered")

st.title("Clarityn")
st.caption("AI-powered speaking intelligence.")
# Feature 5: Better page organization with clearer top controls.
st.write("Upload or record your speech and receive detailed AI-powered speaking analysis, coaching feedback, and progress tracking.")
st.divider()

# Feature 1: Lightweight user profile input near the top, without auth or passwords.
username = st.text_input("Enter your name")
safe_username = sanitize_username(username)
progress_file = None

if username and not safe_username:
    st.warning("Please enter at least one letter or number for your name.")
elif safe_username:
    # Feature 1: Automatically create the user's progress file if it does not exist.
    progress_file = get_progress_file(safe_username)
    ensure_progress_file(progress_file)

# Coaching style selector near the top of the UI.
feedback_mode = st.selectbox(
    "Coaching Style",
    ["Supportive Coach", "Strict Coach", "Debate Coach", "Interview Coach", "TED-style Coach"]
)
st.divider()

# ---------------- Audio Input ----------------
upload_tab, record_tab = st.tabs(["Upload Audio", "Record Live"])

with upload_tab:
    # Feature 5: Better page organization starts with a focused upload section.
    st.subheader("Upload Speech")
    audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])

    if audio_file:
        audio_bytes = audio_file.getvalue()
        audio_path = write_temp_audio(audio_bytes, os.path.splitext(audio_file.name)[1] or ".audio")
        try:
            analyze_audio(
                audio_path,
                (audio_file.name, getattr(audio_file, "size", len(audio_bytes))),
                "upload"
            )
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

with record_tab:
    st.subheader("🎙️ Record Live Speech")

    if mic_recorder is None:
        st.error("Live recording requires the streamlit-mic-recorder package. Add it to requirements.txt and install it locally.")
    else:
        recorded_audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=True,
            key="live_speech_recorder"
        )

        if recorded_audio and recorded_audio.get("bytes"):
            recorded_bytes = recorded_audio["bytes"]
            st.audio(recorded_bytes)

            audio_hash = hashlib.sha1(recorded_bytes).hexdigest()
            suffix = get_audio_suffix(recorded_audio.get("format", "wav"))
            audio_path = write_temp_audio(recorded_bytes, suffix)

            try:
                analyze_audio(audio_path, audio_hash, "recording")
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

st.divider()
