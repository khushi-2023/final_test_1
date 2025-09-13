# main.py
"""
Voice-Guided Git Automation (single-file)
Features:
 - Hybrid ASR: Vosk (offline, streaming) + Google (online) + Whisper fallback (tiny/base)
 - Voice authentication (optional)
 - Fast immediate execution of Git commands (add, commit, push, branch, checkout, status)
 - Auto-detect current git branch and push to origin/<branch>
 - Logs to logs/commands.txt and command_log.txt
"""
import os, sys, time, re, json, wave, subprocess, threading, datetime
import numpy as np

# ASR libraries
try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = None
import pyaudio, sounddevice as sd, speech_recognition as sr

# Whisper (fallback)
try:
    import whisper
except Exception:
    whisper = None

# TTS & audio processing
import pyttsx3
import librosa

# NLP (simple regex-based; spaCy optional)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# ---------------------- CONFIG ----------------------
SAMPLE_RATE = 16000
VOSK_MODEL_PATH = os.path.join(os.getcwd(), "models", "vosk-model-small-en-us-0.15")
WHISPER_MODEL = "tiny"   # tiny | base | small ; tiny is fastest
VOICE_AUTH_REF = os.path.join("auth", "voice_ref.wav")
LOGS_DIR = "logs"
COMMAND_LOG = "command_log.txt"
ENABLE_VOICE_AUTH = False   # set True to require voice auth for push/commit
# ----------------------------------------------------

engine = pyttsx3.init()
def speak(text):
    print("🗣️", text)
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# make logs dir
os.makedirs(LOGS_DIR, exist_ok=True)

def append_log(line):
    ts = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    with open(os.path.join(LOGS_DIR, "commands.txt"), "a", encoding="utf-8") as f:
        f.write(f"{ts} - {line}\n")
    with open(COMMAND_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------------- ASR INIT ----------------------
vosk_model = None
if Model is not None and os.path.exists(VOSK_MODEL_PATH):
    try:
        vosk_model = Model(VOSK_MODEL_PATH)
        print("✅ Vosk model loaded.")
    except Exception as e:
        print("⚠️ Failed to load Vosk model:", e)

whisper_model = None
if whisper is not None:
    try:
        # load whisper model once (tiny by default for speed)
        whisper_model = whisper.load_model(WHISPER_MODEL)
        print(f"✅ Whisper model '{WHISPER_MODEL}' loaded.")
    except Exception as e:
        print("⚠️ Whisper load failed:", e)

# ---------------------- UTIL: record small snippet ----------------------
def record_wav(filename="fallback.wav", duration=3, fs=SAMPLE_RATE):
    """Record short snippet (used for fallback or voice auth)."""
    speak("Recording audio...")
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    data = rec.tobytes()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(data)
    speak("Recording saved.")
    return filename

# ---------------------- OFFLINE ASR (Vosk streaming) ----------------------
def offline_asr_vosk(timeout=6):
    """Listen via PyAudio -> Vosk streaming. Returns recognized text or ''."""
    if vosk_model is None:
        return ""
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=SAMPLE_RATE,
                         input=True,
                         frames_per_buffer=4096)
        stream.start_stream()
        speak("Listening (offline)... speak now.")
        start = time.time()
        text = ""
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text = res.get("text", "").strip()
                break
            # stop if timeout
            if time.time() - start > timeout:
                # try partial
                pres = json.loads(rec.PartialResult())
                text = pres.get("partial", "").strip()
                break
    except Exception as e:
        print("Vosk streaming error:", e)
        text = ""
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
    return text

# ---------------------- CLOUD (Google) ASR ----------------------
def cloud_asr_google(timeout=6):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        speak("Listening (online)... speak now.")
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=8)
        except Exception as e:
            print("Listen error:", e)
            return ""
    try:
        text = r.recognize_google(audio)
        return text.strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("Google API error:", e)
        return ""

# ---------------------- WHISPER fallback ----------------------
def whisper_transcribe_from_file(filename="fallback.wav"):
    if whisper_model is None:
        return ""
    try:
        res = whisper_model.transcribe(filename)
        return res.get("text", "").strip()
    except Exception as e:
        print("Whisper error:", e)
        return ""

# ---------------------- HYBRID ASR ----------------------
def hybrid_listen(mode="offline"):
    """Return recognized text (lowercase) or ''."""
    text = ""

    if mode == "offline":
        # Vosk first
        text = offline_asr_vosk(timeout=5)
        if not text:
            # fallback → Whisper
            record_wav("fallback.wav", duration=3)
            text = whisper_transcribe_from_file("fallback.wav")

    elif mode == "online":
        # Google only (no fallback)
        text = cloud_asr_google(timeout=6)

    else:
        # default = whisper only
        record_wav("fallback.wav", duration=3)
        text = whisper_transcribe_from_file("fallback.wav")

    return (text or "").lower().strip()


# ---------------------- VOICE AUTH (simple MFCC) ----------------------
def extract_mfcc(file):
    y, sr_ = librosa.load(file, sr=SAMPLE_RATE)
    mf = np.mean(librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=13), axis=1)
    return mf

def authenticate_via_voice(timeout=3, threshold=55.0):
    if not os.path.exists(VOICE_AUTH_REF):
        print("Voice reference not found; skipping voice auth.")
        return False
    path = record_wav("auth_tmp.wav", duration=timeout)
    try:
        ref = extract_mfcc(VOICE_AUTH_REF)
        usr = extract_mfcc(path)
        dist = np.linalg.norm(ref - usr)
        print("Voice distance:", dist)
        return dist < threshold
    except Exception as e:
        print("Voice auth error:", e)
        return False

# ---------------------- NLP -> git command parser ----------------------
def parse_git_command(text):
    text = text.lower()
    # commit message "..." or commit message '...'
    m = re.search(r"commit (?:message )?[\"'](.+?)[\"']", text)
    if "commit" in text and m:
        message = m.group(1)
        return ["git add -A", f'git commit -m "{message}"']
    if "commit" in text:
        # default message
        return ["git add -A", 'git commit -m "voice commit"']
    if "push" in text:
        return ["git push"]
    if "pull" in text:
        return ["git pull"]
    if "status" in text:
        return ["git status"]
    if "create branch" in text:
        b = re.search(r"create branch ([\w\-_]+)", text)
        if b:
            bn = b.group(1)
            return [f"git branch {bn}", f"git checkout {bn}"]
        return ["Unknown command"]
    if "switch to" in text or "checkout" in text:
        b = re.search(r"(?:switch to|checkout) (?:branch )?([\w\-_]+)", text)
        if b:
            return [f"git checkout {b.group(1)}"]
    if "undo last commit" in text or "revert last commit" in text or "undo" in text:
        return ["git reset --soft HEAD~1"]
    return ["Unknown command"]

# ---------------------- GIT HELPERS ----------------------
def is_git_repo():
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return res.returncode == 0

def current_branch():
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip()
    return "main"

def run_cmd(cmd):
    """Run shell command list or string; return (rc, stdout+stderr)"""
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    print("⚡ Running:", cmd)
    res = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    print(out)
    return res.returncode, out

def execute_git_sequence(cmds):
    if not is_git_repo():
        speak("Not a Git repository. Please run 'git init' or set the correct folder.")
        return
    for c in cmds:
        rc, out = run_cmd(c)
        if rc != 0:
            speak(f"Command failed: {c.split()[0]}")
            append_log(f"FAILED: {c} -> {out}")
            return False
    return True

def push_current_branch():
    br = current_branch()
    # use safe push: origin <branch>
    rc, out = run_cmd(["git", "push", "origin", br])
    if rc == 0:
        speak(f"Pushed to origin {br}")
        append_log(f"push -> origin/{br}")
        return True
    else:
        speak("Push failed. Check credentials or remote.")
        append_log(f"PUSH FAILED -> origin/{br} : {out}")
        return False

# ---------------------- MAIN LOOP ----------------------
def main():
    speak("Voice Git Automation Started. Say 'offline' or 'online' to choose ASR mode.")
    # initial selection
    mode = "offline"
    attempts = 0
    while attempts < 4:
        text = hybrid_listen(mode="online")  # first try quick online selection (can pick Google)
        if not text:
            attempts += 1
            continue
        if "offline" in text:
            mode = "offline"
            speak("Selected offline mode.")
            break
        if "online" in text:
            mode = "online"
            speak("Selected online mode.")
            break
        attempts += 1
        speak("Please say 'offline' or 'online' to select ASR mode.")
    speak(f"Using {mode} ASR mode. Say a git command, or say 'exit' to stop.")

    try:
        while True:
            text = hybrid_listen(mode=mode)
            if not text:
                continue
            append_log(text)
            print("Recognized:", text)

            # mode switching voice commands
            if "switch to offline" in text:
                mode = "offline"
                speak("Switched to offline mode.")
                continue
            if "switch to online" in text:
                mode = "online"
                speak("Switched to online mode.")
                continue
            if text.strip() in ("exit", "quit", "stop"):
                speak("Exiting. Goodbye.")
                break

            cmds = parse_git_command(text)
            if cmds[0] == "Unknown command":
                speak("Unknown Git command. Try: commit, push, pull, status, create branch, switch to ...")
                continue

            # require voice auth for sensitive ops if enabled
            if ENABLE_VOICE_AUTH and any(k in " ".join(cmds) for k in ("commit", "push", "reset", "branch")):
                speak("Please authenticate with voice.")
                ok = authenticate_via_voice(timeout=3)
                if not ok:
                    speak("Authentication failed.")
                    append_log("AUTH FAIL for command: " + " | ".join(cmds))
                    continue
                speak("Authentication passed.")

            # fast immediate execution
            speak(f"Executing {', '.join(cmds)}")
            ok = execute_git_sequence(cmds)
            if ok and any("push" in c for c in cmds):
                # if user said push, do push_current_branch() to ensure origin/branch
                push_current_branch()
            elif ok and any("commit" in c for c in cmds):
                # after commit, optionally push automatically
                speak("Do you want to push? say 'push' to push now or continue.")
            append_log("EXECUTED: " + " | ".join(cmds))

    except KeyboardInterrupt:
        speak("Stopped by keyboard.")
    except Exception as e:
        print("Main loop error:", e)
    finally:
        try:
            engine.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
