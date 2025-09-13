import streamlit as st
import os

# Paths to log files
COMMAND_LOG_PATH = "command_log.txt"
LOGS_DIR = "logs"
LOGS_FILE_PATH = os.path.join(LOGS_DIR, "commands.txt")

st.set_page_config(page_title="Voice-Guided Git Automation", layout="wide")

st.title("🗣️ Voice-Guided Git Automation Dashboard")
st.markdown("### ✅ Real-time log view of your Git voice commands")

# --- Load logs ---
def load_log_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "⚠️ No logs available yet."

# Display Command Log
st.subheader("📌 Command Execution Log (command_log.txt)")
command_log = load_log_file(COMMAND_LOG_PATH)
st.text_area("Command Log", command_log, height=300)

# Display Detailed Logs
st.subheader("📝 Detailed Logs (logs/commands.txt)")
detailed_log = load_log_file(LOGS_FILE_PATH)
st.text_area("Detailed Log", detailed_log, height=300)

# Refresh button
if st.button("🔄 Refresh Logs"):
    st.rerun()
