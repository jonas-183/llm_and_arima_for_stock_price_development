import os
from dotenv import load_dotenv

# Umgebungsvariablen aus .env Datei laden
load_dotenv()

# ChatGPT Verarbeitungsmodus
# Optionen: "MANUAL" oder "API"
CHATGPT_MODE = os.getenv("CHATGPT_MODE", "MANUAL")

# OpenAI API Konfiguration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")  # oder "gpt-3.5-turbo"

# Dateipfade
OUTPUT_DIR = "output"
PROMPTS_DIR = "prompts"
RESPONSES_DIR = "responses"

# Verzeichnisse erstellen falls sie nicht existieren
for directory in [OUTPUT_DIR, PROMPTS_DIR, RESPONSES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Modell-Parameter
DEFAULT_FORECAST_STEPS = 1
DEFAULT_N_PATHS = 100
DEFAULT_SHOW_LAST_DAYS = 7

# API-Anfrage-Einstellungen
MAX_RETRIES = 3
TIMEOUT = 30  # Sekunden 