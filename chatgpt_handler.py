import os
import json
import time
from typing import Dict, List, Optional
from openai import OpenAI
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from config import (
    CHATGPT_MODE, OPENAI_API_KEY, OPENAI_MODEL,
    PROMPTS_DIR, RESPONSES_DIR, MAX_RETRIES, TIMEOUT, OUTPUT_DIR,
    DEFAULT_FORECAST_STEPS
)
import logging

class ChatGPTHandler:
    def __init__(self):
        self.mode = CHATGPT_MODE
        print(self.mode)
        self.client = None
        if self.mode == "API":
            if not OPENAI_API_KEY:
                print(OPENAI_API_KEY)
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _create_timestamped_dir(self, ticker: str) -> str:
        """
        Erstellt ein zeitstempelbasiertes Verzeichnis für die Speicherung von Prompts und Antworten.
        
        :param ticker: Ticker-Symbol
        :return: Pfad zum erstellten Verzeichnis
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        dir_name = f"adoption_{ticker}_forecast{DEFAULT_FORECAST_STEPS}days_{timestamp}"
        dir_path = os.path.join(OUTPUT_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def _copy_historical_data(self, ticker: str, target_dir: str):
        """
        Kopiert historische Daten in das Zielverzeichnis.
        
        :param ticker: Ticker-Symbol
        :param target_dir: Zielverzeichnis-Pfad
        """
        source_file = os.path.join(OUTPUT_DIR, f"{ticker}_historical_data.csv")
        if os.path.exists(source_file):
            target_file = os.path.join(target_dir, f"{ticker}_historical_data.csv")
            pd.read_csv(source_file).to_csv(target_file, index=False)
            print(f"Copied historical data to {target_file}")

    def _select_file(self, prompt, current_dir):
        """
        Findet Antwortdateien im aktuellen Verzeichnis.
        
        :param prompt: str, anzuzeigender Prompt (nur für Logging verwendet)
        :param current_dir: str, aktueller Verzeichnispfad
        :return: str, ausgewählter Dateipfad
        """
        try:
            # Prompt-Nummern zu erwarteten Dateinamen zuordnen
            filename_map = {
                "Select Response 1 File": "analysis.json",
                "Select Response 2 File": "general.json",
                "Select Response 3 File": "historical.json"
            }
            
            expected_filename = filename_map.get(prompt)
            if not expected_filename:
                raise ValueError(f"Unknown prompt type: {prompt}")
            
            # Nach der Datei im aktuellen Verzeichnis suchen
            file_path = os.path.join(current_dir, expected_filename)
            
            if not os.path.exists(file_path):
                logging.warning(f"Expected file {expected_filename} not found in {current_dir}")
                # Fallback zu manueller Eingabe
                return input(f"{prompt} (or press Enter to skip): ").strip()
            
            logging.info(f"Found response file: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error in file selection: {str(e)}", exc_info=True)
            # Fallback zu manueller Eingabe
            return input(f"{prompt} (or press Enter to skip): ").strip()

    def process_prompts(self, prompts: List[str], ticker: str) -> Dict:
        """
        Verarbeitet Prompts entweder manuell oder über API basierend auf dem konfigurierten Modus.
        
        :param prompts: Liste der zu verarbeitenden Prompts
        :param ticker: Ticker-Symbol für Dateibenennung
        :return: Dictionary mit den Antworten
        """
        if self.mode == "MANUAL":
            return self._process_manual(prompts, ticker)
        else:
            if not self.client:
                raise ValueError("OpenAI client not initialized. Please check your API configuration.")
            return self._process_api(prompts, ticker)

    def _process_manual(self, prompts: List[str], ticker: str) -> Dict:
        """
        Behandelt manuelle Verarbeitung durch Speichern von Prompts und Warten auf Benutzereingabe.
        
        :param prompts: Liste der zu verarbeitenden Prompts
        :param ticker: Ticker-Symbol für Dateibenennung
        :return: Dictionary mit den Antworten
        """
        # Zeitstempelbasiertes Verzeichnis für diesen Lauf erstellen
        current_dir = self._create_timestamped_dir(ticker)
        
        # Historische Daten in das neue Verzeichnis kopieren
        self._copy_historical_data(ticker, current_dir)
        
        # Pfadbeschreibungen aus dem ersten Prompt extrahieren
        path_analysis_prompt = prompts[0]
        paths_descriptions = path_analysis_prompt.split("Here are the paths to analyze:")[1].strip()
        
        # Pfadbeschreibungen in eine separate Datei speichern
        paths_filename = os.path.join(current_dir, f"{ticker}_paths.txt")
        with open(paths_filename, 'w') as f:
            f.write(paths_descriptions)
        
        # Neuen Prompt erstellen der auf die Pfad-Datei verweist
        new_path_analysis_prompt = path_analysis_prompt.split("Here are the paths to analyze:")[0].strip()
        new_path_analysis_prompt += f"\n\nPlease analyze the paths described in the file '{ticker}_paths.txt'."
        
        # Alle Prompts in die Hauptdatei speichern
        prompts_filename = os.path.join(current_dir, f"{ticker}_prompts.txt")
        with open(prompts_filename, 'w') as f:
            f.write("=== Path Analysis Prompt ===\n")
            f.write(new_path_analysis_prompt)
            f.write("\n\n=== General Forecast Prompt ===\n")
            f.write(prompts[1])
            f.write("\n\n=== Historical Forecast Prompt ===\n")
            f.write(prompts[2])
        
        print(f"Saved path descriptions to {paths_filename}")
        print(f"Saved prompts to {prompts_filename}")
        print("\nPlease use these prompts in ChatGPT and save the responses in JSON format.")
        print("After getting the responses, press Enter to continue...")
        input()

        # Antworten laden
        responses = {}
        for i in range(len(prompts)):
            print(f"\nPlease select the file containing response {i+1}:")
            file_path = self._select_file(f"Select Response {i+1} File", current_dir)
            if not file_path:
                raise ValueError(f"No file selected for response {i+1}")
            
            with open(file_path, 'r') as f:
                response = json.load(f)
            responses[f"response_{i+1}"] = response

        return responses

    def _process_api(self, prompts: List[str], ticker: str) -> Dict:
        """
        Verarbeitet Prompts über die OpenAI API.
        
        :param prompts: Liste der zu verarbeitenden Prompts
        :param ticker: Ticker-Symbol für Dateibenennung
        :return: Dictionary mit den Antworten
        """
        responses = {}
        
        # Zeitstempelbasiertes Verzeichnis für diesen Lauf erstellen
        current_dir = self._create_timestamped_dir(ticker)
        
        # Historische Daten in das neue Verzeichnis kopieren
        self._copy_historical_data(ticker, current_dir)
        
        # Pfadbeschreibungen aus dem ersten Prompt extrahieren
        path_analysis_prompt = prompts[0]
        paths_descriptions = path_analysis_prompt.split("Here are the paths to analyze:")[1].strip()
        
        # Pfadbeschreibungen in eine separate Datei für Referenz speichern
        paths_filename = os.path.join(current_dir, f"{ticker}_paths.txt")
        with open(paths_filename, 'w') as f:
            f.write(paths_descriptions)
        
        # Historische Daten lesen
        historical_data_file = os.path.join(current_dir, f"{ticker}_historical_data.csv")
        historical_data = pd.read_csv(historical_data_file)
        historical_data_str = historical_data.to_string()
        
        # Jeden Prompt verarbeiten
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1} via API...")
            
            # Prompts modifizieren um tatsächliche Daten statt Dateireferenzen zu verwenden
            if i == 0:  # Pfad-Analyse-Prompt
                current_prompt = path_analysis_prompt  # Ursprünglichen Prompt mit Pfaden verwenden
            elif i == 2:  # Historischer Prognose-Prompt
                current_prompt = prompt.replace(
                    f"the file {ticker}_historical_data.csv",
                    "the following historical data:\n\n" + historical_data_str
                )
            else:
                current_prompt = prompt
            
            # Prompt für Referenz speichern
            prompt_filename = os.path.join(current_dir, f"{ticker}_prompt_{i+1}.txt")
            with open(prompt_filename, 'w') as f:
                f.write(current_prompt)

            # Mit Wiederholungen verarbeiten
            for attempt in range(MAX_RETRIES):
                try:
                    response = self._make_api_request(current_prompt)
                    responses[f"response_{i+1}"] = response
                    
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise Exception(f"Failed to process prompt {i+1} after {MAX_RETRIES} attempts: {str(e)}")
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponentieller Backoff

        return responses

    def _make_api_request(self, prompt: str) -> Dict:
        """
        Macht eine Anfrage an die OpenAI API.
        
        :param prompt: An die API zu sendender Prompt
        :return: API-Antwort
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please check your API configuration.")
            
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in stock market forecasting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Antwort extrahieren und parsen
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")