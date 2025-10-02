from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import os
import tkinter as tk
from tkinter import filedialog

def generate_path_description(path, ticker, start_date, end_date, last_historical_value):
    """
    Generiert eine flüssige narrative Beschreibung für einen Preis-Pfad mit täglichen Bewegungen.
    
    :param path: np.ndarray, der Preis-Pfad
    :param ticker: str, Name des Tickers
    :param start_date: pd.Timestamp, Startdatum des Pfades
    :param end_date: pd.Timestamp, Enddatum des Pfades
    :param last_historical_value: float, der letzte historische Preiswert
    :return: str, narrative Beschreibung des Preis-Pfades mit täglichen Änderungen
    """
    # Geschäftstage generieren (Wochenenden und Feiertage ausschließend)
    business_days = pd.date_range(start=start_date + pd.Timedelta(days=1), end=end_date, freq='B')
    
    # Feiertage entfernen (US-Feiertage als Beispiel verwenden - für Ihren Markt anpassen)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=business_days[0], end=business_days[-1])
    business_days = business_days[~business_days.isin(holidays)]
    
    # Sicherstellen dass wir genug Geschäftstage haben
    while len(business_days) < len(path) - 1:
        next_day = business_days[-1] + pd.Timedelta(days=1)
        if next_day.weekday() < 5 and next_day not in holidays:
            business_days = business_days.append(pd.DatetimeIndex([next_day]))
    
    description = []
    current_date = start_date
    
    # Einzelschritt-Prognose behandeln
    if len(path) == 1:  # Nur Prognosepunkt
        change = ((path[0] - last_historical_value) / last_historical_value) * 100
        date = business_days[0] if len(business_days) > 0 else start_date + pd.Timedelta(days=1)
        
        # Wochenenden zwischen aktuellem und nächstem Geschäftstag prüfen
        days_between = (date - current_date).days
        if days_between > 1:
            weekend_days = []
            for d in range(1, days_between):
                check_date = current_date + pd.Timedelta(days=d)
                if check_date.weekday() >= 5:
                    weekend_days.append(check_date)
                elif check_date in holidays:
                    description.append(f"On {check_date.strftime('%d.%m.%Y')} the market is closed due to a holiday.")
            
            if weekend_days:
                weekend_text = " and ".join([d.strftime('%d.%m.%Y') for d in weekend_days])
                description.append(f"The next {len(weekend_days)} day(s) ({weekend_text}) are weekend days with no trading.")
        
        # Preisänderungsbeschreibung hinzufügen
        if change > 0:
            if change > 5:
                description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share shows a strong increase of {change:.2f}%.")
            elif change > 2:
                description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share rises by {change:.2f}%.")
            else:
                description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share gains {change:.2f}%.")
        else:
            if change < -5:
                description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share experiences a significant drop of {abs(change):.2f}%.")
            elif change < -2:
                description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share falls by {abs(change):.2f}%.")
            else:
                description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share declines by {abs(change):.2f}%.")
        
        return " ".join(description)
    
    # Mehrschritt-Prognose behandeln
    first_day = True
    previous_change = None
    
    for i, date in enumerate(business_days):
        if i < len(path) - 1:
            # Für den ersten Tag, Änderung vom letzten historischen Wert berechnen
            if first_day:
                change = ((path[i] - last_historical_value) / last_historical_value) * 100
            else:
                change = ((path[i+1] - path[i]) / path[i]) * 100
            
            # Wochenenden zwischen aktuellem und nächstem Geschäftstag prüfen
            days_between = (date - current_date).days
            if days_between > 1:
                weekend_days = []
                for d in range(1, days_between):
                    check_date = current_date + pd.Timedelta(days=d)
                    if check_date.weekday() >= 5:
                        weekend_days.append(check_date)
                    elif check_date in holidays:
                        description.append(f"On {check_date.strftime('%d.%m.%Y')} the market is closed due to a holiday.")
                
                if weekend_days:
                    weekend_text = " and ".join([d.strftime('%d.%m.%Y') for d in weekend_days])
                    description.append(f"The next {len(weekend_days)} day(s) ({weekend_text}) are weekend days with no trading.")
            
            # Preisänderungsbeschreibung mit natürlicherer Sprache hinzufügen
            if first_day:
                if change > 0:
                    if change > 5:
                        description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share shows a strong increase of {change:.2f}%.")
                    elif change > 2:
                        description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share rises by {change:.2f}%.")
                    else:
                        description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share gains {change:.2f}%.")
                else:
                    if change < -5:
                        description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share experiences a significant drop of {abs(change):.2f}%.")
                    elif change < -2:
                        description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share falls by {abs(change):.2f}%.")
                    else:
                        description.append(f"On {date.strftime('%d.%m.%Y')} the {ticker} share declines by {abs(change):.2f}%.")
                first_day = False
                previous_change = change
            else:
                if previous_change is not None:
                    # Zwei Bewegungen in einem Satz kombinieren
                    if previous_change > 0 and change > 0:
                        if previous_change > 5 or change > 5:
                            description.append(f"The {ticker} share continues its upward trend, gaining {previous_change:.2f}% and then {change:.2f}% the next day.")
                        else:
                            description.append(f"The {ticker} share rises by {previous_change:.2f}% and gains another {change:.2f}% the next day.")
                    elif previous_change < 0 and change < 0:
                        if previous_change < -5 or change < -5:
                            description.append(f"The {ticker} share continues its downward trend, falling {abs(previous_change):.2f}% and then {abs(change):.2f}% the next day.")
                        else:
                            description.append(f"The {ticker} share declines by {abs(previous_change):.2f}% and falls another {abs(change):.2f}% the next day.")
                    elif previous_change > 0 and change < 0:
                        description.append(f"After rising by {previous_change:.2f}%, the {ticker} share falls by {abs(change):.2f}% the next day.")
                    else:
                        description.append(f"After falling by {abs(previous_change):.2f}%, the {ticker} share gains {change:.2f}% the next day.")
                    previous_change = None
                else:
                    previous_change = change
            
            current_date = date
    
    # Letzte Bewegung behandeln falls eine übrig ist
    if previous_change is not None:
        if previous_change > 0:
            if previous_change > 5:
                description.append(f"The {ticker} share shows a strong increase of {previous_change:.2f}%.")
            elif previous_change > 2:
                description.append(f"The {ticker} share rises by {previous_change:.2f}%.")
            else:
                description.append(f"The {ticker} share gains {previous_change:.2f}%.")
        else:
            if previous_change < -5:
                description.append(f"The {ticker} share experiences a significant drop of {abs(previous_change):.2f}%.")
            elif previous_change < -2:
                description.append(f"The {ticker} share falls by {abs(previous_change):.2f}%.")
            else:
                description.append(f"The {ticker} share declines by {abs(previous_change):.2f}%.")
    
    return " ".join(description)

def generate_chatgpt_prompt(paths_descriptions, ticker):
    """
    Generiert einen Prompt für ChatGPT um die Pfade zu analysieren.
    
    :param paths_descriptions: str, Beschreibungen aller Preis-Pfade
    :param ticker: str, Name des Tickers
    :return: str, formatierter Prompt für ChatGPT
    """
    prompt = f"""
    IMPORTANT: You must analyze ALL paths provided and follow the EXACT format specified below.

    As an experienced financial analyst, please analyze the following {ticker} stock price paths.
    For each path, assess its probability on a scale of 1-10 (10 = very likely) based on:
    - Current market conditions
    - Historical price patterns
    - Technical indicators
    - Market sentiment
    - Economic factors

    CRITICAL REQUIREMENTS:
    1. You MUST analyze EVERY single path of the 100 paths provided
    2. You MUST provide a probability for EACH path
    3. You MUST follow the EXACT JSON format shown below
    4. Do NOT limit your analysis to just a part of the paths

    Format your response as downloadable JSON with the following structure:
    {{
        "analyses": [
            {{
                "path_id": 1,
                "probability": X,
                "reasoning": "Explanation"
            }},
            ...
        ]
    }}
    
    Here are the paths to analyze:
    {paths_descriptions}
    """
    return prompt

def generate_chatgpt_forecast_prompt(ticker, forecast_steps):
    """
    Generiert einen Prompt für ChatGPT um eine numerische Prognose zu erstellen.
    
    :param ticker: str, Name des Tickers
    :param forecast_steps: int, Anzahl der Tage zu prognostizieren
    :return: str, formatierter Prompt für ChatGPT
    """
    current_date = pd.Timestamp.now().strftime("%d.%m.%Y")
    prompt = f"""
    As an experienced financial analyst, please create a forecast for the {ticker} stock for the next {forecast_steps} trading days, including today ({current_date}).
    Important notes:
    - Only forecast for trading days (Monday to Friday)
    - No forecast is needed for weekends (Saturday and Sunday) as markets are closed
    - All prices should be closing prices (end of trading day)
    - Consider US market holidays when determining trading days

    Please provide a concrete numerical forecast based on your analysis of the current market situation. 
    Consider:
    - Current market trends
    - Economic conditions
    - Industry-specific factors
    - Technical and fundamental signals

    IMPORTANT: You must specify which statistical model or approach you used for your predictions.
    This could be:
    - Time series analysis (e.g., ARIMA, GARCH)
    - Machine learning models (e.g., LSTM, Random Forest)
    - Technical analysis indicators
    - Fundamental analysis metrics
    - Or any other quantitative approach

    Format your response as JSON with the following structure:
    {{
        "forecast": [
            {{
                "date": "DD.MM.YYYY",
                "closing_price": X.XX,
                "confidence": "high/medium/low",
                "reasoning": "Explanation"
            }},
            ...
        ],
        "model_used": {{
            "name": "Name of the statistical model or approach",
            "description": "Brief explanation of how the model works",
            "parameters": "Key parameters or indicators used",
            "rationale": "Why this model was chosen for this particular forecast"
        }}
    }}"""

    return prompt

def generate_chatgpt_forecast_prompt_historical(ticker, forecast_steps):
    """
    Generiert einen Prompt für ChatGPT um historische Daten zu analysieren und eine Prognose zu erstellen.
    
    :param ticker: str, Name des Tickers
    :param forecast_steps: int, Anzahl der Tage zu prognostizieren
    :return: str, formatierter Prompt für ChatGPT
    """
    current_date = pd.Timestamp.now().strftime("%d.%m.%Y")
    prompt = f"""
    As an experienced financial analyst, please create a forecast for the {ticker} stock for the next {forecast_steps} trading days, including today ({current_date}) and using the historical data provided in the file {ticker}_historical_data.csv.
    Important notes:
    - Only forecast for trading days (Monday to Friday)
    - No forecast is needed for weekends (Saturday and Sunday) as markets are closed
    - All prices should be closing prices (end of trading day)
    - Consider US market holidays when determining trading days

    Please provide a concrete numerical forecast based on your analysis of the current market situation. 
    Consider:
    - Current market trends
    - Economic conditions
    - Industry-specific factors
    - Technical and fundamental signals
    - REALLY IMPORTANT: Use the historical data provided in the file {ticker}_historical_data.csv to create the forecast.

    IMPORTANT: You must specify which statistical model or approach you used for your predictions.
    This could be:
    - Time series analysis (e.g., ARIMA, GARCH)
    - Machine learning models (e.g., LSTM, Random Forest)
    - Technical analysis indicators
    - Fundamental analysis metrics
    - Or any other quantitative approach

    Format your response as JSON with the following structure:
    {{
        "forecast": [
            {{
                "date": "DD.MM.YYYY",
                "closing_price": X.XX,
                "confidence": "high/medium/low",
                "reasoning": "Explanation"
            }},
            ...
        ],
        "model_used": {{
            "name": "Name of the statistical model or approach",
            "description": "Brief explanation of how the model works",
            "parameters": "Key parameters or indicators used",
            "rationale": "Why this model was chosen for this particular forecast"
        }}
    }}"""

    return prompt

def save_prompts_to_file(prompts, filename):
    """
    Speichert Prompts und Pfadbeschreibungen in separate Dateien für manuelle ChatGPT-Interaktion.
    
    :param prompts: list of str, zu speichernde Prompts
    :param filename: str, Basisname für die Dateien
    :return: None
    """
    # Pfadbeschreibungen aus dem ersten Prompt extrahieren
    path_analysis_prompt = prompts[0]
    paths_descriptions = path_analysis_prompt.split("Here are the paths to analyze:")[1].strip()
    
    # Pfadbeschreibungen in eine separate Datei speichern
    paths_filename = filename.split(".")[0] + "_paths.txt"
    with open(paths_filename, 'w') as f:
        f.write(paths_descriptions)
    
    # Neuen Prompt erstellen der auf die Pfad-Datei verweist
    new_path_analysis_prompt = path_analysis_prompt.split("Here are the paths to analyze:")[0].strip()
    new_path_analysis_prompt += f"\n\nPlease analyze the paths described in the file '{paths_filename}'."
    
    # Alle Prompts in die Hauptdatei speichern
    with open(filename, 'w') as f:
        f.write("=== Path Analysis Prompt ===\n")
        f.write(new_path_analysis_prompt)
        f.write("\n\n=== General Forecast Prompt ===\n")
        f.write(prompts[1])
        f.write("\n\n=== Historical Forecast Prompt ===\n")
        f.write(prompts[2])
    
    print(f"Saved path descriptions to {paths_filename}")
    print(f"Saved prompts to {filename}")

def load_chatgpt_responses(filename=None):
    """
    Lädt ChatGPT-Antworten aus einer JSON-Datei.
    Falls kein Dateiname angegeben ist, kann der Benutzer eine Datei auswählen.
    
    :param filename: str, optional, Name der Datei mit den Antworten
    :return: dict, geladene JSON-Antworten
    """
    if filename is None:
        # Dateidialog erstellen um die JSON-Datei auszuwählen
        root = tk.Tk()
        root.withdraw()  # Hauptfenster ausblenden
        
        # Aktuelles Verzeichnis abrufen
        current_dir = os.getcwd()
        
        # Dateidialog anzeigen
        file_path = filedialog.askopenfilename(
            initialdir=current_dir,
            title="Select ChatGPT Response File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:  # Benutzer hat Dialog abgebrochen
            raise FileNotFoundError("No file was selected")
            
        filename = file_path
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {filename}")
    except json.JSONDecodeError:
        raise ValueError(f"The file {filename} does not contain valid JSON data")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {str(e)}")