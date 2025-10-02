import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from datetime import datetime

def select_folder():
    """
    Öffnet einen Ordner-Dialog um ein Verzeichnis mit adoption.json Datei auszuwählen.
    
    :return: str, ausgewählter Ordner-Pfad oder None falls abgebrochen
    """
    root = tk.Tk()
    root.withdraw()  # Hauptfenster ausblenden
    
    folder_path = filedialog.askdirectory(
        title="Select folder containing adoption.json file"
    )
    
    if not folder_path:
        print("No folder selected. Exiting.")
        return None
    
    return folder_path

def find_adoption_file(folder_path):
    """
    Findet die adoption.json Datei im angegebenen Ordner.
    
    :param folder_path: str, Pfad zum Ordner
    :return: str, Pfad zur adoption.json Datei oder None falls nicht gefunden
    """
    adoption_file = os.path.join(folder_path, "adoption.json")
    
    if os.path.exists(adoption_file):
        return adoption_file
    else:
        print(f"adoption.json not found in {folder_path}")
        return None

def find_forecasts_file(folder_path):
    """
    Findet die Prognosedatei im angegebenen Ordner.
    
    :param folder_path: str, Pfad zum Ordner
    :return: str, Pfad zur Prognosedatei oder None falls nicht gefunden
    """
    # Nach verschiedenen möglichen Prognosedatei-Namen suchen
    possible_names = [
        "*_forecasts.json",
        "*_forecasts_fixed.json"
    ]
    
    for pattern in possible_names:
        import glob
        files = glob.glob(os.path.join(folder_path, pattern))
        if files:
            return files[0]  # Ersten Treffer zurückgeben
    
    print(f"No forecasts file found in {folder_path}")
    return None

def find_csv_file(folder_path):
    """
    Findet die CSV-Datei mit Simulationspfaden im angegebenen Ordner.
    
    :param folder_path: str, Pfad zum Ordner
    :return: str, Pfad zur CSV-Datei oder None falls nicht gefunden
    """
    # Nach CSV-Dateien im Ordner suchen
    import glob
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if csv_files:
        # Falls mehrere CSV-Dateien, bevorzuge eine mit 'paths' oder 'simulation' im Namen
        for csv_file in csv_files:
            filename = os.path.basename(csv_file).lower()
            if 'path' in filename or 'simulation' in filename or 'simulated' in filename:
                return csv_file
        # Falls kein spezifischer Treffer, erste CSV-Datei zurückgeben
        return csv_files[0]
    
    print(f"No CSV file found in {folder_path}")
    return None

def load_simulation_paths_from_csv(csv_file_path):
    """
    Lädt Simulationspfade aus einer CSV-Datei.
    
    :param csv_file_path: str, Pfad zur CSV-Datei
    :return: np.ndarray, Array der Simulationspfade
    """
    import pandas as pd
    
    try:
        # CSV-Datei lesen
        df = pd.read_csv(csv_file_path)
        
        # Zu numpy Array konvertieren
        # Annahme: Jede Zeile repräsentiert einen Pfad und Spalten repräsentieren Zeitschritte
        paths = df.values
        
        print(f"Loaded {len(paths)} simulation paths from CSV")
        print(f"Each path has {len(paths[0]) if len(paths) > 0 else 0} time steps")
        
        return paths
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def calculate_threshold_weighted_forecast(simulated_paths, weights, threshold=4):
    """
    Berechnet eine Prognose durch Mittelung nur der Pfade mit Wahrscheinlichkeiten über einem Schwellenwert.
    
    :param simulated_paths: np.ndarray, Array der simulierten Pfade
    :param weights: np.ndarray, Gewichtungen für jeden Pfad (Wahrscheinlichkeiten zwischen 1-10)
    :param threshold: float, Mindestwahrscheinlichkeitsschwelle für die Berücksichtigung eines Pfades
    :return: np.ndarray, gemittelte Prognose aus Pfaden über dem Schwellenwert
    """
    # Maske für Pfade mit Wahrscheinlichkeit über dem Schwellenwert erstellen
    mask = weights >= threshold
    
    # Prüfen, ob Pfade den Schwellenwert erfüllen
    if not any(mask):
        print(f"Warning: No paths meet the threshold of {threshold}. Using all paths.")
        mask = np.ones_like(weights, dtype=bool)
    
    # Nur Pfade über dem Schwellenwert auswählen
    filtered_paths = simulated_paths[mask]
    
    # Durchschnittliche Prognose berechnen
    weighted_forecast = np.average(filtered_paths, axis=0)
    
    return weighted_forecast

def get_user_threshold():
    """
    Holt den ChatGPT-Schwellenwert von der Benutzereingabe.
    
    :return: float, benutzerdefinierter Schwellenwert
    """
    root = tk.Tk()
    root.withdraw()  # Hauptfenster ausblenden
    
    threshold = simpledialog.askfloat(
        "ChatGPT Threshold",
        "Enter probability threshold for ChatGPT weighted forecast (1-10):",
        minvalue=1.0,
        maxvalue=10.0,
        initialvalue=5.0
    )
    
    if threshold is None:
        print("No threshold entered. Using default value of 5.0")
        threshold = 5.0
    
    return threshold

def process_adoption_forecasts():
    """
    Hauptfunktion um adoption.json Dateien zu verarbeiten und gewichtete Prognosen zu generieren.
    """
    print("=== Adoption Forecasts Processor ===")
    print()
    
    # Ordner auswählen
    folder_path = select_folder()
    if not folder_path:
        return
    
    print(f"Selected folder: {folder_path}")
    
    # adoption.json Datei finden
    adoption_file = find_adoption_file(folder_path)
    if not adoption_file:
        return
    
    print(f"Found adoption.json: {adoption_file}")
    
    # Prognosedatei finden
    forecasts_file = find_forecasts_file(folder_path)
    if not forecasts_file:
        return
    
    print(f"Found forecasts file: {forecasts_file}")
    
    # CSV-Datei mit Simulationspfaden finden
    csv_file_path = find_csv_file(folder_path)
    if not csv_file_path:
        print("Error: No CSV file found containing simulation paths.")
        return
    
    print(f"Found CSV file: {csv_file_path}")
    
    try:
        # adoption.json laden
        with open(adoption_file, 'r') as f:
            adoption_data = json.load(f)
        
        # Prognosedatei laden
        with open(forecasts_file, 'r') as f:
            forecasts_data = json.load(f)
        
        # Wahrscheinlichkeiten aus adoption-Daten extrahieren
        if 'analyses' not in adoption_data:
            print("Error: No 'analyses' field found in adoption.json")
            return
        
        probabilities = np.array([a["probability"] for a in adoption_data["analyses"]])
        print(f"Loaded {len(probabilities)} path probabilities")
        
        # Simulationspfade aus CSV laden
        simulated_paths = load_simulation_paths_from_csv(csv_file_path)
        if simulated_paths is None:
            print("Error: Could not load simulation paths from CSV.")
            return
        
        # Benutzer-Schwellenwert für ChatGPT gewichtete Prognose holen
        chatgpt_threshold = get_user_threshold()
        print(f"ChatGPT threshold set to: {chatgpt_threshold}")
        
        # Schwellenwert gewichtete Prognose berechnen (Schwellenwert = 4)
        print("Calculating threshold weighted forecast (threshold = 4)...")
        threshold_weighted_forecast = calculate_threshold_weighted_forecast(
            simulated_paths, probabilities, threshold=4
        )
        
        # ChatGPT gewichtete Prognose berechnen (Benutzer-Schwellenwert)
        print(f"Calculating ChatGPT weighted forecast (threshold = {chatgpt_threshold})...")
        chatgpt_weighted_forecast = calculate_threshold_weighted_forecast(
            simulated_paths, probabilities, threshold=chatgpt_threshold
        )
        
        # Neue Prognosedaten mit den gewichteten Prognosen erstellen
        new_forecasts_data = forecasts_data.copy()
        
        # arima_forecast Einträge mit neuen adoption gewichteten Prognosen aktualisieren
        for i, forecast_entry in enumerate(new_forecasts_data['arima_forecast']):
            if i < len(threshold_weighted_forecast):
                forecast_entry['adoption_threshold_weighted_forecast'] = float(threshold_weighted_forecast[i])
            if i < len(chatgpt_weighted_forecast):
                forecast_entry['adoption_chatgpt_weighted_forecast'] = float(chatgpt_weighted_forecast[i])
        
        # Top-Level-Arrays für die neuen adoption Prognosen hinzufügen
        new_forecasts_data['adoption_threshold_weighted_forecast'] = threshold_weighted_forecast.tolist()
        new_forecasts_data['adoption_chatgpt_weighted_forecast'] = chatgpt_weighted_forecast.tolist()
        new_forecasts_data['adoption_chatgpt_threshold'] = float(chatgpt_threshold)
        
        # Ausgabedateiname mit Zeitstempel generieren
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = new_forecasts_data.get('ticker', 'Unknown')
        output_filename = f"{ticker}_forecasts_processed_{timestamp}.json"
        output_path = os.path.join(folder_path, output_filename)
        
        # Neue Prognosedaten speichern
        with open(output_path, 'w') as f:
            json.dump(new_forecasts_data, f, indent=4)
        
        print(f"\n=== Processing Complete ===")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_adoption_forecasts() 