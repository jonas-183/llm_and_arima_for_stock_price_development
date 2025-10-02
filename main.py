import pandas as pd
from data_preparation import download_stock_data, prepare_stationary_data
from arima_modeling import fit_arima, generate_forecast_paths, transform_to_original_scale
from path_descriptions import (generate_path_description, generate_chatgpt_prompt, 
                             generate_chatgpt_forecast_prompt, generate_chatgpt_forecast_prompt_historical)
from visualization import plot_forecast_comparison, print_forecast_details
from chatgpt_handler import ChatGPTHandler
from config import (
    CHATGPT_MODE, DEFAULT_FORECAST_STEPS, DEFAULT_N_PATHS,
    DEFAULT_SHOW_LAST_DAYS, OUTPUT_DIR
)
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import json
import os
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

def calculate_threshold_weighted_forecast(simulated_paths, weights, threshold=2):
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

def main():
    try:
        logging.info("Starting main function")
        
        # Parameter definieren
        tickers = {
            "Alphabet": "GOOGL"     # Alphabet Inc. (Google)
        }
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d") # Heutiges Datum
        forecast_steps = DEFAULT_FORECAST_STEPS
        n_paths = DEFAULT_N_PATHS
        show_last_days = DEFAULT_SHOW_LAST_DAYS
        
        logging.info(f"Parameters set: start_date={start_date}, end_date={end_date}, forecast_steps={forecast_steps}")
        
        # ChatGPT Handler initialisieren
        logging.info("Initializing ChatGPT handler")
        chatgpt_handler = ChatGPTHandler()
        print(f"Running in {CHATGPT_MODE} mode")
        
        # Daten herunterladen und vorbereiten
        logging.info("Downloading and preparing data")
        data = download_stock_data(tickers, start_date, end_date)
        
        # Historische Daten in CSV exportieren
        for ticker in tickers.keys():
            csv_filename = os.path.join(OUTPUT_DIR, f"{ticker}_historical_data.csv")
            data[ticker].to_csv(csv_filename)
            logging.info(f"Exported historical data for {ticker} to {csv_filename}")

        logging.info("Preparing stationary data")
        stationary_data = prepare_stationary_data(data, tickers)
        
        # Jeden Ticker verarbeiten
        for ticker in tickers.keys():
            logging.info(f"Processing {ticker}")
            
            # ARIMA Modell anpassen und Pfade generieren
            logging.info("Fitting ARIMA model")
            best_model, best_order, best_model_fit = fit_arima(stationary_data[ticker])
            if best_model_fit is None:
                logging.error(f"Could not fit ARIMA model for {ticker}")
                continue
                
            logging.info(f"Best ARIMA parameters for {ticker}: {best_order}")
            
            # Prognose-Pfade generieren
            logging.info("Generating forecast paths")
            forecast, simulated_paths = generate_forecast_paths(best_model_fit, forecast_steps, n_paths)

            # Zur ursprünglichen Skala transformieren
            last_original_value = data[ticker].iloc[-1]
            last_date = data[ticker].index[-1]

            logging.info("Transforming to original scale")
            forecast_original, simulated_paths_original, forecast_dates = transform_to_original_scale(
                forecast, simulated_paths, last_original_value, last_date
            )

            # Pfadbeschreibungen und Prompts generieren
            logging.info("Generating path descriptions")
            paths_descriptions = []
            for i, path in enumerate(simulated_paths_original):
                description = generate_path_description(
                    path, ticker, last_date, forecast_dates[-1], last_original_value
                )
                paths_descriptions.append(f"Path {i+1}:\n{description}")
            
            # Prompts generieren
            logging.info("Generating prompts")
            path_analysis_prompt = generate_chatgpt_prompt("\n\n".join(paths_descriptions), ticker)
            forecast_prompt = generate_chatgpt_forecast_prompt(ticker, forecast_steps)
            forecast_prompt_historical = generate_chatgpt_forecast_prompt_historical(ticker, forecast_steps)
            
            # Prompts mit ChatGPT Handler verarbeiten
            logging.info("Processing prompts with ChatGPT")
            responses = chatgpt_handler.process_prompts(
                [path_analysis_prompt, forecast_prompt, forecast_prompt_historical],
                ticker
            )
            
            # Aktuelles Verzeichnis vom Handler abrufen
            current_dir = chatgpt_handler._create_timestamped_dir(ticker)
            
            # Antworten extrahieren
            analysis = responses["response_1"]
            chatgpt_forecast = responses["response_2"]
            chatgpt_forecast_historical = responses["response_3"]
            
            # Pfadwahrscheinlichkeiten aus der Analyse extrahieren
            probabilities = np.array([a["probability"] for a in analysis["analyses"]])
            
            # Schwellenwert-basierte gewichtete Prognose berechnen (Pfade mit Wahrscheinlichkeit > 2)
            logging.info("Calculating threshold weighted forecasts")
            threshold_weighted_forecast = calculate_threshold_weighted_forecast(
                simulated_paths_original, 
                probabilities, 
                threshold=2
            )
            
            # Nach benutzerdefiniertem Schwellenwert basierend auf ChatGPT-Analyse fragen
            
            chatgpt_threshold = float(input("\nBased on ChatGPT's analysis, enter a probability threshold for filtering paths (1-10): "))
            
            # Benutzer-Schwellenwert gewichtete Prognose berechnen
            chatgpt_weighted_forecast = calculate_threshold_weighted_forecast(
                simulated_paths_original, 
                probabilities, 
                threshold=chatgpt_threshold
            )
            
            # Ergebnisse visualisieren
            logging.info("Visualizing results")
            plot_forecast_comparison(
                data[ticker], forecast_original, None,  # weighted_forecast auf None setzen
                chatgpt_forecast, chatgpt_forecast_historical, simulated_paths_original,
                ticker, forecast_steps, show_last_days, current_dir,
                threshold_weighted_forecast=threshold_weighted_forecast,
                chatgpt_weighted_forecast=chatgpt_weighted_forecast,
                chatgpt_threshold=chatgpt_threshold
            )
            
            # Prognose-Details ausgeben
            print_forecast_details(analysis, chatgpt_forecast, chatgpt_forecast_historical)
            
            # Zusätzliche gewichtete Prognosen in JSON speichern
            forecast_file = os.path.join(current_dir, f'{ticker}_forecasts.json')
            if os.path.exists(forecast_file):
                with open(forecast_file, 'r') as f:
                    forecast_data = json.load(f)
                
                # Standard gewichtete Prognose entfernen falls vorhanden
                if 'weighted_forecast' in forecast_data:
                    del forecast_data['weighted_forecast']
                
                # Schwellenwert-Prognosen hinzufügen
                forecast_data['threshold_weighted_forecast'] = threshold_weighted_forecast.tolist()
                forecast_data['chatgpt_weighted_forecast'] = chatgpt_weighted_forecast.tolist()
                forecast_data['chatgpt_threshold'] = chatgpt_threshold
                
                with open(forecast_file, 'w') as f:
                    json.dump(forecast_data, f, indent=4)
            
            # Auf tatsächliche Daten warten um Prognosen zu bewerten
            print("Waiting for actual data to evaluate forecasts...")
            print("After the forecast period has passed, run the evaluation script.")
            print("Press Enter to continue to next ticker...")
            input()
            
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()