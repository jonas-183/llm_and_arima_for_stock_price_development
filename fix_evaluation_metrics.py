import os
import json
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(actual_values, predicted_values):
    """
    Bewertungsmetriken für Prognosegenauigkeit berechnen.
    
    :param actual_values: list or array of actual values
    :param predicted_values: list or array of predicted values
    :return: dict with MSE, RMSE, MAE, and MAPE
    """
    actual = np.array(actual_values)
    predicted = np.array(predicted_values)
    
    # Metriken berechnen
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # MAPE (Mean Absolute Percentage Error) berechnen
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape)
    }

def recalculate_weighted_metrics(evaluation_data, forecasts_data):
    """
    Gewichtete Prognose-Metriken basierend auf den aktualisierten Prognosen neu berechnen.
    
    :param evaluation_data: dict, die Bewertungsmetriken-Daten
    :param forecasts_data: dict, die aktualisierten Prognosedaten
    :return: dict, aktualisierte Bewertungsdaten
    """
    try:
        # Tatsächliche Werte abrufen
        actual_prices = evaluation_data.get('actual_values', {}).get('prices', [])
        if not actual_prices:
            print(f"  ⚠ No actual values found in evaluation data")
            return evaluation_data
        
        # Aktualisierte gewichtete Prognosen abrufen
        threshold_weighted_forecast = forecasts_data.get('threshold_weighted_forecast', [])
        chatgpt_weighted_forecast = forecasts_data.get('chatgpt_weighted_forecast', [])
        
        # Schwellenwert gewichtete Metriken neu berechnen falls Daten verfügbar
        if threshold_weighted_forecast and len(threshold_weighted_forecast) == len(actual_prices):
            threshold_metrics = calculate_metrics(actual_prices, threshold_weighted_forecast)
            evaluation_data['threshold_weighted_metrics'] = threshold_metrics
            print(f"  ✓ Recalculated threshold weighted metrics")
        
        # ChatGPT gewichtete Metriken neu berechnen falls Daten verfügbar
        if chatgpt_weighted_forecast and len(chatgpt_weighted_forecast) == len(actual_prices):
            chatgpt_weighted_metrics = calculate_metrics(actual_prices, chatgpt_weighted_forecast)
            evaluation_data['chatgpt_weighted_metrics'] = chatgpt_weighted_metrics
            print(f"  ✓ Recalculated ChatGPT weighted metrics")
        
        # ARIMA-Metriken mit dem neuen simulierten Durchschnitt neu berechnen
        arima_forecast = [entry['arima_forecast'] for entry in forecasts_data.get('arima_forecast', [])]
        if arima_forecast and len(arima_forecast) == len(actual_prices):
            arima_metrics = calculate_metrics(actual_prices, arima_forecast)
            evaluation_data['arima_metrics'] = arima_metrics
            print(f"  ✓ Recalculated ARIMA metrics (simulated average)")
        
        return evaluation_data
        
    except Exception as e:
        print(f"  ✗ Error recalculating metrics: {str(e)}")
        return evaluation_data

def adjust_evaluation_metrics(evaluation_data, forecasts_data):
    """
    Bewertungsmetriken basierend auf den aktualisierten Prognosen und Schwellenwerten anpassen.
    
    :param evaluation_data: dict, die Bewertungsmetriken-Daten
    :param forecasts_data: dict, die aktualisierten Prognosedaten
    :return: dict, aktualisierte Bewertungsdaten
    """
    try:
        # chatgpt_threshold aktualisieren um mit den aktualisierten Prognosen übereinzustimmen
        new_threshold = forecasts_data.get('chatgpt_threshold', None)
        if new_threshold is not None:
            old_threshold = evaluation_data.get('chatgpt_threshold', None)
            evaluation_data['chatgpt_threshold'] = float(new_threshold)
            print(f"  ✓ Updated chatgpt_threshold from {old_threshold} to {new_threshold}")
        
        # Metriken basierend auf aktualisierten Prognosen neu berechnen
        evaluation_data = recalculate_weighted_metrics(evaluation_data, forecasts_data)
        
        return evaluation_data
        
    except Exception as e:
        print(f"  ✗ Error adjusting evaluation metrics: {str(e)}")
        return evaluation_data

def process_evaluation_file(evaluation_file_path):
    """
    Eine einzelne Bewertungsmetriken-Datei verarbeiten und Anpassungen anwenden.
    
    :param evaluation_file_path: str, Pfad zur evaluation_metrics.json Datei
    :return: bool, True wenn erfolgreich, False sonst
    """
    try:
        output_dir = os.path.dirname(evaluation_file_path)
        ticker = os.path.basename(output_dir).split('_')[0]
        
        print(f"\nProcessing evaluation metrics: {output_dir}")
        
        # Bewertungsmetriken-Daten lesen
        with open(evaluation_file_path, 'r') as f:
            evaluation_data = json.load(f)
        
        # Entsprechende korrigierte Prognosedaten lesen
        forecasts_file = os.path.join(output_dir, f'{ticker}_forecasts_fixed.json')
        forecasts_data = {}
        if os.path.exists(forecasts_file):
            with open(forecasts_file, 'r') as f:
                forecasts_data = json.load(f)
            print(f"  ✓ Loaded fixed forecasts data from: {forecasts_file}")
        else:
            print(f"  ⚠ Fixed forecasts file not found: {forecasts_file}")
            print(f"  ⚠ Will only update threshold, not recalculate metrics")
        
        # Anpassungen anwenden
        evaluation_data = adjust_evaluation_metrics(evaluation_data, forecasts_data)
        
        # Aktualisierte Bewertungsdaten in neue Datei mit "_fixed" Suffix speichern
        fixed_evaluation_path = evaluation_file_path.replace('_evaluation_metrics.json', '_evaluation_metrics_fixed.json')
        with open(fixed_evaluation_path, 'w') as f:
            json.dump(evaluation_data, f, indent=4)
        
        print(f"  ✓ Updated evaluation metrics saved to: {fixed_evaluation_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {evaluation_file_path}: {str(e)}")
        return False

def main():
    """
    Hauptfunktion um alle Bewertungsmetriken-Dateien rekursiv zu verarbeiten.
    """
    print("Starting evaluation metrics fixes...")
    print("=" * 60)
    
    # Alle evaluation_metrics.json Dateien im output Verzeichnis finden
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found!")
        return
    
    # glob verwenden um alle evaluation_metrics.json Dateien rekursiv zu finden
    evaluation_files = glob.glob(os.path.join(output_dir, "**", "*_evaluation_metrics.json"), recursive=True)
    
    # Bestehende "_fixed" Dateien herausfiltern um sie nicht zu verarbeiten
    evaluation_files = [f for f in evaluation_files if not f.endswith('_fixed.json')]
    
    if not evaluation_files:
        print(f"No evaluation metrics files found in {output_dir}")
        return
    
    print(f"Found {len(evaluation_files)} evaluation metrics files to process")
    
    # Jede Datei verarbeiten
    successful = 0
    failed = 0
    
    for evaluation_file in sorted(evaluation_files):
        if process_evaluation_file(evaluation_file):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(evaluation_files)}")
    
    if failed > 0:
        print(f"\n⚠ {failed} files failed to process. Check the error messages above.")
    else:
        print(f"\n✓ All files processed successfully!")
    print(f"\n📁 Fixed evaluation metrics saved with '_fixed' suffix")

if __name__ == "__main__":
    main() 