import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import glob

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

def recalculate_weighted_forecasts(forecasts_data, analysis_data, output_dir):
    """
    Gewichtete Prognosen basierend auf den neuen Schwellenwerten neu berechnen.
    
    :param forecasts_data: dict, die Prognosedaten
    :param analysis_data: dict, die Analysedaten mit Pfadwahrscheinlichkeiten
    :param output_dir: str, Verzeichnis mit den Dateien
    :return: dict, aktualisierte Prognosedaten
    """
    try:
        # Simulierte Pfade extrahieren
        simulated_paths = np.array(forecasts_data.get('simulated_paths', []))
        if len(simulated_paths) == 0:
            print(f"  ⚠ No simulated paths found, skipping weighted forecast recalculation")
            return forecasts_data
        
        # Wahrscheinlichkeiten aus der Analyse extrahieren
        if 'analyses' in analysis_data:
            probabilities = np.array([a["probability"] for a in analysis_data["analyses"]])
        else:
            print(f"  ⚠ No analysis data found, skipping weighted forecast recalculation")
            return forecasts_data
        
        # Neuen Schwellenwert abrufen (ursprünglich + 1)
        new_threshold = forecasts_data.get('chatgpt_threshold', 6.0)  # Standard 6 falls nicht gefunden
        
        # Schwellenwert gewichtete Prognose neu berechnen (Schwellenwert = 4)
        threshold_weighted_forecast = calculate_threshold_weighted_forecast(
            simulated_paths, probabilities, threshold=4
        )
        
        # ChatGPT gewichtete Prognose neu berechnen (neuer Schwellenwert)
        chatgpt_weighted_forecast = calculate_threshold_weighted_forecast(
            simulated_paths, probabilities, threshold=new_threshold
        )
        
        # Gewichtete Prognosen innerhalb der arima_forecast Array-Einträge aktualisieren
        for i, forecast_entry in enumerate(forecasts_data['arima_forecast']):
            if i < len(threshold_weighted_forecast):
                forecast_entry['threshold_weighted_forecast'] = float(threshold_weighted_forecast[i])
            if i < len(chatgpt_weighted_forecast):
                forecast_entry['chatgpt_weighted_forecast'] = float(chatgpt_weighted_forecast[i])
        
        # Auch die Top-Level-Arrays für Kompatibilität aktualisieren
        forecasts_data['threshold_weighted_forecast'] = threshold_weighted_forecast.tolist()
        forecasts_data['chatgpt_weighted_forecast'] = chatgpt_weighted_forecast.tolist()
        
        print(f"  ✓ Recalculated weighted forecasts:")
        print(f"    - Threshold weighted forecast (threshold=4): {len(threshold_weighted_forecast)} values")
        print(f"    - ChatGPT weighted forecast (threshold={new_threshold}): {len(chatgpt_weighted_forecast)} values")
        
        return forecasts_data
        
    except Exception as e:
        print(f"  ✗ Error recalculating weighted forecasts: {str(e)}")
        return forecasts_data

def fix_arima_forecast_with_simulated_average(forecasts_data):
    """
    ARIMA-Prognose durch den Durchschnitt aller Simulationspfade pro Zeitschritt ersetzen.
    
    :param forecasts_data: dict, die Prognosedaten
    :return: dict, aktualisierte Prognosedaten
    """
    if 'simulated_paths' in forecasts_data and forecasts_data['simulated_paths']:
        # Durchschnitt über alle Simulationspfade für jeden Zeitschritt berechnen
        simulated_paths = np.array(forecasts_data['simulated_paths'])
        arima_forecast_average = np.average(simulated_paths, axis=0)
        
        # ARIMA-Prognose in den Daten aktualisieren
        for i, forecast_entry in enumerate(forecasts_data['arima_forecast']):
            forecast_entry['arima_forecast'] = float(arima_forecast_average[i])
        
        print(f"  ✓ Updated ARIMA forecast with simulated paths average")
        return forecasts_data
    else:
        print(f"  ⚠ No simulated paths found, keeping original ARIMA forecast")
        return forecasts_data

def adjust_thresholds(forecasts_data):
    """
    Schwellenwert für sowohl weighted_forecast als auch chatgpt_weighted_forecast um +1 erhöhen.
    
    :param forecasts_data: dict, die Prognosedaten
    :return: dict, aktualisierte Prognosedaten
    """
    # chatgpt_threshold aktualisieren
    if 'chatgpt_threshold' in forecasts_data:
        old_threshold = forecasts_data['chatgpt_threshold']
        forecasts_data['chatgpt_threshold'] = float(forecasts_data['chatgpt_threshold'] + 1.0)
        print(f"  ✓ Increased chatgpt_threshold from {old_threshold} to {forecasts_data['chatgpt_threshold']}")
    
    return forecasts_data

def regenerate_plots_with_black_theme(forecasts_data, output_dir, ticker):
    """
    Plots mit nur schwarzer Farbe und verschiedenen Linienstilen neu generieren.
    
    :param forecasts_data: dict, die Prognosedaten
    :param output_dir: str, Verzeichnis zum Speichern der Plots
    :param ticker: str, Ticker-Symbol
    """
    try:
        # Daten extrahieren
        arima_forecast = [entry['arima_forecast'] for entry in forecasts_data['arima_forecast']]
        threshold_weighted_forecast = forecasts_data.get('threshold_weighted_forecast', [])
        chatgpt_weighted_forecast = forecasts_data.get('chatgpt_weighted_forecast', [])
        chatgpt_threshold = forecasts_data.get('chatgpt_threshold', None)
        simulated_paths = forecasts_data.get('simulated_paths', [])
        
        # ChatGPT-Prognosen
        chatgpt_forecast = forecasts_data.get('chatgpt_forecast', {})
        chatgpt_forecast_historical = forecasts_data.get('chatgpt_historical_forecast', {})
        
        # Historische Daten abrufen
        historical_file = os.path.join(output_dir, f'{ticker}_historical_data.csv')
        if os.path.exists(historical_file):
            historical_data = pd.read_csv(historical_file, index_col=0, parse_dates=True)
            original_series = historical_data['Close']
        else:
            print(f"  ⚠ Historical data file not found: {historical_file}")
            return
        
        # Prognosedaten abrufen
        forecast_start_date = pd.to_datetime(forecasts_data['forecast_start_date'])
        forecast_end_date = pd.to_datetime(forecasts_data['forecast_end_date'])
        forecast_steps = len(arima_forecast)
        
        # Geschäftstage für Prognosezeitraum generieren
        cal = USFederalHolidayCalendar()
        max_days = forecast_steps * 4
        end_date = forecast_start_date + timedelta(days=max_days)
        holidays = cal.holidays(start=forecast_start_date, end=end_date)
        
        forecast_dates = []
        current_date = forecast_start_date
        
        while len(forecast_dates) < forecast_steps:
            if current_date.weekday() < 5 and current_date not in holidays:
                forecast_dates.append(current_date)
            current_date += timedelta(days=1)
        
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        
        # Plot mit schwarzem Theme erstellen
        plt.figure(figsize=(15, 8))
        
        # Letzte 30 Tage historischer Daten anzeigen
        show_last_days = 30
        historical_to_show = original_series.tail(show_last_days)
        
        # Ursprüngliche Serie plotten (durchgezogene schwarze Linie)
        plt.plot(historical_to_show.index, historical_to_show.values, 
                label='Historical Data', color='black', linestyle='-', linewidth=2)
        plt.scatter(historical_to_show.index, historical_to_show.values, 
                   color='black', s=30, alpha=0.7)
        
        # Letzten historischen Datenpunkt abrufen
        last_historical_date = historical_to_show.index[-1]
        last_historical_value = historical_to_show.iloc[-1]
        
        # ARIMA-Prognose plotten (gestrichelte schwarze Linie)
        arima_plot_dates = [last_historical_date] + list(forecast_dates)
        arima_plot_values = [last_historical_value] + list(arima_forecast)
        plt.plot(arima_plot_dates, arima_plot_values, 
                label='ARIMA Forecast (Simulated Average)', color='black', 
                linestyle='--', linewidth=2)
        plt.scatter(forecast_dates, arima_forecast, color='black', s=50, alpha=0.8)
        
        # Schwellenwert-gewichtete Prognose plotten (gepunktete schwarze Linie)
        if threshold_weighted_forecast:
            threshold_plot_dates = [last_historical_date] + list(forecast_dates)
            threshold_plot_values = [last_historical_value] + list(threshold_weighted_forecast)
            plt.plot(threshold_plot_dates, threshold_plot_values, 
                    label='Threshold Weighted Forecast', color='black', 
                    linestyle=':', linewidth=2)
            plt.scatter(forecast_dates, threshold_weighted_forecast, color='black', s=50, alpha=0.8)
        
        # ChatGPT gewichtete Prognose plotten (Strich-Punkt schwarze Linie)
        if chatgpt_weighted_forecast:
            chatgpt_plot_dates = [last_historical_date] + list(forecast_dates)
            chatgpt_plot_values = [last_historical_value] + list(chatgpt_weighted_forecast)
            plt.plot(chatgpt_plot_dates, chatgpt_plot_values, 
                    label=f'ChatGPT Weighted Forecast (threshold={chatgpt_threshold})', 
                    color='black', linestyle='-.', linewidth=2)
            plt.scatter(forecast_dates, chatgpt_weighted_forecast, color='black', s=50, alpha=0.8)
        
        # ChatGPT-Rohprognose plotten (durchgezogene schwarze Linie mit anderem Marker)
        if chatgpt_forecast and 'forecast' in chatgpt_forecast:
            chatgpt_dates = [pd.to_datetime(f['date'], format='%d.%m.%Y') for f in chatgpt_forecast['forecast']]
            chatgpt_prices = [f['closing_price'] for f in chatgpt_forecast['forecast']]
            chatgpt_plot_dates = [last_historical_date] + chatgpt_dates
            chatgpt_plot_values = [last_historical_value] + chatgpt_prices
            plt.plot(chatgpt_plot_dates, chatgpt_plot_values, 
                    label='ChatGPT Raw Forecast', color='black', 
                    linestyle='-', linewidth=1.5, alpha=0.7)
            plt.scatter(chatgpt_dates, chatgpt_prices, color='black', s=40, alpha=0.8, marker='s')
        
        # ChatGPT-historische Prognose plotten (gestrichelte schwarze Linie mit anderem Marker)
        if chatgpt_forecast_historical and 'forecast' in chatgpt_forecast_historical:
            chatgpt_hist_dates = [pd.to_datetime(f['date'], format='%d.%m.%Y') for f in chatgpt_forecast_historical['forecast']]
            chatgpt_hist_prices = [f['closing_price'] for f in chatgpt_forecast_historical['forecast']]
            chatgpt_hist_plot_dates = [last_historical_date] + chatgpt_hist_dates
            chatgpt_hist_plot_values = [last_historical_value] + chatgpt_hist_prices
            plt.plot(chatgpt_hist_plot_dates, chatgpt_hist_plot_values, 
                    label='ChatGPT Historical Forecast', color='black', 
                    linestyle='--', linewidth=1.5, alpha=0.7)
            plt.scatter(chatgpt_hist_dates, chatgpt_hist_prices, color='black', s=40, alpha=0.8, marker='^')
        
        # Simulierte Pfade plotten (sehr helles Grau)
        if simulated_paths:
            for path in simulated_paths:
                sim_plot_dates = [last_historical_date] + list(forecast_dates)
                sim_plot_values = [last_historical_value] + list(path)
                plt.plot(sim_plot_dates, sim_plot_values, color='lightgray', alpha=0.1, linewidth=0.5)
            
            # Konfidenzintervalle hinzufügen (hellgraue Füllung)
            lower_bound = np.percentile(simulated_paths, 2.5, axis=0)
            upper_bound = np.percentile(simulated_paths, 97.5, axis=0)
            plt.fill_between(forecast_dates, lower_bound, upper_bound, 
                           color='lightgray', alpha=0.3, label='95% Confidence Interval')
        
        plt.title(f'{ticker} - Forecast Comparison (Black Theme)', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Plot mit "_fixed" Suffix speichern
        plot_path = os.path.join(output_dir, f'{ticker}_forecast_comparison_fixed.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Regenerated plot with black theme: {plot_path}")
        
    except Exception as e:
        print(f"  ✗ Error regenerating plot: {str(e)}")
        plt.close('all')

def process_forecast_file(forecasts_file_path):
    """
    Eine einzelne Prognosedatei verarbeiten und alle Korrekturen anwenden, in neue Dateien speichern.
    
    :param forecasts_file_path: str, Pfad zur forecasts.json Datei
    :return: bool, True wenn erfolgreich, False sonst
    """
    try:
        output_dir = os.path.dirname(forecasts_file_path)
        ticker = os.path.basename(output_dir).split('_')[0]
        
        print(f"\nProcessing: {output_dir}")
        
        # Prognosedaten lesen
        with open(forecasts_file_path, 'r') as f:
            forecasts_data = json.load(f)
        
        # Analysedaten lesen
        analysis_file = os.path.join(output_dir, 'analysis.json')
        analysis_data = {}
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            print(f"  ✓ Loaded analysis data from: {analysis_file}")
        else:
            print(f"  ⚠ Analysis file not found: {analysis_file}")
        
        # Korrekturen anwenden
        forecasts_data = fix_arima_forecast_with_simulated_average(forecasts_data)
        forecasts_data = adjust_thresholds(forecasts_data)
        forecasts_data = recalculate_weighted_forecasts(forecasts_data, analysis_data, output_dir)
        
        # Aktualisierte Prognosedaten in neue Datei mit "_fixed" Suffix speichern
        fixed_forecasts_path = forecasts_file_path.replace('_forecasts.json', '_forecasts_fixed.json')
        with open(fixed_forecasts_path, 'w') as f:
            json.dump(forecasts_data, f, indent=4)
        
        print(f"  ✓ Updated forecasts data saved to: {fixed_forecasts_path}")
        
        # Plots neu generieren
        regenerate_plots_with_black_theme(forecasts_data, output_dir, ticker)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {forecasts_file_path}: {str(e)}")
        return False

def main():
    """
    Hauptfunktion um alle Prognosedateien rekursiv zu verarbeiten.
    """
    print("Starting forecast fixes and plot regeneration...")
    print("=" * 60)
    
    # Alle forecasts.json Dateien im output Verzeichnis finden
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found!")
        return
    
    # glob verwenden um alle forecasts.json Dateien rekursiv zu finden
    forecast_files = glob.glob(os.path.join(output_dir, "**", "*_forecasts.json"), recursive=True)
    
    # Bestehende "_fixed" Dateien herausfiltern um sie nicht zu verarbeiten
    forecast_files = [f for f in forecast_files if not f.endswith('_fixed.json')]
    
    if not forecast_files:
        print(f"No forecast files found in {output_dir}")
        return
    
    print(f"Found {len(forecast_files)} forecast files to process")
    
    # Jede Datei verarbeiten
    successful = 0
    failed = 0
    
    for forecast_file in sorted(forecast_files):
        if process_forecast_file(forecast_file):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(forecast_files)}")
    
    if failed > 0:
        print(f"\n⚠ {failed} files failed to process. Check the error messages above.")
    else:
        print(f"\n✓ All files processed successfully!")
    print(f"\n📁 Fixed files saved with '_fixed' suffix")
    print(f"📊 New plots saved with '_fixed' suffix")

if __name__ == "__main__":
    main() 