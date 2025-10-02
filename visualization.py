import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import timedelta
import os

def plot_forecast_comparison(original_series, arima_forecast, weighted_forecast, 
                           chatgpt_forecast, chatgpt_forecast_historical, simulated_paths, title, 
                           forecast_steps, show_last_days, current_dir=None,
                           threshold_weighted_forecast=None, chatgpt_weighted_forecast=None, chatgpt_threshold=None):
    """
    Plottet den Vergleich verschiedener Prognosen.
    
    :param original_series: pd.Series, ursprüngliche Preiserie
    :param arima_forecast: array, ARIMA-Prognosewerte
    :param weighted_forecast: array, gewichtete Prognosewerte (veraltet, für Kompatibilität beibehalten)
    :param chatgpt_forecast: dict, ChatGPT-Prognosewerte
    :param chatgpt_forecast_historical: dict, ChatGPT-historische Prognosewerte
    :param simulated_paths: array, simulierte Pfade
    :param title: str, Titel für das Plot
    :param forecast_steps: int, Anzahl der Prognoseschritte
    :param show_last_days: int, Anzahl der letzten Tage anzuzeigen
    :param current_dir: str, Verzeichnis zum Speichern von Dateien
    :param threshold_weighted_forecast: array, Prognose gewichtet nach Pfaden mit Wahrscheinlichkeit > 2
    :param chatgpt_weighted_forecast: array, Prognose gewichtet nach Pfaden mit Wahrscheinlichkeit > Benutzerschwelle
    :param chatgpt_threshold: float, benutzerdefinierte Wahrscheinlichkeitsschwelle
    :return: None
    """
    try:
        plt.figure(figsize=(15, 8))
        
        # Historische Daten zum Anzeigen abrufen
        historical_to_show = original_series.tail(show_last_days)
        
        # Ursprüngliche Serie mit Punkten plotten
        plt.plot(historical_to_show.index, historical_to_show.values, label='Original', color='blue')
        plt.scatter(historical_to_show.index, historical_to_show.values, color='blue', s=50, alpha=0.6)
        
        # Letzten historischen Datenpunkt abrufen und sicherstellen dass er zeitzonenfrei ist
        last_historical_date = historical_to_show.index[-1]
        if last_historical_date.tz is not None:
            last_historical_date = last_historical_date.tz_localize(None)
        last_historical_value = historical_to_show.iloc[-1]
        
        # Geschäftstage generieren
        cal = USFederalHolidayCalendar()
        max_days = forecast_steps * 4
        end_date = last_historical_date + timedelta(days=max_days)
        holidays = cal.holidays(start=last_historical_date, end=end_date)
        
        forecast_dates = []
        current_date = last_historical_date + timedelta(days=1)
        
        while len(forecast_dates) < forecast_steps:
            if current_date.weekday() < 5 and current_date not in holidays:
                forecast_dates.append(current_date)
            current_date += timedelta(days=1)
        
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        
        # Daten und Preise aus ChatGPT-Prognosen extrahieren und zeitzonenfrei sicherstellen
        chatgpt_dates = [pd.to_datetime(f['date'], format='%d.%m.%Y').tz_localize(None) for f in chatgpt_forecast['forecast']]
        chatgpt_prices = [f['closing_price'] for f in chatgpt_forecast['forecast']]
        
        chatgpt_historical_dates = [pd.to_datetime(f['date'], format='%d.%m.%Y').tz_localize(None) for f in chatgpt_forecast_historical['forecast']]
        chatgpt_historical_prices = [f['closing_price'] for f in chatgpt_forecast_historical['forecast']]
        
        # Punkte für Plotting mit verbindenden Linien kombinieren
        arima_plot_dates = [last_historical_date] + list(forecast_dates)
        arima_plot_values = [last_historical_value] + list(arima_forecast)
        
        # ARIMA-Prognose mit verbindender Linie und Punkten plotten
        plt.plot(arima_plot_dates, arima_plot_values, label='ARIMA Forecast', color='red', linestyle='--')
        plt.scatter(forecast_dates, arima_forecast, color='red', s=50, alpha=0.6)
        
        # Standard gewichtete Prognose nur plotten wenn explizit bereitgestellt und nicht None
        if weighted_forecast is not None:
            weighted_plot_dates = [last_historical_date] + list(forecast_dates)
            weighted_plot_values = [last_historical_value] + list(weighted_forecast)
            plt.plot(weighted_plot_dates, weighted_plot_values, label='Standard Weighted Forecast', color='green', linestyle='--')
            plt.scatter(forecast_dates, weighted_forecast, color='green', s=50, alpha=0.6)
        
        # Schwellenwert-gewichtete Prognose plotten falls bereitgestellt
        if threshold_weighted_forecast is not None:
            threshold_plot_dates = [last_historical_date] + list(forecast_dates)
            threshold_plot_values = [last_historical_value] + list(threshold_weighted_forecast)
            plt.plot(threshold_plot_dates, threshold_plot_values, label='Threshold (>=2.0) Weighted', color='cyan', linestyle='--')
            plt.scatter(forecast_dates, threshold_weighted_forecast, color='cyan', s=50, alpha=0.6)
        
        # Benutzer-Schwellenwert gewichtete Prognose plotten falls bereitgestellt
        if chatgpt_weighted_forecast is not None and chatgpt_threshold is not None:
            user_plot_dates = [last_historical_date] + list(forecast_dates)
            user_plot_values = [last_historical_value] + list(chatgpt_weighted_forecast)
            plt.plot(user_plot_dates, user_plot_values, label=f'ChatGPT Threshold (>={chatgpt_threshold}) Weighted', color='magenta', linestyle='--')
            plt.scatter(forecast_dates, chatgpt_weighted_forecast, color='magenta', s=50, alpha=0.6)
        
        # ChatGPT-Prognose mit verbindender Linie und Punkten plotten
        chatgpt_plot_dates = [last_historical_date] + chatgpt_dates
        chatgpt_plot_values = [last_historical_value] + chatgpt_prices
        plt.plot(chatgpt_plot_dates, chatgpt_plot_values, label='ChatGPT Raw Forecast', color='purple', linestyle='--')
        plt.scatter(chatgpt_dates, chatgpt_prices, color='purple', s=50, alpha=0.6)
        
        # ChatGPT-historische Prognose mit verbindender Linie und Punkten plotten
        chatgpt_hist_plot_dates = [last_historical_date] + chatgpt_historical_dates
        chatgpt_hist_plot_values = [last_historical_value] + chatgpt_historical_prices
        plt.plot(chatgpt_hist_plot_dates, chatgpt_hist_plot_values, label='ChatGPT with historical dataForecast', color='orange', linestyle='--')
        plt.scatter(chatgpt_historical_dates, chatgpt_historical_prices, color='orange', s=50, alpha=0.6)
        
        # Simulierte Pfade mit verbindenden Linien plotten
        for path in simulated_paths:
            sim_plot_dates = [last_historical_date] + list(forecast_dates)
            sim_plot_values = [last_historical_value] + list(path)
            plt.plot(sim_plot_dates, sim_plot_values, color='gray', alpha=0.1)
        
        # Konfidenzintervalle hinzufügen
        lower_bound = np.percentile(simulated_paths, 2.5, axis=0)
        upper_bound = np.percentile(simulated_paths, 97.5, axis=0)
        plt.fill_between(forecast_dates, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% CI')
        
        plt.title(f'{title} - Forecast Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Plot speichern
        if current_dir:
            plot_path = os.path.join(current_dir, f'{title}_forecast_comparison.png')
        else:
            plot_path = f'{title}_forecast_comparison.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        # Prognose-Pfade nach dem Plotting exportieren
        export_forecast_paths(original_series, arima_forecast, weighted_forecast, 
                            chatgpt_forecast, chatgpt_forecast_historical, simulated_paths, title, forecast_dates, current_dir,
                            threshold_weighted_forecast, chatgpt_weighted_forecast, chatgpt_threshold)
    except Exception as e:
        print(f"Error in plot_forecast_comparison: {str(e)}")
        plt.close('all')  # Alle offenen Plots bereinigen
        raise

def export_forecast_paths(original_series, arima_forecast, weighted_forecast, 
                         chatgpt_forecast, chatgpt_forecast_historical, simulated_paths, title, forecast_dates, current_dir=None,
                         threshold_weighted_forecast=None, chatgpt_weighted_forecast=None, chatgpt_threshold=None):
    """
    Exportiert Prognose-Pfade in CSV- und JSON-Dateien für spätere Bewertung.
    
    :param original_series: pd.Series, ursprüngliche Zeitreihe
    :param arima_forecast: np.ndarray, ARIMA-Prognose
    :param weighted_forecast: np.ndarray, gewichtete Prognose basierend auf ChatGPT-Analyse (veraltet)
    :param chatgpt_forecast: dict, ChatGPT-Prognose
    :param chatgpt_forecast_historical: dict, ChatGPT-ARIMA-Prognose
    :param simulated_paths: np.ndarray, simulierte Pfade
    :param title: str, Titel für die Dateien
    :param forecast_dates: pd.DatetimeIndex, Daten für die Prognose
    :param current_dir: str, Verzeichnis zum Speichern von Dateien
    :param threshold_weighted_forecast: array, Prognose gewichtet nach Pfaden mit Wahrscheinlichkeit > 2
    :param chatgpt_weighted_forecast: array, Prognose gewichtet nach Pfaden mit Wahrscheinlichkeit > Benutzerschwelle
    :param chatgpt_threshold: float, benutzerdefinierte Wahrscheinlichkeitsschwelle
    :return: None
    """
    # DataFrame für ARIMA- und gewichtete Prognosen erstellen
    arima_df = pd.DataFrame({
        'date': forecast_dates,
        'arima_forecast': arima_forecast,
    })
    
    # Standard gewichtete Prognose nur hinzufügen wenn explizit bereitgestellt
    if weighted_forecast is not None:
        arima_df['weighted_forecast'] = weighted_forecast
    
    # Schwellenwert gewichtete Prognosen hinzufügen falls verfügbar
    if threshold_weighted_forecast is not None:
        arima_df['threshold_weighted_forecast'] = threshold_weighted_forecast
    
    # Benutzer gewichtete Prognosen hinzufügen falls verfügbar
    if chatgpt_weighted_forecast is not None:
        arima_df['chatgpt_weighted_forecast'] = chatgpt_weighted_forecast
    
    # ChatGPT-Prognosedaten extrahieren
    chatgpt_dates = [pd.to_datetime(f['date'], format='%d.%m.%Y') for f in chatgpt_forecast['forecast']]
    chatgpt_prices = [f['closing_price'] for f in chatgpt_forecast['forecast']]
    chatgpt_confidence = [f['confidence'] for f in chatgpt_forecast['forecast']]
    
    chatgpt_df = pd.DataFrame({
        'date': chatgpt_dates,
        'chatgpt_forecast': chatgpt_prices,
        'confidence': chatgpt_confidence
    })
    
    # ChatGPT-historische Prognosedaten extrahieren
    chatgpt_historical_dates = [pd.to_datetime(f['date'], format='%d.%m.%Y') for f in chatgpt_forecast_historical['forecast']]
    chatgpt_historical_prices = [f['closing_price'] for f in chatgpt_forecast_historical['forecast']]
    chatgpt_historical_confidence = [f['confidence'] for f in chatgpt_forecast_historical['forecast']]
    
    chatgpt_historical_df = pd.DataFrame({
        'date': chatgpt_historical_dates,
        'chatgpt_forecast_historical': chatgpt_historical_prices,
        'confidence': chatgpt_historical_confidence
    })
    
    # DataFrame für simulierte Pfade erstellen
    simulated_df = pd.DataFrame(simulated_paths.T, columns=[f'path_{i}' for i in range(simulated_paths.shape[0])])
    simulated_df['date'] = forecast_dates
    
    # Daten zu Strings für JSON-Serialisierung konvertieren
    arima_forecast_json = [{
        'date': date.strftime('%Y-%m-%d'),
        'arima_forecast': float(forecast),
    } for date, forecast in zip(forecast_dates, arima_forecast)]
    
    # Standard gewichtete Prognose zu JSON nur hinzufügen wenn explizit bereitgestellt
    if weighted_forecast is not None:
        for i, weighted_value in enumerate(weighted_forecast):
            arima_forecast_json[i]['weighted_forecast'] = float(weighted_value)
    
    # Schwellenwert gewichtete Prognose zu JSON hinzufügen falls verfügbar
    if threshold_weighted_forecast is not None:
        for i, threshold_value in enumerate(threshold_weighted_forecast):
            arima_forecast_json[i]['threshold_weighted_forecast'] = float(threshold_value)
    
    # Benutzer gewichtete Prognose zu JSON hinzufügen falls verfügbar
    if chatgpt_weighted_forecast is not None:
        for i, user_value in enumerate(chatgpt_weighted_forecast):
            arima_forecast_json[i]['chatgpt_weighted_forecast'] = float(user_value)
    
    # Mit zusätzlichen Metadaten in JSON speichern
    export_data = {
        'ticker': title,
        'forecast_start_date': forecast_dates[0].strftime('%Y-%m-%d'),
        'forecast_end_date': forecast_dates[-1].strftime('%Y-%m-%d'),
        'last_actual_price': float(original_series.iloc[-1]),
        'last_actual_date': original_series.index[-1].strftime('%Y-%m-%d'),
        'arima_forecast': arima_forecast_json,
        'chatgpt_forecast': chatgpt_forecast,
        'chatgpt_historical_forecast': chatgpt_forecast_historical,
        'simulated_paths': [[float(x) for x in path] for path in simulated_paths]
    }
    
    # Standard gewichtete Prognose zu Exportdaten nur hinzufügen wenn explizit bereitgestellt
    if weighted_forecast is not None:
        export_data['weighted_forecast'] = [float(x) for x in weighted_forecast]
    
    # Schwellenwert gewichtete Prognose zu Exportdaten hinzufügen falls verfügbar
    if threshold_weighted_forecast is not None:
        export_data['threshold_weighted_forecast'] = [float(x) for x in threshold_weighted_forecast]
    
    # Benutzer gewichtete Prognose zu Exportdaten hinzufügen falls verfügbar
    if chatgpt_weighted_forecast is not None:
        export_data['chatgpt_weighted_forecast'] = [float(x) for x in chatgpt_weighted_forecast]
        export_data['chatgpt_threshold'] = float(chatgpt_threshold)
    
    if current_dir:
        json_path = os.path.join(current_dir, f'{title}_forecasts.json')
    else:
        json_path = f'{title}_forecasts.json'
    
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=4)
    
    print(f"Exported forecast data to {json_path}")

def print_forecast_details(analysis, chatgpt_forecast, chatgpt_forecast_historical):
    """
    Gibt detaillierte Informationen über die Prognosen aus.
    
    :param analysis: dict, Analyseergebnisse von ChatGPT
    :param chatgpt_forecast: dict, Prognose von ChatGPT
    :param chatgpt_forecast_historical: dict, historische Prognose von ChatGPT
    :return: None
    """
    print("\nForecast Details:")
    print("----------------")
    
    # Pfad-Analyseergebnisse ausgeben
    print("\nPath Analysis Results:")
    for a in analysis['analyses']:
        print(f"Path {a['path_id']}:")
        print(f"  Probability: {a['probability']}/10")
        print(f"  Reasoning: {a['reasoning']}")
    
    # ChatGPT-Prognose ausgeben
    print("\nChatGPT Forecast:")
    for f in chatgpt_forecast['forecast']:
        print(f"Date: {f['date']}")
        print(f"  Price: {f['closing_price']:.2f}")
        print(f"  Confidence: {f['confidence']}")
        print(f"  Reasoning: {f['reasoning']}")
    
    # ChatGPT-ARIMA-Prognose ausgeben
    print("\nChatGPT historical Forecast:")
    print("Model Information:")
    model_info = chatgpt_forecast_historical['model_used']
    print(f"  Model: {model_info['name']}")
    print(f"  Description: {model_info['description']}")
    print(f"  Parameters: {model_info['parameters']}")
    print(f"  Rationale: {model_info['rationale']}")
    
    print("\nForecast Details:")
    for f in chatgpt_forecast_historical['forecast']:
        print(f"Date: {f['date']}")
        print(f"  Price: {f['closing_price']:.2f}")
        print(f"  Confidence: {f['confidence']}")
        print(f"  Reasoning: {f['reasoning']}") 