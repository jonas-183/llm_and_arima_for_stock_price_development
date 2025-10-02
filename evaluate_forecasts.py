import pandas as pd
import numpy as np
import yfinance as yf
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from config import OUTPUT_DIR, DEFAULT_FORECAST_STEPS, DEFAULT_SHOW_LAST_DAYS

def load_forecast_data(ticker):
    """
    Lädt Prognosedaten aus dem Verzeichnis das gestern von chatgpt_handler erstellt wurde.
    
    :param ticker: str, Ticker-Symbol
    :return: dict, Prognosedaten
    """
    # Gestern Datum im Format das von chatgpt_handler verwendet wird
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    
    # Erwarteten Verzeichnisnamen konstruieren
    expected_dir = f"{ticker}_forecast{DEFAULT_FORECAST_STEPS}days_{yesterday}"
    dir_path = os.path.join(OUTPUT_DIR, expected_dir)
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Nach der Prognosedatei im Verzeichnis suchen
    forecast_file = os.path.join(dir_path, f'{ticker}_forecasts.json')
    
    if not os.path.exists(forecast_file):
        raise FileNotFoundError(f"Forecast file not found in {dir_path}")
    
    with open(forecast_file, 'r') as f:
        return json.load(f)

def download_actual_prices(ticker, start_date, end_date):
    """
    Lädt tatsächliche Aktienpreise von Yahoo Finance herunter.
    
    :param ticker: str, Ticker-Symbol
    :param start_date: str, Startdatum im Format YYYY-MM-DD
    :param end_date: str, Enddatum im Format YYYY-MM-DD
    :return: pd.Series, tatsächliche Preise
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # Zeitzone und Zeitkomponente entfernen so dass nur das *Datum* übrig bleibt
    data.index = (
        data.index
            .tz_localize(None)     # Zeitzone entfernen
            .normalize()           # Zeit auf 00:00:00 setzen
    )
    return data['Close']

def load_historical_data(ticker, dir_path):
    """
    Lädt historische Daten aus der CSV-Datei.
    
    :param ticker: str, Ticker-Symbol
    :param dir_path: str, Verzeichnispfad wo historische Daten gespeichert sind
    :return: pd.DataFrame, historische Daten
    """
    historical_file = os.path.join(dir_path, f'{ticker}_historical_data.csv')
    
    if not os.path.exists(historical_file):
        historical_file = os.path.join(OUTPUT_DIR, f'{ticker}_historical_data.csv')
        
    if not os.path.exists(historical_file):
        raise FileNotFoundError(f"Historical data file not found at {historical_file}")
    
    historical_data = pd.read_csv(historical_file)
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    historical_data.set_index('Date', inplace=True)
    
    return historical_data['Close']

def calculate_metrics(actual, forecast):
    """
    Berechnet verschiedene Prognosegenauigkeits-Metriken.
    
    :param actual: np.ndarray, tatsächliche Werte
    :param forecast: np.ndarray, prognostizierte Werte
    :return: dict, Metriken
    """
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def plot_comparison(actual_prices, forecasts, ticker, dir_path):
    """
    Plottet Vergleich von tatsächlichen Preisen und Prognosen basierend auf plot_forecast_comparison.
    
    :param actual_prices: pd.Series, tatsächliche Preise
    :param forecasts: dict, Prognosedaten
    :param ticker: str, Ticker-Symbol
    :param dir_path: str, Verzeichnispfad zum Speichern des Plots
    :return: None
    """
    try:
        plt.figure(figsize=(15, 8))
        
        # Prognosedaten und Werte extrahieren
        forecast_dates = pd.to_datetime([f['date'] for f in forecasts['arima_forecast']])
        if forecast_dates[0].tz is not None:
            forecast_dates = forecast_dates.tz_localize(None)
            
        arima_forecast = np.array([f['arima_forecast'] for f in forecasts['arima_forecast']])
        
        # Prüfen ob Standard gewichtete Prognose existiert (Legacy)
        weighted_forecast = None
        if 'weighted_forecast' in forecasts:
            weighted_forecast = np.array(forecasts['weighted_forecast'])
        
        # Schwellenwert gewichtete Prognose extrahieren falls verfügbar
        threshold_weighted_forecast = None
        if 'threshold_weighted_forecast' in forecasts:
            threshold_weighted_forecast = np.array(forecasts['threshold_weighted_forecast'])
        
        # Benutzer gewichtete Prognose extrahieren falls verfügbar
        chatgpt_weighted_forecast = None
        chatgpt_threshold = None
        if 'chatgpt_weighted_forecast' in forecasts and 'chatgpt_threshold' in forecasts:
            chatgpt_weighted_forecast = np.array(forecasts['chatgpt_weighted_forecast'])
            chatgpt_threshold = forecasts['chatgpt_threshold']
        
        # ChatGPT-Prognose mit korrektem Datumsparsing extrahieren
        chatgpt_dates = pd.to_datetime([f['date'] for f in forecasts['chatgpt_forecast']['forecast']], format='%d.%m.%Y')
        if chatgpt_dates[0].tz is not None:
            chatgpt_dates = chatgpt_dates.tz_localize(None)
            
        chatgpt_prices = np.array([f['closing_price'] for f in forecasts['chatgpt_forecast']['forecast']])
        
        # ChatGPT-historische Prognose mit korrektem Datumsparsing extrahieren
        chatgpt_historical_dates = pd.to_datetime([f['date'] for f in forecasts['chatgpt_historical_forecast']['forecast']], format='%d.%m.%Y')
        if chatgpt_historical_dates[0].tz is not None:
            chatgpt_historical_dates = chatgpt_historical_dates.tz_localize(None)
            
        chatgpt_historical_prices = np.array([f['closing_price'] for f in forecasts['chatgpt_historical_forecast']['forecast']])
        
        # Simulierte Pfade abrufen
        simulated_paths = np.array(forecasts['simulated_paths'])
        
        # Letzten historischen Datenpunkt aus den Prognosedaten abrufen
        last_historical_date = pd.to_datetime(forecasts['last_actual_date'])
        if last_historical_date.tz is not None:
            last_historical_date = last_historical_date.tz_localize(None)
        last_historical_value = float(forecasts['last_actual_price'])
        
        # Sicherstellen dass tatsächliche Preise zeitzonenfreien Index haben
        if actual_prices.index[0].tz is not None:
            actual_prices.index = actual_prices.index.tz_localize(None)
        
        # Tatsächliche Preise mit Punkten plotten (sowohl historisch als auch Prognosezeitraum)
        plt.plot(actual_prices.index, actual_prices.values, label='Actual Price', color='blue', linewidth=2)
        plt.scatter(actual_prices.index, actual_prices.values, color='blue', s=50, alpha=0.6)
        
        # ARIMA-Prognose mit verbindender Linie und Punkten plotten
        plt.plot([last_historical_date, forecast_dates[0]], [last_historical_value, arima_forecast[0]], 
                color='red', linestyle='--', alpha=0.5)
        plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='red', linestyle='--')
        plt.scatter(forecast_dates, arima_forecast, color='red', s=50, alpha=0.6)
        
        # Standard gewichtete Prognose nur plotten wenn explizit bereitgestellt und nicht None (Legacy)
        if weighted_forecast is not None:
            plt.plot([last_historical_date, forecast_dates[0]], [last_historical_value, weighted_forecast[0]], 
                    color='green', linestyle='--', alpha=0.5)
            plt.plot(forecast_dates, weighted_forecast, label='Standard Weighted Forecast', color='green', linestyle='--')
            plt.scatter(forecast_dates, weighted_forecast, color='green', s=50, alpha=0.6)
        
        # Schwellenwert gewichtete Prognose plotten falls verfügbar
        if threshold_weighted_forecast is not None:
            plt.plot([last_historical_date, forecast_dates[0]], [last_historical_value, threshold_weighted_forecast[0]], 
                    color='cyan', linestyle='--', alpha=0.5)
            plt.plot(forecast_dates, threshold_weighted_forecast, label='Threshold (>2) Weighted', color='cyan', linestyle='--')
            plt.scatter(forecast_dates, threshold_weighted_forecast, color='cyan', s=50, alpha=0.6)
        
        # Benutzer-Schwellenwert gewichtete Prognose plotten falls verfügbar
        if chatgpt_weighted_forecast is not None and chatgpt_threshold is not None:
            plt.plot([last_historical_date, forecast_dates[0]], [last_historical_value, chatgpt_weighted_forecast[0]], 
                    color='magenta', linestyle='--', alpha=0.5)
            plt.plot(forecast_dates, chatgpt_weighted_forecast, label=f'ChatGPT Threshold (>{chatgpt_threshold}) Weighted', color='magenta', linestyle='--')
            plt.scatter(forecast_dates, chatgpt_weighted_forecast, color='magenta', s=50, alpha=0.6)
        
        # ChatGPT-Prognose mit verbindender Linie und Punkten plotten
        plt.plot([last_historical_date, chatgpt_dates[0]], [last_historical_value, chatgpt_prices[0]], 
                color='purple', linestyle='--', alpha=0.5)
        plt.plot(chatgpt_dates, chatgpt_prices, label='ChatGPT Raw Forecast', color='purple', linestyle='--')
        plt.scatter(chatgpt_dates, chatgpt_prices, color='purple', s=50, alpha=0.6)
        
        # ChatGPT-historische Prognose mit verbindender Linie und Punkten plotten
        plt.plot([last_historical_date, chatgpt_historical_dates[0]], [last_historical_value, chatgpt_historical_prices[0]], 
                color='orange', linestyle='--', alpha=0.5)
        plt.plot(chatgpt_historical_dates, chatgpt_historical_prices, label='ChatGPT with historical dataForecast', color='orange', linestyle='--')
        plt.scatter(chatgpt_historical_dates, chatgpt_historical_prices, color='orange', s=50, alpha=0.6)
        
        plt.title(f'{ticker} - Forecast vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Plot speichern
        plot_path = os.path.join(dir_path, f'{ticker}_forecast_evaluation.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"Evaluation plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error in plot_comparison: {str(e)}")
        plt.close('all')  # Alle offenen Plots bereinigen
        raise

def main():
    # Ticker vom Benutzer abrufen
    ticker = "Alphabet"
    
    try:
        # Prognosedaten laden
        forecasts = load_forecast_data(ticker)
        
        # Verzeichnispfad aus der geladenen Prognosedatei abrufen
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        dir_path = os.path.join(OUTPUT_DIR, f"{ticker}_forecast{DEFAULT_FORECAST_STEPS}days_{yesterday}")
        
        # Tatsächliche Preise herunterladen einschließlich einiger historischer Daten für Kontext
        start_date = pd.to_datetime(forecasts['last_actual_date']) - timedelta(days=DEFAULT_SHOW_LAST_DAYS)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = (datetime.strptime(forecasts['forecast_end_date'], '%Y-%m-%d') + 
                   timedelta(days=1)).strftime('%Y-%m-%d')
        actual_prices = download_actual_prices("GOOGL", start_date, end_date)
        
        # Prognosedaten und Werte extrahieren
        forecast_dates = pd.to_datetime([f['date'] for f in forecasts['arima_forecast']])
        arima_forecast = np.array([f['arima_forecast'] for f in forecasts['arima_forecast']])
        
        # Prüfen ob Standard gewichtete Prognose existiert (Legacy)
        weighted_forecast = None
        if 'weighted_forecast' in forecasts:
            weighted_forecast = np.array(forecasts['weighted_forecast'])
        
        # Schwellenwert gewichtete Prognose extrahieren falls verfügbar
        threshold_weighted_forecast = None
        if 'threshold_weighted_forecast' in forecasts:
            threshold_weighted_forecast = np.array(forecasts['threshold_weighted_forecast'])
        
        # Benutzer gewichtete Prognose extrahieren falls verfügbar
        chatgpt_weighted_forecast = None
        if 'chatgpt_weighted_forecast' in forecasts:
            chatgpt_weighted_forecast = np.array(forecasts['chatgpt_weighted_forecast'])
        
        # ChatGPT-Prognose mit korrektem Datumsparsing extrahieren
        chatgpt_dates = pd.to_datetime([f['date'] for f in forecasts['chatgpt_forecast']['forecast']], format='%d.%m.%Y')
        chatgpt_prices = np.array([f['closing_price'] for f in forecasts['chatgpt_forecast']['forecast']])
        
        # ChatGPT-historische Prognose mit korrektem Datumsparsing extrahieren
        chatgpt_historical_dates = pd.to_datetime([f['date'] for f in forecasts['chatgpt_historical_forecast']['forecast']], format='%d.%m.%Y')
        chatgpt_historical_prices = np.array([f['closing_price'] for f in forecasts['chatgpt_historical_forecast']['forecast']])
        
        # Tatsächliche Preise für Prognosedaten abrufen
        actual_values = actual_prices[forecast_dates].values
        
        # Metriken für alle Prognosen berechnen
        arima_metrics = calculate_metrics(actual_values, arima_forecast)
        chatgpt_metrics = calculate_metrics(actual_values, chatgpt_prices)
        chatgpt_historical_metrics = calculate_metrics(actual_values, chatgpt_historical_prices)
        
        # Metriken für gewichtete Prognosen berechnen falls verfügbar
        weighted_metrics = None
        if weighted_forecast is not None:
            weighted_metrics = calculate_metrics(actual_values, weighted_forecast)
            
        # Metriken für Schwellenwert- und Benutzer-gewichtete Prognosen berechnen falls verfügbar
        threshold_weighted_metrics = None
        if threshold_weighted_forecast is not None:
            threshold_weighted_metrics = calculate_metrics(actual_values, threshold_weighted_forecast)
            
        chatgpt_weighted_metrics = None
        if chatgpt_weighted_forecast is not None:
            chatgpt_weighted_metrics = calculate_metrics(actual_values, chatgpt_weighted_forecast)
        
        # Ergebnisse ausgeben
        print("\nForecast Evaluation Results:")
        print(f"Ticker: {ticker}")
        print(f"Evaluation Period: {forecasts['forecast_start_date']} to {forecasts['forecast_end_date']}")
        
        print("\nARIMA Forecast Metrics:")
        for metric, value in arima_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Standard gewichtete Metriken ausgeben falls verfügbar (Legacy)
        if weighted_metrics:
            print("\nStandard Weighted Forecast Metrics:")
            for metric, value in weighted_metrics.items():
                print(f"{metric}: {value:.4f}")
        
        if threshold_weighted_metrics:
            print("\nThreshold (>2) Weighted Forecast Metrics:")
            for metric, value in threshold_weighted_metrics.items():
                print(f"{metric}: {value:.4f}")
                
        if chatgpt_weighted_metrics and 'chatgpt_threshold' in forecasts:
            print(f"\nChatGPT Threshold (>{forecasts['chatgpt_threshold']}) Weighted Forecast Metrics:")
            for metric, value in chatgpt_weighted_metrics.items():
                print(f"{metric}: {value:.4f}")
            
        print("\nChatGPT Forecast Metrics:")
        for metric, value in chatgpt_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nChatGPT Historical Forecast Metrics:")
        for metric, value in chatgpt_historical_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Erweiterten Plot-Vergleich erstellen
        plot_comparison(actual_prices, forecasts, ticker, dir_path)
        
        # Metriken in Datei im gleichen Verzeichnis speichern
        metrics = {
            'ticker': ticker,
            'evaluation_period': {
                'start': forecasts['forecast_start_date'],
                'end': forecasts['forecast_end_date']
            },
            'actual_values': {
                'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                'prices': actual_values.tolist()
            },
            'arima_metrics': arima_metrics,
            'chatgpt_metrics': chatgpt_metrics,
            'chatgpt_historical_metrics': chatgpt_historical_metrics
        }
        
        # Standard gewichtete Metriken hinzufügen falls verfügbar (Legacy)
        if weighted_metrics:
            metrics['weighted_metrics'] = weighted_metrics
        
        # Schwellenwert- und Benutzer-gewichtete Metriken hinzufügen falls verfügbar
        if threshold_weighted_metrics:
            metrics['threshold_weighted_metrics'] = threshold_weighted_metrics
            
        if chatgpt_weighted_metrics:
            metrics['chatgpt_weighted_metrics'] = chatgpt_weighted_metrics
            if 'chatgpt_threshold' in forecasts:
                metrics['chatgpt_threshold'] = forecasts['chatgpt_threshold']
        
        metrics_path = os.path.join(dir_path, f'{ticker}_evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nEvaluation results saved to {metrics_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()