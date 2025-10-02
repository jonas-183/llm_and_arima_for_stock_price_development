import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def fit_arima(series):
    """
    Passt ein ARIMA-Modell an die Zeitreihe an unter Verwendung von Grid-Search für optimale Parameter.
    
    :param series: pd.Series, die anzupassende Zeitreihe
    :return: tuple mit (best_model, best_order, best_model_fit)
             - best_model: das beste ARIMA-Modell
             - best_order: tuple der (p,d,q) Parameter
             - best_model_fit: das angepasste Modell
    """
    try:
        # Parameterbereiche definieren
        p_range = range(0, 6)  # AR-Ordnung
        q_range = range(0, 6)  # MA-Ordnung
        
        best_aic = float('inf')
        best_order = None
        best_model = None
        
        # Grid-Search für optimale Parameter
        for p in p_range:
            for q in q_range:
                try:
                    # ARIMA-Modell anpassen
                    model = ARIMA(series, order=(p,0,q))
                    model_fit = model.fit()
                    
                    # Prüfen ob dieses Modell besser ist
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p,0,q)
                        best_model = model_fit
                        
                except:
                    continue
        
        if best_model is None:
            print("Could not find a suitable ARIMA model")
            return None, None, None
            
        print(f"Best ARIMA parameters: {best_order}")
        print(f"AIC: {best_aic:.2f}")
        
        return best_model, best_order, best_model
        
    except Exception as e:
        print(f"Error fitting ARIMA model: {str(e)}")
        return None, None, None

def generate_forecast_paths(model_fit, steps, n_paths):
    """
    Generiert Prognose-Pfade mit dem angepassten ARIMA-Modell.
    
    :param model_fit: angepasstes ARIMA-Modell
    :param steps: int, Anzahl der Geschäftstage zu prognostizieren
    :param n_paths: int, Anzahl der zu generierenden Pfade
    :return: tuple mit (forecast, simulated_paths)
             - forecast: Array der prognostizierten Werte
             - simulated_paths: Array der simulierten Pfade
    """
    # Punktprognose generieren
    forecast = model_fit.forecast(steps=steps)
    
    # Simulierte Pfade generieren
    simulated_paths = []
    for _ in range(n_paths):
        sim = model_fit.simulate(nsimulations=steps)
        simulated_paths.append(sim)
    
    return forecast, np.array(simulated_paths)

def transform_to_original_scale(forecast, simulated_paths, last_original_value, last_date):
    """
    Transformiert differenzierte Prognosen zurück zur ursprünglichen Skala.
    
    :param forecast: array, prognostizierte Werte
    :param simulated_paths: array, simulierte Pfade
    :param last_original_value: float, letzter Wert der ursprünglichen Reihe
    :param last_date: datetime, letztes Datum der ursprünglichen Reihe
    :return: tuple von (transformierte Prognose, transformierte Pfade, Prognose-Daten)
    """
    # Geschäftstage generieren (Wochenenden und US-Feiertage ausschließend)
    from pandas.tseries.offsets import BDay
    from pandas.tseries.holiday import USFederalHolidayCalendar
    
    # Geschäftstag-Kalender erstellen
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=last_date, end=last_date + BDay(forecast.shape[0] + 10))
    
    # Geschäftstage generieren
    forecast_dates = []
    current_date = last_date + BDay(1)
    
    while len(forecast_dates) < forecast.shape[0]:
        # Prüfen ob es ein Wochentag ist (0-4 = Montag-Freitag) und kein Feiertag
        if current_date.weekday() < 5 and current_date not in holidays:
            forecast_dates.append(current_date)
        current_date += BDay(1)
    
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    
    # Prognose transformieren
    forecast_original = np.cumsum(forecast) + last_original_value
    
    # Simulierte Pfade transformieren
    simulated_paths_original = np.cumsum(simulated_paths, axis=1) + last_original_value
    
    return forecast_original, simulated_paths_original, forecast_dates 