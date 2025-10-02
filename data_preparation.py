import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def download_stock_data(tickers, start_date, end_date):
    """
    Lädt Aktiendaten für die gegebenen Ticker und den Zeitraum herunter.
    
    :param tickers: dict, Zuordnung von Ticker-Namen zu Symbolen
    :param start_date: str, Startdatum im Format 'YYYY-MM-DD'
    :param end_date: str, Enddatum im Format 'YYYY-MM-DD'
    :return: dict, Zuordnung von Ticker-Namen zu Preiserien
    """
    data = {}
    for name, symbol in tickers.items():
        print(f"Downloading data for {name} ({symbol})...")
        stock = yf.Ticker(symbol)
        data[name] = stock.history(start=start_date, end=end_date)['Close']
    return data

def check_stationarity(series):
    """
    Führt einen Augmented Dickey-Fuller Test durch um Stationarität zu prüfen.
    
    :param series: pd.Series, zu testende Zeitreihe
    :return: tuple mit (is_stationary, p_value)
             - is_stationary: bool, ob die Reihe stationär ist
             - p_value: float, p-Wert aus dem ADF Test
    """
    result = adfuller(series)
    p_value = result[1]
    return p_value <= 0.05, p_value

def difference_series(series):
    """
    Differenziert die Zeitreihe um Stationarität zu erreichen.
    
    :param series: pd.Series, zu differenzierende Zeitreihe
    :return: pd.Series, differenzierte Zeitreihe
    """
    return series.diff().dropna()

def prepare_stationary_data(data, tickers):
    """
    Bereitet stationäre Daten für jeden Ticker vor durch Anwendung von Differenzierung bis Stationarität erreicht ist.
    
    :param data: dict, Dictionary von DataFrames mit Aktiendaten
    :param tickers: dict, Dictionary von Ticker-Symbolen
    :return: dict, Dictionary von stationären DataFrames
    """
    stationary_data = {}
    
    for ticker in tickers.keys():
        print(f"\nPreparing stationary data for {ticker}...")
        
        # Mit ursprünglichen Daten beginnen
        current_data = data[ticker]
        is_stationary = False
        diff_order = 0
        
        while not is_stationary and diff_order < 3:  # Auf 3 Differenzen begrenzen um Überdifferenzierung zu vermeiden
            # Augmented Dickey-Fuller Test durchführen
            result = adfuller(current_data)
            p_value = result[1]
            
            print(f"ADF test p-value for {diff_order} differences: {p_value:.4f}")
            
            if p_value < 0.05:  # Wenn p-Wert kleiner als 0.05 ist, sind die Daten stationär
                is_stationary = True
                print(f"Data is stationary after {diff_order} differences")
            else:
                # Differenzierung anwenden
                current_data = np.diff(current_data)
                diff_order += 1
                print(f"Applying difference {diff_order}...")
        
        if not is_stationary:
            print(f"Warning: Could not achieve stationarity for {ticker} after {diff_order} differences")
            # Trotzdem die letzte differenzierte Daten verwenden
            current_data = difference_series(current_data)
        
        stationary_data[ticker] = current_data
        
        ## ACF und PACF für die stationären Daten plotten
        #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # plot_acf(current_data, ax=ax1, lags=40)
        # plot_pacf(current_data, ax=ax2, lags=40)
        # plt.tight_layout()
        # plt.show()
    
    return stationary_data 