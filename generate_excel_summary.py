import os
import json
import pandas as pd
import glob
from datetime import datetime
import numpy as np

def extract_forecast_data_from_folder(folder_path):
    """
    Extrahiert alle Prognosedaten aus einem einzelnen Ordner.
    
    :param folder_path: str, Pfad zum Prognose-Ordner
    :return: Liste von Dictionaries mit Prognosedaten
    """
    data_rows = []
    
    try:
        # Ticker aus Ordnernamen abrufen
        ticker = os.path.basename(folder_path).split('_')[0]
        
        # Prognosedaten lesen (zuerst fixed Version versuchen, dann Original)
        forecasts_file = os.path.join(folder_path, f'{ticker}_forecasts_fixed.json')
        if not os.path.exists(forecasts_file):
            forecasts_file = os.path.join(folder_path, f'{ticker}_forecasts.json')
        
        if not os.path.exists(forecasts_file):
            print(f"  ⚠ No forecasts file found in {folder_path}")
            return data_rows
        
        with open(forecasts_file, 'r') as f:
            forecasts_data = json.load(f)
        
        # Bewertungsmetriken lesen (zuerst fixed Version versuchen, dann Original)
        eval_file = os.path.join(folder_path, f'{ticker}_evaluation_metrics_fixed.json')
        if not os.path.exists(eval_file):
            eval_file = os.path.join(folder_path, f'{ticker}_evaluation_metrics.json')
        
        actual_values = {}
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
                actual_values = eval_data.get('actual_values', {})
        
        # Prognosedaten und Daten extrahieren
        forecast_entries = forecasts_data.get('arima_forecast', [])
        
        for entry in forecast_entries:
            date = entry.get('date', '')
            arima_forecast = entry.get('arima_forecast', '')
            threshold_weighted_forecast = entry.get('threshold_weighted_forecast', '')
            chatgpt_weighted_forecast = entry.get('chatgpt_weighted_forecast', '')
            
            # Entsprechende ChatGPT-Prognose finden
            chatgpt_forecast = None
            chatgpt_confidence = ''
            chatgpt_reasoning = ''
            
            chatgpt_data = forecasts_data.get('chatgpt_forecast', {})
            if 'forecast' in chatgpt_data:
                for cf in chatgpt_data['forecast']:
                    cf_date = cf.get('date', '')
                    # Datumsformat konvertieren falls nötig
                    if cf_date:
                        try:
                            cf_date_obj = datetime.strptime(cf_date, '%d.%m.%Y')
                            entry_date_obj = datetime.strptime(date, '%Y-%m-%d')
                            if cf_date_obj.date() == entry_date_obj.date():
                                chatgpt_forecast = cf.get('closing_price', '')
                                chatgpt_confidence = cf.get('confidence', '')
                                chatgpt_reasoning = cf.get('reasoning', '')
                                break
                        except:
                            pass
            
            # Entsprechende ChatGPT-historische Prognose finden
            chatgpt_hist_forecast = None
            chatgpt_hist_confidence = ''
            chatgpt_hist_reasoning = ''
            
            chatgpt_hist_data = forecasts_data.get('chatgpt_historical_forecast', {})
            if 'forecast' in chatgpt_hist_data:
                for cf in chatgpt_hist_data['forecast']:
                    cf_date = cf.get('date', '')
                    # Datumsformat konvertieren falls nötig
                    if cf_date:
                        try:
                            cf_date_obj = datetime.strptime(cf_date, '%d.%m.%Y')
                            entry_date_obj = datetime.strptime(date, '%Y-%m-%d')
                            if cf_date_obj.date() == entry_date_obj.date():
                                chatgpt_hist_forecast = cf.get('closing_price', '')
                                chatgpt_hist_confidence = cf.get('confidence', '')
                                chatgpt_hist_reasoning = cf.get('reasoning', '')
                                break
                        except:
                            pass
            
            # Modellinformationen abrufen
            chatgpt_model_name = ''
            chatgpt_model_description = ''
            chatgpt_model_parameters = ''
            chatgpt_model_rationale = ''
            
            if 'model_used' in chatgpt_data:
                model = chatgpt_data['model_used']
                chatgpt_model_name = model.get('name', '')
                chatgpt_model_description = model.get('description', '')
                chatgpt_model_parameters = model.get('parameters', '')
                chatgpt_model_rationale = model.get('rationale', '')
            
            chatgpt_hist_model_name = ''
            chatgpt_hist_model_description = ''
            chatgpt_hist_model_parameters = ''
            chatgpt_hist_model_rationale = ''
            
            if 'model_used' in chatgpt_hist_data:
                model = chatgpt_hist_data['model_used']
                chatgpt_hist_model_name = model.get('name', '')
                chatgpt_hist_model_description = model.get('description', '')
                chatgpt_hist_model_parameters = model.get('parameters', '')
                chatgpt_hist_model_rationale = model.get('rationale', '')
            
            # Schwellenwerte abrufen
            threshold_value = 4  # Standard-Schwellenwert für threshold_weighted_forecast
            chatgpt_threshold = forecasts_data.get('chatgpt_threshold', '')
            
            # Tatsächlichen Wert für dieses Datum finden
            actual_value = ''
            if actual_values:
                actual_dates = actual_values.get('dates', [])
                actual_prices = actual_values.get('prices', [])
                for i, act_date in enumerate(actual_dates):
                    if act_date == date:
                        actual_value = actual_prices[i] if i < len(actual_prices) else ''
                        break
            
            # Datenzeile erstellen
            data_row = {
                'Date': date,
                'ARIMA_Forecast': arima_forecast,
                'ChatGPT_Forecast': chatgpt_forecast,
                'ChatGPT_Confidence': chatgpt_confidence,
                'ChatGPT_Reasoning': chatgpt_reasoning,
                'ChatGPT_Historical_Forecast': chatgpt_hist_forecast,
                'ChatGPT_Historical_Confidence': chatgpt_hist_confidence,
                'ChatGPT_Historical_Reasoning': chatgpt_hist_reasoning,
                'ChatGPT_Model_Name': chatgpt_model_name,
                'ChatGPT_Model_Description': chatgpt_model_description,
                'ChatGPT_Model_Parameters': chatgpt_model_parameters,
                'ChatGPT_Model_Rationale': chatgpt_model_rationale,
                'ChatGPT_Historical_Model_Name': chatgpt_hist_model_name,
                'ChatGPT_Historical_Model_Description': chatgpt_hist_model_description,
                'ChatGPT_Historical_Model_Parameters': chatgpt_hist_model_parameters,
                'ChatGPT_Historical_Model_Rationale': chatgpt_hist_model_rationale,
                'Threshold_Weighted_Forecast': threshold_weighted_forecast,
                'Threshold_Value': threshold_value,
                'ChatGPT_Weighted_Forecast': chatgpt_weighted_forecast,
                'ChatGPT_Threshold': chatgpt_threshold,
                'Actual_Value': actual_value,
                'Folder': os.path.basename(folder_path)
            }
            
            data_rows.append(data_row)
        
        print(f"  ✓ Extracted {len(data_rows)} forecast entries from {folder_path}")
        
    except Exception as e:
        print(f"  ✗ Error processing {folder_path}: {str(e)}")
    
    return data_rows

def generate_excel_summary():
    """
    Generiert eine umfassende Excel-Datei mit allen Prognosedaten.
    """
    print("Starting Excel summary generation...")
    print("=" * 60)
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found!")
        return
    
    # Alle Prognose-Ordner finden
    forecast1day_folders = glob.glob(os.path.join(output_dir, "*_forecast1days_*"))
    forecast5day_folders = glob.glob(os.path.join(output_dir, "*_forecast5days_*"))
    
    print(f"Found {len(forecast1day_folders)} 1-day forecast folders")
    print(f"Found {len(forecast5day_folders)} 5-day forecast folders")
    
    # 1-Tage-Prognosen verarbeiten
    print("\nProcessing 1-day forecasts...")
    forecast1day_data = []
    for folder in sorted(forecast1day_folders):
        data_rows = extract_forecast_data_from_folder(folder)
        forecast1day_data.extend(data_rows)
    
    # 5-Tage-Prognosen verarbeiten
    print("\nProcessing 5-day forecasts...")
    forecast5day_data = []
    for folder in sorted(forecast5day_folders):
        data_rows = extract_forecast_data_from_folder(folder)
        forecast5day_data.extend(data_rows)
    
    # DataFrames erstellen
    df_1day = pd.DataFrame(forecast1day_data)
    df_5day = pd.DataFrame(forecast5day_data)
    
    # Excel-Datei erstellen
    excel_filename = "forecast_summary.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 1-Tage-Prognosen-Blatt schreiben
        if not df_1day.empty:
            df_1day.to_excel(writer, sheet_name='Forecast_1_Day', index=False)
            print(f"  ✓ Wrote {len(df_1day)} rows to 'Forecast_1_Day' sheet")
        else:
            print(f"  ⚠ No 1-day forecast data to write")
        
        # 5-Tage-Prognosen-Blatt schreiben
        if not df_5day.empty:
            df_5day.to_excel(writer, sheet_name='Forecast_5_Day', index=False)
            print(f"  ✓ Wrote {len(df_5day)} rows to 'Forecast_5_Day' sheet")
        else:
            print(f"  ⚠ No 5-day forecast data to write")
    
    print(f"\n✓ Excel file created: {excel_filename}")
    print(f"  - 1-day forecasts: {len(df_1day)} rows")
    print(f"  - 5-day forecasts: {len(df_5day)} rows")
    
    # Zusammenfassungsstatistiken ausgeben
    if not df_1day.empty:
        print(f"\n1-Day Forecast Summary:")
        print(f"  - Date range: {df_1day['Date'].min()} to {df_1day['Date'].max()}")
        print(f"  - Unique folders: {df_1day['Folder'].nunique()}")
    
    if not df_5day.empty:
        print(f"\n5-Day Forecast Summary:")
        print(f"  - Date range: {df_5day['Date'].min()} to {df_5day['Date'].max()}")
        print(f"  - Unique folders: {df_5day['Folder'].nunique()}")

if __name__ == "__main__":
    generate_excel_summary() 