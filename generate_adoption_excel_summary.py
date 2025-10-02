import os
import json
import pandas as pd
import glob
from datetime import datetime
import numpy as np

def extract_adoption_forecast_data_from_folder(folder_path):
    """
    Extrahiert alle Prognosedaten aus einem einzelnen adoption-Ordner.
    
    :param folder_path: str, Pfad zum adoption-Prognose-Ordner
    :return: Liste von Dictionaries mit Prognosedaten
    """
    data_rows = []
    
    try:
        # Ticker aus Ordnernamen abrufen
        folder_name = os.path.basename(folder_path)
        ticker = folder_name.split('_')[1]  # adoption_Alphabet_forecast1days_20250718
        
        # Prognosedaten lesen (zuerst fixed Version versuchen, dann Original)
        forecasts_file = os.path.join(folder_path, f'{ticker}_forecasts_fixed.json')
        if not os.path.exists(forecasts_file):
            forecasts_file = os.path.join(folder_path, f'{ticker}_forecasts.json')
        
        if not os.path.exists(forecasts_file):
            print(f"  ⚠ No forecasts file found in {folder_path}")
            return data_rows
        
        with open(forecasts_file, 'r') as f:
            forecasts_data = json.load(f)
        
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
            
            # Zusätzliche adoption-spezifische Daten abrufen
            forecast_start_date = forecasts_data.get('forecast_start_date', '')
            forecast_end_date = forecasts_data.get('forecast_end_date', '')
            last_actual_price = forecasts_data.get('last_actual_price', '')
            last_actual_date = forecasts_data.get('last_actual_date', '')
            
            # Simulierte Pfade-Daten abrufen
            simulated_paths = forecasts_data.get('simulated_paths', [])
            simulated_paths_count = len(simulated_paths) if simulated_paths else 0
            
            # Statistiken aus simulierten Pfaden berechnen
            simulated_stats = {}
            if simulated_paths:
                # Pfade abflachen (jeder Pfad ist eine Liste mit einem Wert für 1-Tage-Prognosen)
                flat_values = [path[0] if path and len(path) > 0 else None for path in simulated_paths]
                flat_values = [v for v in flat_values if v is not None]
                
                if flat_values:
                    simulated_stats = {
                        'simulated_mean': np.mean(flat_values),
                        'simulated_std': np.std(flat_values),
                        'simulated_min': np.min(flat_values),
                        'simulated_max': np.max(flat_values),
                        'simulated_median': np.median(flat_values)
                    }
            
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
                'Forecast_Start_Date': forecast_start_date,
                'Forecast_End_Date': forecast_end_date,
                'Last_Actual_Price': last_actual_price,
                'Last_Actual_Date': last_actual_date,
                'Simulated_Paths_Count': simulated_paths_count,
                'Simulated_Mean': simulated_stats.get('simulated_mean', ''),
                'Simulated_Std': simulated_stats.get('simulated_std', ''),
                'Simulated_Min': simulated_stats.get('simulated_min', ''),
                'Simulated_Max': simulated_stats.get('simulated_max', ''),
                'Simulated_Median': simulated_stats.get('simulated_median', ''),
                'Folder': os.path.basename(folder_path)
            }
            
            data_rows.append(data_row)
        
        print(f"  ✓ Extracted {len(data_rows)} forecast entries from {folder_path}")
        
    except Exception as e:
        print(f"  ✗ Error processing {folder_path}: {str(e)}")
    
    return data_rows

def generate_adoption_excel_summary():
    """
    Generiert eine umfassende Excel-Datei mit allen adoption-Prognosedaten.
    """
    print("Starting Adoption Excel summary generation...")
    print("=" * 60)
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found!")
        return
    
    # Alle adoption-Prognose-Ordner finden
    adoption_forecast1day_folders = glob.glob(os.path.join(output_dir, "adoption_*_forecast1days_*"))
    adoption_forecast5day_folders = glob.glob(os.path.join(output_dir, "adoption_*_forecast5days_*"))
    
    print(f"Found {len(adoption_forecast1day_folders)} adoption 1-day forecast folders")
    print(f"Found {len(adoption_forecast5day_folders)} adoption 5-day forecast folders")
    
    # 1-Tage-Prognosen verarbeiten
    print("\nProcessing adoption 1-day forecasts...")
    adoption_forecast1day_data = []
    for folder in sorted(adoption_forecast1day_folders):
        data_rows = extract_adoption_forecast_data_from_folder(folder)
        adoption_forecast1day_data.extend(data_rows)
    
    # 5-Tage-Prognosen verarbeiten
    print("\nProcessing adoption 5-day forecasts...")
    adoption_forecast5day_data = []
    for folder in sorted(adoption_forecast5day_folders):
        data_rows = extract_adoption_forecast_data_from_folder(folder)
        adoption_forecast5day_data.extend(data_rows)
    
    # DataFrames erstellen
    df_1day = pd.DataFrame(adoption_forecast1day_data)
    df_5day = pd.DataFrame(adoption_forecast5day_data)
    
    # Excel-Datei erstellen
    excel_filename = "adoption_forecast_summary.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 1-Tage-Prognosen-Blatt schreiben
        if not df_1day.empty:
            df_1day.to_excel(writer, sheet_name='Adoption_Forecast_1_Day', index=False)
            print(f"  ✓ Wrote {len(df_1day)} rows to 'Adoption_Forecast_1_Day' sheet")
        else:
            print(f"  ⚠ No adoption 1-day forecast data to write")
        
        # 5-Tage-Prognosen-Blatt schreiben
        if not df_5day.empty:
            df_5day.to_excel(writer, sheet_name='Adoption_Forecast_5_Day', index=False)
            print(f"  ✓ Wrote {len(df_5day)} rows to 'Adoption_Forecast_5_Day' sheet")
        else:
            print(f"  ⚠ No adoption 5-day forecast data to write")
    
    print(f"\n✓ Excel file created: {excel_filename}")
    print(f"  - Adoption 1-day forecasts: {len(df_1day)} rows")
    print(f"  - Adoption 5-day forecasts: {len(df_5day)} rows")
    
    # Zusammenfassungsstatistiken ausgeben
    if not df_1day.empty:
        print(f"\nAdoption 1-Day Forecast Summary:")
        print(f"  - Date range: {df_1day['Date'].min()} to {df_1day['Date'].max()}")
        print(f"  - Unique folders: {df_1day['Folder'].nunique()}")
        print(f"  - Average simulated paths per forecast: {df_1day['Simulated_Paths_Count'].mean():.1f}")
    
    if not df_5day.empty:
        print(f"\nAdoption 5-Day Forecast Summary:")
        print(f"  - Date range: {df_5day['Date'].min()} to {df_5day['Date'].max()}")
        print(f"  - Unique folders: {df_5day['Folder'].nunique()}")
        print(f"  - Average simulated paths per forecast: {df_5day['Simulated_Paths_Count'].mean():.1f}")

if __name__ == "__main__":
    generate_adoption_excel_summary() 