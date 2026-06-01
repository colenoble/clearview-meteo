import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import xlsxwriter.utility as xl_util
import pvlib
import pytz
import sys

# --- Configuration ---
COMMISSIONING_DIR = Path(r"T:\Shared\93 Solar USA\Projects\9300001\Internal\14. Commissioning and Testing\Performance Testing Share Folder")
LOCAL_INPUTS_DIR = Path(r"inputs")
SCRIPT_NAME = Path(__file__).stem if '__file__' in globals() else 'calculated_albedo_processing'
OUTPUT_DIR = Path(r"outputs") / SCRIPT_NAME

# --- File Loading Rules ---
HIGH_RES_DATE_THRESHOLD = "2026-05-01" 
LEGACY_FILE_IDENTIFIER = "MET_"     
HIGH_RES_FILE_IDENTIFIER = "export" 
EXCLUDE_IDENTIFIERS = ["alarm", "event", "fault", "scada", "log"] 

# --- Variable Inputs ---
LATITUDE = 39.90
LONGITUDE = -84.22
TZ = pytz.FixedOffset(-300)  

STATIONS = ['02', '16', '22', '37']
REFERENCE_GHI_STATION = '37'  

# Expanded templates to catch high-res export formats
GHI_COL_TEMPLATES = ["MET{}/GHI2", "MET{}/GHI", "MET{}_GHI", "Station{}_GHI", "GHI_{}"]   
RHI_COL_TEMPLATES = ["MET{}/RHI", "MET{}_RHI", "Station{}_RHI", "RHI_{}"]
ACTUAL_ALBEDO_TEMPLATES = ["MET{}/ARRAY_ALBEDO", "MET{}_ARRAY_ALBEDO", "Station{}_ARRAY_ALBEDO", "Albedo_{}"] 

TIMESTAMP_COL = "t_stamp" 
FALLBACK_TIMESTAMP_COLS = ["timestamp", "date", "time", "datetime", "date/time"]

START_DATE = None 
END_DATE = None

ACTUAL_ALBEDO_DATE_THRESHOLD = "2026-05-01" 

# --- Filtering Parameters ---
START_TIME = "08:00"
END_TIME = "18:00"
MIN_GHI = 50        
MAX_ALBEDO = 1   
MIN_ALBEDO = 0.01   

MIN_EFF_ALBEDO = 0.0
MAX_EFF_ALBEDO = 2.0 

# --- Helper Function ---
def get_combined_series(df, templates, station):
    combined = pd.Series(np.nan, index=df.index)
    for t in templates:
        col = t.format(station)
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            combined = combined.fillna(series)
    return combined

def process_and_export(df_valid, output_folder, file_prefix, plot_title_prefix):
    output_folder.mkdir(parents=True, exist_ok=True)
    
    df_filt = pd.DataFrame(index=df_valid.index)
    
    if 'Unified_Ref_GHI' in df_valid.columns:
        df_filt[f'MET{REFERENCE_GHI_STATION}_GHI'] = df_valid['Unified_Ref_GHI']
    
    exclusion_stats = {}
    export_cols = [c for c in df_valid.columns if 'Calc_Albedo' in c or 'Eff_Albedo' in c or 'Actual_Albedo' in c]

    for col_name in export_cols:
        raw_data = df_valid[col_name]
        count_valid_data = raw_data.notna().sum()
        
        if 'Eff_Albedo' in col_name:
            min_mask = raw_data < MIN_EFF_ALBEDO
            max_mask = raw_data > MAX_EFF_ALBEDO
        else:
            min_mask = raw_data < MIN_ALBEDO
            max_mask = raw_data > MAX_ALBEDO
        
        if count_valid_data > 0:
            min_pct = min_mask.sum() / count_valid_data
            max_pct = max_mask.sum() / count_valid_data
        else:
            min_pct = max_pct = 0
            
        exclusion_stats[col_name] = {'min': min_pct, 'max': max_pct}
        out_of_bounds_mask = raw_data.notna() & (min_mask | max_mask)
        df_filt[col_name] = raw_data.where(~out_of_bounds_mask).replace([np.inf, -np.inf], np.nan)

    if 'Mean_GHI' in df_valid.columns:
        df_filt['Mean_GHI'] = df_valid['Mean_GHI']

    excel_path = output_folder / f"{file_prefix}_timeseries.xlsx"
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_filt.to_excel(writer, sheet_name='Filtered (Bounds)')
        df_filt.to_excel(writer, sheet_name='Filtered (GHI Weighted)')
        
        workbook = writer.book
        ws_bounds = writer.sheets['Filtered (Bounds)']
        ws_weight = writer.sheets['Filtered (GHI Weighted)']
        bold_fmt = workbook.add_format({'bold': True})
        pct_fmt = workbook.add_format({'bold': True, 'num_format': '0.00%'})
        
        num_rows = len(df_filt)
        calc_row_idx = num_rows + 1 
        
        ws_bounds.write_string(calc_row_idx, 0, "Period Median", bold_fmt)
        ws_bounds.write_string(calc_row_idx + 1, 0, f"Min Exclusions (<{MIN_ALBEDO})", bold_fmt)
        ws_bounds.write_string(calc_row_idx + 2, 0, f"Max Exclusions (>{MAX_ALBEDO})", bold_fmt)
        
        for col_num, col_name in enumerate(df_filt.columns, start=1):
            if "Albedo" in col_name:
                col_letter = xl_util.xl_col_to_name(col_num)
                ws_bounds.write_formula(calc_row_idx, col_num, f"=MEDIAN({col_letter}2:{col_letter}{num_rows+1})", bold_fmt)
                if col_name in exclusion_stats:
                    ws_bounds.write_number(calc_row_idx + 1, col_num, exclusion_stats[col_name]['min'], pct_fmt)
                    ws_bounds.write_number(calc_row_idx + 2, col_num, exclusion_stats[col_name]['max'], pct_fmt)
                
        ws_weight.write_string(calc_row_idx, 0, "Period GHI-Weighted Average", bold_fmt)
        ghi_letter = xl_util.xl_col_to_name(1) 
        
        for col_num, col_name in enumerate(df_filt.columns, start=1):
            if "Albedo" in col_name:
                col_letter = xl_util.xl_col_to_name(col_num)
                formula = f'=SUMPRODUCT(IF(ISNUMBER({col_letter}2:{col_letter}{num_rows+1}), {col_letter}2:{col_letter}{num_rows+1}, 0), IF(ISNUMBER({col_letter}2:{col_letter}{num_rows+1}), {ghi_letter}2:{ghi_letter}{num_rows+1}, 0)) / SUMIFS({ghi_letter}2:{ghi_letter}{num_rows+1}, {col_letter}2:{col_letter}{num_rows+1}, "<>")'
                ws_weight.write_array_formula(calc_row_idx, col_num, calc_row_idx, col_num, formula, bold_fmt)

    df_filt['Date'] = df_filt.index.date
    colors = plt.cm.tab10.colors 
    
    monthly_dev_records = []
    
    grouped_by_date = list(df_filt.groupby('Date'))
    total_days = len(grouped_by_date)
    
    for day_idx, (date, group) in enumerate(grouped_by_date, start=1):
        print(f"  -> Plotting daily data: [{day_idx}/{total_days}] processing {date} ...".ljust(80), end='\r')
        sys.stdout.flush()
        
        times = pd.date_range(start=f"{date} 00:00:00", end=f"{date} 23:59:00", freq='1min', tz=TZ)
        solpos = pvlib.solarposition.get_solarposition(times, LATITUDE, LONGITUDE)
        solar_noon = solpos['zenith'].idxmin().tz_localize(None) 
        
        win_start = solar_noon - pd.Timedelta(minutes=30)
        win_end = solar_noon + pd.Timedelta(minutes=30)
        noon_time = solar_noon.round('1min')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1_twin = ax1.twinx() 
        plotted_any = False
        
        solar_noon_str = solar_noon.strftime('%H:%M:%S')
        std_text1 = f"Solar Noon: {solar_noon_str}\n\nCalculated Values at Exact Noon:\n"
        std_text2 = f"Solar Noon: {solar_noon_str}\n\nValues & Ratio (R) to Top Albedo at Exact Noon:\n"
        
        noon_vals1 = []
        noon_vals2 = [] 
        
        if 'Mean_GHI' in group.columns:
            day_data_ghi = group['Mean_GHI'].dropna()
            if not day_data_ghi.empty:
                ax1_twin.plot(day_data_ghi.index, day_data_ghi, marker='', linestyle='--', 
                              linewidth=1.5, color='gray', alpha=0.6, label="Mean GHI (All Stations)")

        for i, station in enumerate(STATIONS):
            color = colors[i % len(colors)]
            
            calc_col = f'MET{station}_Calc_Albedo'
            actual_col = f'MET{station}_Actual_Albedo'
            eff1_col = f'MET{station}_Eff_Albedo_1' 
            eff2_col = f'MET{station}_Eff_Albedo_2' 
            
            day_data_calc_full = group[calc_col] if calc_col in group.columns else pd.Series(dtype=float)
            day_data_act_full = group[actual_col] if actual_col in group.columns else pd.Series(dtype=float)
            day_data_eff1_full = group[eff1_col] if eff1_col in group.columns else pd.Series(dtype=float)
            day_data_eff2_full = group[eff2_col] if eff2_col in group.columns else pd.Series(dtype=float)

            valid_calc = day_data_calc_full.dropna()
            if not valid_calc.empty:
                idx1 = valid_calc.index[np.argmin(np.abs(valid_calc.index - noon_time))]
                val1 = valid_calc.loc[idx1]
            else:
                val1 = np.nan
                
            valid_eff1 = day_data_eff1_full.dropna()
            if not valid_eff1.empty:
                idx2_1 = valid_eff1.index[np.argmin(np.abs(valid_eff1.index - noon_time))]
                val2_1 = valid_eff1.loc[idx2_1]
            else:
                val2_1 = np.nan
                
            valid_eff2 = day_data_eff2_full.dropna()
            if not valid_eff2.empty:
                idx2_2 = valid_eff2.index[np.argmin(np.abs(valid_eff2.index - noon_time))]
                val2_2 = valid_eff2.loc[idx2_2]
            else:
                val2_2 = np.nan
                
            if pd.notna(val1): noon_vals1.append(val1)
            
            if pd.notna(val2_1): noon_vals2.append(val2_1)
            if pd.notna(val2_2): noon_vals2.append(val2_2)
            
            # Explicitly plot Calculated Albedo 
            if not valid_calc.empty:
                ax1.plot(valid_calc.index, valid_calc, marker='.', linestyle='-', markersize=4, 
                         color=color, alpha=0.9, label=f"MET{station} (Calc)")
                
                v1_str = f"{val1*100:.2f}%" if pd.notna(val1) else "N/A"
                std_text1 += f"MET{station}: {v1_str}\n"
                plotted_any = True
                
            # Plot Actual Albedo as dashed background line if it exists
            valid_act = day_data_act_full.dropna()
            if not valid_act.empty:
                ax1.plot(valid_act.index, valid_act, marker='', linestyle='--', linewidth=1.5, 
                         color=color, alpha=0.5, label=f"MET{station} (Actual)")
                plotted_any = True
                    
            day_data_eff1 = day_data_eff1_full.dropna()
            day_data_eff1 = day_data_eff1[(day_data_eff1.index >= win_start) & (day_data_eff1.index <= win_end)]
            if not day_data_eff1.empty:
                ax2.plot(day_data_eff1.index, day_data_eff1, marker='.', linestyle=':', markersize=4, 
                         color=color, alpha=0.9, label=f"MET{station} S1")
                plotted_any = True
                
            day_data_eff2 = day_data_eff2_full.dropna()
            day_data_eff2 = day_data_eff2[(day_data_eff2.index >= win_start) & (day_data_eff2.index <= win_end)]
            if not day_data_eff2.empty:
                ax2.plot(day_data_eff2.index, day_data_eff2, marker='', linestyle='-', 
                         color=color, alpha=0.9, label=f"MET{station} S2")
                plotted_any = True

            if not day_data_eff1_full.empty or not day_data_eff2_full.empty:
                v2_1_str = f"{val2_1*100:.2f}%" if pd.notna(val2_1) else "N/A"
                v2_2_str = f"{val2_2*100:.2f}%" if pd.notna(val2_2) else "N/A"
                
                r1 = (val2_1 / val1) if (pd.notna(val1) and pd.notna(val2_1) and val1 != 0) else np.nan
                r2 = (val2_2 / val1) if (pd.notna(val1) and pd.notna(val2_2) and val1 != 0) else np.nan
                
                r1_str = f"{r1:.3f}" if pd.notna(r1) else "N/A"
                r2_str = f"{r2:.3f}" if pd.notna(r2) else "N/A"
                
                std_text2 += f"MET{station}  S1: {v2_1_str} (R: {r1_str}) | S2: {v2_2_str} (R: {r2_str})\n"

        day_record = {'Date': date}
        
        if plotted_any:
            std1 = np.std(noon_vals1) if len(noon_vals1) > 1 else np.nan
            std2 = np.std(noon_vals2) if len(noon_vals2) > 1 else np.nan  
            
            day_record['Overall_Sensor_Std_Dev'] = (std2) if pd.notna(std2) else np.nan
            
            std_text1 += f"---\nCross-Station Std Dev: {std1*100:.2f}%" if pd.notna(std1) else "---\nCross-Station Std Dev: N/A"
            std_text2 += f"---\nOverall Sensor Std Dev (n={len(noon_vals2)}): {std2*100:.2f}%" if pd.notna(std2) else "---\nOverall Sensor Std Dev: N/A"

            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            
            for ax in [ax1, ax2]:
                ax.axvline(solar_noon, color='gray', linestyle='--', alpha=0.6, label='Solar Noon')
                ax.grid(True, linestyle=':', alpha=0.6)

            ax1.set_ylim(bottom=0, top=MAX_ALBEDO + 0.05) 
            ax1_twin.set_ylim(bottom=0)
            
            ax1.set_title(f'{plot_title_prefix} and Mean GHI - {date}\n(Filtered: {START_TIME}-{END_TIME}, GHI>{MIN_GHI})')
            ax1.set_ylabel('Calculated Albedo (RHI/GHI)')
                
            ax1_twin.set_ylabel('Mean GHI (W/m²)', color='dimgray')
            ax1_twin.tick_params(axis='y', labelcolor='dimgray')
            
            ax2.set_title('Effective Albedo (Specific RPOA / Mean POA) - +/- 30 Min Solar Noon')
            ax2.set_ylabel('Effective Albedo Ratio')
            ax2.set_xlabel('Time')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='upper left', bbox_to_anchor=(1.14, 1.0))
            ax2.legend(loc='upper left', bbox_to_anchor=(1.14, 1.0))

            ax1.text(1.14, 0.0, std_text1.strip(), transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
            ax2.text(1.14, 0.0, std_text2.strip(), transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', bbox=props)
            
            fig.savefig(output_folder / f"{file_prefix}_{date}.png", dpi=300, bbox_inches='tight')
        else:
            day_record['Overall_Sensor_Std_Dev'] = np.nan
            
        plt.close(fig)
        monthly_dev_records.append(day_record)

    print("  -> Plotting complete!                                                                 ")
    return pd.DataFrame(monthly_dev_records)

def main():
    possible_files = []
    
    # 1. Process Network Commissioning Directory (Strict "MET_" rule or high-res export)
    if COMMISSIONING_DIR.exists():
        for f in COMMISSIONING_DIR.rglob('*.csv'):
            if any(ex_word in f.name.lower() for ex_word in EXCLUDE_IDENTIFIERS):
                continue
            
            # Must strictly contain "MET_" or match the high-res identifier
            if "MET_" in f.name or HIGH_RES_FILE_IDENTIFIER.lower() in f.name.lower():
                possible_files.append(f)
    else:
        print(f"[WARNING] Commissioning directory not found: {COMMISSIONING_DIR}")

    # 2. Process Local Inputs Directory (Any CSV rule, bypassing exclusions)
    if LOCAL_INPUTS_DIR.exists():
        for f in LOCAL_INPUTS_DIR.rglob('*.csv'):
            if any(ex_word in f.name.lower() for ex_word in EXCLUDE_IDENTIFIERS):
                continue
            possible_files.append(f)
    else:
        print(f"[INFO] Local 'inputs' directory not found: {LOCAL_INPUTS_DIR}. Skipping.")
        
    total_files = len(possible_files)
    if not possible_files:
        print(f"[ERROR] No valid CSV files found in specified directories.")
        return
        
    print(f"\n[INFO] Found {total_files} potential target files. Enforcing date thresholds & resolution deduplication...")
    
    best_data_for_date = {}
    best_res_for_date = {}

    for i, f in enumerate(possible_files, start=1):
        print(f"  -> Parsing file [{i}/{total_files}]: {f.name}".ljust(100), end='\r')
        sys.stdout.flush()
        
        try:
            file_df = pd.read_csv(f)
            file_df.columns = file_df.columns.str.strip()
            
            ts_col = None
            if TIMESTAMP_COL in file_df.columns:
                ts_col = TIMESTAMP_COL
            else:
                for fallback in FALLBACK_TIMESTAMP_COLS:
                    matched_cols = [c for c in file_df.columns if c.lower() == fallback]
                    if matched_cols:
                        ts_col = matched_cols[0]
                        break
            
            if not ts_col:
                print(f"\n[WARNING] Missing recognized timestamp column in {f.name}. Found headers: {list(file_df.columns[:5])}")
                continue
                
            file_df[ts_col] = pd.to_datetime(file_df[ts_col])
            file_df = file_df.sort_values(ts_col).set_index(ts_col)
            
            for date, group in file_df.groupby(file_df.index.date):
                is_after_threshold = date >= pd.to_datetime(HIGH_RES_DATE_THRESHOLD).date()
                
                if is_after_threshold and HIGH_RES_FILE_IDENTIFIER and HIGH_RES_FILE_IDENTIFIER.lower() not in f.name.lower():
                    continue 
                # Strict case enforcement for legacy file check
                if not is_after_threshold and LEGACY_FILE_IDENTIFIER and LEGACY_FILE_IDENTIFIER not in f.name:
                    continue 

                if len(group) < 2:
                    current_res = pd.Timedelta(days=99)
                else:
                    current_res = group.index.to_series().diff().median()
                
                if date not in best_data_for_date:
                    best_data_for_date[date] = group
                    best_res_for_date[date] = current_res
                else:
                    if current_res < best_res_for_date[date]:
                        best_data_for_date[date] = group
                        best_res_for_date[date] = current_res
                    elif current_res == best_res_for_date[date]:
                        if len(group) > len(best_data_for_date[date]):
                            best_data_for_date[date] = group
                            best_res_for_date[date] = current_res
                            
        except Exception as e:
            print(f"\n[WARNING] Failed to parse {f.name}. Error: {e}") 
            
    print("\n[INFO] File parsing complete.                                                                  ")

    if not best_data_for_date:
        print("\n[ERROR] No valid dates found after applying parsing rules.")
        return

    print(f"[DEBUG] Assembled {len(best_data_for_date)} unique days of correct-resolution data.")
    
    df = pd.concat(best_data_for_date.values())
    df = df.sort_index()
    df = df.groupby(level=0).first()
    
    if START_DATE:
        df = df[df.index >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df.index <= pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if df.empty:
        print("\n[ERROR] Dataframe is empty after applying the date filter.")
        return

    ref_ghi_series = get_combined_series(df, GHI_COL_TEMPLATES, REFERENCE_GHI_STATION)
    if ref_ghi_series.isna().all():
        print(f"\n[ERROR] Reference GHI columns not found across any templates.")
        return
        
    df = df.copy()
    df['Unified_Ref_GHI'] = ref_ghi_series

    time_mask = pd.Series(False, index=df.index)
    time_index = df.between_time(START_TIME, END_TIME).index
    time_mask.loc[time_index] = True
    
    ghi_mask = df['Unified_Ref_GHI'] > MIN_GHI

    df_valid = df[time_mask & ghi_mask].copy()
    print(f"[DEBUG] Kept {len(df_valid)} valid daylight timesteps out of {len(df)} merged rows.\n")

    print("[INFO] Processing Standard Albedos, Specific Effective Albedos, and Mean GHI...")
    
    ghi_df = pd.DataFrame(index=df_valid.index)
    for station in STATIONS:
        series = get_combined_series(df_valid, GHI_COL_TEMPLATES, station)
        if not series.isna().all():
            ghi_df[f'MET{station}_GHI'] = series
            
    if not ghi_df.empty:
        df_valid['Mean_GHI'] = ghi_df.mean(axis=1)

    for station in STATIONS:
        ghi_series = get_combined_series(df_valid, GHI_COL_TEMPLATES, station)
        rhi_series = get_combined_series(df_valid, RHI_COL_TEMPLATES, station)
        
        if not rhi_series.isna().all() and not ghi_series.isna().all():
            df_valid[f'MET{station}_Calc_Albedo'] = rhi_series / ghi_series.replace(0, np.nan)
            
        actual_series = get_combined_series(df_valid, ACTUAL_ALBEDO_TEMPLATES, station)
        if not actual_series.isna().all():
             df_valid[f'MET{station}_Actual_Albedo'] = actual_series

    print("\n--- Diagnostic Check: Effective Albedo ---")
    for station in STATIONS:
        rpoa_cols = [f"MET{station}/RPOA_1", f"MET{station}/RPOA_2"]
        poa_cols = [f"MET{station}/POA_1", f"MET{station}/POA_2"]
        
        avail_rpoa = [c for c in rpoa_cols if c in df_valid.columns]
        avail_poa = [c for c in poa_cols if c in df_valid.columns]
        
        if avail_rpoa and avail_poa:
            df_poa = df_valid[avail_poa].apply(pd.to_numeric, errors='coerce')
            poa_mean = df_poa.mean(axis=1)
            
            df_rpoa = df_valid[avail_rpoa].apply(pd.to_numeric, errors='coerce')
            rpoa_mean = df_rpoa.mean(axis=1)
            eff_alb = rpoa_mean / poa_mean.replace(0, np.nan)
            df_valid[f'MET{station}_Eff_Albedo'] = eff_alb
            
            for c in avail_rpoa:
                sensor_idx = c.split('_')[-1] 
                series = pd.to_numeric(df_valid[c], errors='coerce')
                df_valid[f'MET{station}_Eff_Albedo_{sensor_idx}'] = series / poa_mean.replace(0, np.nan)
            
            valid_counts = eff_alb.notna().sum()
            print(f"MET{station} | Valid rows: {valid_counts} | Min: {eff_alb.min():.4f} | Max: {eff_alb.max():.4f}")
        else:
             print(f"MET{station} | Missing Required RPOA or POA Columns")
    print("------------------------------------------\n")

    calculated_cols = [c for c in df_valid.columns if 'Calc_Albedo' in c or 'Eff_Albedo' in c or 'Actual_Albedo' in c]
    if not calculated_cols:
        print("\n[WARNING] No standard or effective albedos calculated.")
        return

    df_valid['YearMonth'] = df_valid.index.strftime('%Y-%m')
    all_deviations = []
    
    for ym, group_df in df_valid.groupby('YearMonth'):
        print(f"\n[INFO] Generating exports and plots for {ym}...")
        ym_output_folder = OUTPUT_DIR / ym
        dev_df = process_and_export(
            df_valid=group_df,
            output_folder=ym_output_folder,
            file_prefix=f"Calculated_Albedo_{ym}",
            plot_title_prefix="Calculated Albedo"
        )
        if not dev_df.empty:
            all_deviations.append(dev_df)
            
    if all_deviations:
        summary_df = pd.concat(all_deviations, ignore_index=True)
        summary_df = summary_df.dropna(subset=['Overall_Sensor_Std_Dev'])
        summary_df.set_index('Date', inplace=True)
        
        summary_path = OUTPUT_DIR / "Overall_Sensor_Std_Deviation_Summary.xlsx"
        with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Daily Overall Std Dev')
            
            workbook = writer.book
            ws = writer.sheets['Daily Overall Std Dev']
            pct_fmt = workbook.add_format({'num_format': '0.00%'})
            
            ws.set_column(0, 0, 15) 
            ws.set_column(1, 1, 25, pct_fmt)
                
        print(f"\n[SUCCESS] Exported Overall Deviation Summary to: {summary_path.name}")

    print("\nProcessing complete.")

if __name__ == '__main__':
    main()