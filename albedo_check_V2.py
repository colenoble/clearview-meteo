import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Forces non-interactive backend to prevent GUI crashes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import xlsxwriter.utility as xl_util

# --- Configuration ---
INPUT_DIR = Path(r"inputs/AlbedoData")
SCRIPT_NAME = Path(__file__).stem if '__file__' in globals() else 'calculated_albedo_processing'
OUTPUT_DIR = Path(r"outputs") / SCRIPT_NAME

# --- Variable Inputs ---
STATIONS = ['02', '16', '22', '37']
REFERENCE_GHI_STATION = '37'  
GHI_COL_TEMPLATE = "MET{}/GHI"
RHI_COL_TEMPLATE = "MET{}/RHI"
TIMESTAMP_COL = "t_stamp" 

START_DATE = '2026-05-02' 
END_DATE = '2026-05-03'

# --- Filtering Parameters ---
START_TIME = "08:00"
END_TIME = "18:00"
MIN_GHI = 50        
MAX_ALBEDO = 1   
MIN_ALBEDO = 0.01   

def process_and_export(df_valid, data_dict, output_folder, file_prefix, plot_title_prefix):
    """
    Handles bounds filtering, Excel export with summary formulas, and daily plotting.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    
    df_filt = pd.DataFrame(index=df_valid.index)
    ref_ghi_col = GHI_COL_TEMPLATE.format(REFERENCE_GHI_STATION)
    df_filt[f'MET{REFERENCE_GHI_STATION}_GHI'] = df_valid[ref_ghi_col]
    
    exclusion_stats = {}
    export_cols = []

    # 1. Apply Bounds and Calculate Stats
    for clean_col_name, raw_data in data_dict.items():
        count_valid_data = raw_data.notna().sum()
        min_mask = raw_data < MIN_ALBEDO
        max_mask = raw_data > MAX_ALBEDO
        
        if count_valid_data > 0:
            min_pct = min_mask.sum() / count_valid_data
            max_pct = max_mask.sum() / count_valid_data
        else:
            min_pct = max_pct = 0
            
        exclusion_stats[clean_col_name] = {'min': min_pct, 'max': max_pct}
        
        # Filter out bounds by leaving them blank
        out_of_bounds_mask = raw_data.notna() & (min_mask | max_mask)
        df_filt[clean_col_name] = raw_data.where(~out_of_bounds_mask).replace([np.inf, -np.inf], np.nan)
        export_cols.append(clean_col_name)

    # 2. Export Timeseries to Excel with Formulas
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
        
        # Format Sheet 1: Medians & Exclusion Stats
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
                
        # Format Sheet 2: GHI-Weighted Average
        ws_weight.write_string(calc_row_idx, 0, "Period GHI-Weighted Albedo", bold_fmt)
        ghi_letter = xl_util.xl_col_to_name(1) 
        
        for col_num, col_name in enumerate(df_filt.columns, start=1):
            if "Albedo" in col_name:
                col_letter = xl_util.xl_col_to_name(col_num)
                formula = f'=SUMPRODUCT(IF(ISNUMBER({col_letter}2:{col_letter}{num_rows+1}), {col_letter}2:{col_letter}{num_rows+1}, 0), IF(ISNUMBER({col_letter}2:{col_letter}{num_rows+1}), {ghi_letter}2:{ghi_letter}{num_rows+1}, 0)) / SUMIFS({ghi_letter}2:{ghi_letter}{num_rows+1}, {col_letter}2:{col_letter}{num_rows+1}, "<>")'
                ws_weight.write_array_formula(calc_row_idx, col_num, calc_row_idx, col_num, formula, bold_fmt)

    print(f"Exported dataset to: {excel_path}")
    
    # 3. Generate Daily Plots
    df_filt['Date'] = df_filt.index.date
    colors = plt.cm.tab10.colors 
    
    for date, group in df_filt.groupby('Date'):
        fig, ax = plt.subplots(figsize=(12, 7))
        plotted_any = False
        
        for i, col in enumerate(export_cols):
            station_name = col.split('_')[0] 
            day_data = group[col].dropna()
            
            if day_data.empty:
                continue
            
            color = colors[i % len(colors)]
            ax.plot(day_data.index, day_data, marker='.', linestyle='-', markersize=4, 
                    color=color, alpha=0.8, label=station_name)
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_title(f'{plot_title_prefix} - {date}\n(Filtered: {START_TIME}-{END_TIME}, GHI>{MIN_GHI}, Bounds: {MIN_ALBEDO}-{MAX_ALBEDO})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Albedo Ratio')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylim(bottom=0, top=MAX_ALBEDO + 0.05) 
        
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plot_path = output_folder / f"{file_prefix}_{date}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return

    files = list(INPUT_DIR.rglob('*.csv'))
    if not files:
        print(f"No CSV files found in {INPUT_DIR}")
        return
        
    print(f"Found {len(files)} files. Compiling data...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()

    if TIMESTAMP_COL not in df.columns:
        print(f"\n[ERROR] Column '{TIMESTAMP_COL}' not found.")
        return

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).set_index(TIMESTAMP_COL)
    
    if START_DATE:
        df = df[df.index >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df.index <= pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if df.empty:
        print("Dataframe is empty after applying the date filter.")
        return

    # Base Filtering (Time & Ref GHI)
    ref_ghi_col = GHI_COL_TEMPLATE.format(REFERENCE_GHI_STATION)
    if ref_ghi_col not in df.columns:
        print(f"\n[ERROR] Reference GHI column '{ref_ghi_col}' not found. Cannot filter timesteps.")
        return

    time_mask = pd.Series(False, index=df.index)
    time_index = df.between_time(START_TIME, END_TIME).index
    time_mask.loc[time_index] = True
    ghi_mask = df[ref_ghi_col] > MIN_GHI

    df_valid = df[time_mask & ghi_mask].copy()
    print(f"Kept {len(df_valid)} valid daylight timesteps out of {len(df)} total.\n")

    # --- Calculated Albedo (RHI / GHI) ---
    print("Processing Calculated Albedos (RHI / Station GHI)...")
    calculated_data = {}
    for station in STATIONS:
        rhi_col = RHI_COL_TEMPLATE.format(station)
        ghi_col = GHI_COL_TEMPLATE.format(station)
        
        if rhi_col in df_valid.columns and ghi_col in df_valid.columns:
            # Calculate RHI / GHI, substituting 0 division with NaN to handle math errors
            calc_series = df_valid[rhi_col] / df_valid[ghi_col].replace(0, np.nan)
            calculated_data[f'MET{station}_Calc_Albedo'] = calc_series
        else:
            print(f"[WARNING] Required RHI or GHI columns for Station {station} missing.")

    if calculated_data:
        process_and_export(
            df_valid=df_valid,
            data_dict=calculated_data,
            output_folder=OUTPUT_DIR,
            file_prefix="Calculated_Albedo",
            plot_title_prefix="Calculated Albedo (RHI/GHI)"
        )
        
    print("\nProcessing complete.")

if __name__ == '__main__':
    main()