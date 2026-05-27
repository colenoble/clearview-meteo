
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

# --- Configuration ---
INPUT_DIR = Path(r"inputs/AlbedoData")
SCRIPT_NAME = Path(__file__).stem if '__file__' in globals() else 'calculated_albedo_processing'
OUTPUT_DIR = Path(r"outputs") / SCRIPT_NAME

# --- Variable Inputs ---
LATITUDE = 39.90
LONGITUDE = -84.22
TZ = pytz.FixedOffset(-300)  # UTC -5

STATIONS = ['02', '16', '22', '37']
REFERENCE_GHI_STATION = '37'  
GHI_COL_TEMPLATE = "MET{}/GHI2"   
RHI_COL_TEMPLATE = "MET{}/RHI"
TIMESTAMP_COL = "t_stamp" 

START_DATE = '2026-05-02' 
END_DATE = '2026-05-27'

# --- Filtering Parameters ---
START_TIME = "08:00"
END_TIME = "18:00"
MIN_GHI = 50        
MAX_ALBEDO = 1   
MIN_ALBEDO = 0.01   

# Separate bounds for Effective Albedo to prevent clipping low bifacial gains
MIN_EFF_ALBEDO = 0.0
MAX_EFF_ALBEDO = 2.0 

def process_and_export(df_valid, output_folder, file_prefix, plot_title_prefix):
    output_folder.mkdir(parents=True, exist_ok=True)
    
    df_filt = pd.DataFrame(index=df_valid.index)
    ref_ghi_col = GHI_COL_TEMPLATE.format(REFERENCE_GHI_STATION)
    if ref_ghi_col in df_valid.columns:
        df_filt[f'MET{REFERENCE_GHI_STATION}_GHI'] = df_valid[ref_ghi_col]
    
    exclusion_stats = {}
    export_cols = [c for c in df_valid.columns if 'Calc_Albedo' in c or 'Eff_Albedo' in c]

    # 1. Apply Bounds and Calculate Stats
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

    # 2. Export Timeseries to Excel
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
        ws_bounds.write_string(calc_row_idx + 1, 0, "Min Exclusions", bold_fmt)
        ws_bounds.write_string(calc_row_idx + 2, 0, "Max Exclusions", bold_fmt)
        
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

    print(f"Exported dataset to: {excel_path}")
    
    # 3. Generate Daily Plots
    df_filt['Date'] = df_filt.index.date
    colors = plt.cm.tab10.colors 
    
    for date, group in df_filt.groupby('Date'):
        times = pd.date_range(start=f"{date} 00:00:00", end=f"{date} 23:59:00", freq='1min', tz=TZ)
        solpos = pvlib.solarposition.get_solarposition(times, LATITUDE, LONGITUDE)
        solar_noon = solpos['zenith'].idxmin().tz_localize(None) 
        
        win_start = solar_noon - pd.Timedelta(minutes=30)
        win_end = solar_noon + pd.Timedelta(minutes=30)

        noon_time = solar_noon.round('1min')
        if not group.empty:
            closest_pos = np.argmin(np.abs(group.index - noon_time))
            closest_noon_idx = group.index[closest_pos]
        else:
            closest_noon_idx = None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1_twin = ax1.twinx() 
        
        plotted_any = False
        
        solar_noon_str = solar_noon.strftime('%H:%M:%S')
        std_text1 = f"Solar Noon: {solar_noon_str}\n\nValues at Exact Noon:\n"
        std_text2 = f"Solar Noon: {solar_noon_str}\n\nValues & Ratio at Exact Noon:\n"
        
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
            eff_col = f'MET{station}_Eff_Albedo'
            day_data1_full = group[calc_col] if calc_col in group.columns else pd.Series(dtype=float)
            
            # Safe extraction preventing pandas Series ambiguity
            if closest_noon_idx is not None:
                v1_raw = group.loc[closest_noon_idx, calc_col] if calc_col in group.columns else np.nan
                v2_raw = group.loc[closest_noon_idx, eff_col] if eff_col in group.columns else np.nan
                val1 = v1_raw.iloc[0] if isinstance(v1_raw, pd.Series) else v1_raw
                val2 = v2_raw.iloc[0] if isinstance(v2_raw, pd.Series) else v2_raw
            else:
                val1, val2 = np.nan, np.nan
                
            if pd.notna(val1): noon_vals1.append(val1)
            if pd.notna(val2): noon_vals2.append(val2)
            
            # --- Top Plot: Albedo ---
            if calc_col in group.columns:
                day_data1 = day_data1_full.dropna()
                if not day_data1.empty:
                    ax1.plot(day_data1.index, day_data1, marker='.', linestyle='-', markersize=4, 
                             color=color, alpha=0.9, label=f"MET{station}")
                    
                    v1_str = f"{val1*100:.2f}%" if pd.notna(val1) else "N/A"
                    std_text1 += f"MET{station}: {v1_str}\n"
                    plotted_any = True
                    
            # --- Bottom Plot: Effective Albedo ---
            if eff_col in group.columns:
                day_data2 = group[eff_col].dropna()
                day_data2 = day_data2[(day_data2.index >= win_start) & (day_data2.index <= win_end)]
                
                if not day_data2.empty:
                    ax2.plot(day_data2.index, day_data2, marker='.', linestyle='-', markersize=4, 
                             color=color, alpha=0.9, label=f"MET{station}")
                    
                    v2_str = f"{val2*100:.2f}%" if pd.notna(val2) else "N/A"
                    ratio = (val2 / val1) if (pd.notna(val1) and pd.notna(val2) and val1 != 0) else np.nan
                    r_str = f"{ratio:.3f}" if pd.notna(ratio) else "N/A"
                    std_text2 += f"MET{station}: {v2_str} | Ratio: {r_str}\n"
                    plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        std1 = np.std(noon_vals1) if len(noon_vals1) > 1 else np.nan
        std2 = np.std(noon_vals2) if len(noon_vals2) > 1 else np.nan
        
        std1_str = f"{std1*100:.2f}%" if pd.notna(std1) else "N/A"
        std2_str = f"{std2*100:.2f}%" if pd.notna(std2) else "N/A"
        
        std_text1 += f"---\nCross-Station Std Dev: {std1_str}"
        std_text2 += f"---\nCross-Station Std Dev: {std2_str}"

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

        ax2.set_title('Effective Albedo (Mean RPOA / Mean POA) - +/- 30 Min Solar Noon')
        ax2.set_ylabel('Effective Albedo Ratio')
        ax2.set_xlabel('Time')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='upper left', bbox_to_anchor=(1.14, 1.0))
        
        ax2.legend(loc='upper left', bbox_to_anchor=(1.14, 1.0))

        ax1.text(1.14, 0.0, std_text1.strip(), transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=props)
        ax2.text(1.14, 0.0, std_text2.strip(), transform=ax2.transAxes, fontsize=10,
                 verticalalignment='bottom', bbox=props)
        
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
    
    # --- Deduplicate Index to Prevent Ambiguous Series Returns ---
    df = df[~df.index.duplicated(keep='first')]
    
    if START_DATE:
        df = df[df.index >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df.index <= pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if df.empty:
        print("Dataframe is empty after applying the date filter.")
        return

    ref_ghi_col = GHI_COL_TEMPLATE.format(REFERENCE_GHI_STATION)
    if ref_ghi_col not in df.columns:
        print(f"\n[ERROR] Reference GHI column '{ref_ghi_col}' not found. Check template definition.")
        return

    time_mask = pd.Series(False, index=df.index)
    time_index = df.between_time(START_TIME, END_TIME).index
    time_mask.loc[time_index] = True
    
    df[ref_ghi_col] = pd.to_numeric(df[ref_ghi_col], errors='coerce')
    ghi_mask = df[ref_ghi_col] > MIN_GHI

    df_valid = df[time_mask & ghi_mask].copy()
    print(f"Kept {len(df_valid)} valid daylight timesteps out of {len(df)} total.\n")

    print("Processing Standard Albedos, Effective Albedos, and Mean GHI...")
    
    ghi_cols = [GHI_COL_TEMPLATE.format(station) for station in STATIONS]
    avail_ghi = [c for c in ghi_cols if c in df_valid.columns]
    if avail_ghi:
        df_valid['Mean_GHI'] = df_valid[avail_ghi].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    for station in STATIONS:
        rhi_col = RHI_COL_TEMPLATE.format(station)
        ghi_col = GHI_COL_TEMPLATE.format(station)
        if rhi_col in df_valid.columns and ghi_col in df_valid.columns:
            df_valid[rhi_col] = pd.to_numeric(df_valid[rhi_col], errors='coerce')
            df_valid[ghi_col] = pd.to_numeric(df_valid[ghi_col], errors='coerce')
            df_valid[f'MET{station}_Calc_Albedo'] = df_valid[rhi_col] / df_valid[ghi_col].replace(0, np.nan)

    print("\n--- Diagnostic Check: Effective Albedo ---")
    for station in STATIONS:
        rpoa_cols = [f"MET{station}/RPOA_1", f"MET{station}/RPOA_2"]
        poa_cols = [f"MET{station}/POA_1", f"MET{station}/POA_2"]
        
        avail_rpoa = [c for c in rpoa_cols if c in df_valid.columns]
        avail_poa = [c for c in poa_cols if c in df_valid.columns]
        
        if avail_rpoa and avail_poa:
            df_rpoa = df_valid[avail_rpoa].apply(pd.to_numeric, errors='coerce')
            df_poa = df_valid[avail_poa].apply(pd.to_numeric, errors='coerce')
            
            rpoa_mean = df_rpoa.mean(axis=1)
            poa_mean = df_poa.mean(axis=1)
            
            eff_alb = rpoa_mean / poa_mean.replace(0, np.nan)
            df_valid[f'MET{station}_Eff_Albedo'] = eff_alb
            
            valid_counts = eff_alb.notna().sum()
            print(f"MET{station} | Valid rows: {valid_counts} | Min: {eff_alb.min():.4f} | Max: {eff_alb.max():.4f}")
        else:
             print(f"MET{station} | Missing Required RPOA or POA Columns")
    print("------------------------------------------\n")

    calculated_cols = [c for c in df_valid.columns if 'Calc_Albedo' in c or 'Eff_Albedo' in c]
    if calculated_cols:
        process_and_export(
            df_valid=df_valid,
            output_folder=OUTPUT_DIR,
            file_prefix="Calculated_Albedo",
            plot_title_prefix="Calculated Albedo"
        )
    else:
        print("\n[WARNING] No standard or effective albedos calculated.")
        
    print("Processing complete.")

if __name__ == '__main__':
    main()

