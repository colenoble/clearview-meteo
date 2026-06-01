import pandas as pd
import numpy as np
from pathlib import Path
import xlsxwriter.utility as xl_util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

# --- Configuration ---
INPUT_DIR = Path(r"inputs")

SCRIPT_NAME = Path(__file__).stem if '__file__' in globals() else 'capacity_test_albedo'
OUTPUT_DIR = Path(r"outputs") / SCRIPT_NAME
OUTPUT_FILE = OUTPUT_DIR / "capacity_test_albedo_formulas.xlsx"
PLOT_DIR = OUTPUT_DIR / "plots"

# Configurable Date Range
START_DATE = "2026-05-02"
END_DATE = "2026-05-29"

# Filtering Parameters
STATIONS = ['02', '16', '22', '37']
START_TIME = "07:30"
END_TIME = "18:30"
MIN_GHI = 50

def get_combined_series(df, templates, station):
    combined = pd.Series(np.nan, index=df.index)
    for t in templates:
        col = t.format(station)
        if col in df.columns:
            combined = combined.fillna(pd.to_numeric(df[col], errors='coerce'))
    return combined

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load High-Res Data
    # Updated to accept any CSV file
    possible_files = list(INPUT_DIR.rglob('*.csv'))
    if not possible_files:
        print("[ERROR] No files found matching '*.csv'. Check your inputs folder.")
        return

    print(f"[INFO] Parsing {len(possible_files)} files...")
    df_list = []
    for f in possible_files:
        temp_df = pd.read_csv(f)
        temp_df.columns = temp_df.columns.str.strip()
        
        ts_col = next((c for c in temp_df.columns if c.lower() in ["t_stamp", "timestamp", "date/time", "datetime"]), None)
        if not ts_col:
            continue
            
        temp_df[ts_col] = pd.to_datetime(temp_df[ts_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[ts_col])
        temp_df = temp_df.set_index(ts_col)
        
        if temp_df.index.tz is not None:
            temp_df.index = temp_df.index.tz_convert(None)
            
        temp_df.index = temp_df.index.round('min')
        df_list.append(temp_df)

    if not df_list:
        print("[ERROR] Could not parse datetime index in any files.")
        return

    df = pd.concat(df_list).sort_index()
    df = df.groupby(level=0).first()

    print(f"[DIAGNOSTIC] Raw data loaded spans from: {df.index.min()} to {df.index.max()}")

    # 2. Apply Strict Date and Time Filters
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df = df.loc[mask]
    df = df.between_time(START_TIME, END_TIME)
    
    if df.empty:
        print(f"[ERROR] Dataframe is empty after applying filters.")
        return

    print(f"[DIAGNOSTIC] After applying {START_DATE} to {END_DATE} filter, data ends on: {df.index.max().date()}")

    # 3. Extract Core Columns 
    out_df = pd.DataFrame(index=df.index)
    
    ghi_cols = []
    rhi_cols = []
    poa_cols = []
    
    for st in STATIONS:
        ghi_col = f'MET{st}_GHI2'
        rhi_col = f'MET{st}_RHI'
        poa_col = f'MET{st}_POA'
        
        out_df[ghi_col] = get_combined_series(df, ["MET{}/GHI2", "MET{}_GHI2", "Station{}_GHI2", "MET{}/GHI_2", "MET{}_GHI_2"], st)
        out_df[rhi_col] = get_combined_series(df, ["MET{}/RHI", "MET{}_RHI", "Station{}_RHI"], st)
        out_df[poa_col] = get_combined_series(df, ["MET{}/POA", "MET{}_POA", "Station{}_POA", "MET{}/POA_1", "MET{}_POA_1", "MET{}/POA_Avg"], st)
        
        ghi_cols.append(ghi_col)
        rhi_cols.append(rhi_col)
        poa_cols.append(poa_col)

    # Calculate Mean GHI for filtering threshold
    out_df['Mean_GHI'] = out_df[ghi_cols].mean(axis=1)
    
    # Pre-calculate aggregate columns specifically for the plots
    out_df['Sum_RHI_div_Sum_GHI'] = out_df[rhi_cols].sum(axis=1) / out_df[ghi_cols].sum(axis=1).replace(0, np.nan)
    out_df['Median_POA'] = out_df[poa_cols].median(axis=1)

    # 4. Generate Continuous Time-Series Plots
    print("[INFO] Generating continuous time-series plots...")
    
    for date, group in out_df.groupby(out_df.index.date):
        daily_15m = group[['Sum_RHI_div_Sum_GHI', 'Median_POA', 'Mean_GHI']].resample('15min').mean()
        daily_15m = daily_15m.dropna(subset=['Sum_RHI_div_Sum_GHI', 'Median_POA'], how='all')
        
        if daily_15m.empty:
            continue
            
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        ax1.fill_between(
            group.index, 0, 1, 
            where=(group['Mean_GHI'] <= MIN_GHI), 
            color='lightcoral', alpha=0.2, 
            transform=ax1.get_xaxis_transform(), 
            label=f'Excluded (Mean GHI <= {MIN_GHI})'
        )
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Weighted Albedo (ΣRHI / ΣGHI)', color=color1)
        l1, = ax1.plot(daily_15m.index, daily_15m['Sum_RHI_div_Sum_GHI'], color=color1, marker='.', linestyle='-', label='15-Min Weighted Albedo')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        current_max = daily_15m['Sum_RHI_div_Sum_GHI'].max()
        if pd.notna(current_max):
            ax1.set_ylim(bottom=0, top=min(1.0, current_max + 0.1))
        
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Irradiance - Median POA (W/m²)', color=color2)
        l2, = ax2.plot(daily_15m.index, daily_15m['Median_POA'], color=color2, marker='', linestyle='--', label='15-Min Irradiance (POA)')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0)
        
        plt.title(f'15-Min Weighted Albedo and Irradiance vs. Time - {date}\n(Red shading indicates excluded low-irradiance periods)')
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        
        plt.grid(True, alpha=0.3, linestyle=':')
        fig.tight_layout()
        
        plot_path = PLOT_DIR / f"Albedo_and_Irradiance_Timeseries_{date}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 5. Filter Data, Apply Grouped Headers, and Write to Excel
    # Physically drop rows where Mean GHI is below threshold
    excel_df = out_df[out_df['Mean_GHI'] > MIN_GHI].copy()
    
    # Enforce strict column layout for Excel formulas
    expected_order = ghi_cols + rhi_cols + poa_cols
    for col in expected_order:
        if col not in excel_df.columns:
            excel_df[col] = np.nan
    excel_df = excel_df[expected_order]
    
    print(f"[INFO] Writing {len(excel_df)} valid rows to Excel with grouped formatting...")
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        # Start writing raw data at Row 7 (index 6) to leave room for the summary box and grouped headers
        excel_df.to_excel(writer, sheet_name='Albedo_Data', startrow=6, header=False)
        
        workbook = writer.book
        ws = writer.sheets['Albedo_Data']
        
        # --- Formatting Toolbelt ---
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        num_fmt = workbook.add_format({'num_format': '0.0'})
        time_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm'})
        
        box_hdr_fmt = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 2, 'bg_color': '#E7E6E6'})
        sub_hdr_fmt = workbook.add_format({'bold': True, 'align': 'center', 'bottom': 2, 'top': 1, 'left': 1, 'right': 1, 'bg_color': '#F2F2F2'})
        
        sum_title_fmt = workbook.add_format({'bold': True, 'border': 2, 'bg_color': '#D9E1F2', 'align': 'center'})
        sum_label_fmt = workbook.add_format({'bold': True, 'left': 2, 'top': 1, 'bottom': 1, 'right': 1})
        sum_val_fmt = workbook.add_format({'bold': True, 'left': 1, 'top': 1, 'bottom': 1, 'right': 2, 'num_format': '0.00%'})
        sum_label_bot_fmt = workbook.add_format({'bold': True, 'left': 2, 'top': 1, 'bottom': 2, 'right': 1})
        sum_val_bot_fmt = workbook.add_format({'bold': True, 'left': 1, 'top': 1, 'bottom': 2, 'right': 2, 'num_format': '0.00%'})

        last_row = 6 + len(excel_df)
        
        ws.set_column('A:A', 18, time_fmt)
        ws.set_column('B:M', 11, num_fmt)     # GHI, RHI, POA
        ws.set_column('N:R', 12, pct_fmt)     # Station Albedos, Site Aggregate
        ws.set_column('S:S', 12, num_fmt)     # Irradiance
        
        # --- Top Summary Section ---
        ws.merge_range('A1:B1', 'PERIOD SUMMARY', sum_title_fmt)
        
        ws.write_string('A2', 'Albedo (Total RHI / Total GHI):', sum_label_fmt)
        ws.write_string('A3', 'POA-Weighted Albedo Average:', sum_label_bot_fmt)
        
        # Data is already pre-filtered by MIN_GHI, so pure SUM and SUMPRODUCT are safe
        ws.write_formula('B2', f'=IFERROR(SUM(F7:I{last_row})/SUM(B7:E{last_row}), "N/A")', sum_val_fmt)
        ws.write_formula('B3', f'=IFERROR(SUMPRODUCT(R7:R{last_row}, S7:S{last_row})/SUM(S7:S{last_row}), "N/A")', sum_val_bot_fmt)
        
        # --- Grouped Headers (Row 5) ---
        ws.write_string(4, 0, "Timestamp", box_hdr_fmt)
        ws.merge_range(4, 1, 4, 4, "GHI (W/m²)", box_hdr_fmt)
        ws.merge_range(4, 5, 4, 8, "RHI (W/m²)", box_hdr_fmt)
        ws.merge_range(4, 9, 4, 12, "POA (W/m²)", box_hdr_fmt)
        ws.merge_range(4, 13, 4, 16, "Station Albedos", box_hdr_fmt)
        ws.write_string(4, 17, "Site Aggregate", box_hdr_fmt)
        ws.write_string(4, 18, "Irradiance", box_hdr_fmt)
        
        # --- Sub-Headers (Row 6) ---
        sub_headers = [""] + STATIONS + STATIONS + STATIONS + STATIONS + ["ΣRHI / ΣGHI", "Median POA"]
        for col_num, text in enumerate(sub_headers):
            ws.write_string(5, col_num, f"MET{text}" if text in STATIONS else text, sub_hdr_fmt)
            
        # --- Row Formulas (Rows 7 and down) ---
        for row in range(7, last_row + 1):
            idx = row - 1
            
            # Station Albedos (Cols N-Q)
            for col_offset in range(4):
                rhi_col = xl_util.xl_col_to_name(5 + col_offset) # F-I
                ghi_col = xl_util.xl_col_to_name(1 + col_offset) # B-E
                write_col = 13 + col_offset # N-Q
                ws.write_formula(idx, write_col, f'=IFERROR(IF({ghi_col}{row}>0, {rhi_col}{row}/{ghi_col}{row}, ""), "")')
                
            # Site Aggregate Albedo (Col R)
            ws.write_formula(idx, 17, f'=IFERROR(IF(SUM(B{row}:E{row})>0, SUM(F{row}:I{row})/SUM(B{row}:E{row}), ""), "")')
            
            # Median POA (Col S)
            ws.write_formula(idx, 18, f'=IFERROR(MEDIAN(J{row}:M{row}), "")')

    print(f"[SUCCESS] File saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()