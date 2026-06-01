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

# Dynamically set output folder based on script name
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
    possible_files = list(INPUT_DIR.rglob('*export*.csv'))
    if not possible_files:
        print("[ERROR] No high-res export CSV files found.")
        return

    print(f"[INFO] Parsing {len(possible_files)} files...")
    df_list = []
    for f in possible_files:
        temp_df = pd.read_csv(f)
        temp_df.columns = temp_df.columns.str.strip()
        
        ts_col = next((c for c in temp_df.columns if c.lower() in ["t_stamp", "timestamp", "date/time", "datetime"]), None)
        if not ts_col:
            continue
            
        temp_df[ts_col] = pd.to_datetime(temp_df[ts_col])
        temp_df = temp_df.set_index(ts_col)
        
        # SNAP TIMESTAMPS
        temp_df.index = temp_df.index.round('min')
        
        df_list.append(temp_df)

    if not df_list:
        print("[ERROR] Could not parse datetime index.")
        return

    df = pd.concat(df_list).sort_index()
    df = df.groupby(level=0).first()

    # 2. Apply Date and Time Filters
    df = df.loc[START_DATE:END_DATE]
    df = df.between_time(START_TIME, END_TIME)
    
    if df.empty:
        print(f"[ERROR] Dataframe is empty after applying filters ({START_DATE} to {END_DATE}, {START_TIME}-{END_TIME}).")
        return

    # 3. Extract Core Columns with Expanded Templates
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

    # Calculate aggregate columns in pandas for plotting and thresholding
    out_df['Mean_GHI'] = out_df[ghi_cols].mean(axis=1)
    out_df['Sum_RHI_div_Sum_GHI'] = out_df[rhi_cols].sum(axis=1) / out_df[ghi_cols].sum(axis=1).replace(0, np.nan)
    out_df['Median_POA'] = out_df[poa_cols].median(axis=1)

    # 4. Generate 15-Minute Continuous Time-Series Plots
    print("[INFO] Generating continuous time-series plots...")
    
    for date, group in out_df.groupby(out_df.index.date):
        daily_15m = group[['Sum_RHI_div_Sum_GHI', 'Median_POA', 'Mean_GHI']].resample('15min').mean()
        daily_15m = daily_15m.dropna(subset=['Sum_RHI_div_Sum_GHI', 'Median_POA'], how='all')
        
        if daily_15m.empty:
            continue
            
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Shade excluded GHI periods
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

    # 5. Filter Data, STRICTLY REORDER COLUMNS, and Write to Excel
    excel_df = out_df[out_df['Mean_GHI'] > MIN_GHI].copy()
    
    # STRICT COLUMN ENFORCEMENT
    expected_order = ghi_cols + rhi_cols + poa_cols
    for col in expected_order:
        if col not in excel_df.columns:
            excel_df[col] = np.nan
    excel_df = excel_df[expected_order]
    
    print(f"[INFO] Writing {len(excel_df)} valid test rows to Excel with formulas...")
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        excel_df.to_excel(writer, sheet_name='Albedo_Data')
        workbook = writer.book
        ws = writer.sheets['Albedo_Data']
        bold_fmt = workbook.add_format({'bold': True})
        
        formula_headers = [
            *[f'MET{st}_Albedo' for st in STATIONS], 
            'Sum_RHI_div_Sum_GHI', 
            'Median_POA'
        ]
        
        start_col = len(excel_df.columns) + 1 
        for i, header in enumerate(formula_headers):
            ws.write_string(0, start_col + i, header, bold_fmt)

        num_rows = len(excel_df)
        for row in range(2, num_rows + 2):
            idx = row - 1 
            
            # GHI (B-E), RHI (F-I) -> Albedo (N-Q)
            for col_offset in range(4):
                rhi_col = xl_util.xl_col_to_name(5 + col_offset)
                ghi_col = xl_util.xl_col_to_name(1 + col_offset)
                write_col = 13 + col_offset
                ws.write_formula(idx, write_col, f'=IF({ghi_col}{row}>0, {rhi_col}{row}/{ghi_col}{row}, "")')
            
            # Col R (17) and S (18)
            ws.write_formula(idx, 17, f'=IF(SUM(B{row}:E{row})>0, SUM(F{row}:I{row})/SUM(B{row}:E{row}), "")')
            ws.write_formula(idx, 18, f'=MEDIAN(J{row}:M{row})')

        summary_row = num_rows + 3
        ws.write_string(summary_row, 0, "PERIOD SUMMARY", bold_fmt)
        
        ws.write_string(summary_row + 1, 0, "Albedo (Total RHI / Total GHI):", bold_fmt)
        ws.write_formula(summary_row + 1, 1, f'=SUM(F2:I{num_rows+1})/SUM(B2:E{num_rows+1})')
        
        # Single weighting: Aggregate Albedo (Col R) weighted by Median POA (Col S)
        ws.write_string(summary_row + 2, 0, "POA-Weighted Albedo Average:", bold_fmt)
        ws.write_formula(summary_row + 2, 1, f'=SUMPRODUCT(R2:R{num_rows+1}, S2:S{num_rows+1})/SUM(S2:S{num_rows+1})')

    print(f"[SUCCESS] File saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()