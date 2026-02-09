import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- 1. Path Configuration ---
input_folder = Path("inputs/GHI&GHI_tilt")
script_name = Path(__file__).stem
output_base = Path("outputs")
output_folder = output_base / script_name
output_folder.mkdir(parents=True, exist_ok=True)

# --- 2. Feature Toggles & Settings ---
PlotDetailed = True  # Set to True for per-station plots; False for summary only
start_time_limit = "11:00:00"
end_time_limit = "14:00:00"

station_colors = {
    'MET02': 'tab:blue',
    'MET16': 'tab:green',
    'MET22': 'tab:red',
    'MET37': 'tab:orange'
}

line_styles = {'POA_1': '-', 'POA_2': ':'}

# --- 3. File Discovery ---
excel_files = list(input_folder.glob("*.xlsx"))
if not excel_files:
    print(f"No Excel files found in {input_folder}")
    sys.exit()

# --- 4. Processing ---
for file_path in excel_files:
    print(f"Processing File: {file_path.name}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"  - Error loading file: {e}")
        continue

    df['t_stamp_dt'] = pd.to_datetime(df['t_stamp'], errors='coerce')
    df['date'] = df['t_stamp_dt'].dt.date
    df['time_str'] = df['t_stamp_dt'].dt.strftime('%H:%M:%S')

    # Column patterns
    poa_cols = [c for c in df.columns if '/POA_' in c and 'TILT' not in c and 'RPOA' not in c]
    tilt_cols = [c for c in df.columns if '/POA_' in c and 'TILT_ANGLE' in c]
    unique_days = sorted(df['date'].dropna().unique())

    for day in unique_days:
        day_df = df[df['date'] == day].copy()
        mask = (day_df['time_str'] >= start_time_limit) & (day_df['time_str'] <= end_time_limit)
        df_filtered = day_df.loc[mask].copy()

        if df_filtered.empty:
            continue

        # --- Sub-Routine: Detailed Plots (Per Station) ---
        if PlotDetailed:
            stations = sorted(list(set([c.split('/')[0] for c in poa_cols])))
            for st in stations:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                
                # Plot POA 1 and 2 for this station
                st_poa = [c for c in poa_cols if c.startswith(st)]
                for col in st_poa:
                    p_type = col.split('/')[1]
                    ax1.plot(df_filtered['time_str'], df_filtered[col], 
                             label=f"{st} {p_type}", color=station_colors[st], linestyle=line_styles[p_type])
                
                ax1.set_title(f'Detailed POA | {st} | {day}', fontsize=14)
                ax1.set_ylabel('POA [W/m²]')
                ax1.legend(loc='upper right')
                ax1.grid(True, linestyle='--', alpha=0.5)

                # Plot Tilts for this station
                st_tilt = [c for c in tilt_cols if c.startswith(st)]
                for col in st_tilt:
                    p_type = col.split('/')[1].replace('_TILT_ANGLE', '')
                    ax2.plot(df_filtered['time_str'], df_filtered[col], 
                             label=f"{st} {p_type} Tilt", color=station_colors[st], linestyle=line_styles[p_type])

                ax2.set_title(f'Detailed Tilt | {st} | {day}', fontsize=14)
                ax2.set_ylabel('Sensor Tilt [°]')
                ax2.set_xlabel('Time (HH:MM:SS)')
                ax2.legend(loc='upper right')
                ax2.grid(True, linestyle='--', alpha=0.5)

                # Tick Formatting
                tick_spacing = max(1, len(df_filtered) // 12)
                ax2.set_xticks(df_filtered['time_str'].iloc[::tick_spacing])
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                save_name = f"{file_path.stem}_{day}_{st}_Detailed.png"
                plt.savefig(output_folder / save_name, dpi=150)
                plt.close(fig)

        # --- Sub-Routine: Average Plot (Always Run) ---
        fig_avg, (ax1_a, ax2_a) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        df_filtered['avg_poa'] = df_filtered[poa_cols].mean(axis=1)
        df_filtered['avg_tilt'] = df_filtered[tilt_cols].mean(axis=1)
        
        ax1_a.plot(df_filtered['time_str'], df_filtered['avg_poa'], color='black', linewidth=2, label='Mean POA')
        ax1_a.set_title(f'Site Average POA | {day}', fontsize=14)
        ax1_a.set_ylabel('POA [W/m²]')
        ax1_a.legend()
        ax1_a.grid(True, linestyle='--', alpha=0.5)

        ax2_a.plot(df_filtered['time_str'], df_filtered['avg_tilt'], color='black', linewidth=2, label='Mean Tilt')
        ax2_a.set_title(f'Site Average Tilt | {day}', fontsize=14)
        ax2_a.set_ylabel('Sensor Tilt [°]')
        ax2_a.set_xlabel('Time (HH:MM:SS)')
        ax2_a.legend()
        ax2_a.grid(True, linestyle='--', alpha=0.5)

        ax2_a.set_xticks(df_filtered['time_str'].iloc[::tick_spacing])
        plt.setp(ax2_a.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        
        avg_name = f"{file_path.stem}_{day}_SiteAverage.png"
        plt.savefig(output_folder / avg_name, dpi=150)
        plt.close(fig_avg)
        
        print(f"  - Generated plots for {day}")

print("\nProcessing complete.")