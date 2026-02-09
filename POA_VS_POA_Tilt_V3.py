import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
import pvlib

# --- 1. Path Configuration ---
input_folder = Path("inputs/GHI&GHI_tilt")
script_name = Path(__file__).stem
output_base = Path("outputs")
output_folder = output_base / script_name
output_folder.mkdir(parents=True, exist_ok=True)

# --- 2. Site & Analysis Settings ---
PlotDetailed = True
start_time_limit = "11:00:00"
end_time_limit = "14:00:00"

# Coordinates (Current: Burlington, ON / User Snippet: 40.26, -83.99)
LAT = 40.26
LON = -83.99
TZ = 'America/Toronto' 

station_colors = {'MET02': 'tab:blue', 'MET16': 'tab:green', 'MET22': 'tab:red', 'MET37': 'tab:orange'}
line_styles = {'POA_1': '-', 'POA_2': ':'}

# --- 3. Helper Functions ---
def get_precise_solar_noon(date_obj):
    """Calculates exact solar transit (noon) using pvlib."""
    times = pd.date_range(start=date_obj, periods=1, freq='D', tz=TZ)
    location = pvlib.location.Location(LAT, LON, tz=TZ)
    # Using .iloc[0] to retrieve the value from the Series
    solar_noon = location.get_sun_rise_set_transit(times)['transit'].iloc[0]
    return solar_noon

def add_stats_box(ax, data_row, cols, label, unit):
    """Adds a box with Median, Average, and Range: Min - Max (Delta)."""
    if data_row.empty: return
    vals = data_row[cols].iloc[0].astype(float)
    v_min, v_max = vals.min(), vals.max()
    delta = v_max - v_min
    
    stats_text = (f"{label} @ Noon\n"
                  f"Median: {vals.median():.2f} {unit}\n"
                  f"Average: {vals.mean():.2f} {unit}\n"
                  f"Range: {v_min:.1f} - {v_max:.1f} {unit}\n"
                  f"Delta (Δ): {delta:.1f} {unit}")
    
    ax.text(0.01, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# --- 4. Processing ---
excel_files = list(input_folder.glob("*.xlsx"))
if not excel_files:
    print(f"No Excel files found in {input_folder}")
    sys.exit()

for file_path in excel_files:
    print(f"Processing File: {file_path.name}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"  - Error loading file: {e}")
        continue

    # Clean and sort data strictly
    df['t_stamp_dt'] = pd.to_datetime(df['t_stamp'], errors='coerce')
    df = df.sort_values('t_stamp_dt')
    df['date'] = df['t_stamp_dt'].dt.date
    
    poa_cols = [c for c in df.columns if '/POA_' in c and 'TILT' not in c and 'RPOA' not in c]
    tilt_cols = [c for c in df.columns if '/POA_' in c and 'TILT_ANGLE' in c]
    unique_days = sorted(df['date'].dropna().unique())

    for day in unique_days:
        day_df = df[df['date'] == day].copy()
        
        # Calculate precise noon
        precise_noon_dt = get_precise_solar_noon(day)
        noon_str = precise_noon_dt.strftime('%H:%M:%S')
        noon_naive = precise_noon_dt.replace(tzinfo=None)
        
        # Slicing the window
        start_ts = pd.to_datetime(f"{day} {start_time_limit}")
        end_ts = pd.to_datetime(f"{day} {end_time_limit}")
        df_filtered = day_df[(day_df['t_stamp_dt'] >= start_ts) & (day_df['t_stamp_dt'] <= end_ts)].copy()
        
        if df_filtered.empty: continue
        
        # Find row closest to solar noon
        noon_row = df_filtered.iloc[(df_filtered['t_stamp_dt'] - noon_naive).abs().argsort()[:1]]

        plot_configs = [('Combined', poa_cols, tilt_cols)]
        if PlotDetailed:
            stations = sorted(list(set([c.split('/')[0] for c in poa_cols])))
            for st in stations:
                plot_configs.append((st, [c for c in poa_cols if c.startswith(st)], [c for c in tilt_cols if c.startswith(st)]))
        
        for name, p_cols, t_cols in plot_configs:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
            x_axis = df_filtered['t_stamp_dt']

            for ax in [ax1, ax2]:
                ax.axvline(x=noon_naive, color='red', linestyle='--', linewidth=1.5, label=f'Solar Noon ({noon_str})')
                # Format X-axis with 5-minute intervals
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.grid(True, which='both', linestyle='--', alpha=0.3)

            # Plot POA
            for c in p_cols:
                st, p_type = c.split('/')
                ax1.plot(x_axis, df_filtered[c], label=f"{st} {p_type}", 
                         color=station_colors.get(st, 'black'), linestyle=line_styles.get(p_type, '-'))
            add_stats_box(ax1, noon_row, p_cols, f"{name} POA", " W/m²")

            # Plot Tilt
            for c in t_cols:
                st, t_type = c.split('/')
                p_type = t_type.replace('_TILT_ANGLE', '')
                ax2.plot(x_axis, df_filtered[c], label=f"{st} {p_type} Tilt", 
                         color=station_colors.get(st, 'black'), linestyle=line_styles.get(p_type, '-'))
            add_stats_box(ax2, noon_row, t_cols, f"{name} Tilt", "°")

            # Updated Title with Solar Noon verification
            ax1.set_title(f'POA Analysis | {name} | {day} | Solar Noon: {noon_str}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Irradiance [W/m²]')
            ax2.set_ylabel('Sensor Tilt [°]')
            ax2.set_xlabel('Time (HH:MM)')
            ax1.legend(loc='upper right', ncol=2, fontsize='small')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            save_path = output_folder / f"{file_path.stem}_{day}_{name}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

        # Site Average Plot
        fig_avg, (ax1_a, ax2_a) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
        df_filtered['avg_poa'] = df_filtered[poa_cols].mean(axis=1)
        df_filtered['avg_tilt'] = df_filtered[tilt_cols].mean(axis=1)
        
        for ax in [ax1_a, ax2_a]:
            ax.axvline(x=noon_naive, color='red', linestyle='--', linewidth=1.5)
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.grid(True, which='both', linestyle='--', alpha=0.3)

        ax1_a.plot(x_axis, df_filtered['avg_poa'], color='black', linewidth=2, label='Mean POA')
        ax2_a.plot(x_axis, df_filtered['avg_tilt'], color='black', linewidth=2, label='Mean Tilt')
        
        add_stats_box(ax1_a, noon_row, poa_cols, "Site POA", " W/m²")
        add_stats_box(ax2_a, noon_row, tilt_cols, "Site Tilt", "°")

        ax1_a.set_title(f'Site Average | {day} | Solar Noon: {noon_str}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_folder / f"{file_path.stem}_{day}_SiteAverage.png", dpi=150)
        plt.close(fig_avg)

print(f"\nProcessing complete. Check '{output_folder}'.")