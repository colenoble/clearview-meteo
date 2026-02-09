import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- 1. Path Configuration ---
# Input directory where your .xlsx files are stored
input_folder = Path("inputs/GHI&GHI_tilt")

# Output directory: outputs/[Script_Name]/
script_name = Path(__file__).stem
output_base = Path("outputs")
output_folder = output_base / script_name

# Create folders if they don't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Analysis time window
start_time_limit = "11:00:00"
end_time_limit = "14:00:00"

# --- 2. File Discovery ---
excel_files = list(input_folder.glob("*.xlsx"))

if not excel_files:
    print(f"No Excel files found in {input_folder}")
    sys.exit()

print(f"Found {len(excel_files)} files to process.")
print(f"Saving plots to: {output_folder}\n")

# --- 3. Batch Processing ---
for file_path in excel_files:
    print(f"Processing File: {file_path.name}")
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"  - Error loading file: {e}")
        continue

    # Prepare time and date components
    df['t_stamp_dt'] = pd.to_datetime(df['t_stamp'], errors='coerce')
    df['date'] = df['t_stamp_dt'].dt.date
    df['time_str'] = df['t_stamp_dt'].dt.strftime('%H:%M:%S')

    # Identify relevant columns
    ghi_cols = [col for col in df.columns if '/GHI' in col and 'TILT' not in col]
    tilt_cols = [col for col in df.columns if '/GHI_TILT_ANGLE' in col]

    unique_days = sorted(df['date'].dropna().unique())

    for day in unique_days:
        # Slicing the data for the specific day and time window
        day_df = df[df['date'] == day].copy()
        mask = (day_df['time_str'] >= start_time_limit) & (day_df['time_str'] <= end_time_limit)
        df_filtered = day_df.loc[mask].copy()

        if df_filtered.empty:
            print(f"  - Skipping {day}: No data found for the 11:00-14:00 window.")
            continue

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot 1: GHI Data
        for col in ghi_cols:
            ax1.plot(df_filtered['time_str'], df_filtered[col], label=col.split('/')[0])
        ax1.set_title(f'GHI Comparison | {day} | Source: {file_path.name}', fontsize=12)
        ax1.set_ylabel('GHI [W/m²]')
        ax1.legend(loc='upper right', title="Sensors")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot 2: SR30 Internal Tilt (Leveling)
        for col in tilt_cols:
            ax2.plot(df_filtered['time_str'], df_filtered[col], label=col.split('/')[0])
        ax2.set_title(f'SR30 Internal Tilt Angle | {day}', fontsize=12)
        ax2.set_ylabel('Sensor Tilt [°]')
        ax2.set_xlabel('Time (HH:MM:SS)')
        ax2.legend(loc='upper right', title="Sensors")
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Adjust ticks for readability
        tick_spacing = max(1, len(df_filtered) // 12)
        ax2.set_xticks(df_filtered['time_str'].iloc[::tick_spacing])
        plt.xticks(rotation=45)

        plt.tight_layout()

        # --- 4. Save/Overwrite Plot ---
        # Filename format: [SourceFile]_[Date].png
        clean_name = f"{file_path.stem}_{day}".replace(" ", "_")
        save_path = output_folder / f"{clean_name}.png"
        
        # plt.savefig overwrites by default if the filename is identical
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  - Saved: {save_path.name}")

print("\nProcessing complete.")