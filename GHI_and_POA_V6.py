import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
import shutil
from datetime import datetime

# pvlib required for precise solar noon
try:
    import pvlib
except ImportError:
    pvlib = None
    print("Warning: pvlib not installed. Solar noon calculations will fail.")


# ------------------------------------------------------------
# 1) CONFIGURATION
# ------------------------------------------------------------
input_folder = Path("inputs/GHI&GHI_tilt")

# --- DYNAMIC OUTPUT FOLDER NAME ---
script_name = Path(__file__).stem  # Uses the actual filename (e.g., "GHI_and_POA_V1")
output_folder = Path("outputs") / script_name
output_folder.mkdir(parents=True, exist_ok=True)

# Comparison Dates
CompareDates = ("2026-02-10", "2026-03-09")

# Time Window for Plots
start_time_limit = "12:00:00"
end_time_limit = "13:00:00"

# Site Coords / Timezone
LAT = 40.26
LON = -83.99
TZ = "Etc/GMT+5"  # Locked to Standard Time (UTC-5)

# Data Smoothing (Assumes 1-minute logging frequency)
AVERAGING_WINDOW_MINS = 1  # 1 = no averaging. E.g., 3, 5, 10 for rolling mean.

# Station Filtering & Styling
SENSORS_TO_INCLUDE = [
    # POA Sensors
    "MET02/POA_1", "MET02/POA_2",
    "MET16/POA_1", "MET16/POA_2",
    "MET22/POA_1", "MET22/POA_2",
    "MET37/POA_1", "MET37/POA_2",
    # GHI Sensors (patterns usually match 'METxx/GHI')
]

station_colors = {
    "MET02": "tab:blue",
    "MET16": "tab:green",
    "MET22": "tab:red",
    "MET37": "tab:orange",
}
line_styles = {"POA_1": "-", "POA_2": ":", "GHI": "-"}

# Tick Formatting
MAJOR_TICK_MINUTES = 15
MINOR_TICK_MINUTES = 5


# ------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ------------------------------------------------------------

def parse_date(date_str: str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes timestamps and applies rolling average if configured.
    Assumes logger data is in fixed Standard Time (EST).
    """
    df = df.copy()
    df["t_stamp_dt"] = pd.to_datetime(df["t_stamp"], errors="coerce")
    df = df.sort_values("t_stamp_dt").reset_index(drop=True)
    
    # Apply rolling average to numeric data (irradiance/tilt)
    if AVERAGING_WINDOW_MINS > 1:
        num_cols = df.select_dtypes(include='number').columns
        # center=True prevents the rolling window from shifting peaks forward in time
        df[num_cols] = df[num_cols].rolling(window=AVERAGING_WINDOW_MINS, center=True, min_periods=1).mean()
        
    df["date"] = df["t_stamp_dt"].dt.date
    return df

def filter_day_time(df: pd.DataFrame, day):
    day_df = df[df["date"] == day].copy()
    start_ts = pd.to_datetime(f"{day} {start_time_limit}")
    end_ts = pd.to_datetime(f"{day} {end_time_limit}")
    return day_df[(day_df["t_stamp_dt"] >= start_ts) & (day_df["t_stamp_dt"] <= end_ts)].copy()

def get_precise_solar_noon(date_obj):
    if pvlib is None:
        return None
    times = pd.date_range(start=date_obj, periods=1, freq="D", tz=TZ)
    location = pvlib.location.Location(LAT, LON, tz=TZ)
    return location.get_sun_rise_set_transit(times)["transit"].iloc[0]

def format_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=MAJOR_TICK_MINUTES))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if MINOR_TICK_MINUTES:
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=MINOR_TICK_MINUTES))
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)

def add_stats_box(ax, target_row, cols, label, unit):
    """Adds the stats box with the % Deviation logic."""
    if target_row is None or target_row.empty or not cols:
        return

    present_cols = [c for c in cols if c in target_row.columns]
    if not present_cols:
        return

    vals = target_row[present_cols].iloc[0].astype(float)
    v_min, v_max = vals.min(), vals.max()
    delta = v_max - v_min

    stats_text = (
        f"{label} @ Noon\n"
        f"Median: {vals.median():.2f}{unit}\n"
        f"Average: {vals.mean():.2f}{unit}\n"
        f"Range: {v_min:.1f} - {v_max:.1f}{unit}\n"
        f"Delta (Δ): {delta:.1f}{unit}"
    )

    if "W/m" in unit:
        if v_min != 0:
            pct_dev = (delta / v_min) * 100
            check_str = "YES" if pct_dev < 2.0 else "NO"
            stats_text += f"\nΔ % of Min: {pct_dev:.2f}%"
            stats_text += f"\n< 2% Check: {check_str}"
        else:
            stats_text += f"\nΔ % of Min: N/A"
            stats_text += f"\n< 2% Check: N/A"

    ax.text(
        0.01, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

def set_top_right_legend(ax, title="Sensors"):
    handles, labels = ax.get_legend_handles_labels()
    if not handles: return
    dedup = {l: h for h, l in zip(handles, labels) if l != "_nolegend_"}
    ax.legend(
        dedup.values(), dedup.keys(),
        title=title, loc="upper right",
        bbox_to_anchor=(0.98, 0.98), bbox_transform=ax.transAxes,
        fontsize="small", framealpha=0.9
    )

def find_best_source_for_day(loaded_items, target_day):
    best = None
    best_count = 0
    for item in loaded_items:
        df_filt = filter_day_time(item["df"], target_day)
        n = len(df_filt)
        if n > best_count:
            best_count = n
            best = {
                "file": item["file"],
                "df_filtered": df_filt,
                "poa_cols": item["poa_cols"],
                "ghi_cols": item["ghi_cols"],
                "poa_tilt": item["poa_tilt"],
                "ghi_tilt": item["ghi_tilt"]
            }
    return best if best_count > 0 else None

def calculate_medians(df, ghi_cols, poa_cols):
    """Calculates median GHI and POA arrays for the dataframe."""
    df = df.copy()
    valid_ghi = [c for c in ghi_cols if c in df.columns]
    valid_poa = [c for c in poa_cols if c in df.columns]
    
    if valid_ghi:
        df["Median_GHI"] = df[valid_ghi].median(axis=1)
    if valid_poa:
        df["Median_POA"] = df[valid_poa].median(axis=1)
        
    return df


# ------------------------------------------------------------
# 3) PLOTTING ENGINE
# ------------------------------------------------------------

def plot_compare_2x2(df1, df2, cols1, tilts1, cols2, tilts2,
                     day1, day2, noon1, noon2, noon_row1, noon_row2,
                     title_prefix, filename_suffix, mode="POA"):
    """
    Generic 2x2 Plotter.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey="row", sharex="col")
    ax_irr_1, ax_irr_2 = axes[0, 0], axes[0, 1]
    ax_tilt_1, ax_tilt_2 = axes[1, 0], axes[1, 1]
    
    noon1_naive = noon1.replace(tzinfo=None)
    noon2_naive = noon2.replace(tzinfo=None)
    noon1_str = noon1.round("1min").strftime("%H:%M")
    noon2_str = noon2.round("1min").strftime("%H:%M")
    
    # --- Day 1 (Left) ---
    x1 = df1["t_stamp_dt"]
    for c in cols1:
        if c in df1.columns:
            st = c.split("/")[0]
            ls = "-"
            if "POA_2" in c: ls = ":"
            label_name = c.split("/")[1] if mode == "POA" else st
            ax_irr_1.plot(x1, df1[c], label=f"{st} {label_name}", 
                          color=station_colors.get(st, "black"), linestyle=ls)
            
    ax_irr_1.axvline(noon1_naive, color="red", ls="--")
    ax_irr_1.set_title(f"{mode} • {day1} • Noon: {noon1_str}", fontsize=12)
    ax_irr_1.set_ylabel(f"{mode} [W/m²]")
    add_stats_box(ax_irr_1, noon_row1, cols1, mode, " W/m²")

    for c in tilts1:
        if c in df1.columns:
            st = c.split("/")[0]
            label_name = "Tilt"
            ax_tilt_1.plot(x1, df1[c], label=f"{st} {label_name}",
                           color=station_colors.get(st, "black"))
    ax_tilt_1.axvline(noon1_naive, color="red", ls="--")
    ax_tilt_1.set_ylabel("Tilt [°]")
    add_stats_box(ax_tilt_1, noon_row1, tilts1, "Tilt", "°")

    # --- Day 2 (Right) ---
    x2 = df2["t_stamp_dt"]
    for c in cols2:
        if c in df2.columns:
            st = c.split("/")[0]
            ls = "-"
            if "POA_2" in c: ls = ":"
            label_name = c.split("/")[1] if mode == "POA" else st
            ax_irr_2.plot(x2, df2[c], label=f"{st} {label_name}",
                          color=station_colors.get(st, "black"), linestyle=ls)

    ax_irr_2.axvline(noon2_naive, color="red", ls="--")
    ax_irr_2.set_title(f"{mode} • {day2} • Noon: {noon2_str}", fontsize=12)
    add_stats_box(ax_irr_2, noon_row2, cols2, mode, " W/m²")

    for c in tilts2:
        if c in df2.columns:
            st = c.split("/")[0]
            ax_tilt_2.plot(x2, df2[c], label=f"{st} {label_name}",
                           color=station_colors.get(st, "black"))
    ax_tilt_2.axvline(noon2_naive, color="red", ls="--")
    add_stats_box(ax_tilt_2, noon_row2, tilts2, "Tilt", "°")

    # Formatting
    for ax in axes.flatten():
        format_time_axis(ax)
        if ax in [ax_tilt_1, ax_tilt_2]:
            ax.set_xlabel("Time (HH:MM)")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        set_top_right_legend(ax)

    fig.suptitle(title_prefix, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    save_path = output_folder / f"{mode}_COMPARE_{day1}_VS_{day2}_{filename_suffix}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path.name}")

def plot_median_ghi_poa_overlay(df, day, noon, start_time, end_time):
    """
    Overlays Median GHI and Median POA for a specific time window,
    with vertical lines for empirical and calculated solar noon.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    x = df["t_stamp_dt"]
    noon_naive = noon.replace(tzinfo=None)
    
    # 1. Plot the Median Irradiance Lines
    if "Median_GHI" in df.columns:
        ax.plot(x, df["Median_GHI"], label="Median GHI", color="tab:blue", linestyle="-", linewidth=2)
        
        # Calculate and plot empirical noon for GHI (Max)
        if not df["Median_GHI"].isna().all():
            idx_max_ghi = df["Median_GHI"].idxmax()
            time_max_ghi = df.loc[idx_max_ghi, "t_stamp_dt"]
            ax.axvline(time_max_ghi, color="tab:blue", ls="--", alpha=0.8, 
                       label=f"Est Noon GHI (Max): {time_max_ghi.strftime('%H:%M:%S')}")

    if "Median_POA" in df.columns:
        ax.plot(x, df["Median_POA"], label="Median POA", color="tab:orange", linestyle="-", linewidth=2)
        
        # Calculate and plot empirical noon for POA (Min)
        if not df["Median_POA"].isna().all():
            idx_min_poa = df["Median_POA"].idxmin()
            time_min_poa = df.loc[idx_min_poa, "t_stamp_dt"]
            ax.axvline(time_min_poa, color="tab:orange", ls="--", alpha=0.8,
                       label=f"Est Noon POA (Min): {time_min_poa.strftime('%H:%M:%S')}")

    # Plot Calculated Solar Noon
    noon_str_exact = noon.round("1s").strftime("%H:%M:%S")
    ax.axvline(noon_naive, color="red", ls="--", linewidth=1.5, label=f"Calculated Noon: {noon_str_exact}")

    # Title and Labels
    start_str = start_time[:5]
    end_str = end_time[:5]
    noon_str_title = noon.round("1min").strftime("%H:%M")
    
    avg_text = f" ({AVERAGING_WINDOW_MINS}-min avg)" if AVERAGING_WINDOW_MINS > 1 else ""
    ax.set_title(f"Median GHI & POA Overlay{avg_text} • {day} ({start_str} - {end_str}) • Noon: {noon_str_title}", fontsize=13)
    ax.set_ylabel("Irradiance [W/m²]")
    ax.set_xlabel("Time (HH:MM)")
    
    # 2. X-Axis Tick Formatting Optimization
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.legend(title="Aggregated Irradiance", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    
    save_path = output_folder / f"MEDIAN_GHI_POA_OVERLAY_{day}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path.name}")


# ------------------------------------------------------------
# 4) EXCEL GENERATION
# ------------------------------------------------------------

def write_excel_sheet(wb, sheet_name, date_val, records):
    """Writes a single tab for a specific date (Before/After)."""
    if not records:
        return

    df_raw = pd.DataFrame(records)
    grouped = df_raw.groupby(["Station", "Type"])
    
    final_rows = []
    max_vals = 0
    
    for (st, m_type), group in grouped:
        row_data = {"Station": st, "Type": m_type}
        sorted_sensors = group.sort_values("Sensor")
        vals = sorted_sensors["Value"].tolist()
        
        if len(vals) > max_vals: max_vals = len(vals)
        for i, v in enumerate(vals): row_data[f"Val_{i+1}"] = v
        final_rows.append(row_data)

    df_out = pd.DataFrame(final_rows)
    base_cols = ["Station", "Type"]
    val_cols = [f"Val_{i+1}" for i in range(max_vals)]
    df_out = df_out[base_cols + val_cols]

    df_out.to_excel(wb, sheet_name=sheet_name, startrow=2, header=False, index=False)
    worksheet = wb.sheets[sheet_name]
    workbook = wb.book
    
    fmt_header = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#EFEFEF'})
    fmt_dec = workbook.add_format({'num_format': '0.00'})
    fmt_pct = workbook.add_format({'num_format': '0.00%'})
    fmt_pass = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    fmt_fail = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    fmt_border = workbook.add_format({'border': 1})
    fmt_bold_border = workbook.add_format({'bold': True, 'border': 1})
    
    worksheet.write(0, 0, "Date", fmt_header)
    worksheet.write(0, 1, str(date_val), fmt_border)
    
    col_headers = list(df_out.columns) + ["Min", "Max", "Delta", "% Deviation", "Check (<2%)"]
    for i, h in enumerate(col_headers): worksheet.write(1, i, h, fmt_header)
        
    start_row = 2
    num_rows = len(df_out)
    col_idx_vals_start = 2
    col_idx_vals_end = 2 + max_vals - 1
    col_idx_min = col_idx_vals_end + 1
    col_idx_max = col_idx_min + 1
    col_idx_delta = col_idx_max + 1
    col_idx_pct = col_idx_delta + 1
    col_idx_check = col_idx_pct + 1

    for r in range(num_rows):
        row_num = start_row + r + 1 
        first_val_char = chr(65 + col_idx_vals_start)
        last_val_char = chr(65 + col_idx_vals_end)
        range_vals = f"{first_val_char}{row_num}:{last_val_char}{row_num}"

        worksheet.write_formula(start_row + r, col_idx_min, f'=MIN({range_vals})', fmt_dec)
        worksheet.write_formula(start_row + r, col_idx_max, f'=MAX({range_vals})', fmt_dec)
        
        col_min_char = chr(65 + col_idx_min)
        col_max_char = chr(65 + col_idx_max)
        
        formula_delta = f'=IF(COUNT({range_vals})>1, {col_max_char}{row_num}-{col_min_char}{row_num}, "")'
        worksheet.write_formula(start_row + r, col_idx_delta, formula_delta, fmt_dec)
        
        col_delta_char = chr(65 + col_idx_delta)
        formula_pct = f'=IF(ISNUMBER({col_delta_char}{row_num}), IF({col_min_char}{row_num}<>0, {col_delta_char}{row_num}/{col_min_char}{row_num}, 0), "")'
        worksheet.write_formula(start_row + r, col_idx_pct, formula_pct, fmt_pct)
        
        col_pct_char = chr(65 + col_idx_pct)
        formula_check = f'=IF(ISNUMBER({col_pct_char}{row_num}), IF({col_pct_char}{row_num}<0.02, "YES", "NO"), "")'
        worksheet.write_formula(start_row + r, col_idx_check, formula_check)

        worksheet.conditional_format(start_row+r, col_idx_check, start_row+r, col_idx_check,
                                     {'type': 'cell', 'criteria': '==', 'value': '"YES"', 'format': fmt_pass})
        worksheet.conditional_format(start_row+r, col_idx_check, start_row+r, col_idx_check,
                                     {'type': 'cell', 'criteria': '==', 'value': '"NO"', 'format': fmt_fail})

    summary_start_row = start_row + num_rows + 2
    worksheet.write(summary_start_row, col_idx_vals_start, "Range (min max)", fmt_bold_border)
    worksheet.write(summary_start_row+1, col_idx_vals_start, "Range (W/m^2)", fmt_bold_border)
    worksheet.write(summary_start_row+2, col_idx_vals_start, "% difference from Min Irradiance", fmt_bold_border)
    
    first_cell = f"{chr(65+col_idx_vals_start)}3"
    last_cell = f"{chr(65+col_idx_vals_end)}{start_row+num_rows}"
    full_data_range = f"{first_cell}:{last_cell}"
    
    worksheet.write_formula(summary_start_row, col_idx_vals_start+1, f'=MIN({full_data_range})', fmt_border)
    worksheet.write_formula(summary_start_row, col_idx_vals_start+2, f'=MAX({full_data_range})', fmt_border)
    
    cell_min_ref = f"{chr(65+col_idx_vals_start+1)}{summary_start_row+1}"
    cell_max_ref = f"{chr(65+col_idx_vals_start+2)}{summary_start_row+1}"
    worksheet.write_formula(summary_start_row+1, col_idx_vals_start+1, f'={cell_max_ref}-{cell_min_ref}', fmt_border)
    
    cell_range_w_ref = f"{chr(65+col_idx_vals_start+1)}{summary_start_row+2}"
    worksheet.write_formula(summary_start_row+2, col_idx_vals_start+1, f'={cell_range_w_ref}/{cell_min_ref}', fmt_pct)
    
    worksheet.write(summary_start_row, col_idx_pct, "Max Deviation", fmt_bold_border)
    pct_col_char = chr(65+col_idx_pct)
    pct_col_range = f"{pct_col_char}3:{pct_col_char}{start_row+num_rows}"
    worksheet.write_formula(summary_start_row, col_idx_check, f'=MAX({pct_col_range})', fmt_pct)
    worksheet.set_column(0, col_idx_check+1, 15)

def write_median_timeseries_sheet(wb, sheet_name, df, noon):
    """Writes the time, median GHI, and median POA to a new sheet and appends summary statistics."""
    cols_to_keep = ["t_stamp", "t_stamp_dt", "Median_GHI", "Median_POA"]
    available_cols = [c for c in cols_to_keep if c in df.columns]
    
    df_export = df[available_cols].copy()
    
    # Calculate empirical noon times
    noon_naive = noon.replace(tzinfo=None)
    
    if "Median_GHI" in df.columns and not df_export["Median_GHI"].isna().all():
        idx_max_ghi = df_export["Median_GHI"].idxmax()
        time_max_ghi = df_export.loc[idx_max_ghi, "t_stamp_dt"]
        diff_ghi = time_max_ghi - noon_naive
        str_max_ghi = time_max_ghi.strftime("%H:%M:%S")
        diff_ghi_mins = int(diff_ghi.total_seconds() / 60)
    else:
        str_max_ghi = "N/A"
        diff_ghi_mins = "N/A"

    if "Median_POA" in df.columns and not df_export["Median_POA"].isna().all():
        idx_min_poa = df_export["Median_POA"].idxmin()
        time_min_poa = df_export.loc[idx_min_poa, "t_stamp_dt"]
        diff_poa = time_min_poa - noon_naive
        str_min_poa = time_min_poa.strftime("%H:%M:%S")
        diff_poa_mins = int(diff_poa.total_seconds() / 60)
    else:
        str_min_poa = "N/A"
        diff_poa_mins = "N/A"

    # Drop t_stamp_dt before exporting to avoid confusing excel
    if "t_stamp_dt" in df_export.columns:
        df_export = df_export.drop(columns=["t_stamp_dt"])

    df_export.to_excel(wb, sheet_name=sheet_name, index=False)
    
    worksheet = wb.sheets[sheet_name]
    workbook = wb.book
    
    worksheet.set_column("A:A", 20)
    worksheet.set_column("B:C", 15)
    
    fmt_bold = workbook.add_format({'bold': True})
    fmt_border = workbook.add_format({'border': 1})
    fmt_bold_border = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#EFEFEF'})
    
    # Write summary box below the data
    last_row = len(df_export) + 2
    
    worksheet.write(last_row, 0, "Solar Noon Estimations", fmt_bold_border)
    worksheet.write(last_row, 1, "Time", fmt_bold_border)
    worksheet.write(last_row, 2, "Diff vs Calculated (mins)", fmt_bold_border)
    
    worksheet.write(last_row + 1, 0, "Calculated Solar Noon", fmt_bold)
    worksheet.write(last_row + 1, 1, noon_naive.strftime("%H:%M:%S"), fmt_border)
    worksheet.write(last_row + 1, 2, 0, fmt_border)
    
    worksheet.write(last_row + 2, 0, "Est. Noon (Max Median GHI)", fmt_bold)
    worksheet.write(last_row + 2, 1, str_max_ghi, fmt_border)
    worksheet.write(last_row + 2, 2, diff_ghi_mins, fmt_border)
    
    worksheet.write(last_row + 3, 0, "Est. Noon (Min Median POA)", fmt_bold)
    worksheet.write(last_row + 3, 1, str_min_poa, fmt_border)
    worksheet.write(last_row + 3, 2, diff_poa_mins, fmt_border)


def create_full_excel_report(output_path, records1, day1, records2, day2, df_median, noon2):
    wb = pd.ExcelWriter(output_path, engine="xlsxwriter")
    
    write_excel_sheet(wb, f"Before_{day1}", day1, records1)
    write_excel_sheet(wb, f"After_{day2}", day2, records2)
    write_median_timeseries_sheet(wb, f"Median_Timeseries_{day2}", df_median, noon2)
    
    wb.close()
    print(f"Saved Excel Analysis: {output_path.name}")


# ------------------------------------------------------------
# 5) MAIN EXECUTION
# ------------------------------------------------------------

def main():
    excel_files = list(input_folder.glob("*.xlsx"))
    if not excel_files:
        print(f"No files in {input_folder}"); sys.exit()

    print(f"Loading {len(excel_files)} files...")
    loaded = []
    for f in excel_files:
        try:
            df = prep_dataframe(pd.read_excel(f))
            poa_cols = [c for c in df.columns if "/POA_" in c and "TILT" not in c and c in SENSORS_TO_INCLUDE]
            ghi_cols = [c for c in df.columns if "/GHI" in c and "TILT" not in c] 
            poa_tilt = [c for c in df.columns if "/POA_" in c and "TILT" in c and c.replace("_TILT_ANGLE","") in SENSORS_TO_INCLUDE]
            ghi_tilt = [c for c in df.columns if "/GHI_TILT" in c]
            
            loaded.append({
                "file": f, "df": df,
                "poa_cols": poa_cols, "ghi_cols": ghi_cols,
                "poa_tilt": poa_tilt, "ghi_tilt": ghi_tilt
            })
        except Exception as e:
            print(f"Skip {f.name}: {e}")

    day1, day2 = parse_date(CompareDates[0]), parse_date(CompareDates[1])
    src1 = find_best_source_for_day(loaded, day1)
    src2 = find_best_source_for_day(loaded, day2)

    if not src1 or not src2:
        print("Missing data for comparison dates."); sys.exit()

    noon1 = get_precise_solar_noon(day1)
    noon2 = get_precise_solar_noon(day2)
    noon1_naive = noon1.replace(tzinfo=None)
    noon2_naive = noon2.replace(tzinfo=None)
    
    df1, df2 = src1["df_filtered"], src2["df_filtered"]
    row1 = df1.iloc[(df1["t_stamp_dt"] - noon1_naive).abs().argsort()[:1]]
    row2 = df2.iloc[(df2["t_stamp_dt"] - noon2_naive).abs().argsort()[:1]]

    print("\nGenerating Plots...")
    plot_compare_2x2(
        df1, df2, src1["ghi_cols"], src1["ghi_tilt"], src2["ghi_cols"], src2["ghi_tilt"],
        day1, day2, noon1, noon2, row1, row2,
        title_prefix="GHI & Tilt Comparison", filename_suffix="Combined", mode="GHI"
    )
    
    plot_compare_2x2(
        df1, df2, src1["poa_cols"], src1["poa_tilt"], src2["poa_cols"], src2["poa_tilt"],
        day1, day2, noon1, noon2, row1, row2,
        title_prefix="POA & Tilt Comparison", filename_suffix="Combined", mode="POA"
    )
    
    stations = sorted(list(set([c.split("/")[0] for c in src1["poa_cols"]])))
    for st in stations:
        p1 = [c for c in src1["poa_cols"] if c.startswith(st)]
        t1 = [c for c in src1["poa_tilt"] if c.startswith(st)]
        p2 = [c for c in src2["poa_cols"] if c.startswith(st)]
        t2 = [c for c in src2["poa_tilt"] if c.startswith(st)]
        
        if p1 and p2:
            st_dir = output_folder / st
            st_dir.mkdir(exist_ok=True)
            fname = f"POA_COMPARE_{day1}_VS_{day2}_{st}.png"
            source_path = output_folder / fname
            target_path = st_dir / fname
            
            plot_compare_2x2(
                df1, df2, p1, t1, p2, t2,
                day1, day2, noon1, noon2, row1, row2,
                title_prefix=f"POA & Tilt Comparison ({st})", filename_suffix=st, mode="POA"
            )
            if target_path.exists(): target_path.unlink()
            source_path.rename(target_path)
            print(f"Moved {fname} to {st_dir}")

    print("\nGenerating Overlay Plot...")
    df2_medians = calculate_medians(df2, src2["ghi_cols"], src2["poa_cols"])
    plot_median_ghi_poa_overlay(df2_medians, day2, noon2, start_time_limit, end_time_limit)

    print("\nGenerating Excel Analysis...")
    def get_records(row, cols, m_type):
        recs = []
        for c in cols:
            if c in row.columns:
                recs.append({"Station": c.split("/")[0], "Type": m_type, "Sensor": c, "Value": float(row[c].iloc[0])})
        return recs

    recs1 = get_records(row1, src1["poa_cols"], "POA") + get_records(row1, src1["ghi_cols"], "GHI")
    recs2 = get_records(row2, src2["poa_cols"], "POA") + get_records(row2, src2["ghi_cols"], "GHI")

    create_full_excel_report(output_folder / "Solar_Noon_Analysis.xlsx", recs1, day1, recs2, day2, df2_medians, noon2)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()