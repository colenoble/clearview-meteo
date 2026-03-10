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
CompareDates = ("2026-02-10", "2026-03-08")

# Time Window for Plots
start_time_limit = "12:00:00"
end_time_limit = "15:00:00"

# Site Coords / Timezone
LAT = 40.26
LON = -83.99
TZ = "America/Toronto"

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
    Standardizes timestamps and applies DST offset for dates >= March 8.
    """
    df = df.copy()
    df["t_stamp_dt"] = pd.to_datetime(df["t_stamp"], errors="coerce")
    
    # DST Fix: Add 1 hour if date is >= March 8, 2026
    dst_start = pd.Timestamp("2026-03-08")
    dst_mask = df["t_stamp_dt"] >= dst_start
    if dst_mask.any():
        df.loc[dst_mask, "t_stamp_dt"] += pd.Timedelta(hours=1)
        
    df = df.sort_values("t_stamp_dt")
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

    # % Deviation Check (Only for Irradiance W/m²)
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


# ------------------------------------------------------------
# 4) EXCEL GENERATION (Split Tabs + Summary Box)
# ------------------------------------------------------------

def write_excel_sheet(wb, sheet_name, date_val, records):
    """
    Writes a single tab for a specific date (Before/After).
    """
    if not records:
        return

    # Create DataFrame
    df_raw = pd.DataFrame(records)
    
    # 1. Structure Data: One row per station/type
    grouped = df_raw.groupby(["Station", "Type"])
    
    final_rows = []
    max_vals = 0
    
    for (st, m_type), group in grouped:
        row_data = {
            "Station": st,
            "Type": m_type,
        }
        
        # Sort so Val_1 is consistent (POA_1 vs POA_2)
        sorted_sensors = group.sort_values("Sensor")
        vals = sorted_sensors["Value"].tolist()
        
        if len(vals) > max_vals:
            max_vals = len(vals)
            
        for i, v in enumerate(vals):
            row_data[f"Val_{i+1}"] = v
            
        final_rows.append(row_data)

    df_out = pd.DataFrame(final_rows)
    
    # Column Order: Station, Type, Val_1, Val_2...
    base_cols = ["Station", "Type"]
    val_cols = [f"Val_{i+1}" for i in range(max_vals)]
    df_out = df_out[base_cols + val_cols]

    # Write Data to Sheet
    df_out.to_excel(wb, sheet_name=sheet_name, startrow=2, header=False, index=False)
    worksheet = wb.sheets[sheet_name]
    workbook = wb.book
    
    # --- Formatting ---
    fmt_header = workbook.add_format({'bold': True, 'border': 1, 'bg_color': '#EFEFEF'})
    fmt_dec = workbook.add_format({'num_format': '0.00'})
    fmt_pct = workbook.add_format({'num_format': '0.00%'})
    fmt_pass = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    fmt_fail = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    fmt_border = workbook.add_format({'border': 1})
    fmt_bold_border = workbook.add_format({'bold': True, 'border': 1})
    
    # Write Date at Top
    worksheet.write(0, 0, "Date", fmt_header)
    worksheet.write(0, 1, str(date_val), fmt_border)
    
    # Write Table Headers (Row 2, 0-indexed is 1)
    col_headers = list(df_out.columns) + ["Min", "Max", "Delta", "% Deviation", "Check (<2%)"]
    for i, h in enumerate(col_headers):
        worksheet.write(1, i, h, fmt_header)
        
    # --- Write Main Table Formulas ---
    # Data starts at Row 3 (Index 2). 
    start_row = 2
    num_rows = len(df_out)
    
    # Indexes for columns
    # 0=Station, 1=Type, 2=Val_1 ... 
    col_idx_vals_start = 2
    col_idx_vals_end = 2 + max_vals - 1
    
    col_idx_min = col_idx_vals_end + 1
    col_idx_max = col_idx_min + 1
    col_idx_delta = col_idx_max + 1
    col_idx_pct = col_idx_delta + 1
    col_idx_check = col_idx_pct + 1
    
    # To store range for summary calculation later
    all_vals_range = [] 

    for r in range(num_rows):
        row_num = start_row + r + 1 # Excel 1-based row
        
        # Get Letters
        first_val_char = chr(65 + col_idx_vals_start)
        last_val_char = chr(65 + col_idx_vals_end)
        
        range_vals = f"{first_val_char}{row_num}:{last_val_char}{row_num}"
        all_vals_range.append(range_vals)

        # Min
        worksheet.write_formula(start_row + r, col_idx_min, f'=MIN({range_vals})', fmt_dec)
        # Max
        worksheet.write_formula(start_row + r, col_idx_max, f'=MAX({range_vals})', fmt_dec)
        
        # Delta: =MAX - MIN (Only if Count > 1)
        col_min_char = chr(65 + col_idx_min)
        col_max_char = chr(65 + col_idx_max)
        
        formula_delta = f'=IF(COUNT({range_vals})>1, {col_max_char}{row_num}-{col_min_char}{row_num}, "")'
        worksheet.write_formula(start_row + r, col_idx_delta, formula_delta, fmt_dec)
        
        # % Deviation
        col_delta_char = chr(65 + col_idx_delta)
        formula_pct = f'=IF(ISNUMBER({col_delta_char}{row_num}), IF({col_min_char}{row_num}<>0, {col_delta_char}{row_num}/{col_min_char}{row_num}, 0), "")'
        worksheet.write_formula(start_row + r, col_idx_pct, formula_pct, fmt_pct)
        
        # Check
        col_pct_char = chr(65 + col_idx_pct)
        formula_check = f'=IF(ISNUMBER({col_pct_char}{row_num}), IF({col_pct_char}{row_num}<0.02, "YES", "NO"), "")'
        worksheet.write_formula(start_row + r, col_idx_check, formula_check)

        # Conditional Format for Check
        worksheet.conditional_format(start_row+r, col_idx_check, start_row+r, col_idx_check,
                                     {'type': 'cell', 'criteria': '==', 'value': '"YES"', 'format': fmt_pass})
        worksheet.conditional_format(start_row+r, col_idx_check, start_row+r, col_idx_check,
                                     {'type': 'cell', 'criteria': '==', 'value': '"NO"', 'format': fmt_fail})

    # --- SUMMARY BOX (Bottom) ---
    summary_start_row = start_row + num_rows + 2
    
    # Labels
    worksheet.write(summary_start_row, col_idx_vals_start, "Range (min max)", fmt_bold_border)
    worksheet.write(summary_start_row+1, col_idx_vals_start, "Range (W/m^2)", fmt_bold_border)
    worksheet.write(summary_start_row+2, col_idx_vals_start, "% difference from Min Irradiance", fmt_bold_border)
    
    # We need a range that covers ALL Values (Val_1 to Val_N) for ALL rows.
    # Ex: C3:D10
    first_cell = f"{chr(65+col_idx_vals_start)}3"
    last_cell = f"{chr(65+col_idx_vals_end)}{start_row+num_rows}"
    full_data_range = f"{first_cell}:{last_cell}"
    
    # 1. Range (Min Max)
    # Min Cell
    worksheet.write_formula(summary_start_row, col_idx_vals_start+1, f'=MIN({full_data_range})', fmt_border)
    # Max Cell
    worksheet.write_formula(summary_start_row, col_idx_vals_start+2, f'=MAX({full_data_range})', fmt_border)
    
    # 2. Range (W/m^2) [Max - Min]
    # Refs to the two cells we just wrote
    cell_min_ref = f"{chr(65+col_idx_vals_start+1)}{summary_start_row+1}"
    cell_max_ref = f"{chr(65+col_idx_vals_start+2)}{summary_start_row+1}"
    worksheet.write_formula(summary_start_row+1, col_idx_vals_start+1, f'={cell_max_ref}-{cell_min_ref}', fmt_border)
    
    # 3. % Difference from Min GHI
    # We use the Global Min we found in step 1 as the denominator.
    cell_range_w_ref = f"{chr(65+col_idx_vals_start+1)}{summary_start_row+2}"
    worksheet.write_formula(summary_start_row+2, col_idx_vals_start+1, f'={cell_range_w_ref}/{cell_min_ref}', fmt_pct)
    
    # 4. Max Deviation Box (Right side)
    # Screenshot puts it to the right of the summary
    worksheet.write(summary_start_row, col_idx_pct, "Max Deviation (between individual sensors)", fmt_bold_border)
    
    # MAX of the % Deviation column
    pct_col_char = chr(65+col_idx_pct)
    pct_col_range = f"{pct_col_char}3:{pct_col_char}{start_row+num_rows}"
    worksheet.write_formula(summary_start_row, col_idx_check, f'=MAX({pct_col_range})', fmt_pct)
    
    # Auto-width
    worksheet.set_column(0, col_idx_check+1, 15)


def create_full_excel_report(output_path, records1, day1, records2, day2):
    wb = pd.ExcelWriter(output_path, engine="xlsxwriter")
    
    # Sheet 1: Before
    sheet1_name = f"Before_{day1}"
    write_excel_sheet(wb, sheet1_name, day1, records1)
    
    # Sheet 2: After
    sheet2_name = f"After_{day2}"
    write_excel_sheet(wb, sheet2_name, day2, records2)
    
    wb.close()
    print(f"Saved Excel Analysis: {output_path.name}")


# ------------------------------------------------------------
# 5) MAIN EXECUTION
# ------------------------------------------------------------

def main():
    # A) Load Files
    excel_files = list(input_folder.glob("*.xlsx"))
    if not excel_files:
        print(f"No files in {input_folder}"); sys.exit()

    print(f"Loading {len(excel_files)} files...")
    loaded = []
    for f in excel_files:
        try:
            df = prep_dataframe(pd.read_excel(f))
            # Identifiy Columns
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

    # B) Prepare Compare Data
    day1, day2 = parse_date(CompareDates[0]), parse_date(CompareDates[1])
    src1 = find_best_source_for_day(loaded, day1)
    src2 = find_best_source_for_day(loaded, day2)

    if not src1 or not src2:
        print("Missing data for comparison dates."); sys.exit()

    # Solar Noon Calcs
    noon1 = get_precise_solar_noon(day1)
    noon2 = get_precise_solar_noon(day2)
    noon1_naive = noon1.replace(tzinfo=None)
    noon2_naive = noon2.replace(tzinfo=None)
    
    # Extract Noon Rows
    df1, df2 = src1["df_filtered"], src2["df_filtered"]
    row1 = df1.iloc[(df1["t_stamp_dt"] - noon1_naive).abs().argsort()[:1]]
    row2 = df2.iloc[(df2["t_stamp_dt"] - noon2_naive).abs().argsort()[:1]]

    # C) PLOTTING (Standard V1 Logic)
    print("\nGenerating Plots...")
    # 1. GHI Compare
    plot_compare_2x2(
        df1, df2, 
        src1["ghi_cols"], src1["ghi_tilt"], 
        src2["ghi_cols"], src2["ghi_tilt"],
        day1, day2, noon1, noon2, row1, row2,
        title_prefix="GHI & Tilt Comparison", 
        filename_suffix="Combined", mode="GHI"
    )
    # 2. POA Compare
    plot_compare_2x2(
        df1, df2, 
        src1["poa_cols"], src1["poa_tilt"], 
        src2["poa_cols"], src2["poa_tilt"],
        day1, day2, noon1, noon2, row1, row2,
        title_prefix="POA & Tilt Comparison", 
        filename_suffix="Combined", mode="POA"
    )
    # 3. Station Plots
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
                title_prefix=f"POA & Tilt Comparison ({st})", 
                filename_suffix=st, mode="POA"
            )
            if target_path.exists(): target_path.unlink()
            source_path.rename(target_path)
            print(f"Moved {fname} to {st_dir}")

    # D) EXCEL GENERATION (Updated for Split Tabs)
    print("\nGenerating Excel Analysis...")
    
    def get_records(row, cols, m_type):
        recs = []
        for c in cols:
            if c in row.columns:
                st = c.split("/")[0]
                val = row[c].iloc[0]
                recs.append({
                    "Station": st,
                    "Type": m_type,
                    "Sensor": c, 
                    "Value": float(val)
                })
        return recs

    # Collect Records for Day 1
    recs1 = []
    recs1.extend(get_records(row1, src1["poa_cols"], "POA"))
    recs1.extend(get_records(row1, src1["ghi_cols"], "GHI"))
    
    # Collect Records for Day 2
    recs2 = []
    recs2.extend(get_records(row2, src2["poa_cols"], "POA"))
    recs2.extend(get_records(row2, src2["ghi_cols"], "GHI"))

    # Write Split Tabs
    create_full_excel_report(output_folder / "Solar_Noon_Analysis.xlsx", recs1, day1, recs2, day2)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()