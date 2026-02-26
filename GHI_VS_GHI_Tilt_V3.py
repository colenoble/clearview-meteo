import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
from datetime import datetime

# pvlib is required for precise solar noon
try:
    import pvlib
except ImportError:
    pvlib = None


# ------------------------------------------------------------
# 1) USER CONFIGURATION
# ------------------------------------------------------------

input_folder = Path("inputs/GHI&GHI_tilt")

script_name = Path(__file__).stem
output_folder = Path("outputs") / script_name
output_folder.mkdir(parents=True, exist_ok=True)

# Analysis time window (inclusive)
start_time_limit = "11:00:00"
end_time_limit = "14:00:00"

# Two-day compare
TwoDayCompare = True
CompareDates = ("2026-02-10", "2026-02-25")

# --- Site coordinates (as requested) ---
LAT = 40.26
LON = -83.99
TZ = "America/Toronto"

# Station colors (fixed mapping)
station_colors = {
    "MET02": "tab:blue",
    "MET16": "tab:green",
    "MET22": "tab:red",
    "MET37": "tab:orange",
}

# Legend formatting
LegendFontSize = "small"
LegendNcol = 1
LegendAlpha = 0.9

# --- NEW: Control tick clutter ---
# Major tick label interval in minutes (recommended: 10 or 15)
MAJOR_TICK_MINUTES = 15

# Minor tick interval in minutes (grid only, no labels). Set None to disable.
MINOR_TICK_MINUTES = 5


# ------------------------------------------------------------
# 2) HELPERS
# ------------------------------------------------------------

def parse_date(date_str: str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t_stamp_dt"] = pd.to_datetime(df["t_stamp"], errors="coerce")
    df = df.sort_values("t_stamp_dt")
    df["date"] = df["t_stamp_dt"].dt.date
    return df

def filter_day_time(df: pd.DataFrame, day):
    day_df = df[df["date"] == day].copy()
    start_ts = pd.to_datetime(f"{day} {start_time_limit}")
    end_ts = pd.to_datetime(f"{day} {end_time_limit}")
    return day_df[(day_df["t_stamp_dt"] >= start_ts) & (day_df["t_stamp_dt"] <= end_ts)].copy()

def get_precise_solar_noon(date_obj):
    """
    Precise solar transit (noon) using pvlib (same method as POA script). [1](https://pclconnects-my.sharepoint.com/personal/colenoble_pcl_com/Documents/Microsoft%20Copilot%20Chat%20Files/POA_VS_POA_Tilt_V3.py)
    """
    if pvlib is None:
        raise ImportError("pvlib is not installed. Install with: pip install pvlib")

    times = pd.date_range(start=date_obj, periods=1, freq="D", tz=TZ)
    location = pvlib.location.Location(LAT, LON, tz=TZ)
    return location.get_sun_rise_set_transit(times)["transit"].iloc[0]

def add_stats_box(ax, noon_row, cols, label, unit):
    """Median / Avg / Range / Δ at noon across given columns. [1](https://pclconnects-my.sharepoint.com/personal/colenoble_pcl_com/Documents/Microsoft%20Copilot%20Chat%20Files/POA_VS_POA_Tilt_V3.py)"""
    if noon_row is None or noon_row.empty or not cols:
        return

    present_cols = [c for c in cols if c in noon_row.columns]
    if not present_cols:
        return

    vals = noon_row[present_cols].iloc[0].astype(float)
    v_min, v_max = vals.min(), vals.max()
    delta = v_max - v_min

    stats_text = (
        f"{label} @ Noon\n"
        f"Median: {vals.median():.2f}{unit}\n"
        f"Average: {vals.mean():.2f}{unit}\n"
        f"Range: {v_min:.1f} - {v_max:.1f}{unit}\n"
        f"Delta (Δ): {delta:.1f}{unit}"
    )

    ax.text(
        0.01, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

def find_best_source_for_day(loaded_items, target_day):
    """Choose the file with the most rows in the time window for that day."""
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
                "ghi_cols": item["ghi_cols"],
                "tilt_cols": item["tilt_cols"],
            }
    return best if best_count > 0 else None

def format_time_axis(ax):
    """
    Apply clean time-axis formatting AFTER plotting (so it doesn't get overridden).
    Major ticks every MAJOR_TICK_MINUTES, minor ticks every MINOR_TICK_MINUTES.
    Labels show HH:MM (no seconds). [1](https://pclconnects-my.sharepoint.com/personal/colenoble_pcl_com/Documents/Microsoft%20Copilot%20Chat%20Files/POA_VS_POA_Tilt_V3.py)
    """
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=MAJOR_TICK_MINUTES))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if MINOR_TICK_MINUTES:
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=MINOR_TICK_MINUTES))

    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)

def set_top_right_legend(ax, title="Sensors"):
    """
    Force legend INSIDE each subplot at top-right using axes coordinates.
    Dedup labels to avoid repeats.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    dedup = {}
    for h, l in zip(handles, labels):
        if l and l != "_nolegend_":
            dedup[l] = h
    if not dedup:
        return

    labels_d = list(dedup.keys())
    handles_d = [dedup[l] for l in labels_d]

    ax.legend(
        handles_d, labels_d,
        title=title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        fontsize=LegendFontSize,
        ncol=LegendNcol,
        frameon=True,
        framealpha=LegendAlpha
    )


# ------------------------------------------------------------
# 3) FILE DISCOVERY
# ------------------------------------------------------------

excel_files = list(input_folder.glob("*.xlsx"))
if not excel_files:
    print(f"No Excel files found in {input_folder}")
    sys.exit()

print(f"Found {len(excel_files)} files to process.")
print(f"Saving plots to: {output_folder}\n")


# ------------------------------------------------------------
# 4) LOAD ALL FILES
# ------------------------------------------------------------

loaded = []
for file_path in excel_files:
    print(f"Loading: {file_path.name}")
    try:
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f" - Error loading file: {e}")
        continue

    df = prep_dataframe(df_raw)

    ghi_cols = [c for c in df.columns if "/GHI" in c and "TILT" not in c]
    tilt_cols = [c for c in df.columns if "/GHI_TILT_ANGLE" in c]

    if not ghi_cols:
        print(" - Warning: No GHI columns found (pattern '/GHI' excluding 'TILT').")
    if not tilt_cols:
        print(" - Warning: No tilt columns found (pattern '/GHI_TILT_ANGLE').")

    loaded.append({
        "file": file_path,
        "df": df,
        "ghi_cols": ghi_cols,
        "tilt_cols": tilt_cols
    })

print("")


# ------------------------------------------------------------
# 5) TWO-DAY COMPARE (ONE OUTPUT ONLY) + METRICS
# ------------------------------------------------------------

if TwoDayCompare:
    if not CompareDates or len(CompareDates) != 2:
        raise ValueError("CompareDates must be two strings: ('YYYY-MM-DD','YYYY-MM-DD')")

    if pvlib is None:
        print("ERROR: pvlib is not installed. Install with: pip install pvlib")
        sys.exit()

    day1 = parse_date(CompareDates[0])
    day2 = parse_date(CompareDates[1])

    src1 = find_best_source_for_day(loaded, day1)
    src2 = find_best_source_for_day(loaded, day2)

    missing = []
    if src1 is None:
        missing.append(str(day1))
    if src2 is None:
        missing.append(str(day2))

    if missing:
        print(f"Skipping compare plot: No data in time window for date(s): {', '.join(missing)}")
        print("\nProcessing complete.")
        sys.exit()

    # Solar noon (tz-aware) -> naive for plotting
    noon1 = get_precise_solar_noon(day1)
    noon2 = get_precise_solar_noon(day2)
    noon1_naive = noon1.replace(tzinfo=None)
    noon2_naive = noon2.replace(tzinfo=None)
    noon1_str = noon1.strftime("%H:%M")
    noon2_str = noon2.strftime("%H:%M")

    df1 = src1["df_filtered"]
    df2 = src2["df_filtered"]

    # Row closest to solar noon in each filtered window
    noon_row_1 = df1.iloc[(df1["t_stamp_dt"] - noon1_naive).abs().argsort()[:1]]
    noon_row_2 = df2.iloc[(df2["t_stamp_dt"] - noon2_naive).abs().argsort()[:1]]

    # 2x2 layout, share x by column so top/bottom align
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey="row", sharex="col")
    ax_ghi_1, ax_ghi_2 = axes[0, 0], axes[0, 1]
    ax_tilt_1, ax_tilt_2 = axes[1, 0], axes[1, 1]

    # LEFT column (day1)
    x1 = df1["t_stamp_dt"]
    for col in src1["ghi_cols"]:
        if col in df1.columns:
            st = col.split("/")[0]
            ax_ghi_1.plot(x1, df1[col], label=st, color=station_colors.get(st, "black"), linestyle="-")
    ax_ghi_1.axvline(x=noon1_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_ghi_1.set_title(f"GHI • {day1} • Solar Noon: {noon1_str}", fontsize=12)
    ax_ghi_1.set_ylabel("GHI [W/m²]")
    add_stats_box(ax_ghi_1, noon_row_1, src1["ghi_cols"], "GHI", " W/m²")

    for col in src1["tilt_cols"]:
        if col in df1.columns:
            st = col.split("/")[0]
            ax_tilt_1.plot(x1, df1[col], label=st, color=station_colors.get(st, "black"), linestyle="-")
    ax_tilt_1.axvline(x=noon1_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_tilt_1.set_title(f"SR30 Internal Tilt Angle • {day1} • Solar Noon: {noon1_str}", fontsize=12)
    ax_tilt_1.set_ylabel("Sensor Tilt [°]")
    ax_tilt_1.set_xlabel("Time (HH:MM)")
    add_stats_box(ax_tilt_1, noon_row_1, src1["tilt_cols"], "Tilt", "°")

    # RIGHT column (day2)
    x2 = df2["t_stamp_dt"]
    for col in src2["ghi_cols"]:
        if col in df2.columns:
            st = col.split("/")[0]
            ax_ghi_2.plot(x2, df2[col], label=st, color=station_colors.get(st, "black"), linestyle="-")
    ax_ghi_2.axvline(x=noon2_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_ghi_2.set_title(f"GHI • {day2} • Solar Noon: {noon2_str}", fontsize=12)
    add_stats_box(ax_ghi_2, noon_row_2, src2["ghi_cols"], "GHI", " W/m²")

    for col in src2["tilt_cols"]:
        if col in df2.columns:
            st = col.split("/")[0]
            ax_tilt_2.plot(x2, df2[col], label=st, color=station_colors.get(st, "black"), linestyle="-")
    ax_tilt_2.axvline(x=noon2_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_tilt_2.set_title(f"SR30 Internal Tilt Angle • {day2} • Solar Noon: {noon2_str}", fontsize=12)
    ax_tilt_2.set_xlabel("Time (HH:MM)")
    add_stats_box(ax_tilt_2, noon_row_2, src2["tilt_cols"], "Tilt", "°")

    # IMPORTANT: Apply time formatting AFTER plotting (prevents formatter being overwritten)
    for ax in (ax_ghi_1, ax_tilt_1, ax_ghi_2, ax_tilt_2):
        format_time_axis(ax)

    # Hide top x tick labels (bottom row shows time)
    ax_ghi_1.tick_params(axis="x", labelbottom=False)
    ax_ghi_2.tick_params(axis="x", labelbottom=False)

    # Rotate bottom tick labels
    for ax in (ax_tilt_1, ax_tilt_2):
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Force legends top-right inside each subplot
    set_top_right_legend(ax_ghi_1, title="Sensors")
    set_top_right_legend(ax_ghi_2, title="Sensors")
    set_top_right_legend(ax_tilt_1, title="Sensors")
    set_top_right_legend(ax_tilt_2, title="Sensors")

    fig.suptitle(
        f"Two-Day Compare (GHI + Tilt) with Solar Noon Metrics\n"
        f"Window: {start_time_limit}–{end_time_limit}\n"
        f"Left source: {src1['file'].name} | Right source: {src2['file'].name}",
        fontsize=13,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    save_name = f"COMPARE_{day1}_VS_{day2}".replace(" ", "_")
    save_path = output_folder / f"{save_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved compare plot: {save_path.name}")
    print("\nProcessing complete.")
    sys.exit()

print("\nProcessing complete.")