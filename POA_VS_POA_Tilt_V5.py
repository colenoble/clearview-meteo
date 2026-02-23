import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
from datetime import datetime

# pvlib used for precise solar noon (same method as POA V3)
try:
    import pvlib
except ImportError:
    pvlib = None


# ------------------------------------------------------------
# 1) PATH CONFIGURATION
# ------------------------------------------------------------
input_folder = Path("inputs/GHI&GHI_tilt")
script_name = Path(__file__).stem
output_folder = Path("outputs") / script_name
output_folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2) SITE & ANALYSIS SETTINGS
# ------------------------------------------------------------

# --- Output Organization ---
GroupOutputsByStation = True  # If True, puts station-specific plots into subfolders

# --- Sensor Inclusion List ---
# Only sensors listed here will be processed. To exclude a sensor that is 
# down or distorting the data, simply comment it out or delete it from the list.
# (Note: Using the 'METXX/POA_X' format based on your dataframe headers)
SENSORS_TO_INCLUDE = [
   # "MET02/POA_1",
   # "MET02/POA_2",
    "MET16/POA_1",
    "MET16/POA_2",
    "MET22/POA_1",
    "MET22/POA_2",
    "MET37/POA_1",
    "MET37/POA_2",
]

# Original mode switch (per-day plots) remains available
PlotDetailed = True  

start_time_limit = "11:00:00"
end_time_limit = "14:00:00"

# --- Two-Day Compare Option (side-by-side) ---
TwoDayCompare = True
CompareDates = ("2026-02-10", "2026-02-22")  # YYYY-MM-DD

# --- Coordinates / TZ ---
LAT = 40.26
LON = -83.99
TZ = "America/Toronto"  

# Styling 
station_colors = {
    "MET02": "tab:blue",
    "MET16": "tab:green",
    "MET22": "tab:red",
    "MET37": "tab:orange",
}
line_styles = {"POA_1": "-", "POA_2": ":"}

# Clean time-axis (avoid clutter)
MAJOR_TICK_MINUTES = 15   
MINOR_TICK_MINUTES = 5    

# Legends pinned inside each subplot (top-right)
LegendFontSize = "small"
LegendNcol = 2
LegendAlpha = 0.9


# ------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ------------------------------------------------------------

def parse_date(date_str: str):
    """Parse YYYY-MM-DD into datetime.date."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean/sort and add datetime and date columns."""
    df = df.copy()
    df["t_stamp_dt"] = pd.to_datetime(df["t_stamp"], errors="coerce")
    df = df.sort_values("t_stamp_dt")
    df["date"] = df["t_stamp_dt"].dt.date
    return df

def filter_day_time(df: pd.DataFrame, day):
    """Slice to a day and the configured time window using datetimes."""
    day_df = df[df["date"] == day].copy()
    start_ts = pd.to_datetime(f"{day} {start_time_limit}")
    end_ts = pd.to_datetime(f"{day} {end_time_limit}")
    return day_df[(day_df["t_stamp_dt"] >= start_ts) & (day_df["t_stamp_dt"] <= end_ts)].copy()

def get_precise_solar_noon(date_obj):
    """Exact solar transit (noon) using pvlib."""
    if pvlib is None:
        raise ImportError("pvlib is not installed. Install with: pip install pvlib")
    times = pd.date_range(start=date_obj, periods=1, freq="D", tz=TZ)
    location = pvlib.location.Location(LAT, LON, tz=TZ)
    return location.get_sun_rise_set_transit(times)["transit"].iloc[0]

def add_stats_box(ax, noon_row, cols, label, unit):
    """Adds Median, Average, Range, Delta at solar noon."""
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

def format_time_axis(ax):
    """Clean time ticks AFTER plotting (prevents being overridden; avoids seconds clutter)."""
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=MAJOR_TICK_MINUTES))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if MINOR_TICK_MINUTES:
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=MINOR_TICK_MINUTES))

    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.15)

def set_top_right_legend(ax, title="Sensors"):
    """Force legend inside axes, top-right, and dedupe labels."""
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

def find_best_source_for_day(loaded_items, target_day):
    """
    Across ALL loaded files, choose the file that has the most rows
    for target_day within the time window.
    """
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
                "tilt_cols": item["tilt_cols"],
            }
    return best if best_count > 0 else None

def get_station_list(poa_cols):
    """Extract stations from POA columns."""
    return sorted(list(set([c.split("/")[0] for c in poa_cols])))

def plot_compare_2x2(df1, df2, poa_cols_1, tilt_cols_1, poa_cols_2, tilt_cols_2,
                     day1, day2, noon1_naive, noon2_naive, noon1_str, noon2_str,
                     noon_row_1, noon_row_2, suptitle, save_path):
    """
    Core: Create a 2x2 side-by-side compare:
      left=day1, right=day2; top=POA, bottom=Tilt
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey="row", sharex="col")
    ax_poa_1, ax_poa_2 = axes[0, 0], axes[0, 1]
    ax_tilt_1, ax_tilt_2 = axes[1, 0], axes[1, 1]

    # ---- Day 1 (left) ----
    x1 = df1["t_stamp_dt"]

    for c in poa_cols_1:
        if c in df1.columns:
            st, p_type = c.split("/")
            ax_poa_1.plot(
                x1, df1[c],
                label=f"{st} {p_type}",
                color=station_colors.get(st, "black"),
                linestyle=line_styles.get(p_type, "-")
            )
    ax_poa_1.axvline(x=noon1_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_poa_1.set_title(f"POA • {day1} • Noon: {noon1_str}", fontsize=12)
    ax_poa_1.set_ylabel("Irradiance [W/m²]")
    add_stats_box(ax_poa_1, noon_row_1, poa_cols_1, "POA", " W/m²")

    for c in tilt_cols_1:
        if c in df1.columns:
            st, t_type = c.split("/")
            p_type = t_type.replace("_TILT_ANGLE", "")
            ax_tilt_1.plot(
                x1, df1[c],
                label=f"{st} {p_type} Tilt",
                color=station_colors.get(st, "black"),
                linestyle=line_styles.get(p_type, "-")
            )
    ax_tilt_1.axvline(x=noon1_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_tilt_1.set_title(f"Tilt • {day1} • Noon: {noon1_str}", fontsize=12)
    ax_tilt_1.set_ylabel("Sensor Tilt [°]")
    ax_tilt_1.set_xlabel("Time (HH:MM)")
    add_stats_box(ax_tilt_1, noon_row_1, tilt_cols_1, "Tilt", "°")

    # ---- Day 2 (right) ----
    x2 = df2["t_stamp_dt"]

    for c in poa_cols_2:
        if c in df2.columns:
            st, p_type = c.split("/")
            ax_poa_2.plot(
                x2, df2[c],
                label=f"{st} {p_type}",
                color=station_colors.get(st, "black"),
                linestyle=line_styles.get(p_type, "-")
            )
    ax_poa_2.axvline(x=noon2_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_poa_2.set_title(f"POA • {day2} • Noon: {noon2_str}", fontsize=12)
    add_stats_box(ax_poa_2, noon_row_2, poa_cols_2, "POA", " W/m²")

    for c in tilt_cols_2:
        if c in df2.columns:
            st, t_type = c.split("/")
            p_type = t_type.replace("_TILT_ANGLE", "")
            ax_tilt_2.plot(
                x2, df2[c],
                label=f"{st} {p_type} Tilt",
                color=station_colors.get(st, "black"),
                linestyle=line_styles.get(p_type, "-")
            )
    ax_tilt_2.axvline(x=noon2_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_tilt_2.set_title(f"Tilt • {day2} • Noon: {noon2_str}", fontsize=12)
    ax_tilt_2.set_xlabel("Time (HH:MM)")
    add_stats_box(ax_tilt_2, noon_row_2, tilt_cols_2, "Tilt", "°")

    for ax in (ax_poa_1, ax_tilt_1, ax_poa_2, ax_tilt_2):
        format_time_axis(ax)

    ax_poa_1.tick_params(axis="x", labelbottom=False)
    ax_poa_2.tick_params(axis="x", labelbottom=False)

    for ax in (ax_tilt_1, ax_tilt_2):
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    set_top_right_legend(ax_poa_1, title="Sensors")
    set_top_right_legend(ax_poa_2, title="Sensors")
    set_top_right_legend(ax_tilt_1, title="Sensors")
    set_top_right_legend(ax_tilt_2, title="Sensors")

    fig.suptitle(suptitle, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# 4) LOAD FILES & FILTER SENSORS
# ------------------------------------------------------------
excel_files = list(input_folder.glob("*.xlsx"))
if not excel_files:
    print(f"No Excel files found in {input_folder}")
    sys.exit()

print(f"Found {len(excel_files)} files to process.")
print(f"Saving plots to: {output_folder}\n")

loaded = []
for file_path in excel_files:
    print(f"Loading: {file_path.name}")
    try:
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f" - Error loading file: {e}")
        continue

    df = prep_dataframe(df_raw)

    # Grab all valid columns, then filter against SENSORS_TO_INCLUDE
    raw_poa_cols = [c for c in df.columns if "/POA_" in c and "TILT" not in c and "RPOA" not in c]
    raw_tilt_cols = [c for c in df.columns if "/POA_" in c and "TILT_ANGLE" in c]

    poa_cols = [c for c in raw_poa_cols if c in SENSORS_TO_INCLUDE]
    
    # For tilt, we check if the base sensor name is in the inclusion list
    tilt_cols = [c for c in raw_tilt_cols if c.replace("_TILT_ANGLE", "") in SENSORS_TO_INCLUDE]

    loaded.append({
        "file": file_path,
        "df": df,
        "poa_cols": poa_cols,
        "tilt_cols": tilt_cols
    })

print("")


# ------------------------------------------------------------
# 5) TWO-DAY COMPARE MODE (COMBINED + PER-STATION)
# ------------------------------------------------------------
if TwoDayCompare:
    if pvlib is None:
        print("ERROR: pvlib is not installed. Install with: pip install pvlib")
        sys.exit()

    if not CompareDates or len(CompareDates) != 2:
        raise ValueError("CompareDates must be a tuple/list of TWO strings: ('YYYY-MM-DD','YYYY-MM-DD')")

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
        print(f"Skipping compare plots: No data in time window for date(s): {', '.join(missing)}")
        print("\nProcessing complete.")
        sys.exit()

    df1 = src1["df_filtered"]
    df2 = src2["df_filtered"]

    noon1 = get_precise_solar_noon(day1)
    noon2 = get_precise_solar_noon(day2)
    noon1_naive = noon1.replace(tzinfo=None)
    noon2_naive = noon2.replace(tzinfo=None)
    noon1_str = noon1.strftime("%H:%M")
    noon2_str = noon2.strftime("%H:%M")

    noon_row_1 = df1.iloc[(df1["t_stamp_dt"] - noon1_naive).abs().argsort()[:1]]
    noon_row_2 = df2.iloc[(df2["t_stamp_dt"] - noon2_naive).abs().argsort()[:1]]

    # -------------------------
    # A) COMBINED compare plot
    # -------------------------
    combined_save = output_folder / f"POA_COMPARE_{day1}_VS_{day2}_Combined.png"
    combined_title = (
        f"POA Two-Day Compare (Combined) with Solar Noon Metrics\n"
        f"Window: {start_time_limit}–{end_time_limit}\n"
        f"Left source: {src1['file'].name} | Right source: {src2['file'].name}"
    )

    plot_compare_2x2(
        df1=df1, df2=df2,
        poa_cols_1=src1["poa_cols"], tilt_cols_1=src1["tilt_cols"],
        poa_cols_2=src2["poa_cols"], tilt_cols_2=src2["tilt_cols"],
        day1=day1, day2=day2,
        noon1_naive=noon1_naive, noon2_naive=noon2_naive,
        noon1_str=noon1_str, noon2_str=noon2_str,
        noon_row_1=noon_row_1, noon_row_2=noon_row_2,
        suptitle=combined_title,
        save_path=combined_save
    )
    print(f"Saved combined compare plot: {combined_save.name}")

    # ------------------------------------
    # B) PER-STATION compare plots 
    # ------------------------------------
    stations = get_station_list(src1["poa_cols"])
    if not stations:
        print("No stations found in POA columns for day1; skipping per-station plots.")
        print("\nProcessing complete.")
        sys.exit()

    for st in stations:
        poa1_st = [c for c in src1["poa_cols"] if c.startswith(st + "/")]
        tilt1_st = [c for c in src1["tilt_cols"] if c.startswith(st + "/")]
        poa2_st = [c for c in src2["poa_cols"] if c.startswith(st + "/")]
        tilt2_st = [c for c in src2["tilt_cols"] if c.startswith(st + "/")]

        if (not poa1_st and not tilt1_st) or (not poa2_st and not tilt2_st):
            print(f" - Skipping {st}: station not present on one of the days.")
            continue

        # Subfolder routing logic
        if GroupOutputsByStation:
            st_out_dir = output_folder / st
            st_out_dir.mkdir(parents=True, exist_ok=True)
            st_save = st_out_dir / f"POA_COMPARE_{day1}_VS_{day2}_{st}.png"
        else:
            st_save = output_folder / f"POA_COMPARE_{day1}_VS_{day2}_{st}.png"

        st_title = (
            f"POA Two-Day Compare ({st}) with Solar Noon Metrics\n"
            f"Window: {start_time_limit}–{end_time_limit}\n"
            f"Left source: {src1['file'].name} | Right source: {src2['file'].name}"
        )

        plot_compare_2x2(
            df1=df1, df2=df2,
            poa_cols_1=poa1_st, tilt_cols_1=tilt1_st,
            poa_cols_2=poa2_st, tilt_cols_2=tilt2_st,
            day1=day1, day2=day2,
            noon1_naive=noon1_naive, noon2_naive=noon2_naive,
            noon1_str=noon1_str, noon2_str=noon2_str,
            noon_row_1=noon_row_1, noon_row_2=noon_row_2,
            suptitle=st_title,
            save_path=st_save
        )
        print(f" - Saved station compare plot: {st_save.name}")

    print("\nProcessing complete.")
    sys.exit()


# ------------------------------------------------------------
# 6) ORIGINAL MODE (V3 BEHAVIOR) — runs when TwoDayCompare=False
# ------------------------------------------------------------

if pvlib is None:
    print("ERROR: pvlib is not installed. Install with: pip install pvlib")
    sys.exit()

for item in loaded:
    file_path = item["file"]
    df = item["df"]
    poa_cols = item["poa_cols"]
    tilt_cols = item["tilt_cols"]

    if not poa_cols or not tilt_cols:
        continue

    unique_days = sorted(df["date"].dropna().unique())
    for day in unique_days:
        day_df = df[df["date"] == day].copy()

        precise_noon_dt = get_precise_solar_noon(day)
        noon_str = precise_noon_dt.strftime("%H:%M:%S")
        noon_naive = precise_noon_dt.replace(tzinfo=None)

        df_filtered = filter_day_time(day_df, day)
        if df_filtered.empty:
            continue

        noon_row = df_filtered.iloc[(df_filtered["t_stamp_dt"] - noon_naive).abs().argsort()[:1]]

        plot_configs = [("Combined", poa_cols, tilt_cols)]
        if PlotDetailed:
            stations = sorted(list(set([c.split("/")[0] for c in poa_cols])))
            for st in stations:
                plot_configs.append(
                    (st,
                     [c for c in poa_cols if c.startswith(st)],
                     [c for c in tilt_cols if c.startswith(st)])
                )

        for name, p_cols, t_cols in plot_configs:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
            x_axis = df_filtered["t_stamp_dt"]

            for ax in (ax1, ax2):
                ax.axvline(x=noon_naive, color="red", linestyle="--", linewidth=1.5,
                           label=f"Solar Noon ({noon_str})")
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # POA
            for c in p_cols:
                st, p_type = c.split("/")
                ax1.plot(
                    x_axis, df_filtered[c],
                    label=f"{st} {p_type}",
                    color=station_colors.get(st, "black"),
                    linestyle=line_styles.get(p_type, "-")
                )
            add_stats_box(ax1, noon_row, p_cols, f"{name} POA", " W/m²")

            # Tilt
            for c in t_cols:
                st, t_type = c.split("/")
                p_type = t_type.replace("_TILT_ANGLE", "")
                ax2.plot(
                    x_axis, df_filtered[c],
                    label=f"{st} {p_type} Tilt",
                    color=station_colors.get(st, "black"),
                    linestyle=line_styles.get(p_type, "-")
                )
            add_stats_box(ax2, noon_row, t_cols, f"{name} Tilt", "°")

            ax1.set_title(f"POA Analysis\n{name}\n{day}\nSolar Noon: {noon_str}", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Irradiance [W/m²]")
            ax2.set_ylabel("Sensor Tilt [°]")
            ax2.set_xlabel("Time (HH:MM)")
            ax1.legend(loc="upper right", ncol=2, fontsize="small")

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Subfolder routing logic
            if GroupOutputsByStation and name != "Combined":
                save_dir = output_folder / name
            else:
                save_dir = output_folder
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / f"{file_path.stem}_{day}_{name}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

        # Site Average Plot
        fig_avg, (ax1_a, ax2_a) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

        df_filtered = df_filtered.copy()
        df_filtered["avg_poa"] = df_filtered[poa_cols].mean(axis=1)
        df_filtered["avg_tilt"] = df_filtered[tilt_cols].mean(axis=1)
        x_axis = df_filtered["t_stamp_dt"]

        for ax in (ax1_a, ax2_a):
            ax.axvline(x=noon_naive, color="red", linestyle="--", linewidth=1.5)
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

        ax1_a.plot(x_axis, df_filtered["avg_poa"], color="black", linewidth=2, label="Mean POA")
        ax2_a.plot(x_axis, df_filtered["avg_tilt"], color="black", linewidth=2, label="Mean Tilt")

        add_stats_box(ax1_a, noon_row, poa_cols, "Site POA", " W/m²")
        add_stats_box(ax2_a, noon_row, tilt_cols, "Site Tilt", "°")

        ax1_a.set_title(f"Site Average\n{day}\nSolar Noon: {noon_str}", fontsize=14, fontweight="bold")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_folder / f"{file_path.stem}_{day}_SiteAverage.png", dpi=150)
        plt.close(fig_avg)

print(f"\nProcessing complete. Check '{output_folder}'.")