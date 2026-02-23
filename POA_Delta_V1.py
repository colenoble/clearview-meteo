
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
from datetime import datetime

# pvlib used for precise solar noon (same method as your POA V4)
try:
    import pvlib
except ImportError:
    pvlib = None

# -------------------------------------------------------------------
# 1) PATH CONFIGURATION
# -------------------------------------------------------------------
input_folder = Path("inputs/GHI&GHI_tilt")
script_name = Path(__file__).stem
output_folder = Path("outputs") / script_name
output_folder.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 2) SITE & ANALYSIS SETTINGS
# -------------------------------------------------------------------
PlotDetailed = True  # used only when TwoDayCompare=False (original mode)
start_time_limit = "11:00:00"
end_time_limit = "14:00:00"

# --- Two-Day Compare Option (side-by-side)
TwoDayCompare = True
CompareDates = ("2026-02-01", "2026-02-10")  # YYYY-MM-DD

# --- Coordinates / TZ
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

# Clean time-axis (avoid clutter)
MAJOR_TICK_MINUTES = 15
MINOR_TICK_MINUTES = 5

LegendFontSize = "small"
LegendNcol = 2
LegendAlpha = 0.9

# -------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# -------------------------------------------------------------------

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
    """Slice to a day and the configured time window."""
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
        f"Range: {v_min:.2f} – {v_max:.2f}{unit}\n"
        f"Delta (Δ): {delta:.2f}{unit}"
    )
    ax.text(
        0.01, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )


def format_time_axis(ax):
    """Clean time ticks AFTER plotting."""
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


def get_station_list_from_columns(cols):
    """Extract stations from columns that look like 'STATION/...'."""
    stations = set()
    for c in cols:
        if isinstance(c, str) and "/" in c:
            stations.add(c.split("/")[0])
    return sorted(stations)


def compute_poa_tilt_deltas(df: pd.DataFrame):
    """Create per-station delta columns for POA and TILT: POA_1 - POA_2."""
    df = df.copy()

    # Find candidate stations from any POA-related columns
    candidates = [c for c in df.columns if isinstance(c, str) and "/POA_" in c]
    stations = get_station_list_from_columns(candidates)

    delta_poa_cols = []
    delta_tilt_cols = []

    for st in stations:
        poa1 = f"{st}/POA_1"
        poa2 = f"{st}/POA_2"
        if poa1 in df.columns and poa2 in df.columns:
            col = f"{st}/POA_DELTA"
            df[col] = pd.to_numeric(df[poa1], errors="coerce") - pd.to_numeric(df[poa2], errors="coerce")
            delta_poa_cols.append(col)

        tilt1 = f"{st}/POA_1_TILT_ANGLE"
        tilt2 = f"{st}/POA_2_TILT_ANGLE"
        if tilt1 in df.columns and tilt2 in df.columns:
            col = f"{st}/TILT_DELTA"
            df[col] = pd.to_numeric(df[tilt1], errors="coerce") - pd.to_numeric(df[tilt2], errors="coerce")
            delta_tilt_cols.append(col)

    return df, delta_poa_cols, delta_tilt_cols


def find_best_source_for_day(loaded_items, target_day):
    """Pick the loaded file that has the most rows for target_day within the time window."""
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
                "delta_poa_cols": item["delta_poa_cols"],
                "delta_tilt_cols": item["delta_tilt_cols"],
            }
    return best if best_count > 0 else None


def plot_compare_2x2_delta(
    df1, df2,
    dpoa_cols_1, dtilt_cols_1,
    dpoa_cols_2, dtilt_cols_2,
    day1, day2,
    noon1_naive, noon2_naive,
    noon1_str, noon2_str,
    noon_row_1, noon_row_2,
    suptitle, save_path
):
    """2x2 compare: left=day1, right=day2; top=ΔPOA, bottom=ΔTilt."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey="row", sharex="col")
    ax_poa_1, ax_poa_2 = axes[0, 0], axes[0, 1]
    ax_tilt_1, ax_tilt_2 = axes[1, 0], axes[1, 1]

    # --- Day 1 (left)
    x1 = df1["t_stamp_dt"]
    for c in dpoa_cols_1:
        if c in df1.columns:
            st = c.split("/")[0]
            ax_poa_1.plot(x1, df1[c], label=f"{st} ΔPOA", color=station_colors.get(st, "black"), linewidth=1.6)
    ax_poa_1.axhline(0, color="black", alpha=0.25, linewidth=1, label="_nolegend_")
    ax_poa_1.axvline(x=noon1_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_poa_1.set_title(f"Δ POA (POA_1 − POA_2) • {day1} • Noon: {noon1_str}", fontsize=12)
    ax_poa_1.set_ylabel("Δ Irradiance [W/m²]")
    add_stats_box(ax_poa_1, noon_row_1, dpoa_cols_1, "ΔPOA", " W/m²")

    for c in dtilt_cols_1:
        if c in df1.columns:
            st = c.split("/")[0]
            ax_tilt_1.plot(x1, df1[c], label=f"{st} ΔTilt", color=station_colors.get(st, "black"), linewidth=1.6)
    ax_tilt_1.axhline(0, color="black", alpha=0.25, linewidth=1, label="_nolegend_")
    ax_tilt_1.axvline(x=noon1_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_tilt_1.set_title(f"Δ Tilt (POA_1 − POA_2) • {day1} • Noon: {noon1_str}", fontsize=12)
    ax_tilt_1.set_ylabel("Δ Sensor Tilt [°]")
    ax_tilt_1.set_xlabel("Time (HH:MM)")
    add_stats_box(ax_tilt_1, noon_row_1, dtilt_cols_1, "ΔTilt", "°")

    # --- Day 2 (right)
    x2 = df2["t_stamp_dt"]
    for c in dpoa_cols_2:
        if c in df2.columns:
            st = c.split("/")[0]
            ax_poa_2.plot(x2, df2[c], label=f"{st} ΔPOA", color=station_colors.get(st, "black"), linewidth=1.6)
    ax_poa_2.axhline(0, color="black", alpha=0.25, linewidth=1, label="_nolegend_")
    ax_poa_2.axvline(x=noon2_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_poa_2.set_title(f"Δ POA (POA_1 − POA_2) • {day2} • Noon: {noon2_str}", fontsize=12)
    add_stats_box(ax_poa_2, noon_row_2, dpoa_cols_2, "ΔPOA", " W/m²")

    for c in dtilt_cols_2:
        if c in df2.columns:
            st = c.split("/")[0]
            ax_tilt_2.plot(x2, df2[c], label=f"{st} ΔTilt", color=station_colors.get(st, "black"), linewidth=1.6)
    ax_tilt_2.axhline(0, color="black", alpha=0.25, linewidth=1, label="_nolegend_")
    ax_tilt_2.axvline(x=noon2_naive, color="red", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax_tilt_2.set_title(f"Δ Tilt (POA_1 − POA_2) • {day2} • Noon: {noon2_str}", fontsize=12)
    ax_tilt_2.set_xlabel("Time (HH:MM)")
    add_stats_box(ax_tilt_2, noon_row_2, dtilt_cols_2, "ΔTilt", "°")

    # Format axes
    for ax in (ax_poa_1, ax_tilt_1, ax_poa_2, ax_tilt_2):
        format_time_axis(ax)

    # Hide top-row x labels
    ax_poa_1.tick_params(axis="x", labelbottom=False)
    ax_poa_2.tick_params(axis="x", labelbottom=False)

    for ax in (ax_tilt_1, ax_tilt_2):
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    set_top_right_legend(ax_poa_1, title="Stations")
    set_top_right_legend(ax_poa_2, title="Stations")
    set_top_right_legend(ax_tilt_1, title="Stations")
    set_top_right_legend(ax_tilt_2, title="Stations")

    fig.suptitle(suptitle, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------
# 4) LOAD FILES
# -------------------------------------------------------------------
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

    # Build delta columns once per file
    df, delta_poa_cols, delta_tilt_cols = compute_poa_tilt_deltas(df)

    loaded.append({
        "file": file_path,
        "df": df,
        "delta_poa_cols": delta_poa_cols,
        "delta_tilt_cols": delta_tilt_cols,
    })

print("")

# -------------------------------------------------------------------
# 5) TWO-DAY COMPARE MODE (DELTA)
# -------------------------------------------------------------------
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

    # Solar noon
    noon1 = get_precise_solar_noon(day1)
    noon2 = get_precise_solar_noon(day2)
    noon1_naive = noon1.replace(tzinfo=None)
    noon2_naive = noon2.replace(tzinfo=None)
    noon1_str = noon1.strftime("%H:%M")
    noon2_str = noon2.strftime("%H:%M")

    # Noon rows (closest timestamp)
    noon_row_1 = df1.iloc[(df1["t_stamp_dt"] - noon1_naive).abs().argsort()[:1]]
    noon_row_2 = df2.iloc[(df2["t_stamp_dt"] - noon2_naive).abs().argsort()[:1]]

    # --- A) Combined delta compare plot
    combined_save = output_folder / f"DELTA_COMPARE_{day1}_VS_{day2}_Combined.png"
    combined_title = (
        "ΔPOA & ΔTilt Two-Day Compare (POA_1 − POA_2) with Solar Noon Metrics\n"
        f"Window: {start_time_limit}–{end_time_limit}\n"
        f"Left source: {src1['file'].name}    Right source: {src2['file'].name}"
    )

    plot_compare_2x2_delta(
        df1=df1, df2=df2,
        dpoa_cols_1=src1["delta_poa_cols"], dtilt_cols_1=src1["delta_tilt_cols"],
        dpoa_cols_2=src2["delta_poa_cols"], dtilt_cols_2=src2["delta_tilt_cols"],
        day1=day1, day2=day2,
        noon1_naive=noon1_naive, noon2_naive=noon2_naive,
        noon1_str=noon1_str, noon2_str=noon2_str,
        noon_row_1=noon_row_1, noon_row_2=noon_row_2,
        suptitle=combined_title,
        save_path=combined_save
    )

    print(f"Saved combined delta compare plot: {combined_save.name}")

    # --- B) Per-station delta compare plots
    stations = get_station_list_from_columns(src1["delta_poa_cols"] + src1["delta_tilt_cols"])
    for st in stations:
        dpoa1 = [c for c in src1["delta_poa_cols"] if c.startswith(st + "/")]
        dtilt1 = [c for c in src1["delta_tilt_cols"] if c.startswith(st + "/")]
        dpoa2 = [c for c in src2["delta_poa_cols"] if c.startswith(st + "/")]
        dtilt2 = [c for c in src2["delta_tilt_cols"] if c.startswith(st + "/")]

        # skip if missing entirely on either day
        if (not dpoa1 and not dtilt1) or (not dpoa2 and not dtilt2):
            print(f" - Skipping {st}: station not present on one of the days.")
            continue

        st_save = output_folder / f"DELTA_COMPARE_{day1}_VS_{day2}_{st}.png"
        st_title = (
            f"ΔPOA & ΔTilt Two-Day Compare ({st}) (POA_1 − POA_2) with Solar Noon Metrics\n"
            f"Window: {start_time_limit}–{end_time_limit}\n"
            f"Left source: {src1['file'].name}    Right source: {src2['file'].name}"
        )

        plot_compare_2x2_delta(
            df1=df1, df2=df2,
            dpoa_cols_1=dpoa1, dtilt_cols_1=dtilt1,
            dpoa_cols_2=dpoa2, dtilt_cols_2=dtilt2,
            day1=day1, day2=day2,
            noon1_naive=noon1_naive, noon2_naive=noon2_naive,
            noon1_str=noon1_str, noon2_str=noon2_str,
            noon_row_1=noon_row_1, noon_row_2=noon_row_2,
            suptitle=st_title,
            save_path=st_save
        )

        print(f" - Saved station delta compare plot: {st_save.name}")

    print("\nProcessing complete.")
    sys.exit()

# -------------------------------------------------------------------
# 6) ORIGINAL MODE (per-day deltas) — runs when TwoDayCompare=False
# -------------------------------------------------------------------
if pvlib is None:
    print("ERROR: pvlib is not installed. Install with: pip install pvlib")
    sys.exit()

for item in loaded:
    file_path = item["file"]
    df = item["df"]
    dpoa_cols = item["delta_poa_cols"]
    dtilt_cols = item["delta_tilt_cols"]

    if not dpoa_cols and not dtilt_cols:
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

        plot_configs = [("Combined", dpoa_cols, dtilt_cols)]

        if PlotDetailed:
            stations = get_station_list_from_columns(dpoa_cols + dtilt_cols)
            for st in stations:
                plot_configs.append((
                    st,
                    [c for c in dpoa_cols if c.startswith(st + "/")],
                    [c for c in dtilt_cols if c.startswith(st + "/")],
                ))

        for name, p_cols, t_cols in plot_configs:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
            x_axis = df_filtered["t_stamp_dt"]

            for ax in (ax1, ax2):
                ax.axvline(x=noon_naive, color="red", linestyle="--", linewidth=1.5,
                           label=f"Solar Noon ({noon_str})")
                ax.axhline(0, color="black", alpha=0.25, linewidth=1)
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.grid(True, which="both", linestyle="--", alpha=0.3)

            # ΔPOA
            for c in p_cols:
                st = c.split("/")[0]
                ax1.plot(x_axis, df_filtered[c],
                         label=f"{st} ΔPOA",
                         color=station_colors.get(st, "black"), linewidth=1.6)
            add_stats_box(ax1, noon_row, p_cols, f"{name} ΔPOA", " W/m²")

            # ΔTilt
            for c in t_cols:
                st = c.split("/")[0]
                ax2.plot(x_axis, df_filtered[c],
                         label=f"{st} ΔTilt",
                         color=station_colors.get(st, "black"), linewidth=1.6)
            add_stats_box(ax2, noon_row, t_cols, f"{name} ΔTilt", "°")

            ax1.set_title(
                f"POA Delta Analysis (POA_1 − POA_2)\n{name}\n{day}\nSolar Noon: {noon_str}",
                fontsize=14, fontweight="bold"
            )
            ax1.set_ylabel("Δ Irradiance [W/m²]")
            ax2.set_ylabel("Δ Sensor Tilt [°]")
            ax2.set_xlabel("Time (HH:MM)")

            ax1.legend(loc="upper right", ncol=2, fontsize="small")
            plt.xticks(rotation=45)
            plt.tight_layout()

            save_path = output_folder / f"{file_path.stem}_{day}_{name}_DELTA.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

print(f"\nProcessing complete. Check '{output_folder}'.")
