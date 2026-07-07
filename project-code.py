"""
╔══════════════════════════════════════════════════════════════════════╗
║         Running Biomechanics Analysis Pipeline                       ║
║         Excel Data Automation — SUT Numerical Methods Project         ║
║                                                                      ║
║  Steps:                                                              ║
║   1. Preprocessing   — gravity removal + Butterworth band-pass       ║
║   2. ZUPT Velocity   — integration with Zero Velocity Update         ║
║   3. Gait Segments   — Standing / Walking / Running classification   ║
║   4. Analytics       — per-interval velocity & heel-strike force     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings("ignore")
 
 
# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
 
G_CONST     = 9.81          # gravitational acceleration (m/s²)
DT          = 0.01          # sampling interval (s)
FS          = 1.0 / DT      # sampling frequency = 100 Hz
BODY_MASS   = 70.0          # runner body mass (kg)
 
# Band-pass filter bounds (Hz)
BP_LOW      = 0.15        
BP_HIGH     = 45.0
BP_ORDER    = 4
 
# ZUPT — stance-phase detection
ZUPT_THRESH          = 0.6   # m/s²
MIN_STANCE_SAMPLES   = 5
 
# Gait classification thresholds (m/s, on smoothed speed)
WALK_THRESH = 0.5             # below → Standing
RUN_THRESH  = 1.1              # above → Running, between → Walking
 
# Heel-strike detection (on filtered az, vertical axis)
HS_MIN_HEIGHT   = G_CONST * 1.3   
HS_MIN_DIST_S   = 0.25            
 

ETA_ATTENUATE   = 0.25      
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING & COLUMN MAPPING
# ══════════════════════════════════════════════════════════════════════
 
def load_excel_data(filepath: str) -> pd.DataFrame:
    """
    Loads Excel data, skips metadata headers, validates and maps custom column names
    to generic physics notation (ax, ay, az) for the pipeline.
    """
    print(f"  [Loading] Reading Excel file: {filepath}")
    df = pd.read_excel(filepath, skiprows=10)
    
    required = {"Accelerometer X", "Accelerometer Y", "Accelerometer Z"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel sheet is missing required columns: {missing}")
    
    df = df.rename(columns={
        "Accelerometer X": "ax",
        "Accelerometer Y": "ay",
        "Accelerometer Z": "az"
    })
    

    df["ax"] = df["ax"] * G_CONST
    df["ay"] = df["ay"] * G_CONST
    df["az"] = df["az"] * G_CONST
    # ───────────────────────────
    
    if "time" not in df.columns:
        df.insert(0, "time", np.arange(len(df)) * DT)
        
    return df[["time", "ax", "ay", "az"]].copy()
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
 
def remove_gravity(df: pd.DataFrame) -> pd.DataFrame:
    """Estimates and subtracts static gravity dynamic bias from az."""
    df = df.copy()
    win    = int(2.0 / DT)
    a_mag  = np.sqrt(df.ax**2 + df.ay**2 + df.az**2).values
    stds   = np.array([a_mag[i:i+win].std() for i in range(0, len(a_mag)-win, win)])
    q_idx  = np.argmin(stds) * win
    g_est  = df.az.values[q_idx : q_idx+win].mean()
 
    print(f"  [Gravity] estimated g = {g_est:.4f} m/s²  (ideal: {G_CONST:.4f})")
    df["az"] = df["az"] - g_est   
    return df
 
 
def bandpass_filter(df: pd.DataFrame, low: float = BP_LOW, high: float = BP_HIGH, order: int = BP_ORDER) -> pd.DataFrame:
    """Applies a zero-phase 4th-order Butterworth bandpass filter."""
    df = df.copy()
    nyq    = 0.5 * FS
    b, a   = butter(order, [low/nyq, high/nyq], btype="band")
 
    for col in ("ax", "ay", "az"):
        df[col] = filtfilt(b, a, df[col].values)
 
    return df
 
 
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: gravity removal → band-pass filter."""
    print("\n── STEP 1: Preprocessing ─────────────────────────────────────")
    df = remove_gravity(df)
    df = bandpass_filter(df)
    df["a_mag"] = np.sqrt(df.ax**2 + df.ay**2 + df.az**2)
    print(f"  [Filter ] band-pass {BP_LOW}–{BP_HIGH} Hz, order {BP_ORDER}")
    print(f"  [Signal ] samples={len(df):,}  duration={len(df)*DT/60:.1f} min")
    return df
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — ZUPT VELOCITY INTEGRATION (3D CORRECTED)
# ══════════════════════════════════════════════════════════════════════
 
def detect_stance_phase(a_mag: np.ndarray, thresh: float = ZUPT_THRESH, min_run: int = MIN_STANCE_SAMPLES) -> np.ndarray:
    """Detects stance intervals when the foot is on the ground."""
    candidate = a_mag < thresh
    is_stance = np.zeros(len(a_mag), dtype=bool)
    count     = 0
    for i, c in enumerate(candidate):
        count = count + 1 if c else 0
        if count >= min_run:
            is_stance[i - min_run + 1 : i + 1] = True
    return is_stance
 
 
def integrate_velocity_zupt(df: pd.DataFrame, mass: float = BODY_MASS) -> pd.DataFrame:
    """Performs numerical integration corrected by Zero Velocity Update in 3D axes separately."""
    df = df.copy()
    a_mag     = df["a_mag"].values
    is_stance = detect_stance_phase(a_mag)
 
    ax = df["ax"].values
    ay = df["ay"].values
    az = df["az"].values
 
    vx = np.zeros(len(df))
    vy = np.zeros(len(df))
    vz = np.zeros(len(df))
 
    for i in range(1, len(df)):
        vx[i] = vx[i-1] + ax[i] * DT
        vy[i] = vy[i-1] + ay[i] * DT
        vz[i] = vz[i-1] + az[i] * DT
        
        if is_stance[i]:
            vx[i] = 0.0
            vy[i] = 0.0
            vz[i] = 0.0
 
    v = np.sqrt(vx**2 + vy**2 + vz**2)
 
    df["velocity"]    = v
    df["is_stance"]   = is_stance
    
    df["speed_smooth"] = (df["velocity"]
                          .rolling(window=int(2.0 / DT), center=True, min_periods=1)
                          .mean())
 
    n_stance = is_stance.sum()
    print(f"\n── STEP 2: ZUPT Velocity Integration (3D Corrected) ─────────")
    print(f"  [ZUPT  ] stance samples = {n_stance:,} ({100*n_stance/len(a_mag):.1f} %)")
    print(f"  [Speed ] max = {v.max():.2f} m/s  |  mean = {v.mean():.2f} m/s")
    return df
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — GAIT SEGMENTATION
# ══════════════════════════════════════════════════════════════════════
 
def classify_gait(df: pd.DataFrame) -> pd.DataFrame:
    """Labels segments as Standing, Walking, or Running based on speed thresholds."""
    df  = df.copy()
    spd = df["speed_smooth"].values
 
    labels = np.where(spd < WALK_THRESH, "Standing",
             np.where(spd < RUN_THRESH,  "Walking",
                                         "Running"))
    df["gait"] = labels
 
    min_frag = int(2.0 / DT)
    label_arr = df["gait"].values.copy()
    i = 0
    while i < len(label_arr):
        j = i
        while j < len(label_arr) and label_arr[j] == label_arr[i]:
            j += 1
        seg_len = j - i
        if seg_len < min_frag and i > 0:
            label_arr[i:j] = label_arr[i-1]
        i = j
    df["gait"] = label_arr
 
    counts = pd.Series(label_arr).value_counts()
    print(f"\n── STEP 3: Gait Segmentation ─────────────────────────────────")
    for phase in ("Standing", "Walking", "Running"):
        n = counts.get(phase, 0)
        print(f"  [{phase:<10}] {n:>7,} samples  ({n*DT/60:5.1f} min)")
    return df
 
 
def extract_gait_intervals(df: pd.DataFrame) -> list[dict]:
    intervals = []
    gait = df["gait"].values
    i    = 0
    while i < len(gait):
        j = i
        while j < len(gait) and gait[j] == gait[i]:
            j += 1
        intervals.append({
            "phase"   : gait[i],
            "start"   : i,
            "end"     : j - 1,
            "t_start" : df["time"].iat[i],
            "t_end"   : df["time"].iat[j-1],
            "duration": (j - i) * DT,
        })
        i = j
    return intervals
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — RUNNING ANALYTICS (KINETIC CORRECTED)
# ══════════════════════════════════════════════════════════════════════
 
def detect_heel_strikes(az_seg: np.ndarray, min_height: float = HS_MIN_HEIGHT, min_dist_s: float = HS_MIN_DIST_S) -> np.ndarray:
    az_abs = np.abs(az_seg)
    adaptive_h = max(G_CONST * 0.6, np.percentile(az_abs, 60))
    peaks, _ = find_peaks(az_abs, height=adaptive_h, distance=int(min_dist_s / DT), prominence=1.5)
    return peaks

 
def analyze_running_intervals(df: pd.DataFrame, intervals: list[dict], mass: float = BODY_MASS) -> pd.DataFrame:
    """Calculates numerical kinematics and kinetic metrics with joint damping correction."""
    records = []
    run_ints = [iv for iv in intervals if iv["phase"] == "Running"]
    print(f"\n── STEP 4: Running Analytics (Damped F_GRF) ──────────────────")
    print(f"  Found {len(run_ints)} Running interval(s)\n")
 
    header = (f"  {'#':>3}  {'t_start':>8}  {'t_end':>8}  {'dur(s)':>7}  "
              f"{'v_mean':>7}  {'v_max':>7}  {'v_sd':>6}  "
              f"{'HS':>5}  {'F_max':>8}  {'F_mean':>8}  {'F_sd':>8}")
    print(header)
    print("  " + "─"*len(header.strip()))
 
    for k, iv in enumerate(run_ints):
        s, e  = iv["start"], iv["end"] + 1
        v_seg = df["velocity"].values[s:e]
        a_seg = df["az"].values[s:e]       
 
        v_mean = float(v_seg.mean())
        v_max  = float(v_seg.max())
        v_sd   = float(v_seg.std())
 
        hs_idx = detect_heel_strikes(a_seg)
        if len(hs_idx) > 0:
            a_mag_seg = df["a_mag"].values[s:e] 
            

            forces    = mass * (ETA_ATTENUATE * a_mag_seg[hs_idx] + G_CONST)
            
            f_max     = float(forces.max())
            f_min     = float(forces.min())
            f_sd      = float(forces.std())
            f_mean    = float(forces.mean())
        else:
            f_max = f_min = f_sd = f_mean = float("nan")
 
        records.append({
            "interval"      : k + 1,
            "t_start_s"     : iv["t_start"],
            "t_end_s"       : iv["t_end"],
            "duration_s"    : iv["duration"],
            "v_mean_m_s"    : round(v_mean, 4),
            "v_max_m_s"     : round(v_max,  4),
            "v_sd_m_s"      : round(v_sd,   4),
            "heel_strikes"  : len(hs_idx),
            "force_max_N"   : round(f_max,  2),
            "force_min_N"   : round(f_min,  2),
            "force_sd_N"    : round(f_sd,   2),
            "force_mean_N"  : round(f_mean, 2),
        })
 
        print(f"  {k+1:>3}  "
              f"{iv['t_start']:>8.1f}  {iv['t_end']:>8.1f}  "
              f"{iv['duration']:>7.1f}  "
              f"{v_mean:>7.3f}  {v_max:>7.3f}  {v_sd:>6.3f}  "
              f"{len(hs_idx):>5}  "
              f"{f_max:>8.1f}  {f_mean:>8.1f}  {f_sd:>8.1f}")
 
    return pd.DataFrame(records)
 
 
# ══════════════════════════════════════════════════════════════════════
#  SECTION 6 — SUMMARY & FATIGUE STATISTICS
# ══════════════════════════════════════════════════════════════════════
 
def print_summary(run_df: pd.DataFrame) -> None:
    if run_df.empty:
        print("\n  No Running intervals discovered in your dataset.")
        return
 
    sep = "─" * 52
    print(f"\n{'═'*52}")
    print("  SUMMARY — All Running Intervals Combined")
    print(f"{'═'*52}")
    print(f"  Total run duration : {run_df.duration_s.sum()/60:>7.2f} min")
    print(f"  Intervals          : {len(run_df):>7}")
    print(f"  Total heel strikes : {run_df.heel_strikes.sum():>7,}")
    print(sep)
    print(f"  {'Metric':<28} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print(sep)
 
    metrics = [
        ("Speed mean (m/s)",    "v_mean_m_s"),
        ("Speed max  (m/s)",    "v_max_m_s"),
        ("Speed SD   (m/s)",    "v_sd_m_s"),
        ("Force max  (N)",      "force_max_N"),
        ("Force min  (N)",      "force_min_N"),
        ("Force SD   (N)",      "force_sd_N"),
        ("Force mean (N)",      "force_mean_N"),
    ]
    for label, col in metrics:
        vals = run_df[col].dropna()
        if vals.empty:
            continue
        print(f"  {label:<28} {vals.mean():>8.2f}  {vals.min():>8.2f}  {vals.max():>8.2f}")
 
    print(sep)
    print("  FATIGUE ASSESSMENT")
    print(sep)
 
    def quarter_compare(col: str, label: str, high_is_bad: bool) -> None:
        vals = run_df[col].dropna()
        if len(vals) < 4:
            print(f"  {label:<28} Not enough running intervals to trend fatigue.")
            return
        q     = max(1, len(vals) // 4)
        early = vals.iloc[:q].mean()
        late  = vals.iloc[-q:].mean()
        pct   = (late - early) / (early + 1e-9) * 100
        flag  = ("⚠  fatigue detected" if (pct > 10 and high_is_bad) else "✅ stable")
        print(f"  {label:<28} early={early:>7.2f}  late={late:>7.2f}  Δ={pct:>+6.1f}%  {flag}")
 
    quarter_compare("force_sd_N",   "Force SD   (N)",   high_is_bad=True)
    quarter_compare("force_mean_N", "Force mean (N)",   high_is_bad=True)
    quarter_compare("v_sd_m_s",     "Speed SD   (m/s)", high_is_bad=True)
    print(f"{'═'*52}\n")
 
 
# ══════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

 
def run_pipeline(filepath: str, mass: float = BODY_MASS) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "═"*52)
    print("  Running Biomechanics Analysis Pipeline")
    print("═"*52)
    print(f"  Target File : {filepath}")
    print(f"  Runner Mass : {mass} kg")
 
    raw = load_excel_data(filepath)
    df = preprocess(raw)
    df = integrate_velocity_zupt(df, mass=mass)
    df = classify_gait(df)
    intervals = extract_gait_intervals(df)
    run_df = analyze_running_intervals(df, intervals, mass=mass)
    print_summary(run_df)
    
    plot_results(df)
 
    return df, run_df



def plot_results(df):
    print("\n  [Plotting] Generating and saving biomechanics_plot.png ...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    time_mins = df["time"] / 60.0  # تبدیل ثانیه به دقیقه برای نمایش بهتر

    # نمودار اول: سرعت و فازهای حرکتی
    ax1.plot(time_mins, df["speed_smooth"], color='#1f77b4', linewidth=1.5, label="Smoothed Speed")
    ax1.axhline(y=1.1, color='red', linestyle='--', alpha=0.7, label="Running Threshold (1.1 m/s)")
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label="Walking Threshold (0.5 m/s)")
    ax1.set_ylabel("Speed (m/s)", fontsize=11)
    ax1.set_title("Gait Phase Segmentation based on Corrected ZUPT Velocity", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper right")

    # نمودار دوم: شتاب عمودی (ضربه پاشنه)
    ax2.plot(time_mins, df["az"], color='#9467bd', alpha=0.7, linewidth=1, label="Filtered Vertical Accel (az)")
    ax2.set_xlabel("Time (minutes)", fontsize=11)
    ax2.set_ylabel("Acceleration (m/s²)", fontsize=11)
    ax2.set_title("Vertical Impact Accelerations (Heel Strikes)", fontsize=12, fontweight='bold')
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(r"D:\SUT\term 8\محاسبات عددی\Mohasebat Project\biomechanics_plot.png", dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    target_excel = r'D:\SUT\term 8\محاسبات عددی\Mohasebat Project\TAS1F06180329 (2018-10-24)-IMU.xlsx'
    df, run_df = run_pipeline(filepath=target_excel, mass=BODY_MASS)