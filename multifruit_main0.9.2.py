import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pypylon import pylon
import cv2
import time
import pygame
import csv
import numpy as np
import serial
import serial.tools.list_ports
from serial import Serial
import threading
from matplotlib.backends.backend_pdf import PdfPages
import json
import os

# Load configuration from JSON file
def load_config():
    config_file = "fly_tracking_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    else:
        print("No config file found. Using default values.")
        return {}

# Load settings
config = load_config()

# Configuration parameters - now loaded from config file with fallback defaults
STILLNESS_THRESHOLD = config.get('stillness_threshold', 0)
STILLNESS_OBJECT_COUNT = config.get('stillness_object_count', 0)
LED_THRESHOLD = config.get('led_threshold', 0)
MIN_OBJECT_AREA = config.get('min_object_area', 1)
PIXELS_TO_MM = config.get('pixels_to_mm', 10 / 109)
MIN_TIME_BEFORE_SOUND = 15
SOUND_DURATION = config.get('sound_duration', 150)
SOUND_VOLUME_MODE = config.get('sound_volume_mode', "full")
MOTION_THRESHOLD = config.get('motion_threshold', 0.5)
FRAME_RATE = config.get('frame_rate', 30)

# ✅ NEW CONFIG VALUE (ADDED)
INTERVAL_ITI = config.get('interval_iti', 5)

# --- Stimulus schedule constants ---
STIM_VOLUMES = [0.8, 0.85, 0.9, 0.95]
BOUT_DURATION = 3  # seconds per stimulus bout

# Global variables
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=15, detectShadows=False)
arduino = None
active_chambers = []
last_click_time = {}
DOUBLE_CLICK_TIME = 0.5

class TrackedObject:
    def __init__(self, center, axes, angle):
        self.center = center
        self.axes = axes
        self.angle = angle
        self.last_motion_time = time.time()
        self.id = None
        self.moving = False
        self.trajectory = [center]
        self.stillness_timer = 0
        self.max_stillness_time = 0
        self.velocity_history = []
        self.movement_started = False
        self.real_detection = False
        self.sleep_latency = None
        self.sleep_onset_recorded = False
        self.prev_stillness_timer = 0  # ✅ track previous frame stillness timer

    def update_stillness(self, frame_rate):
        self.max_stillness_time = max(self.max_stillness_time, self.stillness_timer)

def setup_arduino():
    global arduino
    try:
        arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        arduino.write(b'F')
        print("Arduino connected and LED turned off")
        return arduino
    except Exception as e:
        print(f"Error setting up Arduino: {e}")
        return None

def led_on():
    if arduino:
        arduino.write(b'N')
        print("LED ON command sent")

def play_continuous_sound(sound_file, duration=15):
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.mixer.music.load(sound_file)
        
        # --- FULL MODE ---
        if SOUND_VOLUME_MODE == "full":
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()
            time.sleep(duration)
            pygame.mixer.music.stop()
            return
        
        # --- GRADUAL MODE ---
        elif SOUND_VOLUME_MODE == "gradual":
            FADE_IN_DURATION = 15
            pygame.mixer.music.set_volume(0.0)
            pygame.mixer.music.play(-1)
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    pygame.mixer.music.fadeout(1)
                    time.sleep(0.5)
                    break
                if elapsed < FADE_IN_DURATION:
                    volume = elapsed / FADE_IN_DURATION
                    status = f"FADING IN ({elapsed:.1f}s / {FADE_IN_DURATION}s)"
                else:
                    volume = 1.0
                    status = f"FULL VOLUME ({elapsed:.1f}s / {duration}s)"
                pygame.mixer.music.set_volume(volume)
                if int(elapsed) != int(elapsed - 0.1):
                    print(f"[GRADUAL] {status} | Volume: {volume:.2%}")
                time.sleep(0.1)
            return
        
        # --- INTERVAL MODE ---
        elif SOUND_VOLUME_MODE == "intervals":
            for i, v in enumerate(STIM_VOLUMES, 1):
                print(f"[INTERVAL {i}/{len(STIM_VOLUMES)}] Volume: {v:.2%} for {BOUT_DURATION}s")
                pygame.mixer.music.set_volume(v)
                pygame.mixer.music.play()
                time.sleep(BOUT_DURATION)
                pygame.mixer.music.stop()
                if i < len(STIM_VOLUMES):
                    print(f"[ITI] Pausing for {INTERVAL_ITI}s...")
                    time.sleep(INTERVAL_ITI)
            return
        
        else:
            print(f"ERROR: Unknown sound mode '{SOUND_VOLUME_MODE}', using full volume")
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()
            time.sleep(duration)
            pygame.mixer.music.stop()
            
    except Exception as e:
        print(f"Error playing sound: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass

# ===========================================================

def calculate_velocity(obj, current_time, frame_rate):
    if not obj.real_detection:
        return 0
    if len(obj.trajectory) < 2:
        return 0
    if len(obj.trajectory) < 3:
        return 0
    
    n_frames = min(5, len(obj.trajectory) - 1)
    dx = obj.trajectory[-1][0] - obj.trajectory[-n_frames][0]
    dy = obj.trajectory[-1][1] - obj.trajectory[-n_frames][1]
    dt = n_frames / frame_rate
    velocity_px = np.sqrt(dx**2 + dy**2) / dt
    velocity_mm = velocity_px * PIXELS_TO_MM
    
    obj.velocity_history.append(velocity_mm)
    if len(obj.velocity_history) > 5:
        obj.velocity_history.pop(0)
    
    return np.mean(obj.velocity_history)

def define_chamber_boundaries(frame_width, frame_height, num_chambers):
    chamber_width = frame_width // num_chambers
    chambers = []
    for i in range(num_chambers):
        left = i * chamber_width
        right = (i + 1) * chamber_width
        chambers.append((left, 0, right, frame_height))
    return chambers

def get_chamber_from_click(x, y, chambers):
    for i, (x1, y1, x2, y2) in enumerate(chambers):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

def mouse_callback(event, x, y, flags, param):
    global active_chambers, last_click_time
    chambers = param
    if event == cv2.EVENT_LBUTTONDOWN:
        chamber_idx = get_chamber_from_click(x, y, chambers)
        if chamber_idx is not None:
            current_time = time.time()
            if chamber_idx in last_click_time:
                time_since_last_click = current_time - last_click_time[chamber_idx]
                if time_since_last_click < DOUBLE_CLICK_TIME:
                    if chamber_idx in active_chambers:
                        active_chambers.remove(chamber_idx)
                        print(f"Chamber {chamber_idx + 1} DEACTIVATED")
                    else:
                        active_chambers.append(chamber_idx)
                        active_chambers.sort()
                        print(f"Chamber {chamber_idx + 1} ACTIVATED")
            last_click_time[chamber_idx] = current_time

def play_inaudible_sound():
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
    duration = 1
    frequency = 18800
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    buf = np.sin(2 * np.pi * np.arange(num_samples) * frequency / sample_rate).astype(np.float32)
    buf = (buf * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(buf)
    sound.play()

def keep_speaker_active():
    while True:
        play_inaudible_sound()
        time.sleep(900)

def track_objects(frame, prev_objects, current_time, chambers, frame_rate, light_on_time=None, background_reset_duration=3.1):
    global fgbg, active_chambers
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    fgmask = fgbg.apply(gray, learningRate=0.005)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    current_objects = []
    for i, chamber in enumerate(chambers):
        if i not in active_chambers:
            continue
            
        chamber_mask = np.zeros(fgmask.shape, dtype=np.uint8)
        x1, y1, x2, y2 = chamber
        cv2.rectangle(chamber_mask, (x1, y1), (x2, y2), 255, -1)
        chamber_fgmask = cv2.bitwise_and(fgmask, chamber_mask)
        
        contours, _ = cv2.findContours(chamber_fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_OBJECT_AREA]
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        if valid_contours:
            contour = valid_contours[0]
            try:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    if (not np.any(np.isnan(center)) and 
                        not np.any(np.isnan(axes)) and 
                        not np.isnan(angle) and 
                        all(ax > 0 for ax in axes)):
                        obj = TrackedObject(center, axes, angle)
                        obj.id = i
                        obj.real_detection = True
                        current_objects.append(obj)
                    else:
                        raise ValueError("Invalid ellipse parameters")
                else:
                    raise ValueError("Not enough points for ellipse")
            except (cv2.error, ValueError) as e:
                prev_obj = next((obj for obj in prev_objects if obj.id == i), None)
                if prev_obj:
                    current_objects.append(prev_obj)
                else:
                    chamber_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    new_obj = TrackedObject(chamber_center, (10, 10), 0)
                    new_obj.id = i
                    new_obj.real_detection = False
                    current_objects.append(new_obj)
        else:
            prev_obj = next((obj for obj in prev_objects if obj.id == i), None)
            if prev_obj:
                current_objects.append(prev_obj)
            else:
                chamber_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                new_obj = TrackedObject(chamber_center, (10, 10), 0)
                new_obj.id = i
                new_obj.real_detection = False
                current_objects.append(new_obj)
    
    for curr_obj in current_objects:
        prev_obj = next((obj for obj in prev_objects if obj.id == curr_obj.id), None)
        if prev_obj:
            curr_obj.last_motion_time = prev_obj.last_motion_time
            curr_obj.moving = prev_obj.moving
            curr_obj.stillness_timer = prev_obj.stillness_timer
            curr_obj.max_stillness_time = prev_obj.max_stillness_time
            curr_obj.sleep_latency = prev_obj.sleep_latency
            curr_obj.sleep_onset_recorded = prev_obj.sleep_onset_recorded

            if prev_obj.real_detection and curr_obj.real_detection:
                curr_obj.trajectory = prev_obj.trajectory[-49:] + [curr_obj.center]
            elif not prev_obj.real_detection and curr_obj.real_detection:
                curr_obj.trajectory = [curr_obj.center]
            else:
                curr_obj.trajectory = prev_obj.trajectory

            # ✅ Save previous stillness timer BEFORE potentially resetting it
            curr_obj.prev_stillness_timer = prev_obj.stillness_timer

            if np.linalg.norm(np.array(curr_obj.center) - np.array(prev_obj.center)) > MOTION_THRESHOLD:
                curr_obj.last_motion_time = current_time
                curr_obj.moving = True
                curr_obj.stillness_timer = 0  # reset
            else:
                curr_obj.stillness_timer += 1 / frame_rate
                curr_obj.update_stillness(frame_rate)
        else:
            curr_obj.trajectory = [curr_obj.center]
            curr_obj.stillness_timer = 0
            curr_obj.prev_stillness_timer = 0
            curr_obj.max_stillness_time = 0

    return current_objects

def generate_plots(fly_name, velocity_data, movement_thresholds, sound_start_time, led_on_time, expected_objects):
    plt.figure(figsize=(10, 6))
    for i in range(expected_objects):
        if i in movement_thresholds and movement_thresholds[i] is not None:
            plt.bar(f"Object {i+1}", movement_thresholds[i])
    plt.xlabel('Objects')
    plt.ylabel('Volume Percentage at Movement Start')
    plt.title('Volume Percentage When Objects Started Moving')
    plt.ylim(0, 100)
    for i in range(expected_objects):
        if i in movement_thresholds and movement_thresholds[i] is not None:
            plt.text(i, movement_thresholds[i] + 1, f'{movement_thresholds[i]:.2f}%', ha='center')

    with PdfPages(f"{fly_name}_velocities.pdf") as pdf:
        for i in range(expected_objects):
            if i in active_chambers:
                plt.figure(figsize=(12, 6))
                plt.plot(velocity_data["time"], velocity_data[f"object{i+1}_velocity"])
                plt.xlabel('Time (seconds)')
                plt.ylabel('Velocity (mm/s)')
                plt.title(f'Velocity of Object {i+1} Over Time')
                plt.ylim(0, 40)
                if sound_start_time is not None:
                    plt.axvline(x=sound_start_time, color='r', linestyle=':', label='Sound Start')
                if led_on_time is not None:
                    plt.axvspan(sound_start_time, sound_start_time + SOUND_DURATION, alpha=0.2, color='gray', label='Sound Duration')
                plt.legend()
                pdf.savefig()
                plt.close()

def compute_interval_averages_df(df, time_col, velocity_cols, pre_duration, stim_duration, iti_duration, num_stims):
    rows = []
    for col in velocity_cols:
        row = {'subject': col}
        t = 0.0
        for i in range(1, num_stims + 1):
            pre_start = t
            pre_end = t + pre_duration if i == 1 else t + iti_duration
            pre_vals = df[(df[time_col] >= pre_start) & (df[time_col] < pre_end)][col]
            row[f'pre_stimulus{i}'] = pre_vals.mean() if len(pre_vals) > 0 else np.nan
            t = pre_end
            stim_start = t
            stim_end = t + stim_duration
            stim_vals = df[(df[time_col] >= stim_start) & (df[time_col] < stim_end)][col]
            row[f'stimulus{i}'] = stim_vals.mean() if len(stim_vals) > 0 else np.nan
            t = stim_end
        rows.append(row)

    out_df = pd.DataFrame(rows)
    ordered_cols = ['subject']
    for i in range(1, num_stims + 1):
        ordered_cols.append(f'pre_stimulus{i}')
        ordered_cols.append(f'stimulus{i}')
    ordered_cols = [c for c in ordered_cols if c in out_df.columns]
    out_df = out_df[ordered_cols]
    return out_df

# find volume percentage point where locomotor activity exceeds 2 standard deviations from baseline mean. 
def find_2sd_arousal_threshold(df, time_col, subj_col, sound_start_time):
    pre_start = max(0, sound_start_time - 15)
    pre_data = df[(df[time_col] >= pre_start) & (df[time_col] < sound_start_time)][subj_col].dropna()
    if len(pre_data) < 2:
        return {'baseline_mean': np.nan, 'baseline_sd': np.nan, 'threshold_2sd': np.nan,
                'time_of_2sd_crossing': np.nan, 'time_since_sound_2sd': np.nan, 'volume_pct_at_2sd': np.nan,
                'time_of_2sd_1sec_bin': np.nan, 'time_since_sound_2sd_1sec_bin': np.nan,
                'volume_pct_at_2sd_1sec_bin': np.nan}

    

    baseline_mean = pre_data.mean()
    baseline_sd = pre_data.std(ddof=1)

# If baseline is 0, any movement exceeds 2SD — use smallest detectable velocity
    if baseline_sd == 0:
        threshold = baseline_mean + 1e-9
    else:
        threshold = baseline_mean + 2 * baseline_sd
    

    # Stimulus window: only the 15 seconds used as the comparison period - Change this if you want to use a longer period
    stim_window_duration = 15.0
    stim_end_time = sound_start_time + stim_window_duration

    # Single-frame 2SD crossing (original method)
    # Only look within the 15s stimulus window
    post_data = df[
        (df[time_col] >= sound_start_time) & (df[time_col] < stim_end_time)
    ][[time_col, subj_col]].dropna()
    single_frame_result = {
        'time_of_2sd_crossing': np.nan,
        'time_since_sound_2sd': np.nan,
        'volume_pct_at_2sd': np.nan,
    }
    for _, row in post_data.iterrows():
        t = row[time_col]
        vel = row[subj_col]
        if vel >= threshold:
            t_since = t - sound_start_time
            # percentage within the 15s stimulus window
            volume_pct = (t_since / stim_window_duration) * 100.0
            single_frame_result = {
                'time_of_2sd_crossing': t,
                'time_since_sound_2sd': t_since,
                'volume_pct_at_2sd': volume_pct,
            }
            break

    # 1-second bin 2SD crossing (new method) ---
    # Build non-overlapping 1s bins within the 15s stimulus window only.
    # Volume % is based on the bin start time as a fraction of the 15s window.
    bin_result = {
        'time_of_2sd_1sec_bin': np.nan,
        'time_since_sound_2sd_1sec_bin': np.nan,
        'volume_pct_at_2sd_1sec_bin': np.nan,
    }
    if len(post_data) > 0:
        bin_start = sound_start_time
        while bin_start < stim_end_time:
            bin_end = min(bin_start + 1.0, stim_end_time)
            bin_vals = post_data[
                (post_data[time_col] >= bin_start) & (post_data[time_col] < bin_end)
            ][subj_col]
            if len(bin_vals) > 0 and bin_vals.mean() >= threshold:
                t_since = bin_start - sound_start_time
                t_since = max(0.0, t_since)
                # percentage within the 15s stimulus window (0–100%)
                volume_pct = (t_since / stim_window_duration) * 100.0
                bin_result = {
                    'time_of_2sd_1sec_bin': bin_start,
                    'time_since_sound_2sd_1sec_bin': t_since,
                    'volume_pct_at_2sd_1sec_bin': volume_pct,
                }
                break
            bin_start = bin_end

    return {
        'baseline_mean': baseline_mean,
        'baseline_sd': baseline_sd,
        'threshold_2sd': threshold,
        **single_frame_result,
        **bin_result,
    }

def calculate_velocity_averages(fly_name, sound_start_time):
    try:
        df = pd.read_csv(f"{fly_name}_velocities.csv")
        time_col = df.columns[0]
        velocity_cols = df.columns[1:]
        
        if sound_start_time is None:
            print("No sound was played. Using 15 seconds as reference point for averages.")
            sound_start_time = 18.0
        
        results = []
        for subj_col in velocity_cols:
            pre_start = max(0, sound_start_time - 15)
            pre_end = sound_start_time
            pre_data = df[(df[time_col] >= pre_start) & (df[time_col] <= pre_end)][subj_col]
            pre_avg = pre_data.mean() if len(pre_data) > 0 else np.nan
            
            post_start = sound_start_time
            post_end = sound_start_time + 15
            post_data = df[(df[time_col] > post_start) & (df[time_col] <= post_end)][subj_col]
            post_avg = post_data.mean() if len(post_data) > 0 else np.nan

            # Compute both single-frame and 1-second bin 2SD arousal thresholds
            arousal = find_2sd_arousal_threshold(df, time_col, subj_col, sound_start_time)

            exceeded = not np.isnan(arousal['time_of_2sd_crossing'])
            results.append({
                'subject': subj_col,
                'pre_average': pre_avg,
                'post_average': post_avg,
                'difference': post_avg - pre_avg if not (np.isnan(pre_avg) or np.isnan(post_avg)) else np.nan,
                # --- empty placeholder columns for manual fill-in ---
                'sleep_cond': '',
                'genotype': '',
                'file': '',
                'day': '',
                # --- 2SD stats ---
                'baseline_mean': arousal['baseline_mean'],
                'baseline_sd': arousal['baseline_sd'],
                'threshold_2sd': arousal['threshold_2sd'],
                'exceeded_threshold': exceeded,
                # --- single-frame 2SD crossing ---
                'onset_time_threshold': arousal['time_of_2sd_crossing'],
                'time_from_18s_threshold': arousal['time_since_sound_2sd'],
                'volume_percent_threshold': arousal['volume_pct_at_2sd'],
                # --- 1-second bin 2SD crossing ---
                'time_from_18s_sig_1s': arousal['time_since_sound_2sd_1sec_bin'],
                'volume_percent_sig_1s': arousal['volume_pct_at_2sd_1sec_bin'],
            })
        
        output_df = pd.DataFrame(results)
        output_filename = f"{fly_name}_velocity_averages.csv"
        output_df.to_csv(output_filename, index=False)
        print(f"\nVelocity averages saved to {output_filename}")
        print("\nVelocity Averages Summary:")
        print(output_df.to_string(index=False))
        
        try:
            if SOUND_VOLUME_MODE == "intervals":
                num_stims = len(STIM_VOLUMES)
                pre_duration = MIN_TIME_BEFORE_SOUND
                stim_duration = BOUT_DURATION
                iti_duration = INTERVAL_ITI
                interval_df = compute_interval_averages_df(
                    df, time_col, velocity_cols,
                    pre_duration, stim_duration, iti_duration, num_stims
                )
                interval_output_filename = f"{fly_name}_velocity_interval_averages.csv"
                interval_df.to_csv(interval_output_filename, index=False)
                print(f"Interval averages saved to {interval_output_filename}")
        except Exception as e:
            print(f"Error computing interval averages: {e}")

        return output_df
        
    except Exception as e:
        print(f"Error calculating velocity averages: {e}")
        return None

def save_data(fly_name, velocity_data, movement_thresholds, sound_start_time, led_on_time, objects=None):
    with open(f"{fly_name}_movement_thresholds.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sound Start Time (seconds)', sound_start_time if sound_start_time is not None else 'No sound played'])
        writer.writerow(['LED On Time (seconds)', led_on_time if led_on_time is not None else 'LED not turned on'])
        writer.writerow([])
        writer.writerow(['Object', 
                'Sleep Latency (s)', 
                'Volume Percentage at Movement Start', 
                'Max Stillness Time (s)', 
                'Final Stillness Start Time (s)', 
                'Status'])
        
        for obj_id in movement_thresholds.keys():
            threshold = movement_thresholds[obj_id]
            stillness_time = 0
            final_stillness_time = 'Did not reach final stillness'
            status = 'Active' if obj_id in active_chambers else 'Inactive'
            sleep_latency = 'Never reached stillness threshold'

            if objects:
                obj = next((o for o in objects if o.id == obj_id), None)
                if obj:
                    stillness_time = obj.max_stillness_time
                    if obj.sleep_latency is not None:
                        sleep_latency = f'{obj.sleep_latency:.2f}'
                        final_stillness_time = f'{obj.sleep_latency:.2f}'
                    elif hasattr(obj, 'final_stillness_start'):
                        sleep_latency = f'{obj.final_stillness_start:.2f}'
                        final_stillness_time = f'{obj.final_stillness_start:.2f}'
            
            writer.writerow([
                f'Object {obj_id+1}',
                sleep_latency,
                f'{threshold:.2f}%' if threshold is not None else 'Did not move',
                f'{stillness_time:.2f}',
                final_stillness_time,
                status
            ])

    with open(f"{fly_name}_velocities.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Time"] + [f"Object{i+1}_Velocity" for i in range(len(movement_thresholds))]
        writer.writerow(headers)
        
        min_length = min(len(velocity_data["time"]), 
                        *[len(velocity_data[f"object{j+1}_velocity"]) for j in range(len(movement_thresholds))])
        
        for i in range(min_length):
            row = [velocity_data["time"][i]] + [velocity_data[f"object{j+1}_velocity"][i] for j in range(len(movement_thresholds))]
            writer.writerow(row)
    
    calculate_velocity_averages(fly_name, sound_start_time)


def main():
    global active_chambers
    
    arduino = setup_arduino()
    
    fly_name = config.get('fly_name', '')
    if not fly_name:
        fly_name = input("Enter the fly name: ")
    else:
        print(f"Using fly name from config: {fly_name}")
    
    expected_objects = config.get('expected_objects', 6)
    print(f"Tracking {expected_objects} objects")
    
    active_chambers = list(range(expected_objects))

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    
    camera.Width = config.get('camera_width', 1440)
    camera.Height = config.get('camera_height', 200)
    camera.Gain = config.get('camera_gain', 5)
    camera.OffsetY.SetValue(config.get('camera_offset_y', 500))
    camera.ExposureTime.SetValue(config.get('exposure_time', 25000))
    
    frame_rate = config.get('frame_rate', 30)
    video_duration = config.get('video_duration', 5000000)
    
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(frame_rate)
    
    print(f"\nCamera Settings:")
    print(f"  Width: {camera.Width.GetValue()}")
    print(f"  Height: {camera.Height.GetValue()}")
    print(f"  Gain: {camera.Gain.GetValue()}")
    print(f"  Exposure: {camera.ExposureTime.GetValue()} μs")
    print(f"  Frame Rate: {frame_rate} fps")
    print(f"\nMotion Settings:")
    print(f"  MOTION_THRESHOLD: {MOTION_THRESHOLD} px")
    print(f"  Wake detection: stillness timer reset (perfectly consistent with video)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f"{fly_name}.mp4", fourcc, frame_rate, (camera.Width.GetValue(), camera.Height.GetValue()))
    raw_writer = cv2.VideoWriter(f"{fly_name}_raw.mp4", fourcc, frame_rate, (camera.Width.GetValue(), camera.Height.GetValue()))

    speaker_thread = threading.Thread(target=keep_speaker_active, daemon=True)
    speaker_thread.start()
    print("Speaker maintenance thread started")

    chambers = define_chamber_boundaries(camera.Width.GetValue(), camera.Height.GetValue(), expected_objects)
    
    cv2.namedWindow("Fly Tracking")
    cv2.setMouseCallback("Fly Tracking", mouse_callback, chambers)
    print("\nDouble-click on a chamber to activate/deactivate it")
    
    objects = []
    sound_played = False
    led_on_time = None
    sound_start_time = None
    sound_end_time = None
    velocity_data = {f"object{i+1}_velocity": [] for i in range(expected_objects)}
    velocity_data["time"] = []
    movement_thresholds = {i: None for i in range(expected_objects)}
    initial_movement_times = {i: None for i in range(expected_objects)}
    quiescence_start_time = None
    total_stillness_time = 0

    sound_file_path = config.get('sound_file_path', "/home/joeh/Documents/pulse_cont3.wav")
    print(f"Sound file: {sound_file_path}")

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    start_time = time.time()
    while camera.IsGrabbing() and (time.time() - start_time) < video_duration:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            current_time = time.time()
            elapsed_time = current_time - start_time

            if sound_played and (elapsed_time - sound_start_time) >= 30:
                print(f"30 seconds elapsed after sound. Stopping recording.")
                break

            objects = track_objects(frame, objects, current_time, chambers, frame_rate, led_on_time)

            active_objects = [obj for obj in objects if obj.id in active_chambers]
            if len(active_objects) == len(active_chambers):
                velocity_data["time"].append(elapsed_time)
                for i in range(expected_objects):
                    if i in active_chambers:
                        obj = next((o for o in objects if o.id == i), None)
                        if obj:
                            velocity = calculate_velocity(obj, current_time, frame_rate)
                            velocity_data[f"object{i+1}_velocity"].append(velocity)

                            # Record sleep latency when fly first reaches stillness threshold
                            if not obj.sleep_onset_recorded:
                                if obj.stillness_timer >= STILLNESS_THRESHOLD:
                                    obj.sleep_latency = elapsed_time
                                    obj.sleep_onset_recorded = True
                                    print(f"✓ Object {obj.id+1} reached sleep threshold at {elapsed_time:.2f}s (after {obj.stillness_timer:.2f}s still)")

                            if not sound_played:
                                if obj.stillness_timer >= STILLNESS_THRESHOLD and not hasattr(obj, 'final_stillness_start'):
                                    obj.final_stillness_start = elapsed_time
                                    print(f"Object {obj.id+1} reached final stillness at {elapsed_time:.2f} seconds")
                                elif obj.moving and hasattr(obj, 'final_stillness_start'):
                                    delattr(obj, 'final_stillness_start')

                            # ✅ WAKE DETECTION: fires at the exact same moment the stillness
                            # timer resets — no velocity, no smoothing, no separate threshold.
                            # If the stillness timer resets on the video, it WILL be recorded here.
                            if sound_played and initial_movement_times[obj.id] is None:
                                if obj.stillness_timer == 0 and obj.prev_stillness_timer > 0:
                                    initial_movement_times[obj.id] = elapsed_time
                                    time_since_sound_start = elapsed_time - sound_start_time
                                    if time_since_sound_start <= SOUND_DURATION:
                                        volume_percentage = (time_since_sound_start / SOUND_DURATION) * 100
                                        movement_thresholds[obj.id] = volume_percentage
                                        print(f"Object {obj.id+1} woke up at {volume_percentage:.2f}% volume (stillness timer reset)")
                        else:
                            velocity_data[f"object{i+1}_velocity"].append(0)
                    else:
                        velocity_data[f"object{i+1}_velocity"].append(0)

            still_objects = sum(1 for obj in objects if obj.id in active_chambers and obj.stillness_timer >= STILLNESS_THRESHOLD)

            if still_objects >= STILLNESS_OBJECT_COUNT:
                if quiescence_start_time is None:
                    quiescence_start_time = current_time
                quiescence_duration = current_time - quiescence_start_time
            else:
                quiescence_start_time = None

            display_frame = frame.copy()

            for i, chamber in enumerate(chambers):
                x1, y1, x2, y2 = chamber
                if i not in active_chambers:
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), -1)
                    cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, "INACTIVE", 
                              (x1 + 5, y1 + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    obj = next((o for o in objects if o.id == i), None)
                    if obj:
                        stillness_text = f"Still: {obj.stillness_timer:.1f}s"
                        text_color = (0, 255, 0) if obj.stillness_timer >= STILLNESS_THRESHOLD else (255, 255, 255)
                        cv2.putText(display_frame, stillness_text,
                                  (x1 + 5, y2 - 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            for obj in objects:
                if obj.id not in active_chambers:
                    continue
                try:
                    color = (0, 255, 0) if obj.moving else (0, 0, 255)
                    if not (np.any(np.isnan(obj.center)) or np.any(np.isnan(obj.axes)) or np.isnan(obj.angle)):
                        cv2.ellipse(display_frame, 
                                  (int(obj.center[0]), int(obj.center[1])), 
                                  (int(obj.axes[0]/2), int(obj.axes[1]/2)), 
                                  obj.angle, 0, 360, color, 2)
                        cv2.putText(display_frame, f"ID: {obj.id+1}", 
                                  (int(obj.center[0]), int(obj.center[1])), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        if len(obj.trajectory) > 1:
                            cv2.polylines(display_frame, [np.array(obj.trajectory, dtype=np.int32)], 
                                        False, (255, 255, 0), 1)
                except (ValueError, cv2.error):
                    continue

            if still_objects >= STILLNESS_OBJECT_COUNT:
                frame_time = 1/frame_rate
                total_stillness_time += frame_time
                cv2.putText(display_frame, f"Total Still Time: {total_stillness_time:.1f}s", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if sound_played and sound_end_time is None and (elapsed_time - sound_start_time) >= SOUND_DURATION:
                sound_end_time = elapsed_time

            if not sound_played and elapsed_time >= MIN_TIME_BEFORE_SOUND:
                if led_on_time is None and still_objects >= STILLNESS_OBJECT_COUNT:
                    led_on_time = current_time
                    led_on()
                    print(f"LED turned on at {elapsed_time:.2f} seconds")
                elif led_on_time is not None and current_time - led_on_time >= LED_THRESHOLD:
                    sound_played = True
                    sound_start_time = elapsed_time
                    threading.Thread(target=play_continuous_sound, 
                            args=(sound_file_path, SOUND_DURATION)).start()
                    print(f"Sound started at {elapsed_time:.2f} seconds")

            cv2.putText(display_frame, f"Active: {len(active_chambers)}/{expected_objects}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Still Objects: {still_objects}/{STILLNESS_OBJECT_COUNT}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if quiescence_start_time is not None:
                cv2.putText(display_frame, f"Quiescence: {(current_time - quiescence_start_time):.1f}s", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Time: {elapsed_time:.2f}s", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Objects: {len(active_objects)}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            writer.write(display_frame)
            raw_writer.write(frame)
            cv2.imshow("Fly Tracking", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            grabResult.Release()

    # Cleanup
    camera.StopGrabbing()
    camera.Close()
    writer.release()
    raw_writer.release()
    cv2.destroyAllWindows()

    print("\nGenerating plots and saving data...")
    generate_plots(fly_name, velocity_data, movement_thresholds, sound_start_time, led_on_time, expected_objects)
    save_data(fly_name, velocity_data, movement_thresholds, sound_start_time, led_on_time, objects)
    print("Processing complete!")

if __name__ == "__main__":
    print("="*60)
    print("FLY TRACKING SYSTEM")
    print("="*60)
    print("\nLoaded Configuration:")
    print(f"  Stillness Threshold: {STILLNESS_THRESHOLD}s")
    print(f"  Stillness Object Count: {STILLNESS_OBJECT_COUNT}")
    print(f"  LED Threshold: {LED_THRESHOLD}s")
    print(f"  Min Time Before Sound: {MIN_TIME_BEFORE_SOUND}s")
    print(f"  Sound Duration: {SOUND_DURATION}s")
    print(f"  Motion Threshold: {MOTION_THRESHOLD} pixels")
    print(f"  Wake detection: stillness timer reset (perfectly consistent with video)")
    print("="*60)
    print("\nStarting tracking system...\n")
    main()