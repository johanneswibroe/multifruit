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

# Configuration parameters
STILLNESS_THRESHOLD = 10   # seconds
STILLNESS_OBJECT_COUNT = 4  # number of objects that need to be still
LED_THRESHOLD = 3  # seconds before sound start to turn on LED
MIN_OBJECT_AREA = 4  # minimum area for tracked objects
PIXELS_TO_MM = 10 / 240  # conversion factor: 10 mm per 240 pixels
MIN_TIME_BEFORE_SOUND = 10  # minimum seconds before sound can start
SOUND_DURATION = 15  # duration of sound in seconds

# Global variables
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30)
arduino = None

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
        self.max_stillness_time = 0  # Track maximum stillness time
        self.velocity_history = []
        self.movement_started = False

    def update_stillness(self, frame_rate):
        # Update max stillness time if current stillness is higher
        self.max_stillness_time = max(self.max_stillness_time, self.stillness_timer)

def setup_arduino():
    global arduino
    try:
        arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        arduino.write(b'F')  # Turn off LED at start
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
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play(fade_ms=int(duration * 1000))
    start_time = time.time()
    while pygame.mixer.music.get_busy():
        elapsed = time.time() - start_time
        if elapsed >= duration:
            pygame.mixer.music.stop()
            break
        volume = elapsed / duration
        pygame.mixer.music.set_volume(volume)
        time.sleep(0.1)

def calculate_velocity(obj, current_time, frame_rate):
    if len(obj.trajectory) < 2:
        return 0
    
    n_frames = min(3, len(obj.trajectory) - 1)
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

def play_inaudible_sound():
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
    duration = 1
    frequency = 20000
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    buf = np.sin(2 * np.pi * np.arange(num_samples) * frequency / sample_rate).astype(np.float32)
    buf = (buf * 32767).astype(np.int16)
    
    sound = pygame.sndarray.make_sound(buf)
    sound.play()

def keep_speaker_active():
    while True:
        play_inaudible_sound()
        time.sleep(1740)  # Wait for 29 minutes

def track_objects(frame, prev_objects, current_time, chambers, frame_rate, light_on_time=None, background_reset_duration=3.1):
    global fgbg
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Instead of returning prev_objects, we'll use a higher learning rate during LED transition
    if light_on_time is not None and 0 < (current_time - light_on_time) <= background_reset_duration:
        fgmask = fgbg.apply(gray, learningRate=0.5)  # Higher learning rate to adapt to LED change
    else:
        fgmask = fgbg.apply(gray, learningRate=0.01)  # Normal learning rate
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Rest of the function remains the same
    current_objects = []
    for i, chamber in enumerate(chambers):
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
                    current_objects.append(new_obj)
        else:
            prev_obj = next((obj for obj in prev_objects if obj.id == i), None)
            if prev_obj:
                current_objects.append(prev_obj)
            else:
                chamber_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                new_obj = TrackedObject(chamber_center, (10, 10), 0)
                new_obj.id = i
                current_objects.append(new_obj)
    
    for curr_obj in current_objects:
        prev_obj = next((obj for obj in prev_objects if obj.id == curr_obj.id), None)
        if prev_obj:
            curr_obj.last_motion_time = prev_obj.last_motion_time
            curr_obj.moving = prev_obj.moving
            curr_obj.trajectory = prev_obj.trajectory[-49:] + [curr_obj.center]
            curr_obj.stillness_timer = prev_obj.stillness_timer
            curr_obj.max_stillness_time = prev_obj.max_stillness_time

            if np.linalg.norm(np.array(curr_obj.center) - np.array(prev_obj.center)) > 1:
                curr_obj.last_motion_time = current_time
                curr_obj.moving = True
                curr_obj.stillness_timer = 0
            else:
                curr_obj.stillness_timer += 1 / frame_rate
                curr_obj.update_stillness(frame_rate)
        else:
            curr_obj.trajectory = [curr_obj.center]
            curr_obj.stillness_timer = 0
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
            plt.figure(figsize=(12, 6))
            plt.plot(velocity_data["time"], velocity_data[f"object{i+1}_velocity"])
            plt.xlabel('Time (seconds)')
            plt.ylabel('Velocity (mm/s)')
            plt.title(f'Velocity of Object {i+1} Over Time')
            plt.ylim(0, 40)
            if sound_start_time is not None:
                plt.axvline(x=sound_start_time, color='r', linestyle=':', label='Sound Start')
            if led_on_time is not None:
                plt.axvline(x=(sound_start_time - 3), color='orange', linestyle=':', label='LED On')
                plt.axvspan((sound_start_time - 3), sound_start_time, color='orange', label='LED On Duration')
                plt.axvspan(sound_start_time, sound_start_time + SOUND_DURATION, alpha=0.2, color='gray', label='Sound Duration')
            plt.legend()
            pdf.savefig()
            plt.close()

def save_data(fly_name, velocity_data, movement_thresholds, sound_start_time, led_on_time, objects=None):
    # Save movement thresholds, stillness times, sound start time, and LED on time
    with open(f"{fly_name}_movement_thresholds.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sound Start Time (seconds)', sound_start_time if sound_start_time is not None else 'No sound played'])
        writer.writerow(['LED On Time (seconds)', led_on_time if led_on_time is not None else 'LED not turned on'])
        writer.writerow([])
        writer.writerow(['Object', 'Volume Percentage at Movement Start', 'Max Stillness Time (seconds)'])
        
        for obj_id in movement_thresholds.keys():
            threshold = movement_thresholds[obj_id]
            stillness_time = 0
            if objects:
                obj = next((o for o in objects if o.id == obj_id), None)
                if obj:
                    stillness_time = obj.max_stillness_time
            
            writer.writerow([
                f'Object {obj_id}',
                f'{threshold:.2f}%' if threshold is not None else 'Did not move',
                f'{stillness_time:.2f}'
            ])

    # Save velocities
    with open(f"{fly_name}_velocities.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Time"] + [f"Object{i+1}_Velocity" for i in range(len(movement_thresholds))]
        writer.writerow(headers)
        
        min_length = min(len(velocity_data["time"]), 
                        *[len(velocity_data[f"object{j+1}_velocity"]) for j in range(len(movement_thresholds))])
        
        for i in range(min_length):
            row = [velocity_data["time"][i]] + [velocity_data[f"object{j+1}_velocity"][i] for j in range(len(movement_thresholds))]
            writer.writerow(row)

def main():
    # Setup
    arduino = setup_arduino()
    fly_name = input("Enter the fly name: ")
    expected_objects = int(input("Enter the number of objects to track: "))

    # Camera setup
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.Width = 828
    camera.Height = 230
    camera.OffsetY.SetValue(800)
    camera.ExposureTime.SetValue(10000)
    frame_rate = 30
    video_duration = 18000 
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(frame_rate)

    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(f"{fly_name}.mp4", fourcc, frame_rate, (camera.Width.GetValue(), camera.Height.GetValue()))
    raw_writer = cv2.VideoWriter(f"{fly_name}_raw.mp4", fourcc, frame_rate, (camera.Width.GetValue(), camera.Height.GetValue()))

    # Start the speaker maintenance thread
    speaker_thread = threading.Thread(target=keep_speaker_active, daemon=True)
    speaker_thread.start()
    print("Speaker maintenance thread started")

# Initialize variables
    chambers = define_chamber_boundaries(camera.Width.GetValue(), camera.Height.GetValue(), expected_objects)
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

    # Start grabbing frames
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
            current_time = time.time()
            elapsed_time = current_time - start_time

            if sound_played and (elapsed_time - sound_start_time) >= 30:
                print(f"30 seconds elapsed after sound. Stopping recording.")
                break

            objects = track_objects(frame, objects, current_time, chambers, frame_rate, led_on_time)

            # Calculate and store velocities
            if len(objects) == expected_objects:
                velocity_data["time"].append(elapsed_time)
                for obj in objects:
                    velocity = calculate_velocity(obj, current_time, frame_rate)
                    velocity_data[f"object{obj.id+1}_velocity"].append(velocity)

                    # Check for initial movement after sound starts
                    if sound_played and initial_movement_times[obj.id] is None:
                        if velocity > 2.0:  # Threshold for considering movement
                            initial_movement_times[obj.id] = elapsed_time
                            time_since_sound_start = elapsed_time - sound_start_time
                            if time_since_sound_start <= SOUND_DURATION:
                                volume_percentage = (time_since_sound_start / SOUND_DURATION) * 100
                                movement_thresholds[obj.id] = volume_percentage
                                print(f"Object {obj.id+1} started moving at {volume_percentage:.2f}% volume "
                                      f"({time_since_sound_start:.2f} seconds after sound start)")

            # Check for stillness and track quiescence duration
            still_objects = sum(1 for obj in objects if obj.stillness_timer >= STILLNESS_THRESHOLD)

            # Add these debug prints
            print(f"Number of still objects: {still_objects}")
            print(f"Required still objects: {STILLNESS_OBJECT_COUNT}")
            print(f"Stillness timers:", [obj.stillness_timer for obj in objects])

            # Update quiescence tracking
            if still_objects >= STILLNESS_OBJECT_COUNT:
                if quiescence_start_time is None:
                    quiescence_start_time = current_time
                quiescence_duration = current_time - quiescence_start_time
            else:
                quiescence_start_time = None

            # Create display frame
            display_frame = frame.copy()

            # Draw objects and trajectories
            for obj in objects:
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

            # Check for sound end
            if sound_played and sound_end_time is None and (elapsed_time - sound_start_time) >= SOUND_DURATION:
                sound_end_time = elapsed_time

            # Check for stillness and trigger LED/sound
            if not sound_played and elapsed_time >= MIN_TIME_BEFORE_SOUND:
                if led_on_time is None and still_objects >= STILLNESS_OBJECT_COUNT:
                    led_on_time = current_time
                    led_on()
                    print(f"LED turned on at {elapsed_time:.2f} seconds")
                elif led_on_time is not None and current_time - led_on_time >= LED_THRESHOLD:
                    sound_played = True
                    sound_start_time = elapsed_time
                    threading.Thread(target=play_continuous_sound, 
                            args=("/home/joeh/Documents/pulse_cont2.wav", SOUND_DURATION)).start()
                    print(f"Sound started at {elapsed_time:.2f} seconds")

            # Display additional information
            cv2.putText(display_frame, f"Still Objects: {still_objects}/{STILLNESS_OBJECT_COUNT}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if quiescence_start_time is not None:
                cv2.putText(display_frame, f"Quiescence: {(current_time - quiescence_start_time):.1f}s", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Time: {elapsed_time:.2f}s", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Objects: {len(objects)}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw chamber boundaries
            for chamber in chambers:
                x1, y1, x2, y2 = chamber
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

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

    # Generate plots and save data
    generate_plots(fly_name, velocity_data, movement_thresholds, sound_start_time, led_on_time, expected_objects)
    save_data(fly_name, velocity_data, movement_thresholds, sound_start_time, led_on_time, objects)

if __name__ == "__main__":
    main()
