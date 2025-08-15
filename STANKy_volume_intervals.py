import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pypylon import pylon
import cv2
import time
import pygame
import csv
import cProfile
import pstats
import subprocess
import io
import numpy as np
import threading

#time.sleep(120)
# Import OpenCV's OpenCL module
import cv2.ocl

def show_time_without_motion(time_without_motion):
    # Create a black background image
    img = np.zeros((100, 500, 3), np.uint8)
    # Add the text showing the time without motion
    cv2.putText(img, f"Time without motion: {time_without_motion}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Display the image in a new window
    cv2.imshow("Time Without Motion", img)

def prof_to_csv(prof: cProfile.Profile):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)

def play_sound(volume_levels, sound_file):
    pygame.init()
    pygame.mixer.music.load(sound_file)
    for volume in volume_levels:
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.01)  # Wait for the current sound to finish playdifferent_intervals_10ing

fly_name = "test"
rawname = fly_name + "_raw.mp4"
moviename = fly_name + ".mp4"
csv_name = fly_name + ".csv"
plotname = fly_name + ".pdf"

# Connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.Width = 400
camera.Height = 400
camera.OffsetX.SetValue(628)
camera.OffsetY.SetValue(628)

# Enable OpenCL for OpenCV
cv2.ocl.setUseOpenCL(True)

camera.ExposureTime.SetValue(30000)
# Set the desired frame rate
frame_rate = 30

time_list = []
velocity_list = []

# Set the duration of the video directly IN SECONDS
video_duration = 3600

camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(frame_rate)

# Initialize background subtractor and bounding box variables
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=500)
bbox = None

# Initialize variables for calculating velocity
prev_center = None
prev_time = time.time()
velocity = None

writer = cv2.VideoWriter(moviename, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (400, 400))
raw_writer = cv2.VideoWriter(rawname, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (400, 400))

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

start_time = time.time()
elapsed_time = 0

# Get some variables going
frame_count = 0
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
counter = 0
sound_time = None
last_motion = 0
sound_played = False

# Create a profiler instance
profiler = cProfile.Profile()

# Start profiling
profiler.enable()
time_since_motion = 0

while camera.IsGrabbing() and elapsed_time < video_duration:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    elapsed_time = time.time() - start_time

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        frame = img
        writer.write(img)
        raw_writer.write(img)

        # Apply background subtraction
        fgmask = fgbg.apply(img)
        kernel_size = (5, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        kernel = cv2.GaussianBlur(kernel, kernel_size, 0)
        fgmask = cv2.filter2D(fgmask, -1, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        cv2.dilate(fgmask, kernel, dst=fgmask, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 100
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None and len(max_contour) >= 5:
            ellipse = cv2.fitEllipse(max_contour)
            center, axes, angle = ellipse
            major_axis, minor_axis = axes
            angle_rad = np.radians(angle)
            rect_points = cv2.boxPoints(((center[0], center[1]), (major_axis, minor_axis), angle))
            rect_points = np.int0(rect_points)
            cv2.drawContours(img, [rect_points], 0, (0, 255, 0), 2)

            # Update velocity calculation using the rotated bounding box
            if prev_center is not None:
                current_time = time.time()
                dt = current_time - prev_time
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]

                # Rotate velocity vector back to the original image orientation
                rotated_dx = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
                rotated_dy = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)

                velocity = (rotated_dx / dt, rotated_dy / dt)
                prev_time = current_time
            prev_center = center

            if cv2.norm(velocity) > 5:
                time_since_motion = 0
                last_motion = time.time()

        else:
            time_since_motion += elapsed_time
            velocity = (0,0)

        time_without_motion = time.time() - last_motion

        if cv2.norm(velocity) > 5:
            time_without_motion = 0

        cv2.putText(frame, "Time: " + str(round(elapsed_time, 1)), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        raw_writer.write(img)

        cv2.putText(img, f"Time without motion: {time_without_motion}", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        if time_without_motion >= 1 and time_without_motion < 3600 and not sound_played and elapsed_time >= 10:
            sound_played = True
            sound_time = elapsed_time
            threading.Thread(target=play_sound, args=([0.02, 0.05, 0.1, 0.3, 1.0], "/home/joeh/Downloads/mating_call_normal.wav")).start()
                                                        #0.02, 0.05, 0.1, 0.3, 1.0
                                                        #
                                                        
                                                        #if +25% on soundcard 

        if bbox is not None:
            cv2.ellipse(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, "Velocity: {:.2f} pixels/sec".format(cv2.norm(velocity)), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if bbox is None:
            velocity == 0

        resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Motion Detection", resized_img)
        cv2.resizeWindow("Motion Detection", 10, 120)

        # Show the "Time without Motion" text in a separate windowq
        show_time_without_motion(time_without_motion)

        if cv2.waitKey(1) == ord('q'):
            break

        if elapsed_time > 5:  # Filter out data points where time is below 5000 milliseconds
             time_list.append(elapsed_time)
             velocity_list.append(cv2.norm(velocity))

print(elapsed_time)

grabResult.Release()

camera.Close()
cv2.destroyAllWindows()
writer.release()
raw_writer.release()

scaled_velocity_list = []  # Define a list to store scaled velocities

# Open CSV file to write the results
with open(csv_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    pixels_to_mm = 240 / 10  # Conversion factor from pixels to millimeters
    writer.writerow(['Time (milliseconds)', 'Locomotion (mm/sec)', 'First Sound Time', 'Second Sound Time', 'Third Sound Time', 'Fourth Sound Time', 'Fifth Sound Time'])
    for i in range(len(time_list)):
        scaled_velocity_mm = round(velocity_list[i] / pixels_to_mm, 2)
        scaled_velocity_list.append(scaled_velocity_mm)  # Append to the list
        first_sound_time_ms = round(sound_time * 1000, 1) if sound_time is not None else ''
        second_sound_time_ms = round((sound_time + 2.8) * 1000, 1) if sound_time is not None else ''
        third_sound_time_ms = round((sound_time + 2 * 2.8) * 1000, 1) if sound_time is not None else ''
        fourth_sound_time_ms = round((sound_time + 3 * 2.8) * 1000, 1) if sound_time is not None else ''
        fifth_sound_time_ms = round((sound_time + 4 * 2.8) * 1000, 1) if sound_time is not None else ''
        writer.writerow([round(time_list[i] * 1000, 1), scaled_velocity_mm, first_sound_time_ms, second_sound_time_ms, third_sound_time_ms, fourth_sound_time_ms, fifth_sound_time_ms])

# Filter data for Time above 5000 milliseconds
time_above_5000 = [t for t in time_list if t > 5]
scaled_velocity_above_5000 = [scaled_velocity_list[i] for i in range(len(time_list)) if time_list[i] > 5]

# Plot data considering the sound times
if sound_time is not None:
    first_sound_time = sound_time
    second_sound_time = sound_time + 2.8
    third_sound_time = sound_time + 2 * 2.8
    fourth_sound_time = sound_time + 3 * 2.8
    fifth_sound_time = sound_time + 4 * 2.8

    # Extract data 10 seconds before the first sound time
    start_time = max(0, first_sound_time - 10)
    filtered_time = [t for t in time_above_5000 if start_time <= t <= fifth_sound_time + 10]
    filtered_velocity = [v for i, v in enumerate(scaled_velocity_above_5000) if start_time <= time_above_5000[i] <= fifth_sound_time + 10]

    plt.plot(filtered_time, filtered_velocity, label='Velocity (mm/sec)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (mm/sec)')
    plt.title('Locomotion over Time')

    # Add red dotted lines for sound times
    plt.axvline(x=first_sound_time, color='red', linestyle='--', label='First Sound')
    plt.axvline(x=second_sound_time, color='red', linestyle='--', label='Second Sound')
    plt.axvline(x=third_sound_time, color='red', linestyle='--', label='Third Sound')
    plt.axvline(x=fourth_sound_time, color='red', linestyle='--', label='Fourth Sound')
    plt.axvline(x=fifth_sound_time, color='red', linestyle='--', label='Fifth Sound')

    plt.legend()
    plt.savefig(plotname)
    plt.show()

# Stop profiling and print the results
profiler.disable()

# Save profiling results to a CSV file
with open(f'{fly_name}_profiling.csv', 'w') as f:
    f.write(prof_to_csv(profiler))

print("Profiling data saved to profiling.csv")