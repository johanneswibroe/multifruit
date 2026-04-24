# fly_tracking_gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import subprocess
import threading
from pathlib import Path

class FlyTrackingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fly Tracking Configuration")
        self.root.geometry("800x900")

        # Dark mode colors
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.entry_bg = "#2d2d2d"
        self.button_bg = "#0d7377"
        self.button_hover = "#14b8a6"
        self.frame_bg = "#252525"
        self.accent_color = "#00d9ff"

        self.config_file = Path("fly_tracking_config.json")
        self.default_settings = {
            "stillness_threshold": 0,
            "stillness_object_count": 0,
            "led_threshold": 0,
            "min_object_area": 1,
            "pixels_to_mm": 0.1,
            "min_time_before_sound": 15,
            "sound_duration": 150,
            "sound_volume_mode": "full",
            "motion_threshold": 0.9,
            "exposure_time": 25000,
            "camera_gain": 5,
            "camera_width": 1440,
            "camera_height": 200,
            "camera_offset_y": 500,
            "frame_rate": 30,
            "video_duration": 18000000,
            "fly_name": "",
            "expected_objects": 6,
            "sound_file_path": str(Path.home() / "pulse_cont2.wav"),
            "interval_iti": 5,   # default ITI in seconds

        }

        self.settings = self.load_settings()
        self.setup_ui()

    def setup_ui(self):
        self.root.configure(bg=self.bg_color)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background=self.bg_color)
        style.configure('Dark.TLabel', background=self.bg_color, foreground=self.fg_color, font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=self.bg_color, foreground=self.accent_color, font=('Segoe UI', 12, 'bold'))
        style.configure('Dark.TButton', background=self.button_bg, foreground=self.fg_color, borderwidth=0, font=('Segoe UI', 10, 'bold'))
        style.map('Dark.TButton', background=[('active', self.button_hover)])

        # Main canvas + scroll
        main_canvas = tk.Canvas(self.root, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas, style='Dark.TFrame')

        scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        title_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        title_frame.pack(fill='x', padx=20, pady=(20, 10))
        title = ttk.Label(title_frame, text="🔬 Fly Tracking Configuration", style='Header.TLabel', font=('Segoe UI', 16, 'bold'))
        title.pack()

        # Experiment Settings
        self.create_section(scrollable_frame, "Experiment Settings", [
            ("Fly Name:", "fly_name", "entry", "Enter experiment name"),
            ("Expected Objects:", "expected_objects", "spinbox", (1, 20)),
            ("Sound File Path:", "sound_file_path", "filepicker", None)
        ])

        # Detection Thresholds
        self.create_section(scrollable_frame, "Detection Thresholds", [
            ("Stillness Threshold (s):", "stillness_threshold", "spinbox", (0, 60)),
            ("Stillness Object Count:", "stillness_object_count", "spinbox", (0, 20)),
            ("LED Threshold (s):", "led_threshold", "spinbox", (0, 60)),
            ("Min Object Area (px):", "min_object_area", "spinbox", (1, 100)),
            ("Motion Threshold (px):", "motion_threshold", "float_spinbox", (0.1, 10.0))
        ])

        # Camera Settings
        self.create_section(scrollable_frame, "Camera Settings", [
            ("Exposure Time (μs):", "exposure_time", "spinbox", (1000, 100000)),
            ("Camera Gain:", "camera_gain", "spinbox", (0, 20)),
            ("Camera Width:", "camera_width", "spinbox", (320, 4096)),
            ("Camera Height:", "camera_height", "spinbox", (240, 4096)),
            ("Camera Offset Y:", "camera_offset_y", "spinbox", (0, 2000)),
            ("Frame Rate (fps):", "frame_rate", "spinbox", (1, 120))
        ])

        # Sound & Timing
        self.create_section(scrollable_frame, "Sound & Timing", [
            ("Min Time Before Sound (s):", "min_time_before_sound", "spinbox", (0, 300)),
            ("Sound Duration (s):", "sound_duration", "spinbox", (1, 600)),
            ("Sound Volume Mode:", "sound_volume_mode", "dropdown", ["full", "gradual", "intervals"]),
            ("Interval ITI (s):", "interval_iti", "spinbox", (1, 60)),

            


        ])

        # Calibration
        self.create_section(scrollable_frame, "Calibration", [
            ("Pixels to MM Ratio:", "pixels_to_mm", "float_spinbox", (0.01, 1.0))
        ])

        # Buttons
        button_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        button_frame.pack(fill='x', padx=20, pady=20)
        self.save_btn = self.create_custom_button(button_frame, "💾 Save Settings", self.save_settings_action, column=0)
        self.reset_btn = self.create_custom_button(button_frame, "🔄 Reset to Defaults", self.reset_to_defaults, column=1)
        self.start_btn = self.create_custom_button(button_frame, "▶️ Start Tracking", self.start_tracking, column=2, color="#059669")

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def create_section(self, parent, title, fields):
        section_frame = tk.Frame(parent, bg=self.frame_bg, relief='flat', bd=0)
        section_frame.pack(fill='x', padx=20, pady=10)
        header_frame = tk.Frame(section_frame, bg=self.accent_color, height=3)
        header_frame.pack(fill='x')
        title_label = ttk.Label(section_frame, text=title, style='Header.TLabel')
        title_label.pack(anchor='w', padx=15, pady=(10, 5))
        for field in fields:
            self.create_field(section_frame, *field)

    def create_field(self, parent, label_text, setting_key, field_type, options=None):
        field_frame = tk.Frame(parent, bg=self.frame_bg)
        field_frame.pack(fill='x', padx=15, pady=5)
        label = tk.Label(field_frame, text=label_text, bg=self.frame_bg, fg=self.fg_color, font=('Segoe UI', 10), width=25, anchor='w')
        label.pack(side='left')

        if field_type == "entry":
            entry = tk.Entry(field_frame, bg=self.entry_bg, fg=self.fg_color, insertbackground=self.fg_color, relief='flat', font=('Segoe UI', 10))
            entry.insert(0, str(self.settings.get(setting_key, "")))
            entry.pack(side='left', fill='x', expand=True, ipady=5)
            setattr(self, f"{setting_key}_var", entry)

        elif field_type == "spinbox":
            spinbox = tk.Spinbox(field_frame, from_=options[0], to=options[1], bg=self.entry_bg, fg=self.fg_color, buttonbackground=self.button_bg, insertbackground=self.fg_color, relief='flat', font=('Segoe UI', 10))
            spinbox.delete(0, 'end')
            spinbox.insert(0, str(self.settings.get(setting_key, options[0])))
            spinbox.pack(side='left', fill='x', expand=True, ipady=3)
            setattr(self, f"{setting_key}_var", spinbox)

        elif field_type == "float_spinbox":
            spinbox = tk.Spinbox(field_frame, from_=options[0], to=options[1], increment=0.1, bg=self.entry_bg, fg=self.fg_color, buttonbackground=self.button_bg, insertbackground=self.fg_color, relief='flat', font=('Segoe UI', 10))
            spinbox.delete(0, 'end')
            spinbox.insert(0, str(self.settings.get(setting_key, options[0])))
            spinbox.pack(side='left', fill='x', expand=True, ipady=3)
            setattr(self, f"{setting_key}_var", spinbox)

        elif field_type == "dropdown":
            var = tk.StringVar(value=self.settings.get(setting_key, options[0]))
            dropdown = ttk.Combobox(field_frame, textvariable=var, values=options, state='readonly')
            dropdown.pack(side='left', fill='x', expand=True, ipady=3)
            setattr(self, f"{setting_key}_var", var)

        elif field_type == "filepicker":
            # show entry + button
            entry = tk.Entry(field_frame, bg=self.entry_bg, fg=self.fg_color, insertbackground=self.fg_color, relief='flat', font=('Segoe UI', 10))
            entry.insert(0, str(self.settings.get(setting_key, "")))
            entry.pack(side='left', fill='x', expand=True, ipady=5)
            def pick_file():
                path = filedialog.askopenfilename(title="Select sound file", filetypes=[("WAV files","*.wav"),("All files","*.*")])
                if path:
                    entry.delete(0, 'end')
                    entry.insert(0, path)
            pick_btn = tk.Button(field_frame, text="Browse", command=pick_file, bg=self.button_bg, fg=self.fg_color, relief='flat', cursor='hand2')
            pick_btn.pack(side='left', padx=8)
            setattr(self, f"{setting_key}_var", entry)

    def create_custom_button(self, parent, text, command, column, color=None):
        btn = tk.Button(parent, text=text, command=command, bg=color or self.button_bg, fg=self.fg_color, activebackground=self.button_hover, activeforeground=self.fg_color, relief='flat', font=('Segoe UI', 10, 'bold'), cursor='hand2', padx=20, pady=10)
        btn.grid(row=0, column=column, padx=5, sticky='ew')
        parent.grid_columnconfigure(column, weight=1)
        def on_enter(e):
            btn['background'] = color or self.button_hover
        def on_leave(e):
            btn['background'] = color or self.button_bg
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def load_settings(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    return {**self.default_settings, **loaded}
            except Exception as e:
                messagebox.showwarning("Warning", f"Unable to load config, using defaults.\n{e}", parent=self.root)
                return self.default_settings.copy()
        return self.default_settings.copy()

    def save_settings_action(self):
        # Build settings dict by converting widgets to the type of the defaults
        new_settings = {}
        for key, default in self.default_settings.items():
            widget = getattr(self, f"{key}_var", None)
            if widget is None:
                new_settings[key] = default
                continue

            raw_value = None
            if isinstance(widget, tk.StringVar):
                raw_value = widget.get()
            elif isinstance(widget, tk.Entry):
                raw_value = widget.get()
            else:  # Spinbox (tk.Spinbox returns a widget object)
                try:
                    raw_value = widget.get()
                except Exception:
                    raw_value = None

            # Convert to appropriate type using the default as the guide
            try:
                if isinstance(default, bool):
                    new_settings[key] = bool(raw_value)
                elif isinstance(default, int):
                    # allow floats in spinbox but convert to int when default int
                    new_settings[key] = int(float(raw_value)) if raw_value not in (None, "") else default
                elif isinstance(default, float):
                    new_settings[key] = float(raw_value) if raw_value not in (None, "") else default
                else:
                    # string or path
                    new_settings[key] = raw_value if (raw_value is not None) else default
            except Exception:
                new_settings[key] = default

        # Write to file
        try:
            with open(self.config_file, 'w') as f:
                json.dump(new_settings, f, indent=4)
            self.settings = new_settings
            messagebox.showinfo("Success", "Settings saved successfully!", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}", parent=self.root)

    def reset_to_defaults(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?", parent=self.root):
            self.settings = self.default_settings.copy()
            for key, value in self.default_settings.items():
                widget = getattr(self, f"{key}_var", None)
                if widget:
                    if isinstance(widget, tk.StringVar):
                        widget.set(value)
                    elif isinstance(widget, tk.Entry):
                        widget.delete(0, 'end')
                        widget.insert(0, str(value))
                    else:  # Spinbox
                        widget.delete(0, 'end')
                        widget.insert(0, str(value))
            messagebox.showinfo("Reset Complete", "Settings reset to defaults!", parent=self.root)

    def _run_tracking_subprocess(self, tracking_script="tracking.py"):
        # run in a subprocess to avoid freezing GUI. Use system python.
        try:
            # save settings first to ensure tracking script reads latest file
            self.save_settings_action()
            # launch as a new process
            proc = subprocess.Popen(["python3", tracking_script])
            # optionally track proc if needed (not waiting here)
            # re-enable start button after a short wait -> we'll poll
            proc.wait()
            self.start_btn.config(state='normal')
            messagebox.showinfo("Tracking finished", f"Tracking process exited with code {proc.returncode}", parent=self.root)
        except FileNotFoundError:
            self.start_btn.config(state='normal')
            messagebox.showerror("Script not found", f"Could not find {tracking_script} in the current directory.", parent=self.root)
        except Exception as e:
            self.start_btn.config(state='normal')
            messagebox.showerror("Error launching tracking", str(e), parent=self.root)

    def start_tracking(self):
        # Launch the external tracking script in a separate thread to avoid freezing the GUI
        if messagebox.askyesno("Start Tracking", "This will save current settings and start the tracking script. Continue?", parent=self.root):
            self.save_settings_action()
            self.start_btn.config(state='disabled')
            threading.Thread(target=self._run_tracking_subprocess, kwargs={"tracking_script": "tracking.py"}, daemon=True).start()

def main():
    root = tk.Tk()
    app = FlyTrackingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
