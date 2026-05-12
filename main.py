# ============================================================
#   Face Recognition Attendance System (Standalone App)
#   Redesigned & Modernized AI-Powered Cyber Dashboard
#   Uses: OpenCV LBPH recognizer + Haarcascade detector
# ============================================================

import cv2
import numpy as np
import os
import pickle
import pandas as pd
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import threading
import shutil
import pyttsx3
import time

# ─────────────────────────────────────────────────────────────
# VOICE ALERT SYSTEM
# ─────────────────────────────────────────────────────────────
last_speech_time = 0

def speak(message):
    """Voice alert with a 3-second cooldown to avoid spamming."""
    global last_speech_time
    current_time = time.time()
    if current_time - last_speech_time >= 3:
        last_speech_time = current_time
        def tts_thread():
            try:
                # Initialize inside thread to avoid Windows COM errors
                eng = pyttsx3.init()
                eng.say(message)
                eng.runAndWait()
            except Exception:
                pass
        threading.Thread(target=tts_thread, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# PATHS — all folders are auto-created on first run
# ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
ENCODINGS_DIR   = os.path.join(BASE_DIR, "encodings")
ATTENDANCE_DIR  = os.path.join(BASE_DIR, "attendance")
MODEL_FILE      = os.path.join(ENCODINGS_DIR, "lbph_model.xml")
LABELS_FILE     = os.path.join(ENCODINGS_DIR, "labels.pkl")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.xlsx")

# CSV Log paths
REGISTRATION_LOG_FILE = os.path.join(ATTENDANCE_DIR, "registration_log.csv")
ATTENDANCE_LOG_CSV    = os.path.join(ATTENDANCE_DIR, "attendance_log.csv")

for _d in [DATASET_DIR, ENCODINGS_DIR, ATTENDANCE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HAARCASCADE — built-in with OpenCV, no extra download needed
# ─────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ─────────────────────────────────────────────────────────────
# LBPH FACE RECOGNIZER HELPERS
# ─────────────────────────────────────────────────────────────

def _create_recognizer():
    """Create a fresh LBPH recognizer instance."""
    return cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0
    )

def load_model():
    """
    Load the trained LBPH model and label map from disk.
    Returns (recognizer, id_to_name dict) or (None, {}) if not trained yet.
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABELS_FILE):
        return None, {}
    recognizer = _create_recognizer()
    recognizer.read(MODEL_FILE)
    with open(LABELS_FILE, "rb") as f:
        id_to_name = pickle.load(f)
    return recognizer, id_to_name

def train_model():
    """
    Scan dataset/ folder, read all face images, train LBPH model,
    and save model + label map.  Returns (recognizer, id_to_name).
    """
    faces, labels, id_to_name = [], [], {}
    current_id = 0

    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        has_images = False
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            try:
                pil_img  = Image.open(img_path).convert("L")   # grayscale
                img_arr  = np.array(pil_img, dtype=np.uint8)
                faces.append(img_arr)
                labels.append(current_id)
                has_images = True
            except:
                continue
        if has_images:
            id_to_name[current_id] = person_name
            current_id += 1

    if not faces:
        if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
        if os.path.exists(LABELS_FILE): os.remove(LABELS_FILE)
        return None, {}

    recognizer = _create_recognizer()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(id_to_name, f)

    return recognizer, id_to_name

# ─────────────────────────────────────────────────────────────
# LOGGING & ATTENDANCE HELPERS
# ─────────────────────────────────────────────────────────────

def log_registration(user_id, name, image_count):
    """Log registration details to a CSV file."""
    file_exists = os.path.isfile(REGISTRATION_LOG_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")
    
    try:
        with open(REGISTRATION_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["User ID", "Name", "Date", "Time", "Images Captured"])
            writer.writerow([user_id, name, today, now_time, image_count])
    except Exception as e:
        print(f"Error logging registration: {e}")

def mark_attendance(name: str):
    """
    Append a row to attendance.xlsx.
    Returns True if the record is new, False if already marked today,
    and "PermissionError" if the excel file is open somewhere else.
    """
    today    = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    try:
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
        else:
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

        if "Status" not in df.columns:
            df["Status"] = "Present"

        # Duplicate guard — same name + same date
        if ((df["Name"] == name) & (df["Date"] == today)).any():
            return False

        new_row = pd.DataFrame([{"Name": name, "Date": today, "Time": now_time, "Status": "Present"}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
        
        # Log to separate CSV as well
        csv_exists = os.path.isfile(ATTENDANCE_LOG_CSV)
        try:
            with open(ATTENDANCE_LOG_CSV, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not csv_exists:
                    writer.writerow(["Name", "Date", "Time", "Status"])
                writer.writerow([name, today, now_time, "Present"])
        except Exception as e:
            print(f"Error logging attendance to CSV: {e}")
            
        return True
    except PermissionError:
        return "PermissionError"
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# MODERN UI GENERATION UTILITIES (Anti-Aliased Vectors)
# ─────────────────────────────────────────────────────────────

def get_rounded_button_image(width, height, radius, bg_color, border_color=None, border_width=1):
    """Generates an anti-aliased rounded rectangle image using Pillow supersampling."""
    scale = 2
    w_s, h_s, r_s = width * scale, height * scale, radius * scale
    
    img = Image.new("RGBA", (w_s, h_s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw filled rounded rectangle
    draw.rounded_rectangle(
        (0, 0, w_s - 1, h_s - 1),
        radius=r_s,
        fill=bg_color,
        outline=border_color if border_color else None,
        width=border_width * scale if border_color else 0
    )
    
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img

def get_user_avatar(name, DATASET_DIR, default_avatar_photo):
    """Load user's registered face and render it in a clean circular frame with green glow."""
    folder = os.path.join(DATASET_DIR, name)
    if os.path.exists(folder):
        try:
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                img_path = os.path.join(folder, files[0])
                img = Image.open(img_path)
                
                # Center-crop to square
                w, h = img.size
                size = min(w, h)
                left = (w - size) // 2
                top = (h - size) // 2
                right = left + size
                bottom = top + size
                img_cropped = img.crop((left, top, right, bottom))
                img_resized = img_cropped.resize((40, 40), Image.Resampling.LANCZOS)
                
                # Circular Mask
                mask = Image.new("L", (40, 40), 0)
                draw_mask = ImageDraw.Draw(mask)
                draw_mask.ellipse((0, 0, 39, 39), fill=255)
                
                round_img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
                round_img.paste(img_resized, (0, 0), mask=mask)
                
                # Glowing Green Circular Border
                draw_border = ImageDraw.Draw(round_img)
                draw_border.ellipse((0, 0, 39, 39), outline="#10b981", width=2)
                
                return ImageTk.PhotoImage(round_img)
        except Exception as e:
            print(f"Error loading avatar: {e}")
            
    return default_avatar_photo

# ─────────────────────────────────────────────────────────────
# CUSTOM TECH WIDGETS
# ─────────────────────────────────────────────────────────────

class ModernRoundedButton(tk.Button):
    """Premium flat button with supersampled rounded borders and hover glows."""
    def __init__(self, parent, text, command, width=240, height=42, radius=12, bg_color="#1e1b4b", hover_bg="#312e81", fg_color="white", border_color=None, hover_border=None, border_width=1):
        self.parent = parent
        self.text = text
        self.command = command
        
        pil_normal = get_rounded_button_image(width, height, radius, bg_color, border_color, border_width)
        self.normal_img = ImageTk.PhotoImage(pil_normal)
        
        hb = hover_border if hover_border else border_color
        pil_hover = get_rounded_button_image(width, height, radius, hover_bg, hb, border_width)
        self.hover_img = ImageTk.PhotoImage(pil_hover)
        
        super().__init__(
            parent,
            text=text,
            image=self.normal_img,
            compound="center",
            fg=fg_color,
            activeforeground=fg_color,
            activebackground=parent["bg"],
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
            command=command
        )
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(image=self.hover_img)

    def on_leave(self, e):
        self.config(image=self.normal_img)


class GlowingCardFrame(tk.Frame):
    """Responsive card container with a glowing cyber border and optional live dot."""
    def __init__(self, parent, title="", glow_color="#38bdf8", bg_color="#111827", border_width=1, show_live_dot=False, **kwargs):
        super().__init__(parent, bg=glow_color, bd=0, **kwargs)
        
        self.inner_frame = tk.Frame(self, bg=bg_color, bd=0)
        self.inner_frame.pack(fill="both", expand=True, padx=border_width, pady=border_width)
        
        if title:
            header_f = tk.Frame(self.inner_frame, bg=bg_color)
            header_f.pack(fill="x", padx=15, pady=(12, 8))
            
            self.title_lbl = tk.Label(
                header_f,
                text=title,
                font=("Segoe UI", 11, "bold"),
                bg=bg_color,
                fg=glow_color
            )
            self.title_lbl.pack(side="left")
            
            if show_live_dot:
                self.live_lbl = tk.Label(
                    header_f,
                    text="● LIVE MONITOR",
                    font=("Segoe UI", 9, "bold"),
                    bg=bg_color,
                    fg="#ef4444"
                )
                self.live_lbl.pack(side="right")
                self.blink_state = True
                self._blink_live_dot()

    def _blink_live_dot(self):
        if hasattr(self, 'live_lbl') and self.live_lbl.winfo_exists():
            self.blink_state = not self.blink_state
            color = "#ef4444" if self.blink_state else "#111827"
            self.live_lbl.config(fg=color)
            self.after(800, self._blink_live_dot)


class FeatureCard(GlowingCardFrame):
    """Horizontal tech card that pulses bright blue upon hover."""
    def __init__(self, parent, icon, title, desc, normal_glow="#1e293b", active_glow="#38bdf8"):
        super().__init__(parent, title="", glow_color=normal_glow, bg_color="#0f172a", border_width=1)
        self.normal_glow = normal_glow
        self.active_glow = active_glow
        
        self.icon_lbl = tk.Label(
            self.inner_frame,
            text=icon,
            font=("Segoe UI", 20),
            bg="#0f172a",
            fg="#38bdf8"
        )
        self.icon_lbl.pack(side="left", padx=(15, 12), pady=12)
        
        text_f = tk.Frame(self.inner_frame, bg="#0f172a")
        text_f.pack(side="left", fill="both", expand=True, pady=10)
        
        self.title_lbl = tk.Label(
            text_f,
            text=title,
            font=("Segoe UI", 10, "bold"),
            bg="#0f172a",
            fg="white",
            anchor="w"
        )
        self.title_lbl.pack(anchor="w")
        
        self.desc_lbl = tk.Label(
            text_f,
            text=desc,
            font=("Segoe UI", 8),
            bg="#0f172a",
            fg="#94a3b8",
            anchor="w",
            justify="left",
            wraplength=180
        )
        self.desc_lbl.pack(anchor="w", pady=(2, 0))
        
        for widget in (self.inner_frame, self.icon_lbl, text_f, self.title_lbl, self.desc_lbl):
            widget.bind("<Enter>", self.on_enter)
            widget.bind("<Leave>", self.on_leave)
            
    def on_enter(self, event):
        self.config(bg=self.active_glow)
        
    def on_leave(self, event):
        self.config(bg=self.normal_glow)


class StatCell(tk.Frame):
    """Grid metric block displaying large numbers in appropriate alert colors."""
    def __init__(self, parent, title, num_color, bg_color="#111827"):
        super().__init__(parent, bg=bg_color)
        
        self.title_lbl = tk.Label(
            self,
            text=title,
            font=("Segoe UI", 8, "bold"),
            bg=bg_color,
            fg="#64748b"
        )
        self.title_lbl.pack(anchor="center", pady=(5, 0))
        
        self.num_lbl = tk.Label(
            self,
            text="0",
            font=("Segoe UI", 18, "bold"),
            bg=bg_color,
            fg=num_color
        )
        self.num_lbl.pack(anchor="center", pady=(2, 5))


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

class AttendanceApp(tk.Tk):

    # ── CONFIDENCE THRESHOLD ─────────────────
    CONF_THRESHOLD = 50

    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("1100x720")
        self.attributes('-fullscreen', True)
        self.bind("<Escape>", self._exit_fullscreen)
        self.configure(bg="#0a0b10")

        # Runtime state
        self.cap         = None
        self.running     = False
        self.mode        = None          # "register" | "attendance" | None
        self.frame_count = 0             # For performance (skip frames)
        self.last_faces  = []            # Cache for drawing bounding boxes
        self.last_messages = []          # Cache for UI messages
        self.face_track  = []            # For anti-spoofing (movement check)
        self.reg_name    = ""
        self.reg_count   = 0
        self.REG_SAMPLES = 10            # SPEED OPTIMIZATION
        self.marked_today = set()        # session-level duplicate guard

        # Load recognizer
        self.recognizer, self.id_to_name = load_model()

        # Startup data sync: read excel to find who's marked today
        today = datetime.now().strftime("%Y-%m-%d")
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
                if not df.empty and "Date" in df.columns and "Name" in df.columns:
                    today_present = df[(df["Date"] == today) & (df.get("Status", "Present") == "Present")]["Name"].unique()
                    self.marked_today = set(today_present)
            except Exception as e:
                print(f"Error restoring marked today on startup: {e}")

        # Initialize PIL image assets (Required after Tk initialized)
        self._init_pil_resources()

        self._build_ui()
        
        # Setup automatic status-dot color shifts via trace
        self.status_var.trace_add("write", self._on_status_changed)

        # Trigger initial data draw
        self._load_and_update_stats()
        self._update_recent_logs_panel()
        self._update_recognition_bar(status="idle")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_pil_resources(self):
        """Pre-generate high-quality vectorized default layouts for fast drawing."""
        # Default circular avatar (Small, 40x40)
        img_small = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        draw_small = ImageDraw.Draw(img_small)
        draw_small.ellipse((0, 0, 39, 39), fill="#1f2937", outline="#38bdf8", width=1)
        draw_small.ellipse((14, 8, 25, 19), fill="#94a3b8")
        draw_small.chord((6, 20, 33, 42), start=180, end=0, fill="#94a3b8")
        self.default_avatar_photo = ImageTk.PhotoImage(img_small)
        
        # Default circular avatar (Large, 70x70)
        img_large = Image.new("RGBA", (70, 70), (0, 0, 0, 0))
        draw_large = ImageDraw.Draw(img_large)
        draw_large.ellipse((0, 0, 69, 69), fill="#1f2937", outline="#38bdf8", width=2)
        draw_large.ellipse((24, 14, 45, 35), fill="#94a3b8")
        draw_large.chord((10, 36, 59, 72), start=180, end=0, fill="#94a3b8")
        self.default_avatar_large = ImageTk.PhotoImage(img_large)
        
        # High-tech App Shield Logo (64x64)
        img_logo = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw_logo = ImageDraw.Draw(img_logo)
        draw_logo.polygon([(32, 4), (56, 12), (56, 36), (32, 58), (8, 36), (8, 12)], fill="#1e1b4b", outline="#38bdf8", width=3)
        draw_logo.ellipse((22, 16, 42, 36), fill="#312e81")
        draw_logo.chord((14, 32, 50, 52), start=180, end=0, fill="#4f46e5")
        self.app_logo_img = ImageTk.PhotoImage(img_logo)
        
        try:
            self.iconphoto(False, self.app_logo_img)
        except Exception:
            pass

    def _build_ui(self):
        # ─────────────────────────────────────────────────────────────
        # LEFT SIDEBAR
        # ─────────────────────────────────────────────────────────────
        left = tk.Frame(self, bg="#0d0f17", width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # Sidebar Divider Line (Razor edge)
        divider = tk.Frame(left, bg="#1e293b", width=1)
        divider.pack(side="right", fill="y")

        container = tk.Frame(left, bg="#0d0f17")
        container.pack(fill="both", expand=True, pady=30)

        # Logo & App Info
        logo_lbl = tk.Label(container, image=self.app_logo_img, bg="#0d0f17")
        logo_lbl.pack(pady=(10, 10))

        tk.Label(container, text="Attendance System",
                 font=("Segoe UI", 16, "bold"),
                 bg="#0d0f17", fg="#f8fafc", justify="center").pack()
        tk.Label(container, text="Smart  •  Secure  •  Accurate",
                 font=("Segoe UI", 9, "bold"), bg="#0d0f17", fg="#6366f1").pack(pady=(2, 20))

        ttk.Separator(container).pack(fill="x", padx=30, pady=(0, 20))

        # Dynamic Variable Containers (Bridges for original backend methods)
        self.status_var = tk.StringVar(value="Ready ✔")
        self.users_var  = tk.StringVar(value=self._users_label())
        self.info_var   = tk.StringVar(value="Press a button to begin")

        # Sidebar Rounded Action Links (Hover & glow configured)
        ModernRoundedButton(container, "➕  Register New User", self._start_register, bg_color="#4f46e5", hover_bg="#6366f1").pack(pady=7)
        ModernRoundedButton(container, "🔍  Take Attendance", self._start_attendance, bg_color="#059669", hover_bg="#10b981").pack(pady=7)
        ModernRoundedButton(container, "🚫  Mark Absentees", self._mark_absentees, bg_color="#d97706", hover_bg="#f59e0b").pack(pady=7)
        ModernRoundedButton(container, "📊  View Attendance", self._view_attendance, bg_color="#2563eb", hover_bg="#3b82f6").pack(pady=7)
        ModernRoundedButton(container, "🗑️  Delete User", self._delete_user, bg_color="#db2777", hover_bg="#ec4899").pack(pady=7)
        ModernRoundedButton(container, "⏹️  Stop Camera", self._stop_camera, bg_color="#475569", hover_bg="#64748b").pack(pady=7)
        ModernRoundedButton(container, "🚪  Exit System", self._on_close, bg_color="#dc2626", hover_bg="#ef4444").pack(pady=(25, 7))

        # Bottom System Status Monitor Card in Sidebar
        status_card = tk.Frame(left, bg="#0d0f17")
        status_card.pack(side="bottom", fill="x", padx=25, pady=(0, 30))
        
        ttk.Separator(status_card).pack(fill="x", pady=(0, 15))
        
        status_row = tk.Frame(status_card, bg="#0d0f17")
        status_row.pack(fill="x")
        
        # LED Indicator Dot (Drawn via Canvas)
        self.status_dot = tk.Canvas(status_row, width=12, height=12, bg="#0d0f17", highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8))
        self._set_status_dot_color("#38bdf8")
        
        self.status_lbl = tk.Label(
            status_row,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#0d0f17"
        )
        self.status_lbl.pack(side="left")
        
        self.known_users_lbl = tk.Label(
            status_card,
            textvariable=self.users_var,
            font=("Segoe UI", 9, "italic"),
            fg="#6366f1",
            bg="#0d0f17",
            anchor="w"
        )
        self.known_users_lbl.pack(anchor="w", pady=(4, 0))

        # ─────────────────────────────────────────────────────────────
        # MAIN LAYOUT WORKSPACE
        # ─────────────────────────────────────────────────────────────
        main_container = tk.Frame(self, bg="#0a0b10")
        main_container.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Upper Deck row (Camera feed on Left, Stats & Logs on Right)
        upper_deck = tk.Frame(main_container, bg="#0a0b10")
        upper_deck.pack(side="top", fill="both", expand=True)

        # --- CENTER CAMERA FEED SECTION ---
        center_panel = tk.Frame(upper_deck, bg="#0a0b10")
        center_panel.pack(side="left", fill="both", expand=True, padx=(0, 15))

        self.camera_card = GlowingCardFrame(center_panel, title="LIVE FEED MONITOR", glow_color="#6366f1", show_live_dot=True)
        self.camera_card.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.camera_card.inner_frame,
            bg="#020617",
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # HUD Recognition Card at bottom of Camera Feed
        self.rec_details_bar = tk.Frame(self.camera_card.inner_frame, bg="#161b22", height=85)
        self.rec_details_bar.pack(fill="x", side="bottom", padx=15, pady=(0, 15))
        self.rec_details_bar.pack_propagate(False)
        
        self.rec_avatar_lbl = tk.Label(self.rec_details_bar, image=self.default_avatar_large, bg="#161b22")
        self.rec_avatar_lbl.pack(side="left", padx=(15, 15))
        
        details_stack = tk.Frame(self.rec_details_bar, bg="#161b22")
        details_stack.pack(side="left", fill="y", pady=10)
        
        self.rec_name_lbl = tk.Label(
            details_stack,
            text="SYSTEM STANDBY",
            font=("Segoe UI", 13, "bold"),
            fg="#64748b",
            bg="#161b22",
            anchor="w"
        )
        self.rec_name_lbl.pack(anchor="w")
        
        meta_row = tk.Frame(details_stack, bg="#161b22")
        meta_row.pack(anchor="w", pady=(2, 0))
        
        self.rec_id_lbl = tk.Label(
            meta_row,
            text="ID / ROLL: --",
            font=("Segoe UI", 9),
            fg="#64748b",
            bg="#161b22"
        )
        self.rec_id_lbl.pack(side="left")
        
        self.rec_time_lbl = tk.Label(
            meta_row,
            text="  •  TIME: --",
            font=("Segoe UI", 9),
            fg="#64748b",
            bg="#161b22"
        )
        self.rec_time_lbl.pack(side="left")
        
        self.rec_status_badge = tk.Label(
            self.rec_details_bar,
            text="STANDBY",
            font=("Segoe UI", 10, "bold"),
            bg="#21262d",
            fg="#8b949e",
            padx=12,
            pady=6
        )
        self.rec_status_badge.pack(side="right", padx=(0, 15), pady=22)

        # --- RIGHT STATUS & LOGS PANEL ---
        right_panel = tk.Frame(upper_deck, bg="#0a0b10", width=340)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # Clock Card
        self.clock_card = GlowingCardFrame(right_panel, title="SYSTEM TIME", glow_color="#38bdf8")
        self.clock_card.pack(fill="x", pady=(0, 15))

        clock_container = tk.Frame(self.clock_card.inner_frame, bg="#111827")
        clock_container.pack(fill="both", expand=True, padx=15, pady=(0, 12))

        self.time_lbl = tk.Label(clock_container, text="00:00:00 AM", font=("Segoe UI", 20, "bold"), fg="#38bdf8", bg="#111827")
        self.time_lbl.pack(anchor="w")

        self.date_lbl = tk.Label(clock_container, text="Monday, Jan 01, 2026", font=("Segoe UI", 9), fg="#94a3b8", bg="#111827")
        self.date_lbl.pack(anchor="w", pady=(2, 0))

        # Start Clock Engine
        self._update_clock()

        # Today Summary Card (Grid metrics)
        self.summary_card = GlowingCardFrame(right_panel, title="TODAY SUMMARY", glow_color="#8b5cf6")
        self.summary_card.pack(fill="x", pady=(0, 15))

        grid_f = tk.Frame(self.summary_card.inner_frame, bg="#111827")
        grid_f.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        grid_f.columnconfigure((0, 1), weight=1)
        grid_f.rowconfigure((0, 1), weight=1)
        
        self.cell_total = StatCell(grid_f, "TOTAL STUDENTS", "#8b5cf6")
        self.cell_total.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.stat_total_lbl = self.cell_total.num_lbl
        
        self.cell_present = StatCell(grid_f, "PRESENT TODAY", "#10b981")
        self.cell_present.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.stat_present_lbl = self.cell_present.num_lbl
        
        self.cell_absent = StatCell(grid_f, "ABSENT TODAY", "#ef4444")
        self.cell_absent.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.stat_absent_lbl = self.cell_absent.num_lbl
        
        self.cell_late = StatCell(grid_f, "LATE TODAY", "#f59e0b")
        self.cell_late.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.stat_late_lbl = self.cell_late.num_lbl

        # Recent Logs Feed Card
        self.recent_card = GlowingCardFrame(right_panel, title="RECENT ACTIVITY", glow_color="#10b981")
        self.recent_card.pack(fill="both", expand=True)

        self.recent_list_frame = tk.Frame(self.recent_card.inner_frame, bg="#111827")
        self.recent_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # --- BOTTOM FEATURE DECK ---
        bottom_deck = tk.Frame(main_container, bg="#0a0b10")
        bottom_deck.pack(side="bottom", fill="x", pady=(15, 0))
        bottom_deck.grid_columnconfigure((0, 1, 2, 3), weight=1)

        FeatureCard(bottom_deck, "👁️", "Face Recognition", "Advanced local LBPH facial parsing engine").grid(row=0, column=0, padx=(0, 6), sticky="ew")
        FeatureCard(bottom_deck, "🎯", "High Accuracy", "Robust multi-angle alignment & threshold filter").grid(row=0, column=1, padx=6, sticky="ew")
        FeatureCard(bottom_deck, "🔒", "Data Security", "Secure on-premises database via excel logs").grid(row=0, column=2, padx=6, sticky="ew")
        FeatureCard(bottom_deck, "⚡", "Fast & Efficient", "Skipped framerate loops & multi-threading").grid(row=0, column=3, padx=(6, 0), sticky="ew")

        self.update_idletasks()
        self._draw_placeholder()

    # ── SYSTEM STATUS CONTROLLER ────────────────
    def _on_status_changed(self, *args):
        """Monitor changes to self.status_var to automatically shift LED dot state."""
        val = self.status_var.get()
        if "Taking attendance" in val or "Registering" in val:
            self._set_status_dot_color("#10b981")  # Green
        elif "Training" in val or "Updating model" in val:
            self._set_status_dot_color("#8b5cf6")  # Purple
        elif "stopped" in val or "Ready" in val:
            self._set_status_dot_color("#38bdf8")  # Sky Blue

    def _set_status_dot_color(self, color_hex):
        if hasattr(self, 'status_dot') and self.status_dot.winfo_exists():
            self.status_dot.delete("all")
            self.status_dot.create_oval(1, 1, 11, 11, fill=color_hex, outline="")

    # ── CLOCK SYSTEM ────────────────────────────
    def _update_clock(self):
        now = datetime.now()
        time_str = now.strftime("%I:%M:%S %p")
        date_str = now.strftime("%A, %b %d, %Y")
        
        if hasattr(self, 'time_lbl') and self.time_lbl.winfo_exists():
            self.time_lbl.config(text=time_str)
        if hasattr(self, 'date_lbl') and self.date_lbl.winfo_exists():
            self.date_lbl.config(text=date_str)
            
        self.after(1000, self._update_clock)

    # ── METRIC READERS & FEED DRAWER ────────────
    def _load_and_update_stats(self):
        """Recalculate and render today's attendance variables into cells."""
        total_students = len(self.id_to_name) if self.id_to_name else 0
        present_count = len(self.marked_today)
        absent_count = max(0, total_students - present_count)
        
        late_count = 0
        today = datetime.now().strftime("%Y-%m-%d")
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
                if not df.empty and "Date" in df.columns and "Status" in df.columns:
                    today_present = df[(df["Date"] == today) & (df["Status"] == "Present")]
                    for idx, row in today_present.iterrows():
                        time_str = str(row.get("Time", ""))
                        if time_str and time_str != "--:--:--":
                            try:
                                parts = time_str.split(":")
                                if parts and int(parts[0]) >= 9:
                                    late_count += 1
                            except Exception:
                                pass
            except Exception as e:
                print(f"Error reading stats: {e}")
                
        if hasattr(self, 'stat_total_lbl') and self.stat_total_lbl.winfo_exists():
            self.stat_total_lbl.config(text=str(total_students))
        if hasattr(self, 'stat_present_lbl') and self.stat_present_lbl.winfo_exists():
            self.stat_present_lbl.config(text=str(present_count))
        if hasattr(self, 'stat_absent_lbl') and self.stat_absent_lbl.winfo_exists():
            self.stat_absent_lbl.config(text=str(absent_count))
        if hasattr(self, 'stat_late_lbl') and self.stat_late_lbl.winfo_exists():
            self.stat_late_lbl.config(text=str(late_count))

    def _update_recent_logs_panel(self):
        """Populate recent activity cards with photos & badges."""
        if not hasattr(self, 'recent_list_frame') or not self.recent_list_frame.winfo_exists():
            return
            
        for widget in self.recent_list_frame.winfo_children():
            widget.destroy()
            
        recent_entries = []
        if os.path.exists(ATTENDANCE_FILE):
            try:
                df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
                if not df.empty:
                    df_reversed = df.iloc[::-1]
                    count = 0
                    for idx, row in df_reversed.iterrows():
                        if count >= 4:  # Optimized to display top 4 beautiful cards comfortably
                            break
                        recent_entries.append({
                            "Name": row["Name"],
                            "Date": row["Date"],
                            "Time": row["Time"],
                            "Status": row.get("Status", "Present")
                        })
                        count += 1
            except Exception as e:
                print(f"Error loading recent: {e}")
                
        if not recent_entries:
            placeholder = tk.Label(
                self.recent_list_frame,
                text="No logs recorded today.",
                font=("Segoe UI", 9, "italic"),
                bg="#111827",
                fg="#64748b"
            )
            placeholder.pack(pady=40)
            return
            
        self.recent_avatar_refs = []  # Garbage collector anchor
        
        for entry in recent_entries:
            row_frame = tk.Frame(self.recent_list_frame, bg="#161b22", height=45)
            row_frame.pack(fill="x", pady=4)
            row_frame.pack_propagate(False)
            
            # Draw circular avatar dynamically
            name = entry["Name"]
            avatar_img = get_user_avatar(name, DATASET_DIR, self.default_avatar_photo)
            self.recent_avatar_refs.append(avatar_img)
            
            img_lbl = tk.Label(row_frame, image=avatar_img, bg="#161b22")
            img_lbl.pack(side="left", padx=(10, 10))
            
            # Texts Block
            info_sub = tk.Frame(row_frame, bg="#161b22")
            info_sub.pack(side="left", fill="y", pady=4)
            
            clean_name = name.replace("_", " ").title()
            name_lbl = tk.Label(info_sub, text=clean_name, font=("Segoe UI", 9, "bold"), fg="white", bg="#161b22", anchor="w")
            name_lbl.pack(anchor="w")
            
            time_lbl = tk.Label(info_sub, text=f"{entry['Time']}", font=("Segoe UI", 8), fg="#94a3b8", bg="#161b22", anchor="w")
            time_lbl.pack(anchor="w")
            
            # Badge Alert Status
            status = entry["Status"]
            badge_bg = "#065f46" if status == "Present" else "#7f1d1d"
            badge_fg = "#34d399" if status == "Present" else "#f87171"
            
            badge_lbl = tk.Label(
                row_frame,
                text=status.upper(),
                font=("Segoe UI", 7, "bold"),
                bg=badge_bg,
                fg=badge_fg,
                padx=8,
                pady=2
            )
            badge_lbl.pack(side="right", padx=(0, 10))

    def _update_recognition_bar(self, name=None, status="idle"):
        """Shift states & layouts of HUD bar beneath live monitor."""
        if not hasattr(self, 'rec_details_bar') or not self.rec_details_bar.winfo_exists():
            return
            
        if status == "idle":
            self.rec_avatar_lbl.config(image=self.default_avatar_large)
            self.rec_name_lbl.config(text="MONITOR STANDBY", fg="#64748b")
            self.rec_id_lbl.config(text="ID / ROLL: --", fg="#64748b")
            self.rec_time_lbl.config(text="  •  TIME: --", fg="#64748b")
            self.rec_status_badge.config(text="CAMERA OFF", bg="#21262d", fg="#8b949e")
        elif status == "scanning":
            self.rec_avatar_lbl.config(image=self.default_avatar_large)
            self.rec_name_lbl.config(text="SCANNING FOR TARGETS", fg="#38bdf8")
            self.rec_id_lbl.config(text="ID / ROLL: --", fg="#64748b")
            self.rec_time_lbl.config(text="  •  TIME: --", fg="#64748b")
            self.rec_status_badge.config(text="ACTIVE SCAN", bg="#1e3a8a", fg="#93c5fd")
        elif status == "recognized" and name:
            parts = name.split("_")
            if parts and parts[-1].isdigit():
                roll = parts[-1]
                disp_name = " ".join(parts[:-1]).title()
            else:
                roll = "N/A"
                disp_name = name.replace("_", " ").title()
                
            now_time = datetime.now().strftime("%I:%M:%S %p")
            
            avatar_img = self._get_large_avatar(name)
            self.large_avatar_ref = avatar_img  # Reference anchor
            self.rec_avatar_lbl.config(image=avatar_img)
            
            self.rec_name_lbl.config(text=disp_name, fg="#10b981")
            self.rec_id_lbl.config(text=f"ID / ROLL: {roll}", fg="#e2e8f0")
            self.rec_time_lbl.config(text=f"  •  DETECTED: {now_time}", fg="#e2e8f0")
            self.rec_status_badge.config(text="RECOGNIZED", bg="#065f46", fg="#34d399")
        elif status == "unknown":
            self.rec_avatar_lbl.config(image=self.default_avatar_large)
            self.rec_name_lbl.config(text="UNKNOWN FACE DETECTED", fg="#ef4444")
            self.rec_id_lbl.config(text="ID: UNREGISTERED", fg="#f87171")
            self.rec_time_lbl.config(text="  •  TIME: --", fg="#64748b")
            self.rec_status_badge.config(text="UNAUTHORIZED", bg="#7f1d1d", fg="#f87171")

    def _get_large_avatar(self, name):
        """Helper to center-crop large user files for the recognition HUD."""
        folder = os.path.join(DATASET_DIR, name)
        if os.path.exists(folder):
            try:
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    img_path = os.path.join(folder, files[0])
                    img = Image.open(img_path)
                    
                    # Center-crop to square
                    w, h = img.size
                    size = min(w, h)
                    left = (w - size) // 2
                    top = (h - size) // 2
                    right = left + size
                    bottom = top + size
                    img_cropped = img.crop((left, top, right, bottom))
                    img_resized = img_cropped.resize((70, 70), Image.Resampling.LANCZOS)
                    
                    # Round mask
                    mask = Image.new("L", (70, 70), 0)
                    draw_mask = ImageDraw.Draw(mask)
                    draw_mask.ellipse((0, 0, 69, 69), fill=255)
                    
                    round_img = Image.new("RGBA", (70, 70), (0, 0, 0, 0))
                    round_img.paste(img_resized, (0, 0), mask=mask)
                    
                    # Success circular green outline
                    draw_border = ImageDraw.Draw(round_img)
                    draw_border.ellipse((0, 0, 69, 69), outline="#10b981", width=2)
                    
                    return ImageTk.PhotoImage(round_img)
            except Exception as e:
                print(f"Error drawing large avatar: {e}")
        return self.default_avatar_large

    def _users_label(self):
        return f"Registered Encodings: {len(self.id_to_name)}"

    def _draw_placeholder(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10: cw = 700
        if ch < 10: ch = 480
        self.canvas.create_text(
            cw // 2, ch // 2,
            text="🛡️\n\nMONITOR STANDBY\n(Select actions to start live camera • Escape for standard window)",
            font=("Segoe UI", 13, "bold"), fill="#334155",
            justify="center", tags="placeholder")

    # ── CAMERA THREAD ────────────────────────
    def _start_camera(self):
        if self.cap and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Cannot open webcam.\nEnsure device is connected and not locked by other applications.")
            return
        self.running = True
        
        # Shift bar to active scan status
        self._update_recognition_bar(status="scanning")
        
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        """Background thread reading camera inputs & pushing updates."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10: cw = 700
            if ch < 10: ch = 480
            display = cv2.resize(frame, (cw, ch))

            if self.mode == "attendance":
                display = self._attendance_frame(frame, display)
            elif self.mode == "register":
                display = self._register_frame(frame, display)

            # BGR → RGB → PIL → Tk
            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Main thread synchronization
            self.after(0, self._render_canvas, imgtk)

        self._stop_camera()

    def _render_canvas(self, imgtk):
        """Updates canvas inside the main Tk event thread."""
        if hasattr(self, 'canvas') and self.canvas.winfo_exists():
            self.canvas.imgtk = imgtk          # Garbage collection barrier
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)

    # ── ATTENDANCE MODE ───────────────────────
    def _start_attendance(self):
        if self.recognizer is None:
            messagebox.showwarning(
                "Not Trained",
                "No trained LBPH mathematical model found.\n"
                "Please register and record a user folder first.")
            return
        self.mode = "attendance"
        self.error_shown = False
        self.shutdown_time = None
        self.status_var.set("📸 Taking attendance…")
        self.info_var.set("")
        self._start_camera()

    def _attendance_frame(self, frame, display):
        self.frame_count += 1
        
        # Camera automatic delay shutdown logic
        if hasattr(self, 'shutdown_time') and self.shutdown_time:
            time_left = self.shutdown_time - time.time()
            if time_left <= 0:
                self.shutdown_time = None
                speak("Camera shutting down")
                self.after(0, self._stop_camera)
                return display
            else:
                cv2.putText(display, f"Camera shutting down in {int(time_left)+1}s", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        sx = display.shape[1] / frame.shape[1]
        sy = display.shape[0] / frame.shape[0]

        # Heavy processing every 2nd frame
        if self.frame_count % 2 == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = face_cascade.detectMultiScale(
                small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]
            
            self.last_faces = []
            self.last_messages = []

            if len(faces) == 0:
                self.face_track = []
                # Re-verify scanning status in main thread
                self.after(0, lambda: self._update_recognition_bar(status="scanning"))
            elif len(faces) > 1:
                self.last_messages.append(("MULTIPLE FACES DETECTED", (0, 0, 255)))
                speak("Please come one by one")
                for (x, y, w, h) in faces:
                    self.last_faces.append((x, y, w, h, "Multiple", (0, 0, 255)))
                self.face_track = []
            elif len(faces) == 1:
                (x, y, w, h) = faces[0]
                
                # Proximity thresholds
                if w < 80:
                    self.last_messages.append(("MOVE CLOSER", (0, 165, 255)))
                    speak("Please come closer")
                    self.last_faces.append((x, y, w, h, "Too far", (0, 165, 255)))
                    self.face_track = []
                elif w > 200:
                    self.last_messages.append(("TOO CLOSE", (0, 165, 255)))
                    speak("Too close, step back")
                    self.last_faces.append((x, y, w, h, "Too close", (0, 165, 255)))
                    self.face_track = []
                else:
                    # Anti-Spoofing: Motion variation check
                    cx, cy = x + w/2, y + h/2
                    self.face_track.append((cx, cy))
                    if len(self.face_track) > 6:
                        self.face_track.pop(0)
                    
                    is_static = False
                    if len(self.face_track) == 6:
                        xs = [pt[0] for pt in self.face_track]
                        ys = [pt[1] for pt in self.face_track]
                        max_dx = max(xs) - min(xs)
                        max_dy = max(ys) - min(ys)
                        if max_dx < 3 and max_dy < 3:
                            is_static = True

                    face_roi = gray[y:y+h, x:x+w]
                    face_roi_resized = cv2.resize(face_roi, (200, 200))
                    label_id, confidence = self.recognizer.predict(face_roi_resized)

                    if confidence < self.CONF_THRESHOLD:
                        name  = self.id_to_name.get(label_id, "Unknown")
                        color = (50, 220, 50)           # Green
                        conf_text = f"{name} ({confidence:.0f})"

                        # Mark attendance
                        if name not in self.marked_today:
                            added = mark_attendance(name)
                            if added is True:
                                self.marked_today.add(name)
                                speak(f"Attendance marked for {name.replace('_', ' ')}")
                                self.after(0, lambda n=name: self._show_marked(n))
                                if not self.shutdown_time:
                                    self.shutdown_time = time.time() + 3
                            elif added == "PermissionError":
                                if getattr(self, "error_shown", False) is False:
                                    self.error_shown = True
                                    self.after(0, lambda: messagebox.showerror(
                                        "File Open Blockage", 
                                        "attendance.xlsx is currently locked/opened in Excel.\nPlease close the spreadsheet first so logs can update!"))
                        else:
                            if not self.shutdown_time:
                                self.shutdown_time = time.time() + 3

                        self.last_faces.append((x, y, w, h, conf_text, color))
                        # Update recognized HUD
                        self.after(0, lambda n=name: self._update_recognition_bar(n, status="recognized"))
                    else:
                        name      = "Unknown"
                        color     = (0, 165, 255)       # Cyber Warning Amber
                        conf_text = "Unknown Face"
                        self.last_faces.append((x, y, w, h, conf_text, color))
                        # Update unauthorized HUD
                        self.after(0, lambda: self._update_recognition_bar(status="unknown"))

        # Render active face boxes
        for (x, y, w, h, text, color) in self.last_faces:
            dx, dy = int(x*sx), int(y*sy)
            dw, dh = int(w*sx), int(h*sy)
            cv2.rectangle(display, (dx, dy), (dx+dw, dy+dh), color, 2)
            cv2.rectangle(display, (dx, dy+dh-32), (dx+dw, dy+dh), color, cv2.FILLED)
            cv2.putText(display, text.replace('_', ' ').title(), (dx+6, dy+dh-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
        # Overlay active system telemetry texts
        y_offset = 40
        for msg, color in self.last_messages:
            cv2.putText(display, msg, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            y_offset += 40

        return display

    def _show_marked(self, name):
        self.status_var.set(f"✅ Attendance marked — {name}")
        self.info_var.set(f"✅  {name} — Recorded Successfully")
        
        # Real-time dashboard updates (Dispatched thread-safely)
        self._load_and_update_stats()
        self._update_recent_logs_panel()
        
        messagebox.showinfo("Attendance Recorded",
                            f"✅  Attendance successfully written to database for:\n\n    {name.replace('_', ' ').title()}")

    # ── REGISTER MODE ─────────────────────────
    def _start_register(self):
        name = simpledialog.askstring(
            "Register Target Enrollee",
            "Enter Full Name or Roll ID of registration subject:",
            parent=self)
        if not name or not name.strip():
            return

        self.reg_name  = name.strip().replace(" ", "_")
        self.reg_count = 0
        self.mode      = "register"

        person_dir = os.path.join(DATASET_DIR, self.reg_name)
        os.makedirs(person_dir, exist_ok=True)

        self.status_var.set(f"📷 Registering: {self.reg_name}\nCapturing {self.REG_SAMPLES} face images…")
        self.info_var.set("")
        self._start_camera()

    def _register_frame(self, frame, display):
        self.frame_count += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces = face_cascade.detectMultiScale(
            small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
        faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]

        sx = display.shape[1] / frame.shape[1]
        sy = display.shape[0] / frame.shape[0]

        if len(faces) > 1:
            cv2.putText(display, "MULTIPLE TARGETS! Align single user.", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return display
            
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Reduced blur filter threshold for fast registration
            variance = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            if variance < 40:
                cv2.putText(display, "STABILIZE DEVICE! Blurry input.", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return display

            # Check if face already registered before starting
            if self.reg_count == 0 and self.recognizer is not None:
                face_roi_resized = cv2.resize(face_roi, (200, 200))
                label_id, confidence = self.recognizer.predict(face_roi_resized)
                if confidence < self.CONF_THRESHOLD:
                    name = self.id_to_name.get(label_id, "Unknown")
                    speak("Face already registered")
                    self.mode = None
                    self.after(0, self._stop_camera)
                    self.after(0, lambda: messagebox.showerror(
                        "Registration Rejected", 
                        f"Target face has already been logged under identity: '{name.replace('_', ' ').title()}'."))
                    return display

            # Draw registration box (Purple cyber lock)
            cv2.rectangle(display,
                          (int(x*sx), int(y*sy)),
                          (int((x+w)*sx), int((y+h)*sy)),
                          (180, 50, 230), 2)

            # Record facial crop
            if self.reg_count < self.REG_SAMPLES and self.frame_count % 2 == 0:
                face_roi_resized = cv2.resize(face_roi, (200, 200))
                save_path = os.path.join(
                    DATASET_DIR, self.reg_name,
                    f"{self.reg_count:04d}.jpg")
                cv2.imwrite(save_path, face_roi_resized)
                self.reg_count += 1

        # Render loading percentages
        pct = min(int(self.reg_count / self.REG_SAMPLES * 100), 100)
        cv2.putText(display,
                    f"Biometric Scans: {min(self.reg_count, self.REG_SAMPLES)}/{self.REG_SAMPLES}  ({pct}%)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 50, 230), 2)

        # Loading slide bar
        bar_w = int(display.shape[1] * pct / 100)
        cv2.rectangle(display, (0, display.shape[0]-8),
                      (bar_w, display.shape[0]), (180, 50, 230), cv2.FILLED)

        if self.reg_count >= self.REG_SAMPLES:
            self.mode = None
            self.after(0, self._finish_register)

        return display

    def _finish_register(self):
        self._stop_camera()
        self.status_var.set("🔄 Training LBPH model…")
        self.update()
        
        # Train model locally
        self.recognizer, self.id_to_name = train_model()
        
        user_id = "N/A"
        for uid, name in self.id_to_name.items():
            if name == self.reg_name:
                user_id = uid
                break
                
        clean_name = self.reg_name.replace('_', ' ')
        log_registration(user_id, clean_name, self.reg_count)
        
        self.users_var.set(self._users_label())
        self.status_var.set("Ready ✔")
        self.info_var.set("Registration complete!")
        
        # Refresh widgets
        self._load_and_update_stats()
        self._update_recent_logs_panel()
        
        messagebox.showinfo(
            "Registration Successful",
            f"✅  '{clean_name.title()}' biometric profile registered successfully!\n"
            f"Total active system encodings: {len(self.id_to_name)}\n"
            f"AI classifier retrained in real-time.")

    # ── DELETE USER ───────────────────────────
    def _delete_user(self):
        if not self.id_to_name:
            messagebox.showinfo("Empty Registries", "No registered profiles found in database.")
            return

        win = tk.Toplevel(self)
        win.title("De-Register User Enrollee")
        win.geometry("380x210")
        win.configure(bg="#0d0f17")
        win.resizable(False, False)
        win.grab_set()

        # Styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Cyber.TCombobox", 
                        fieldbackground="#161b22", 
                        background="#0d0f17", 
                        foreground="white", 
                        arrowcolor="#38bdf8",
                        font=("Segoe UI", 10))

        tk.Label(win, text="🛡️ De-Register User", font=("Segoe UI", 12, "bold"), bg="#0d0f17", fg="#ef4444").pack(pady=(15, 5))
        tk.Label(win, text="Select biometric identity to purge:", font=("Segoe UI", 10), bg="#0d0f17", fg="#94a3b8").pack(pady=2)
        
        users = sorted(list(set(self.id_to_name.values())))
        # Render clean names in combobox
        display_users = [u.replace("_", " ").title() for u in users]
        combo_map = {u.replace("_", " ").title(): u for u in users}
        
        combo = ttk.Combobox(win, values=display_users, state="readonly", font=("Segoe UI", 10), style="Cyber.TCombobox")
        combo.pack(pady=12)
        if display_users:
            combo.current(0)
            
        def do_delete():
            disp_user = combo.get()
            if not disp_user: return
            
            user = combo_map.get(disp_user)
            if not user: return
            
            if messagebox.askyesno("Confirm Purge", f"WARNING: Are you sure you want to completely de-register '{disp_user}'?\n\nThis will permanently delete all dataset crops, encodings, and historic attendance sheets.", parent=win):
                # 1. Delete dataset folder
                user_dir = os.path.join(DATASET_DIR, user)
                if os.path.exists(user_dir):
                    shutil.rmtree(user_dir, ignore_errors=True)
                
                # 2. Delete from attendance records
                try:
                    if os.path.exists(ATTENDANCE_FILE):
                        df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
                        if "Name" in df.columns:
                            df = df[df["Name"] != user]
                            df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
                except PermissionError:
                    messagebox.showwarning("Warning", "attendance.xlsx is locked. Purged records in directory, but spreadsheet row skipped.", parent=win)
                
                # 3. Retrain model
                self.status_var.set("🔄 Updating model...")
                self.update()
                self.recognizer, self.id_to_name = train_model()
                self.users_var.set(self._users_label())
                self.status_var.set("Ready ✔")
                
                # Session duplicate set purge
                if user in self.marked_today:
                    self.marked_today.discard(user)
                
                # Refresh widgets
                self._load_and_update_stats()
                self._update_recent_logs_panel()
                
                messagebox.showinfo("Purge Successful", f"Identity of '{disp_user}' has been permanently wiped.", parent=win)
                win.destroy()
                
        # Modern rounded delete button
        btn_frame = tk.Frame(win, bg="#0d0f17")
        btn_frame.pack(pady=10)
        ModernRoundedButton(btn_frame, "Purge Biometric Identity", do_delete, width=220, height=36, radius=10, bg_color="#dc2626", hover_bg="#ef4444").pack()


    # ── MARK ABSENTEES ────────────────────────
    def _mark_absentees(self):
        if not self.id_to_name:
            messagebox.showinfo("Empty Database", "Cannot compute absentees. Registries are currently empty.")
            return

        today = datetime.now().strftime("%Y-%m-%d")
        try:
            if os.path.exists(ATTENDANCE_FILE):
                df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
            else:
                df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

            if "Status" not in df.columns:
                df["Status"] = "Present"

            today_records = df[df["Date"] == today]
            present_users = today_records[today_records["Status"] == "Present"]["Name"].tolist()
            
            all_users = list(set(self.id_to_name.values()))
            
            new_rows = []
            for user in all_users:
                if user not in present_users:
                    if not ((df["Name"] == user) & (df["Date"] == today) & (df["Status"] == "Absent")).any():
                        new_rows.append({"Name": user, "Date": today, "Time": "--:--:--", "Status": "Absent"})
            
            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
                
                # Refresh stats and logs
                self._load_and_update_stats()
                self._update_recent_logs_panel()
                
                messagebox.showinfo("Absentees Wrote", f"✅ Successfully logged {len(new_rows)} absentees into attendance sheets.")
            else:
                messagebox.showinfo("Audit Standby", "All registered users are present or already logged absent today.")
                
        except PermissionError:
            messagebox.showerror("Spreadsheet Lock", "attendance.xlsx is currently open in another program.\nPlease close Excel to register absentees!")
        except Exception as e:
            messagebox.showerror("Audit Error", f"Audit failed:\n{e}")

    # ── VIEW ATTENDANCE ───────────────────────
    def _view_attendance(self):
        if not os.path.exists(ATTENDANCE_FILE):
            messagebox.showinfo("Empty Records", "No attendance records database found.\nStart active scans first.")
            return

        win = tk.Toplevel(self)
        win.title("Biometric Attendance Logbook")
        win.geometry("740x520")
        win.configure(bg="#0d0f17")
        win.grab_set()

        # Title
        tk.Label(win, text="📋  Biometric Attendance Database",
                 font=("Segoe UI", 14, "bold"),
                 bg="#0d0f17", fg="#38bdf8").pack(pady=(15, 4))

        # Main frame wrapper
        frame = tk.Frame(win, bg="#0d0f17")
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        # High-tech Cyber Treeview Theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Cyber.Treeview",
                        background="#111827", 
                        foreground="#f1f5f9",
                        rowheight=28, 
                        fieldbackground="#111827",
                        font=("Segoe UI", 10))
        style.configure("Cyber.Treeview.Heading",
                        background="#1e293b", 
                        foreground="#38bdf8",
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=0)
        style.map("Cyber.Treeview", 
                  background=[("selected", "#3b82f6")],
                  foreground=[("selected", "white")])

        cols = ("Name", "Date", "Time", "Status")
        tree = ttk.Treeview(frame, columns=cols, show="headings", style="Cyber.Treeview")
        
        for col in cols:
            tree.heading(col, text=col.upper())
            tree.column(col, width=160, anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Load Rows
        try:
            df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
            if "Status" not in df.columns:
                df["Status"] = "Present"
                
            # Render beautifully (newest first)
            df_reversed = df.iloc[::-1]
            for _, row in df_reversed.iterrows():
                u_name = str(row["Name"]).replace("_", " ").title()
                tree.insert("", "end",
                            values=(u_name, row["Date"], row["Time"], row.get("Status", "Present")))
        except PermissionError:
            messagebox.showerror("File Access Denied", "attendance.xlsx is open in another program.\nClose the worksheet and try again.", parent=win)
            win.destroy()
            return
        except Exception as e:
            messagebox.showerror("Error", f"Error reading database:\n{e}", parent=win)
            win.destroy()
            return

        # Bottom row inside view logs
        footer = tk.Frame(win, bg="#0d0f17")
        footer.pack(fill="x", side="bottom", padx=20, pady=15)

        tk.Label(footer, text=f"Total records logged: {len(df)}",
                 font=("Segoe UI", 9, "bold"), bg="#0d0f17", fg="#64748b").pack(side="left")

        # Open in Excel Rounded button
        excel_cmd = lambda: os.startfile(ATTENDANCE_FILE)
        btn_wrapper = tk.Frame(footer, bg="#0d0f17")
        btn_wrapper.pack(side="right")
        ModernRoundedButton(btn_wrapper, "Open Excel Sheets", excel_cmd, width=150, height=34, radius=8, bg_color="#1e1b4b", hover_bg="#312e81").pack()

    # ── CLEANUP ───────────────────────────────
    def _stop_camera(self):
        self.running = False
        self.mode    = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_var.set("Camera stopped. Ready ✔")
        self._update_recognition_bar(status="idle")
        self._draw_placeholder()

    def _exit_fullscreen(self, event=None):
        self.attributes('-fullscreen', False)
        self.geometry("1100x720")

    def _on_close(self):
        self._stop_camera()
        self.destroy()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
