# ============================================================
#   Face Recognition Attendance System
#   OPTION 1 — No dlib | Works on Python 3.14+
#   Uses: OpenCV LBPH recognizer + Haarcascade detector
# ============================================================
#   Install (one command):
#       pip install opencv-contrib-python numpy pandas openpyxl Pillow
#   Run:
#       python main.py
# ============================================================

import cv2
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading

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

        id_to_name[current_id] = person_name
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            pil_img  = Image.open(img_path).convert("L")   # grayscale
            img_arr  = np.array(pil_img, dtype=np.uint8)
            faces.append(img_arr)
            labels.append(current_id)
        current_id += 1

    if not faces:
        return None, {}

    recognizer = _create_recognizer()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(id_to_name, f)

    return recognizer, id_to_name

# ─────────────────────────────────────────────────────────────
# ATTENDANCE HELPERS
# ─────────────────────────────────────────────────────────────

def mark_attendance(name: str) -> bool:
    """
    Append a row to attendance.xlsx.
    Returns True if the record is new, False if already marked today.
    """
    today    = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Duplicate guard — same name + same date
    if ((df["Name"] == name) & (df["Date"] == today)).any():
        return False

    new_row = pd.DataFrame([{"Name": name, "Date": today, "Time": now_time}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(ATTENDANCE_FILE, index=False, engine="openpyxl")
    return True

# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

class AttendanceApp(tk.Tk):

    # ── CONFIDENCE THRESHOLD ─────────────────
    # LBPH confidence: lower = better match.
    # Faces with confidence > this are "Unknown".
    CONF_THRESHOLD = 70

    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("1100x680")
        self.resizable(False, False)
        self.configure(bg="#0f0f1a")

        # Runtime state
        self.cap         = None
        self.running     = False
        self.mode        = None          # "register" | "attendance" | None
        self.reg_name    = ""
        self.reg_count   = 0
        self.REG_SAMPLES = 30            # face images captured per user
        self.marked_today = set()        # session-level duplicate guard

        # Load recognizer
        self.recognizer, self.id_to_name = load_model()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ───────────────────────────────────
    def _build_ui(self):
        # Left panel
        left = tk.Frame(self, bg="#16162a", width=280)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="🎓", font=("Segoe UI", 38),
                 bg="#16162a", fg="#a78bfa").pack(pady=(28, 4))
        tk.Label(left, text="Attendance\nSystem",
                 font=("Segoe UI", 15, "bold"),
                 bg="#16162a", fg="#e2e8f0", justify="center").pack()
        tk.Label(left, text="OpenCV  •  LBPH  •  Excel",
                 font=("Segoe UI", 8), bg="#16162a", fg="#6366f1").pack(pady=(2, 18))

        ttk.Separator(left).pack(fill="x", padx=18, pady=4)

        B = dict(font=("Segoe UI", 11, "bold"), bd=0, relief="flat",
                 activeforeground="white", cursor="hand2", width=22, pady=9)

        tk.Button(left, text="➕  Register New User",
                  bg="#6366f1", fg="white", activebackground="#4f46e5",
                  command=self._start_register, **B).pack(pady=7, padx=18)

        tk.Button(left, text="✅  Take Attendance",
                  bg="#10b981", fg="white", activebackground="#059669",
                  command=self._start_attendance, **B).pack(pady=7, padx=18)

        tk.Button(left, text="📋  View Attendance",
                  bg="#f59e0b", fg="white", activebackground="#d97706",
                  command=self._view_attendance, **B).pack(pady=7, padx=18)

        tk.Button(left, text="🔄  Re-train Model",
                  bg="#3b82f6", fg="white", activebackground="#2563eb",
                  command=self._retrain, **B).pack(pady=7, padx=18)

        tk.Button(left, text="⏹  Stop Camera",
                  bg="#64748b", fg="white", activebackground="#475569",
                  command=self._stop_camera, **B).pack(pady=7, padx=18)

        tk.Button(left, text="🚪  Exit",
                  bg="#ef4444", fg="white", activebackground="#dc2626",
                  command=self._on_close, **B).pack(pady=7, padx=18)

        ttk.Separator(left).pack(fill="x", padx=18, pady=8)

        self.status_var = tk.StringVar(value="Ready ✔")
        tk.Label(left, textvariable=self.status_var,
                 font=("Segoe UI", 9), bg="#16162a", fg="#94a3b8",
                 wraplength=240, justify="center").pack(pady=4, padx=10)

        self.users_var = tk.StringVar(value=self._users_label())
        tk.Label(left, textvariable=self.users_var,
                 font=("Segoe UI", 9, "italic"),
                 bg="#16162a", fg="#6366f1").pack(pady=2)

        # Right panel — camera canvas
        right = tk.Frame(self, bg="#0f0f1a")
        right.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(right, bg="#0a0a14", width=800, height=580,
                                highlightthickness=2,
                                highlightbackground="#6366f1")
        self.canvas.pack(padx=10, pady=10)

        self.info_var = tk.StringVar(value="Press a button to begin")
        tk.Label(right, textvariable=self.info_var,
                 font=("Segoe UI", 11), bg="#0f0f1a", fg="#a78bfa").pack()

        self._draw_placeholder()

    def _users_label(self):
        return f"Known users: {len(self.id_to_name)}"

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 290,
            text="📷\n\nCamera feed will appear here",
            font=("Segoe UI", 16), fill="#334155",
            justify="center", tags="placeholder")

    # ── CAMERA THREAD ────────────────────────
    def _start_camera(self):
        if self.cap and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.\nCheck that it is connected and not in use.")
            return
        self.running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        """Background thread: reads frames, dispatches to mode handler."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            display = cv2.resize(frame, (800, 580))

            if self.mode == "attendance":
                display = self._attendance_frame(frame, display)
            elif self.mode == "register":
                display = self._register_frame(frame, display)

            # BGR → RGB → PIL → Tk
            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk          # keep reference alive
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)

        self._stop_camera()

    # ── ATTENDANCE MODE ───────────────────────
    def _start_attendance(self):
        if self.recognizer is None:
            messagebox.showwarning(
                "Not Trained",
                "No trained model found.\n"
                "Please register at least one user first.")
            return
        self.mode = "attendance"
        self.status_var.set("📸 Taking attendance…")
        self.info_var.set("Look at the camera — attendance is being recorded")
        self._start_camera()

    def _attendance_frame(self, frame, display):
        """Detect faces → recognize → mark attendance."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        sx = display.shape[1] / frame.shape[1]   # scale factors
        sy = display.shape[0] / frame.shape[0]

        for (x, y, w, h) in faces:
            # Crop + resize face for LBPH
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label_id, confidence = self.recognizer.predict(face_roi)

            if confidence < self.CONF_THRESHOLD:
                name  = self.id_to_name.get(label_id, "Unknown")
                color = (50, 220, 50)           # green — recognized
                conf_text = f"{name}  ({confidence:.0f})"

                # Mark attendance (thread-safe via after())
                if name not in self.marked_today:
                    added = mark_attendance(name)
                    if added:
                        self.marked_today.add(name)
                        self.after(0, lambda n=name: self._show_marked(n))
            else:
                name      = "Unknown"
                color     = (60, 60, 230)       # blue — unknown
                conf_text = f"Unknown  ({confidence:.0f})"

            # Draw bounding box (scaled to display size)
            dx, dy = int(x*sx), int(y*sy)
            dw, dh = int(w*sx), int(h*sy)
            cv2.rectangle(display, (dx, dy), (dx+dw, dy+dh), color, 2)
            cv2.rectangle(display, (dx, dy+dh-32), (dx+dw, dy+dh), color, cv2.FILLED)
            cv2.putText(display, conf_text,
                        (dx+6, dy+dh-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display

    def _show_marked(self, name):
        self.status_var.set(f"✅ Attendance marked — {name}")
        self.info_var.set(f"✅  {name} — Attendance Recorded Successfully")
        messagebox.showinfo("Attendance Marked",
                            f"✅  Attendance successfully marked for:\n\n    {name}")

    # ── REGISTER MODE ─────────────────────────
    def _start_register(self):
        name = simpledialog.askstring(
            "Register New User",
            "Enter the person's Full Name or Roll Number:",
            parent=self)
        if not name or not name.strip():
            return

        self.reg_name  = name.strip().replace(" ", "_")
        self.reg_count = 0
        self.mode      = "register"

        person_dir = os.path.join(DATASET_DIR, self.reg_name)
        os.makedirs(person_dir, exist_ok=True)

        self.status_var.set(f"📷 Registering: {self.reg_name}\nCapturing {self.REG_SAMPLES} face images…")
        self.info_var.set(f"Look straight at the camera — capturing {self.reg_name}")
        self._start_camera()

    def _register_frame(self, frame, display):
        """Capture face images for a new user."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        sx = display.shape[1] / frame.shape[1]
        sy = display.shape[0] / frame.shape[0]

        for (x, y, w, h) in faces:
            # Draw purple box while registering
            cv2.rectangle(display,
                          (int(x*sx), int(y*sy)),
                          (int((x+w)*sx), int((y+h)*sy)),
                          (180, 50, 230), 2)

            # Save every 2nd frame for variety, up to REG_SAMPLES
            if self.reg_count < self.REG_SAMPLES and self.reg_count % 2 == 0:
                face_roi  = gray[y:y+h, x:x+w]
                face_roi  = cv2.resize(face_roi, (200, 200))
                save_path = os.path.join(
                    DATASET_DIR, self.reg_name,
                    f"{self.reg_count:04d}.jpg")
                cv2.imwrite(save_path, face_roi)

            if len(faces):
                self.reg_count += 1

        # Progress overlay
        pct = min(int(self.reg_count / self.REG_SAMPLES * 100), 100)
        cv2.putText(display,
                    f"Capturing: {min(self.reg_count, self.REG_SAMPLES)}/{self.REG_SAMPLES}  ({pct}%)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 50, 230), 2)

        # Bar
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
        self.recognizer, self.id_to_name = train_model()
        self.users_var.set(self._users_label())
        self.status_var.set("Ready ✔")
        self.info_var.set("Registration complete!")
        messagebox.showinfo(
            "Registration Successful",
            f"✅  '{self.reg_name.replace('_',' ')}' registered successfully!\n"
            f"Total known users: {len(self.id_to_name)}")

    # ── RETRAIN ───────────────────────────────
    def _retrain(self):
        self.status_var.set("🔄 Re-training LBPH model…")
        self.update()
        self.recognizer, self.id_to_name = train_model()
        self.users_var.set(self._users_label())
        self.status_var.set("Ready ✔")
        if self.id_to_name:
            messagebox.showinfo("Done",
                                f"✅ Model re-trained.\n"
                                f"Total users: {len(self.id_to_name)}")
        else:
            messagebox.showwarning("Empty Dataset",
                                   "No images found in dataset/.\n"
                                   "Please register users first.")

    # ── VIEW ATTENDANCE ───────────────────────
    def _view_attendance(self):
        if not os.path.exists(ATTENDANCE_FILE):
            messagebox.showinfo("No Records",
                                "No attendance records found yet.\n"
                                "Mark attendance first.")
            return

        win = tk.Toplevel(self)
        win.title("Attendance Records")
        win.geometry("680x480")
        win.configure(bg="#0f0f1a")
        win.grab_set()

        tk.Label(win, text="📋  Attendance Records",
                 font=("Segoe UI", 14, "bold"),
                 bg="#0f0f1a", fg="#e2e8f0").pack(pady=(14, 4))

        frame = tk.Frame(win, bg="#0f0f1a")
        frame.pack(fill="both", expand=True, padx=14, pady=8)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("A.Treeview",
                        background="#16162a", foreground="#e2e8f0",
                        rowheight=26, fieldbackground="#16162a",
                        font=("Segoe UI", 10))
        style.configure("A.Treeview.Heading",
                        background="#6366f1", foreground="white",
                        font=("Segoe UI", 10, "bold"))
        style.map("A.Treeview", background=[("selected", "#4f46e5")])

        cols = ("Name", "Date", "Time")
        tree = ttk.Treeview(frame, columns=cols, show="headings",
                            style="A.Treeview")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=200, anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        df = pd.read_excel(ATTENDANCE_FILE, engine="openpyxl")
        for _, row in df.iterrows():
            tree.insert("", "end",
                        values=(row["Name"], row["Date"], row["Time"]))

        tk.Label(win, text=f"Total records: {len(df)}",
                 font=("Segoe UI", 9), bg="#0f0f1a", fg="#94a3b8").pack(pady=5)

        tk.Button(win, text="Open Excel File",
                  bg="#6366f1", fg="white", font=("Segoe UI", 9),
                  command=lambda: os.startfile(ATTENDANCE_FILE),
                  cursor="hand2", bd=0, padx=10, pady=4).pack(pady=6)

    # ── CLEANUP ───────────────────────────────
    def _stop_camera(self):
        self.running = False
        self.mode    = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_var.set("Camera stopped. Ready ✔")
        self._draw_placeholder()

    def _on_close(self):
        self._stop_camera()
        self.destroy()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
