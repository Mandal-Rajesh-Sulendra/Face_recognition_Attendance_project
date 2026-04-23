# 🎓 Face Recognition Attendance System

A real-time face recognition attendance system built with Python, OpenCV, and Tkinter.

---

## 📁 Folder Structure

```
Face_recognition_Attendance_project/
│
├── main.py              ← Main application (run this)
├── requirements.txt     ← Python dependencies
├── setup.bat            ← One-click Windows installer
├── README.md            ← This file
│
├── dataset/             ← Face images (auto-created per user)
│   └── John_Doe/
│       ├── 0000.jpg
│       └── ...
│
├── encodings/           ← Saved face encodings (auto-generated)
│   └── encodings.pkl
│
└── attendance/          ← Excel attendance records
    └── attendance.xlsx
```

---

## ⚙️ Installation

### Option A — One-Click Setup (Windows)
```
Double-click setup.bat
```

### Option B — Manual Install
```bash
pip install cmake dlib
pip install face_recognition opencv-python numpy pandas openpyxl Pillow
```

> **Note:** `dlib` requires cmake. On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if dlib fails to install.
>
> **Easy alternative:** Download a pre-compiled dlib wheel from  
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x  
> then run: `pip install dlib‑<version>.whl`

---

## 🚀 Running the App

```bash
python main.py
```

---

## 🖥️ GUI Buttons

| Button | Action |
|---|---|
| ➕ Register New User | Open webcam, enter name, capture 20 face images |
| ✅ Take Attendance | Recognize faces in real-time and log to Excel |
| 📋 View Attendance | View all records in a table inside the app |
| 🔄 Re-train Encodings | Rebuild encodings if you add images manually |
| ⏹ Stop Camera | Stop webcam feed |
| 🚪 Exit | Close the application |

---

## 📊 Excel Output Format

`attendance/attendance.xlsx`

| Name | Date | Time |
|---|---|---|
| John_Doe | 2026-04-23 | 09:15:32 |
| Jane_Smith | 2026-04-23 | 09:17:05 |

- Duplicate entries (same name + same date) are **automatically prevented**.
- New records are **appended** without overwriting existing data.

---

## 📝 How It Works

1. **Register** → Webcam captures 20 images → saved in `dataset/<name>/`  
2. **Encode** → `face_recognition` extracts 128-d face embeddings → saved in `encodings/encodings.pkl`  
3. **Attend** → Live webcam compares faces against stored encodings → marks attendance in Excel  

---

## 🛠️ Tips

- Register in **good lighting** for best accuracy.
- Tolerance is set to **0.5** (lower = stricter). Adjust in `main.py` line with `tolerance=0.5`.
- To remove a user: delete their folder in `dataset/` and click **Re-train Encodings**.
