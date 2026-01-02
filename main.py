import winsound
import win32api
import win32con
import win32gui
import numpy as np
import random
import time
import cv2
import mss
import threading
import tkinter as tk
from tkinter import filedialog, ttk
import os
import json

class Config:
    def __init__(self):
        try:
            self.width = win32api.GetSystemMetrics(0)
            self.height = win32api.GetSystemMetrics(1)
        except:
            self.width = 1920
            self.height = 1080
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.uniformCaptureSize = 240
        self.crosshairUniform = self.uniformCaptureSize // 2
        self.capture_left = self.center_x - self.crosshairUniform
        self.capture_top = self.center_y - self.crosshairUniform
        self.region = {"top": self.capture_top, "left": self.capture_left, "width": self.uniformCaptureSize, "height": self.uniformCaptureSize}

config = Config()
screenCapture = mss.mss()

def get_hsv_range_from_image(path, tolerance=(10, 60, 60)):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        lower = np.array([0, 160, 160], dtype=np.uint8)
        upper = np.array([10, 255, 255], dtype=np.uint8)
        return lower, upper
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    pixel_bgr = img[h//2, w//2].reshape(1, 1, 3)
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    lower = np.array([
        max(int(pixel_hsv[0]) - tolerance[0], 0),
        max(int(pixel_hsv[1]) - tolerance[1], 0),
        max(int(pixel_hsv[2]) - tolerance[2], 0)
    ], dtype=np.uint8)
    upper = np.array([
        min(int(pixel_hsv[0]) + tolerance[0], 179),
        min(int(pixel_hsv[1]) + tolerance[1], 255),
        min(int(pixel_hsv[2]) + tolerance[2], 255)
    ], dtype=np.uint8)
    return lower, upper

lower_hsv, upper_hsv = get_hsv_range_from_image("images/main.png", tolerance=(10, 60, 60))
kernel = np.ones((3, 3), np.uint8)
min_area = 60
crosshairU = config.crosshairUniform
regionC = config.region
robloxSensitivity = 0.55
PF_MouseSensitivity = 0.5
PF_AimSensitivity = 1
PF_sensitivity = PF_MouseSensitivity*PF_AimSensitivity
movementCompensation = 0.0
finalComputerSensitivityMultiplier = ((robloxSensitivity*PF_sensitivity)/0.55) + movementCompensation
deadzone_px = 6
max_step_px = 6
smooth_alpha = 0.18
ema_dx = 0.0
ema_dy = 0.0
kp = 0.45
kd = 0.25
prev_err_x = 0.0
prev_err_y = 0.0
track_cx = None
track_cy = None
roi_radius = 50
lost_frames = 0
lost_threshold = 6
aim_enabled = True
toggle_key = 0x77
prev_toggle_state = 0
fov_radius = 80
lock_strength = 1.0
offset_x = 0
offset_y = 0
trigger_enabled = False
trigger_distance = 5
toggle_key = 0x77  # Default F8 (VK_F8)
trigger_key = 0x79 # Default F10
active_ranges = []
running = False
worker = None
lock_cx = None
lock_cy = None
overlay = None
overlay_canvas = None

def build_mask(frame_hsv):
    if not active_ranges:
        return cv2.inRange(frame_hsv, lower_hsv, upper_hsv)
    m = None
    for lo, up in active_ranges:
        mm = cv2.inRange(frame_hsv, lo, up)
        m = mm if m is None else cv2.bitwise_or(m, mm)
    return m

def hex_to_bgr(s):
    s = s.strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 6:
        r = int(s[0:2],16)
        g = int(s[2:4],16)
        b = int(s[4:6],16)
        return (b,g,r)
    return None

def range_from_hex(s, tol_h=10, tol_s=60, tol_v=60):
    bgr = hex_to_bgr(s)
    if bgr is None:
        return None
    pix = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0,0]
    h = int(hsv[0]); s = int(hsv[1]); v = int(hsv[2])
    lo = np.array([max(h - tol_h, 0), max(s - tol_s, 0), max(v - tol_v, 0)], dtype=np.uint8)
    up = np.array([min(h + tol_h, 179), min(s + tol_s, 255), min(v + tol_v, 255)], dtype=np.uint8)
    return lo, up

def analyze_image_colors(path, k=5):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []
    if img.ndim==3 and img.shape[2]==4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    Z = img.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    ret,label,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    centers = center.astype(np.uint8)
    res=[]
    for c in centers:
        r,g,b = int(c[2]),int(c[1]),int(c[0])
        hx = '#%02X%02X%02X'% (r,g,b)
        res.append(hx)
    return res

def analyze_folder_colors(folder, k=8, max_images=50, sample_per_image=4000):
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    except Exception:
        files = []
    if not files:
        return []
    files = files[:max_images]
    samples = []
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim==3 and img.shape[2]==4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        flat = img.reshape((-1,3))
        n = flat.shape[0]
        if n > sample_per_image:
            idx = np.random.choice(n, sample_per_image, replace=False)
            samples.append(flat[idx])
        else:
            samples.append(flat)
    if not samples:
        return []
    Z = np.vstack(samples).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    ret,label,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    centers = center.astype(np.uint8)
    lbl = label.ravel()
    counts = np.bincount(lbl, minlength=k)
    order = np.argsort(-counts)
    res=[]
    for i in order:
        c = centers[i]
        r,g,b = int(c[2]),int(c[1]),int(c[0])
        hx = '#%02X%02X%02X'% (r,g,b)
        res.append(hx)
    return res

def run_loop():
    global running, track_cx, track_cy, prev_toggle_state, ema_dx, ema_dy, prev_err_x, prev_err_y, lock_cx, lock_cy, aim_enabled, trigger_enabled
    running = True
    s = mss.mss()
    while running:
        time.sleep(0.001)
        GameFrame = np.array(s.grab(regionC))
        GameFrame = cv2.cvtColor(GameFrame, cv2.COLOR_BGRA2BGR)
        
        # Toggle Aim
        tk_state = win32api.GetAsyncKeyState(toggle_key)
        if tk_state < 0 and prev_toggle_state >= 0:
            aim_enabled = not aim_enabled
            try:
                winsound.Beep(1200 if aim_enabled else 800, 80)
            except Exception:
                pass
        prev_toggle_state = tk_state
        
        # Trigger Toggle (Separate Key Check example, or UI Toggle)
        # For now, let's keep trigger enabled via UI checkbox only to avoid key clutter, 
        # or check a specific key if requested. Let's stick to UI variable for enable/disable logic first.

        if win32api.GetAsyncKeyState(0x6) < 0:
            try:
                winsound.Beep(1000, 10)
            except Exception:
                pass
            break
        elif aim_enabled and win32api.GetAsyncKeyState(0x02) < 0:
            frame_hsv = cv2.cvtColor(GameFrame, cv2.COLOR_BGR2HSV)
            mask = build_mask(frame_hsv)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.medianBlur(mask, 5)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                centroids = []
                for c in contours:
                    a = cv2.contourArea(c)
                    if a < min_area:
                        continue
                    M = cv2.moments(c)
                    if M['m00'] == 0:
                        continue
                    cx_i = M['m10'] / M['m00']
                    cy_i = M['m01'] / M['m00']
                    if (cx_i - crosshairU)**2 + (cy_i - crosshairU)**2 <= fov_radius*fov_radius:
                        centroids.append((cx_i, cy_i, a, c))
                if centroids:
                    if track_cx is not None:
                        near = [t for t in centroids if (t[0]-track_cx)**2 + (t[1]-track_cy)**2 <= roi_radius*roi_radius]
                        chosen = min(near, key=lambda t: (t[0]-track_cx)**2 + (t[1]-track_cy)**2) if near else max(centroids, key=lambda t: t[2])
                    else:
                        chosen = max(centroids, key=lambda t: t[2])
                    cx, cy, area, _ = chosen
                    track_cx, track_cy = cx, cy
                    lock_cx, lock_cy = cx, cy
                    
                    # Triggerbot Logic
                    if trigger_enabled and aim_enabled:
                         dist_sq = (crosshairU - cx)**2 + (crosshairU - cy)**2
                         if dist_sq <= trigger_distance**2:
                             win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                             time.sleep(0.01)
                             win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                             time.sleep(0.05) # Delay between shots

                    # Apply Offset
                    target_x = cx + offset_x
                    target_y = cy + offset_y
                    
                    err_x = (-(crosshairU - target_x))
                    err_y = (-(crosshairU - target_y))
                    if abs(err_x) < deadzone_px:
                        err_x = 0.0
                    if abs(err_y) < deadzone_px:
                        err_y = 0.0
                    cross_x = np.sign(err_x) != np.sign(prev_err_x)
                    cross_y = np.sign(err_y) != np.sign(prev_err_y)
                    scale_x = np.tanh(abs(err_x)/10.0)
                    scale_y = np.tanh(abs(err_y)/10.0)
                    finalMult = finalComputerSensitivityMultiplier * lock_strength
                    dx_raw = (kp*err_x + kd*(err_x - prev_err_x))*finalMult*scale_x
                    dy_raw = (kp*err_y + kd*(err_y - prev_err_y))*finalMult*scale_y
                    if cross_x:
                        dx_raw *= 0.5
                    if cross_y:
                        dy_raw *= 0.5
                    dx_raw = float(np.clip(dx_raw, -max_step_px, max_step_px))
                    dy_raw = float(np.clip(dy_raw, -max_step_px, max_step_px))
                    ema_dx = (1 - smooth_alpha)*ema_dx + smooth_alpha*dx_raw
                    ema_dy = (1 - smooth_alpha)*ema_dy + smooth_alpha*dy_raw
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(ema_dx), int(ema_dy), 0, 0)
                    prev_err_x = err_x
                    prev_err_y = err_y
    cv2.destroyAllWindows()

def update_swatch(canvas, hex_str):
    try:
        canvas.delete("all")
        canvas.create_rectangle(0, 0, 48, 24, fill=hex_str, outline="#000000")
    except Exception:
        pass

def on_palette_select(event):
    try:
        sel = palette_list.curselection()
        if sel:
            s = palette_list.get(sel[-1])
            update_swatch(swatch_canvas, s)
    except Exception:
        pass

def on_palette_double_click(event):
    try:
        lb = event.widget
        idx = lb.nearest(event.y)
        s = lb.get(idx)
        hex_var.set(s)
        set_from_hex()
    except Exception:
        pass
def update_palette_canvas(cols):
    try:
        palette_canvas.delete("all")
        h = int(palette_canvas.winfo_height() or 120)
        n = max(1, len(cols))
        bar_h = max(12, h//n)
        for i, hx in enumerate(cols):
            palette_canvas.create_rectangle(0, i*bar_h, 48, (i+1)*bar_h, fill=hx, outline="")
    except Exception:
        pass

def run_worker():
    global worker
    if worker and worker.is_alive():
        return
    worker = threading.Thread(target=run_loop, daemon=True)
    worker.start()
    status_var.set("Running")

def stop_worker():
    global running
    running = False
    status_var.set("Stopped")
    hide_overlay()

def on_start():
    ensure_overlay()
    run_worker()

def set_from_hex():
    global lower_hsv, upper_hsv
    s = hex_var.get()
    rng = range_from_hex(s, tol_h_var.get(), tol_s_var.get(), tol_v_var.get())
    if rng:
        active_ranges.clear()
        active_ranges.append(rng)
        update_swatch(swatch_canvas, s)

def choose_image():
    p = filedialog.askopenfilename(filetypes=[("Image","*.png;*.jpg;*.jpeg;*.bmp")])
    if not p:
        return
    cols = analyze_image_colors(p, k=5)
    palette_list.delete(0, tk.END)
    for hx in cols:
        palette_list.insert(tk.END, hx)
    update_palette_canvas(cols)

def lock_selected_colors():
    sel = palette_list.curselection()
    if not sel:
        return
    active_ranges.clear()
    for i in sel:
        s = palette_list.get(i)
        rng = range_from_hex(s, tol_h_var.get(), tol_s_var.get(), tol_v_var.get())
        if rng:
            active_ranges.append(rng)

def auto_detect_colors_from_folder():
    cols = analyze_folder_colors('images', k=8)
    if not cols:
        return
    palette_list.delete(0, tk.END)
    for hx in cols:
        palette_list.insert(tk.END, hx)
    update_palette_canvas(cols)
    active_ranges.clear()
    for hx in cols[:3]:
        rng = range_from_hex(hx, tol_h_var.get(), tol_s_var.get(), tol_v_var.get())
        if rng:
            active_ranges.append(rng)
    update_swatch(swatch_canvas, cols[0])

def update_params(*args):
    global lock_strength, smooth_alpha, max_step_px, deadzone_px, fov_radius, offset_x, offset_y
    lock_strength = float(strength_var.get())
    smooth_alpha = float(stability_var.get())
    max_step_px = int(max_step_var.get())
    deadzone_px = int(deadzone_var.get())
    fov_radius = int(fov_var.get())
    offset_x = int(offset_x_var.get())
    offset_y = int(offset_y_var.get())
    global trigger_enabled, trigger_distance, toggle_key
    trigger_enabled = trigger_var.get()
    trigger_distance = int(trigger_dist_var.get())
    try:
        toggle_key = int(aim_key_var.get(), 16)
    except:
        pass
    update_overlay()

def ensure_overlay():
    global overlay, overlay_canvas
    if not show_fov_var.get():
        hide_overlay()
        return
    if overlay is None or not overlay.winfo_exists():
        overlay = tk.Toplevel(root)
        overlay.overrideredirect(True)
        overlay.attributes('-topmost', True)
        try:
            overlay.attributes('-transparentcolor', 'magenta')
            bgc = 'magenta'
        except Exception:
            bgc = None
        overlay.configure(bg=bgc if bgc else '')
        overlay_canvas = tk.Canvas(overlay, bg=bgc if bgc else '', highlightthickness=0, width=10, height=10)
        overlay_canvas.pack(fill='both', expand=True)
    update_overlay()

def update_overlay():
    global overlay, overlay_canvas
    if overlay is None or not overlay.winfo_exists():
        return
    if not show_fov_var.get():
        hide_overlay()
        return
    r = int(fov_var.get())
    d = 2*r + 6
    x = config.center_x - d//2
    y = config.center_y - d//2
    overlay.geometry(f"{d}x{d}+{x}+{y}")
    overlay.deiconify()
    overlay_canvas.delete('all')
    overlay_canvas.create_oval(3,3,d-3,d-3, outline='red', width=2)

def hide_overlay():
    global overlay
    try:
        if overlay and overlay.winfo_exists():
            overlay.withdraw()
    except Exception:
        pass

LANG_TEXT = {
    'TH': {
        'frame_color': 'การตั้งค่าสี / พาเลต',
        'hex': 'รหัสสี:',
        'apply_hex': 'ใช้สี',
        'choose_image': 'เลือกรูปภาพ',
        'analyze_images': 'วิเคราะห์ images',
        'lock_colors': 'ล็อคสีที่เลือก',
        'frame_params': 'พารามิเตอร์การทำงาน',
        'lock': 'แรงดูด (Lock)',
        'smooth': 'ความนิ่ง (Smooth)',
        'max_step': 'Max Step',
        'deadzone': 'Deadzone',
        'fov': 'FOV Radius',
        'offset_x': 'Offset X',
        'offset_y': 'Offset Y',
        'frame_trigger': 'Triggerbot / Hotkeys',
        'trigger_enable': 'เปิดใช้งาน Triggerbot',
        'trigger_dist': 'ระยะทำงาน (px)',
        'hotkey_aim': 'ปุ่ม Aim (Hex):',
        'hotkey_trigger': 'ปุ่ม Trigger (Hex):',
        'frame_controls': 'ควบคุม / สถานะ',
        'start': 'เริ่มการทำงาน',
        'stop': 'หยุดการทำงาน',
        'show_fov': 'แสดง FOV Overlay',
        'frame_status': 'สถานะ',
        'lang_button': 'ภาษา: ไทย'
    },
    'EN': {
        'frame_color': 'Color / Palette Settings',
        'hex': 'Hex:',
        'apply_hex': 'Apply Hex',
        'choose_image': 'Pick Image',
        'analyze_images': 'Analyze images',
        'lock_colors': 'Lock Selected Colors',
        'frame_params': 'Operation Parameters',
        'lock': 'Lock Strength',
        'smooth': 'Smoothness',
        'max_step': 'Max Step',
        'deadzone': 'Deadzone',
        'fov': 'FOV Radius',
        'offset_x': 'Offset X',
        'offset_y': 'Offset Y',
        'frame_trigger': 'Triggerbot / Hotkeys',
        'trigger_enable': 'Enable Triggerbot',
        'trigger_dist': 'Trigger Distance',
        'hotkey_aim': 'Aim Key (Hex)',
        'hotkey_trigger': 'Trigger Key (Hex)',
        'frame_controls': 'Controls / Status',
        'start': 'Start',
        'stop': 'Stop',
        'show_fov': 'Show FOV Overlay',
        'frame_status': 'Status',
        'lang_button': 'Language: English'
    }
}

def apply_language():
    t = LANG_TEXT[lang_var.get()]
    label_title_color.configure(text=t['frame_color'])
    hex_label.configure(text=t['hex'])
    apply_hex_btn.configure(text=t['apply_hex'])
    choose_image_btn.configure(text=t['choose_image'])
    analyze_images_btn.configure(text=t['analyze_images'])
    lock_colors_btn.configure(text=t['lock_colors'])
    
    label_title_params.configure(text=t['frame_params'])
    label_lock.configure(text=t['lock'])
    label_smooth.configure(text=t['smooth'])
    label_max_step.configure(text=t['max_step'])
    label_deadzone.configure(text=t['deadzone'])
    label_fov.configure(text=t['fov'])
    label_offset_x.configure(text=t['offset_x'])
    label_offset_y.configure(text=t['offset_y'])
    
    label_title_trigger.configure(text=t['frame_trigger'])
    chk_trigger.configure(text=t['trigger_enable'])
    label_trigger_dist.configure(text=t['trigger_dist'])
    label_hotkey_aim.configure(text=t['hotkey_aim'])
    
    label_title_controls.configure(text=t['frame_controls'])
    start_btn.configure(text=t['start'])
    stop_btn.configure(text=t['stop'])
    show_fov_chk.configure(text=t['show_fov'])
    lang_btn.configure(text=t['lang_button'])

def toggle_language():
    lang_var.set('EN' if lang_var.get() == 'TH' else 'TH')
    apply_language()

import customtkinter

# Color Theme - Grey/Black
BG_DARK = "#121212"
BG_MEDIUM = "#1E1E1E"
BG_LIGHT = "#252525"
PRIMARY_PURPLE = "#404040"
PRIMARY_PURPLE_HOVER = "#505050"
ACCENT_RED = "#E94560"
TEXT_LIGHT = "#FFFFFF"
TEXT_GRAY = "#C0C0C0"

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("Color Aimbot")
root.geometry('700x605')
root.resizable(False, False)
root.configure(fg_color=BG_DARK)

hex_var = customtkinter.StringVar(value="#feffb2")
tol_h_var = customtkinter.IntVar(value=10)
tol_s_var = customtkinter.IntVar(value=60)
tol_v_var = customtkinter.IntVar(value=60)
strength_var = customtkinter.DoubleVar(value=1.0)
stability_var = customtkinter.DoubleVar(value=smooth_alpha)
max_step_var = customtkinter.IntVar(value=max_step_px)
deadzone_var = customtkinter.IntVar(value=deadzone_px)
fov_var = customtkinter.IntVar(value=fov_radius)
offset_x_var = customtkinter.IntVar(value=0)
offset_y_var = customtkinter.IntVar(value=0)
trigger_var = customtkinter.BooleanVar(value=False)
trigger_dist_var = customtkinter.IntVar(value=5)
aim_key_var = customtkinter.StringVar(value="0x77")
show_fov_var = customtkinter.BooleanVar(value=False)
status_var = customtkinter.StringVar(value='Stopped')
lang_var = customtkinter.StringVar(value='TH')

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

left_col = customtkinter.CTkFrame(root, fg_color=BG_MEDIUM)
left_col.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
left_col.grid_columnconfigure(0, weight=1)

right_col = customtkinter.CTkFrame(root, fg_color=BG_MEDIUM)
right_col.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
right_col.grid_columnconfigure(0, weight=1)

# --- Left Column: Color Settings ---
frame_color = customtkinter.CTkFrame(left_col, fg_color=BG_LIGHT)
frame_color.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
frame_color.grid_columnconfigure(0, weight=1)

label_title_color = customtkinter.CTkLabel(frame_color, text="🎨 การตั้งค่าสี / พาเลต", font=("Roboto", 16, "bold"))
label_title_color.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="w", padx=10)

row1 = customtkinter.CTkFrame(frame_color, fg_color="transparent")
row1.grid(row=1, column=0, columnspan=2, sticky='w', pady=(0, 10), padx=10)

hex_label = customtkinter.CTkLabel(row1, text='รหัสสี:')
hex_label.pack(side='left', padx=(0, 4))

hex_entry = customtkinter.CTkEntry(row1, textvariable=hex_var, width=80, fg_color=BG_MEDIUM, border_color=PRIMARY_PURPLE)
hex_entry.pack(side='left', padx=(0, 6))

apply_hex_btn = customtkinter.CTkButton(row1, text='ใช้สี', command=set_from_hex, width=60, fg_color=PRIMARY_PURPLE, hover_color=PRIMARY_PURPLE_HOVER)
apply_hex_btn.pack(side='left', padx=(0, 10))

swatch_canvas = tk.Canvas(row1, width=48, height=24, highlightthickness=1, highlightbackground=BG_MEDIUM, bg=BG_MEDIUM)
swatch_canvas.pack(side='left')
update_swatch(swatch_canvas, hex_var.get())

palette_frame = customtkinter.CTkFrame(frame_color, fg_color="transparent")
palette_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=10)
palette_frame.grid_columnconfigure(0, weight=1)

palette_list = tk.Listbox(palette_frame, selectmode="multiple", height=8, bg=BG_MEDIUM, fg=TEXT_LIGHT, borderwidth=0, highlightthickness=0)
palette_list.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
palette_list.bind('<<ListboxSelect>>', on_palette_select)
palette_list.bind('<Double-Button-1>', on_palette_double_click)

palette_canvas = tk.Canvas(palette_frame, width=48, height=140, highlightthickness=1, highlightbackground=BG_MEDIUM, bg=BG_MEDIUM)
palette_canvas.grid(row=0, column=1, sticky='ns')

btns_palette = customtkinter.CTkFrame(frame_color, fg_color="transparent")
btns_palette.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(10, 10), padx=10)
btns_palette.grid_columnconfigure(0, weight=1)
btns_palette.grid_columnconfigure(1, weight=1)

choose_image_btn = customtkinter.CTkButton(btns_palette, text='เลือกรูปภาพ', command=choose_image, fg_color=PRIMARY_PURPLE, hover_color=PRIMARY_PURPLE_HOVER)
choose_image_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5), pady=(0, 5))

analyze_images_btn = customtkinter.CTkButton(btns_palette, text='วิเคราะห์ images', command=auto_detect_colors_from_folder, fg_color=PRIMARY_PURPLE, hover_color=PRIMARY_PURPLE_HOVER)
analyze_images_btn.grid(row=0, column=1, sticky='ew', padx=(5, 0), pady=(0, 5))

lock_colors_btn = customtkinter.CTkButton(btns_palette, text='ล็อคสี', command=lock_selected_colors, fg_color=PRIMARY_PURPLE, hover_color=PRIMARY_PURPLE_HOVER)
lock_colors_btn.grid(row=1, column=0, columnspan=2, sticky='ew', padx=0, pady=0)


# --- Right Column: Parameters ---
frame_params = customtkinter.CTkFrame(right_col, fg_color=BG_LIGHT)
frame_params.grid(row=0, column=0, sticky='new', padx=10, pady=10)
frame_params.grid_columnconfigure(0, minsize=140)
frame_params.grid_columnconfigure(1, weight=1)

label_title_params = customtkinter.CTkLabel(frame_params, text="⚙️ พารามิเตอร์", font=("Roboto", 16, "bold"))
label_title_params.grid(row=0, column=0, columnspan=3, pady=(10, 5), sticky="w", padx=10)

def add_labeled_scale(parent, label_text, var, frm, to, row_idx):
    lbl = customtkinter.CTkLabel(parent, text=label_text, anchor="w")
    lbl.grid(row=row_idx, column=0, sticky='w', padx=(10, 5), pady=5)
    
    s = customtkinter.CTkSlider(parent, from_=frm, to=to, variable=var, command=update_params, button_color=ACCENT_RED, progress_color=PRIMARY_PURPLE)
    s.grid(row=row_idx, column=1, sticky='ew', padx=5, pady=5)
    
    val_lbl = customtkinter.CTkLabel(parent, textvariable=var, width=30, anchor="e")
    val_lbl.grid(row=row_idx, column=2, sticky='e', padx=(5, 10), pady=5)
    return lbl

label_lock = add_labeled_scale(frame_params, 'แรงดูด (Lock)', strength_var, 0.5, 3.0, 1)
label_smooth = add_labeled_scale(frame_params, 'ความนิ่ง (Smooth)', stability_var, 0.05, 1.00, 2)
label_max_step = add_labeled_scale(frame_params, 'Max Step', max_step_var, 1, 20, 3)
label_deadzone = add_labeled_scale(frame_params, 'Deadzone', deadzone_var, 0, 15, 4)
label_fov = add_labeled_scale(frame_params, 'FOV Radius', fov_var, 30, 140, 5)
label_offset_x = add_labeled_scale(frame_params, 'Offset X', offset_x_var, -100, 100, 6)
label_offset_y = add_labeled_scale(frame_params, 'Offset Y', offset_y_var, -100, 100, 7)


# --- Left Column: Controls ---
frame_controls = customtkinter.CTkFrame(left_col, fg_color=BG_LIGHT)
frame_controls.grid(row=1, column=0, sticky='sew', padx=10, pady=10)
frame_controls.grid_columnconfigure(0, weight=1)

label_title_controls = customtkinter.CTkLabel(frame_controls, text="�️ ควบคุม", font=("Roboto", 16, "bold"))
label_title_controls.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)

start_btn = customtkinter.CTkButton(frame_controls, text='✅ เริ่มการทำงาน', command=on_start, fg_color="#2CC985", hover_color="#229966", text_color="white")
start_btn.grid(row=1, column=0, sticky='ew', pady=(5, 5), padx=10)

stop_btn = customtkinter.CTkButton(frame_controls, text='❌ หยุดการทำงาน', command=stop_worker, fg_color="#C92C2C", hover_color="#992222", text_color="white")
stop_btn.grid(row=2, column=0, sticky='ew', pady=(0, 10), padx=10)

show_fov_chk = customtkinter.CTkCheckBox(frame_controls, text='แสดง FOV Overlay', variable=show_fov_var, command=ensure_overlay, fg_color=PRIMARY_PURPLE, hover_color=PRIMARY_PURPLE_HOVER)
show_fov_chk.grid(row=3, column=0, sticky='w', pady=(0, 10), padx=10)

frame_status = customtkinter.CTkFrame(frame_controls, fg_color="transparent", border_width=1, border_color=PRIMARY_PURPLE)
frame_status.grid(row=4, column=0, sticky='ew', pady=(0, 10), padx=10)
status_label = customtkinter.CTkLabel(frame_status, textvariable=status_var)
status_label.pack(fill='x', expand=True, padx=5, pady=5)

lang_btn = customtkinter.CTkButton(frame_controls, text='ภาษา', command=toggle_language, fg_color="transparent", border_width=1, border_color=PRIMARY_PURPLE, text_color=TEXT_LIGHT)
lang_btn.grid(row=5, column=0, sticky='ew', pady=(0, 10), padx=10)


# --- Right Column: Triggerbot / Hotkeys ---
# Placing it below Parameters (frame_params is row 0)
frame_trigger = customtkinter.CTkFrame(right_col, fg_color=BG_LIGHT)
frame_trigger.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
frame_trigger.grid_columnconfigure(0, minsize=140)
frame_trigger.grid_columnconfigure(1, weight=1)

label_title_trigger = customtkinter.CTkLabel(frame_trigger, text="� Triggerbot / Hotkeys", font=("Roboto", 16, "bold"))
label_title_trigger.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="w", padx=10)

chk_trigger = customtkinter.CTkCheckBox(frame_trigger, text='Enable Triggerbot', variable=trigger_var, command=update_params, fg_color=PRIMARY_PURPLE, hover_color=PRIMARY_PURPLE_HOVER)
chk_trigger.grid(row=1, column=0, columnspan=2, sticky='w', padx=10, pady=5)

label_trigger_dist = add_labeled_scale(frame_trigger, 'Trigger Distance', trigger_dist_var, 1, 50, 3)

# Hotkey Aim
row_hk = customtkinter.CTkFrame(frame_trigger, fg_color="transparent")
row_hk.grid(row=2, column=0, columnspan=3, sticky='w', padx=10, pady=5)

label_hotkey_aim = customtkinter.CTkLabel(row_hk, text='Aim Key (Hex):')
label_hotkey_aim.pack(side='left', padx=(0, 5))

entry_hotkey_aim = customtkinter.CTkEntry(row_hk, textvariable=aim_key_var, width=80, fg_color=BG_MEDIUM, border_color=PRIMARY_PURPLE)
entry_hotkey_aim.pack(side='left', padx=(0, 5))

btn_update_hotkey = customtkinter.CTkButton(row_hk, text='Update', command=update_params, width=60, fg_color=PRIMARY_PURPLE, hover_color=PRIMARY_PURPLE_HOVER)
btn_update_hotkey.pack(side='left')


def load_settings():
    try:
        if not os.path.exists('settings.json'):
            return
        
        # Check if file is empty
        if os.path.getsize('settings.json') == 0:
            return

        with open('settings.json', 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Settings file is corrupted. Using defaults.")
                return
            
        hex_var.set(data.get("hex", "#feffb2"))
        tol_h_var.set(data.get("tol_h", 10))
        tol_s_var.set(data.get("tol_s", 60))
        tol_v_var.set(data.get("tol_v", 60))
        strength_var.set(data.get("lock_strength", 1.0))
        stability_var.set(data.get("smooth_alpha", 0.18))
        max_step_var.set(data.get("max_step", 6))
        deadzone_var.set(data.get("deadzone", 6))
        fov_var.set(data.get("fov", 80))
        offset_x_var.set(data.get("offset_x", 0))
        offset_y_var.set(data.get("offset_y", 0))
        show_fov_var.set(data.get("show_fov", False))
        lang_var.set(data.get("lang", "TH"))
        trigger_var.set(data.get("trigger_enabled", False))
        trigger_dist_var.set(data.get("trigger_distance", 5))
        aim_key_var.set(data.get("aim_key", "0x77"))
        
        palette_list.delete(0, tk.END)
        for c in data.get("palette", []):
            palette_list.insert(tk.END, c)
            
        update_swatch(swatch_canvas, hex_var.get())
        update_params()
        
    except Exception as e:
        print(f"Error loading settings: {e}")

def save_settings():
    try:
        data = {
            "hex": hex_var.get(),
            "tol_h": tol_h_var.get(),
            "tol_s": tol_s_var.get(),
            "tol_v": tol_v_var.get(),
            "lock_strength": strength_var.get(),
            "smooth_alpha": stability_var.get(),
            "max_step": max_step_var.get(),
            "deadzone": deadzone_var.get(),
            "fov": fov_var.get(),
            "offset_x": offset_x_var.get(),
            "offset_y": offset_y_var.get(),
            "show_fov": show_fov_var.get(),
            "lang": lang_var.get(),
            "trigger_enabled": trigger_var.get(),
            "trigger_distance": trigger_dist_var.get(),
            "aim_key": aim_key_var.get(),
            "palette": list(palette_list.get(0, tk.END))
        }
        with open('settings.json', 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")

def on_close():
    stop_worker()
    save_settings()
    root.destroy()

load_settings()
apply_language()
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
