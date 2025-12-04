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

class Config:
    def __init__(self):
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
    global running, track_cx, track_cy, prev_toggle_state, ema_dx, ema_dy, prev_err_x, prev_err_y, lock_cx, lock_cy, aim_enabled
    running = True
    s = mss.mss()
    while running:
        time.sleep(0.001)
        GameFrame = np.array(s.grab(regionC))
        GameFrame = cv2.cvtColor(GameFrame, cv2.COLOR_BGRA2BGR)
        tk_state = win32api.GetAsyncKeyState(toggle_key)
        if tk_state < 0 and prev_toggle_state >= 0:
            aim_enabled = not aim_enabled
            try:
                winsound.Beep(1200 if aim_enabled else 800, 80)
            except Exception:
                pass
        prev_toggle_state = tk_state
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
                    err_x = (-(crosshairU - cx))
                    err_y = (-(crosshairU - cy))
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
    global lock_strength, smooth_alpha, max_step_px, deadzone_px, fov_radius
    lock_strength = float(strength_var.get())
    smooth_alpha = float(stability_var.get())
    max_step_px = int(max_step_var.get())
    deadzone_px = int(deadzone_var.get())
    fov_radius = int(fov_var.get())
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
        'frame_color': '🎨 การตั้งค่าสี / พาเลต',
        'hex': 'รหัสสี:',
        'apply_hex': 'ใช้สี',
        'choose_image': 'เลือกรูปภาพ',
        'analyze_images': 'วิเคราะห์ images',
        'lock_colors': 'ล็อคสีที่เลือก',
        'frame_params': '⚙️ พารามิเตอร์การทำงาน',
        'lock': 'แรงดูด (Lock)',
        'smooth': 'ความนิ่ง (Smooth)',
        'max_step': 'Max Step',
        'deadzone': 'Deadzone',
        'fov': 'FOV Radius',
        'frame_controls': '🕹️ ควบคุม / สถานะ',
        'start': '✅ เริ่มการทำงาน',
        'stop': '❌ หยุดการทำงาน',
        'show_fov': 'แสดง FOV Overlay',
        'frame_status': 'สถานะ',
        'lang_button': 'ภาษา: ไทย'
    },
    'EN': {
        'frame_color': '🎨 Color / Palette Settings',
        'hex': 'Hex:',
        'apply_hex': 'Apply Hex',
        'choose_image': 'Pick Image',
        'analyze_images': 'Analyze images',
        'lock_colors': 'Lock Selected Colors',
        'frame_params': '⚙️ Operation Parameters',
        'lock': 'Lock Strength',
        'smooth': 'Smoothness',
        'max_step': 'Max Step',
        'deadzone': 'Deadzone',
        'fov': 'FOV Radius',
        'frame_controls': '🕹️ Controls / Status',
        'start': '✅ Start',
        'stop': '❌ Stop',
        'show_fov': 'Show FOV Overlay',
        'frame_status': 'Status',
        'lang_button': 'Language: English'
    }
}

def apply_language():
    t = LANG_TEXT[lang_var.get()]
    frame_color.configure(text=t['frame_color'])
    hex_label.configure(text=t['hex'])
    apply_hex_btn.configure(text=t['apply_hex'])
    choose_image_btn.configure(text=t['choose_image'])
    analyze_images_btn.configure(text=t['analyze_images'])
    lock_colors_btn.configure(text=t['lock_colors'])
    frame_params.configure(text=t['frame_params'])
    label_lock.configure(text=t['lock'])
    label_smooth.configure(text=t['smooth'])
    label_max_step.configure(text=t['max_step'])
    label_deadzone.configure(text=t['deadzone'])
    label_fov.configure(text=t['fov'])
    frame_controls.configure(text=t['frame_controls'])
    start_btn.configure(text=t['start'])
    stop_btn.configure(text=t['stop'])
    show_fov_chk.configure(text=t['show_fov'])
    frame_status.configure(text=t['frame_status'])
    lang_btn.configure(text=t['lang_button'])

def toggle_language():
    lang_var.set('EN' if lang_var.get() == 'TH' else 'TH')
    apply_language()


root = tk.Tk()
root.title("Color Aimbot")
root.geometry('650x500')
style = ttk.Style(root)
try:
    style.theme_use('clam')
except Exception:
    pass

hex_var = tk.StringVar(value="#feffb2")
tol_h_var = tk.IntVar(value=10)
tol_s_var = tk.IntVar(value=60)
tol_v_var = tk.IntVar(value=60)
strength_var = tk.DoubleVar(value=1.0)
stability_var = tk.DoubleVar(value=smooth_alpha)
max_step_var = tk.IntVar(value=max_step_px)
deadzone_var = tk.IntVar(value=deadzone_px)
fov_var = tk.IntVar(value=fov_radius)
show_fov_var = tk.BooleanVar(value=False)
status_var = tk.StringVar(value='Stopped')
lang_var = tk.StringVar(value='TH')

root.columnconfigure(0, weight=1, uniform="group1")
root.columnconfigure(1, weight=1, uniform="group1")
root.rowconfigure(0, weight=1)

left_col = ttk.Frame(root, padding=(8, 8, 4, 8))
left_col.grid(row=0, column=0, sticky='nsew')
left_col.columnconfigure(0, weight=1)
left_col.rowconfigure(0, weight=1)
left_col.rowconfigure(1, weight=1)

right_col = ttk.Frame(root, padding=(4, 8, 8, 8))
right_col.grid(row=0, column=1, sticky='nsew')
right_col.columnconfigure(0, weight=1)
right_col.rowconfigure(0, weight=1)
right_col.rowconfigure(1, weight=1)


frame_color = ttk.LabelFrame(left_col, text='🎨 การตั้งค่าสี / พาเลต', padding=(10, 10))
frame_color.grid(row=0, column=0, rowspan=2, sticky='nsew', pady=(6, 12))
frame_color.columnconfigure(0, weight=1)
frame_color.columnconfigure(1, weight=1)

row1 = ttk.Frame(frame_color)
row1.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))
hex_label = ttk.Label(row1, text='รหัสสี:')
hex_label.pack(side='left', padx=(0, 4))
hex_entry = ttk.Entry(row1, textvariable=hex_var, width=10)
hex_entry.pack(side='left', padx=(0, 6))
apply_hex_btn = ttk.Button(row1, text='ใช้สี', command=set_from_hex, width=8)
apply_hex_btn.pack(side='left', padx=(0, 10))
swatch_canvas = tk.Canvas(row1, width=48, height=24, highlightthickness=1, highlightbackground="#999999")
swatch_canvas.pack(side='left')
update_swatch(swatch_canvas, hex_var.get())

palette_frame = ttk.Frame(frame_color)
palette_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
palette_frame.columnconfigure(0, weight=1)
palette_frame.rowconfigure(0, weight=1)

palette_list = tk.Listbox(palette_frame, selectmode="multiple", height=6)
palette_list.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
palette_list.bind('<<ListboxSelect>>', on_palette_select)
palette_list.bind('<Double-Button-1>', on_palette_double_click)
palette_canvas = tk.Canvas(palette_frame, width=48, height=120, highlightthickness=1, highlightbackground="#999999")
palette_canvas.grid(row=0, column=1, sticky='ns')

btns_palette = ttk.Frame(frame_color)
btns_palette.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10, 0))
choose_image_btn = ttk.Button(btns_palette, text='เลือกรูปภาพ', command=choose_image)
choose_image_btn.pack(side='left', fill='x', expand=True)
analyze_images_btn = ttk.Button(btns_palette, text='วิเคราะห์ images', command=auto_detect_colors_from_folder)
analyze_images_btn.pack(side='left', padx=(6, 0), fill='x', expand=True)
lock_colors_btn = ttk.Button(btns_palette, text='ล็อคสีที่เลือก', command=lock_selected_colors)
lock_colors_btn.pack(side='left', padx=(6, 0), fill='x', expand=True)


frame_params = ttk.LabelFrame(right_col, text='⚙️ พารามิเตอร์การทำงาน', padding=(10, 10))
frame_params.grid(row=0, column=0, sticky='new', pady=6)
frame_params.columnconfigure(1, weight=1)

def add_labeled_scale(parent, label_text, var, frm, to, resolution=None, **kw):
    row = ttk.Frame(parent)
    row.pack(fill='x', pady=6)
    row.columnconfigure(1, weight=1)
    lbl = ttk.Label(row, text=label_text, width=15)
    lbl.grid(row=0, column=0, sticky='w')
    s = ttk.Scale(row, from_=frm, to=to, variable=var, command=update_params)
    s.grid(row=0, column=1, sticky='ew', padx=6)
    val_lbl = ttk.Label(row, textvariable=var, width=5, anchor='e')
    val_lbl.grid(row=0, column=2, sticky='e')
    return lbl

label_lock = add_labeled_scale(frame_params, 'แรงดูด (Lock)', strength_var, 0.5, 3.0)
label_smooth = add_labeled_scale(frame_params, 'ความนิ่ง (Smooth)', stability_var, 0.05, 1.00)
label_max_step = add_labeled_scale(frame_params, 'Max Step', max_step_var, 1, 20)
label_deadzone = add_labeled_scale(frame_params, 'Deadzone', deadzone_var, 0, 15)
label_fov = add_labeled_scale(frame_params, 'FOV Radius', fov_var, 30, 140)


frame_controls = ttk.LabelFrame(right_col, text='🕹️ ควบคุม / สถานะ', padding=(10, 10))
frame_controls.grid(row=1, column=0, sticky='sew', pady=6)
frame_controls.columnconfigure(0, weight=1)

start_btn = ttk.Button(frame_controls, text='✅ เริ่มการทำงาน', command=on_start)
start_btn.grid(row=0, column=0, sticky='ew', pady=(0, 6))
stop_btn = ttk.Button(frame_controls, text='❌ หยุดการทำงาน', command=stop_worker)
stop_btn.grid(row=1, column=0, sticky='ew', pady=(0, 10))

show_fov_chk = ttk.Checkbutton(frame_controls, text='แสดง FOV Overlay', variable=show_fov_var, command=ensure_overlay)
show_fov_chk.grid(row=2, column=0, sticky='w', pady=(0, 10))

frame_status = ttk.LabelFrame(frame_controls, text='สถานะ', padding=(10, 10))
frame_status.grid(row=3, column=0, sticky='ew', pady=(0, 10))
status_label = ttk.Label(frame_status, textvariable=status_var)
status_label.pack(fill='x', expand=True, padx=5)

lang_btn = ttk.Button(frame_controls, text='ภาษา', command=toggle_language)
lang_btn.grid(row=4, column=0, sticky='ew', pady=(0, 0))


apply_language() 
root.mainloop()
