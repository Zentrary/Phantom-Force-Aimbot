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
import os
import json
import customtkinter

from tkinter import filedialog, simpledialog, messagebox


class Config:
    def __init__(self):
        try:
            self.width = win32api.GetSystemMetrics(0)
            self.height = win32api.GetSystemMetrics(1)
        except Exception:
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
kernel = np.ones((3, 3), np.uint8)
lower_hsv = np.array([0, 160, 160], dtype=np.uint8)
upper_hsv = np.array([10, 255, 255], dtype=np.uint8)
min_area = 20
max_area = 4000
min_saturation_floor = 180
min_value_floor = 200
shape_filter_enabled = True
shape_min_aspect = 0.15
shape_max_aspect = 3.0
shape_min_solidity = 0.55
shape_min_extent = 0.20
context_check_enabled = True
context_radius = 25
context_inner_radius = 8
context_ratio_pct = 25
context_cyan_h_lo = 85
context_cyan_h_hi = 135
context_cyan_s_min = 120
context_cyan_v_min = 100
context_dark_v_max = 60
adaptive_hsv_enabled = False
adaptive_ranges = []
adaptive_cycle_frames = 30
adaptive_last_cycle = 0
adaptive_samples = []
adaptive_min_sat = 200
adaptive_min_samples = 120
crosshairU = config.crosshairUniform
regionC = config.region
robloxSensitivity = 0.55
PF_MouseSensitivity = 0.5
PF_AimSensitivity = 1.0
movementCompensation = 0.0
PF_sensitivity = PF_MouseSensitivity * PF_AimSensitivity
finalComputerSensitivityMultiplier = ((robloxSensitivity * PF_sensitivity) / 0.55) + movementCompensation
deadzone_px = 4
max_step_px = 6
smooth_alpha = 0.18
ema_dx = 0.0
ema_dy = 0.0
sub_dx = 0.0
sub_dy = 0.0
kp = 0.45
kd = 0.25
prev_err_x = 0.0
prev_err_y = 0.0
lead_frames = 1.5
w_area = 0.20
w_dist = 0.60
w_stick = 0.40
switch_cooldown_ms = 120
hysteresis_pct = 0.15
max_tracks = 8
track_miss_limit = 20
kalman_Q = 0.05
kalman_R = 0.5
tracks = {}
next_track_id = 1
primary_track_id = None
last_switch_time = 0.0
last_lock_cx = None
last_lock_cy = None
lock_velocity = (0.0, 0.0)
roi_radius = 50
lost_frames = 0
lost_threshold = 8
aim_enabled = True
aim_key = 0x10
prev_aim_state = 0
fov_radius = 70
lock_strength = 1.0
offset_x = 0
offset_y = 0
active_ranges = []
active_hexes = []
running = False
worker = None
lock_cx = None
lock_cy = None
overlay = None
overlay_canvas = None
status_lock_hex = "-"
status_ranges = "0"
status_contours = "-"
status_target = "-"
status_err = "-, -"
status_hit = "-"
status_loop_hz = "0"
status_tracks = "0"
status_ctx = "-"
debug_mask_visible = False

class Kalman2D:
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 100.0
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4, dtype=float) * kalman_Q
        self.R = np.eye(2, dtype=float) * kalman_R
        self.initialized = False

    def init(self, x, y):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return 0.0, 0.0, 0.0, 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

    def update(self, zx, zy):
        if not self.initialized:
            self.init(zx, zy)
            return zx, zy
        z = np.array([[zx], [zy]], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float(self.x[0, 0]), float(self.x[1, 0])
        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])

    def pos(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    def vel(self):
        return float(self.x[2, 0]), float(self.x[3, 0])


class Track:
    __slots__ = ("id", "kalman", "hits", "miss", "last_cx", "last_cy", "last_seen", "confidence")

    def __init__(self, tid, cx, cy, frame_idx):
        self.id = tid
        self.kalman = Kalman2D()
        self.kalman.init(cx, cy)
        self.hits = 1
        self.miss = 0
        self.last_cx = cx
        self.last_cy = cy
        self.last_seen = frame_idx
        self.confidence = 1.0


def round_to_2(value):
    return round(float(value), 2)


def format_value(value, digits=2):
    return f"{round_to_2(value):.{digits}f}"


def build_mask(frame_hsv):
    m = None
    for lo, up in active_ranges:
        mm = cv2.inRange(frame_hsv, lo, up)
        m = mm if m is None else cv2.bitwise_or(m, mm)
    for lo, up in adaptive_ranges:
        mm = cv2.inRange(frame_hsv, lo, up)
        m = mm if m is None else cv2.bitwise_or(m, mm)
    if m is None:
        m = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)
    return m


def hex_to_bgr(s):
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6:
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (b, g, r)
        except ValueError:
            return None
    return None


def range_from_hex(s, tol_h=10, tol_s=60, tol_v=60, with_floor=True):
    bgr = hex_to_bgr(s)
    if bgr is None:
        return None
    pix = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
    h = int(hsv[0])
    s_ = int(hsv[1])
    v = int(hsv[2])
    s_lo = max(s_ - tol_s, 0)
    v_lo = max(v - tol_v, 0)
    if with_floor:
        s_lo = max(s_lo, min_saturation_floor)
        v_lo = max(v_lo, min_value_floor)
    lo = np.array([max(h - tol_h, 0), s_lo, v_lo], dtype=np.uint8)
    up = np.array([min(h + tol_h, 179), min(s_ + tol_s, 255), min(v + tol_v, 255)], dtype=np.uint8)
    return lo, up


def range_from_hex_bright(s, tol_h=8):
    bgr = hex_to_bgr(s)
    if bgr is None:
        return None
    pix = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
    h = int(hsv[0])
    lo = np.array([max(h - tol_h, 0), min_saturation_floor, min_value_floor], dtype=np.uint8)
    up = np.array([min(h + tol_h, 179), 255, 255], dtype=np.uint8)
    return lo, up


def color_is_marker_grade(hex_str):
    bgr = hex_to_bgr(hex_str)
    if bgr is None:
        return False
    pix = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
    s_ = int(hsv[1])
    v = int(hsv[2])
    return s_ >= 170 and v >= 200


def analyze_image_colors(path, k=5):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = center.astype(np.uint8)
    res = []
    for c in centers:
        r, g, b = int(c[2]), int(c[1]), int(c[0])
        res.append("#%02X%02X%02X" % (r, g, b))
    return res


def analyze_folder_colors(folder, k=8, max_images=50, sample_per_image=4000):
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
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
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        flat = img.reshape((-1, 3))
        n = flat.shape[0]
        if n > sample_per_image:
            idx = np.random.choice(n, sample_per_image, replace=False)
            samples.append(flat[idx])
        else:
            samples.append(flat)
    if not samples:
        return []
    Z = np.vstack(samples).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = center.astype(np.uint8)
    lbl = label.ravel()
    counts = np.bincount(lbl, minlength=k)
    order = np.argsort(-counts)
    res = []
    for i in order:
        c = centers[i]
        r, g, b = int(c[2]), int(c[1]), int(c[0])
        res.append("#%02X%02X%02X" % (r, g, b))
    return res


def rebuild_active_ranges_from_hexes():
    global active_ranges
    active_ranges = []
    for hx in active_hexes:
        r1 = range_from_hex(hx, tol_h_var.get(), tol_s_var.get(), tol_v_var.get(), with_floor=True)
        if r1:
            active_ranges.append(r1)
        r2 = range_from_hex_bright(hx)
        if r2:
            active_ranges.append(r2)


def get_key_name(key_code):
    if key_code == 0x01:
        return "Left Mouse"
    if key_code == 0x02:
        return "Right Mouse"
    if key_code == 0x04:
        return "Middle Mouse"
    if key_code == 0x05:
        return "Mouse 4"
    if key_code == 0x06:
        return "Mouse 5"
    if key_code == 0x10:
        return "Left Shift"
    if key_code == 0x11:
        return "Left Ctrl"
    if key_code == 0x12:
        return "Left Alt"
    if key_code == 0x20:
        return "Space"
    if key_code == 0x0D:
        return "Enter"
    if key_code == 0x1B:
        return "Escape"
    if key_code == 0x09:
        return "Tab"
    if key_code == 0x14:
        return "Caps Lock"
    if key_code == 0x08:
        return "Backspace"
    if key_code == 0x2E:
        return "Delete"
    if key_code == 0x2D:
        return "Insert"
    if key_code == 0x24:
        return "Home"
    if key_code == 0x23:
        return "End"
    if key_code == 0x21:
        return "Page Up"
    if key_code == 0x22:
        return "Page Down"
    if key_code == 0x25:
        return "Left Arrow"
    if key_code == 0x26:
        return "Up Arrow"
    if key_code == 0x27:
        return "Right Arrow"
    if key_code == 0x28:
        return "Down Arrow"
    if 0x70 <= key_code <= 0x7B:
        return f"F{key_code - 0x6F}"
    if key_code == 0xA0:
        return "Left Shift"
    if key_code == 0xA1:
        return "Right Shift"
    if key_code == 0xA2:
        return "Left Ctrl"
    if key_code == 0xA3:
        return "Right Ctrl"
    if key_code == 0xA4:
        return "Left Alt"
    if key_code == 0xA5:
        return "Right Alt"
    if 0x60 <= key_code <= 0x69:
        return f"Numpad {key_code - 0x60}"
    if 0x30 <= key_code <= 0x39:
        return chr(key_code)
    if 0x41 <= key_code <= 0x5A:
        return chr(key_code)
    return f"Key 0x{key_code:02X}"


KEYSYM_TO_VK = {
    "Shift_L": 0xA0,
    "Shift_R": 0xA1,
    "Control_L": 0xA2,
    "Control_R": 0xA3,
    "Alt_L": 0xA4,
    "Alt_R": 0xA5,
    "Caps_Lock": 0x14,
    "Escape": 0x1B,
    "Return": 0x0D,
    "BackSpace": 0x08,
    "Tab": 0x09,
    "space": 0x20,
    "Delete": 0x2E,
    "Insert": 0x2D,
    "Home": 0x24,
    "End": 0x23,
    "Prior": 0x21,
    "Next": 0x22,
    "Left": 0x25,
    "Up": 0x26,
    "Right": 0x27,
    "Down": 0x28,
}


def _add_keysym_maps():
    for i in range(1, 13):
        KEYSYM_TO_VK[f"F{i}"] = 0x6F + i
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        KEYSYM_TO_VK[c.lower()] = ord(c)
        KEYSYM_TO_VK[c] = ord(c)
    for c in "0123456789":
        KEYSYM_TO_VK[c] = ord(c)


_add_keysym_maps()


def shape_ok(contour, area):
    if not shape_filter_enabled:
        return True
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return False
    aspect = w / float(h)
    if aspect < shape_min_aspect or aspect > shape_max_aspect:
        return False
    extent = area / float(w * h)
    if extent < shape_min_extent:
        return False
    try:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < shape_min_solidity:
                return False
    except Exception:
        pass
    return True


def context_ok(frame_hsv, cx, cy):
    if not context_check_enabled:
        return True
    h, w = frame_hsv.shape[:2]
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    y0 = max(0, cy_i - context_radius)
    y1 = min(h, cy_i + context_radius + 1)
    x0 = max(0, cx_i - context_radius)
    x1 = min(w, cx_i + context_radius + 1)
    if (x1 - x0) < 6 or (y1 - y0) < 6:
        return False
    patch = frame_hsv[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    ccx = cx_i - x0
    ccy = cy_i - y0
    yy, xx = np.ogrid[:ph, :pw]
    d2 = (xx - ccx) ** 2 + (yy - ccy) ** 2
    annulus = (d2 >= context_inner_radius * context_inner_radius) & (d2 <= context_radius * context_radius)
    total = int(annulus.sum())
    if total == 0:
        return False
    Hc = patch[:, :, 0]
    Sc = patch[:, :, 1]
    Vc = patch[:, :, 2]
    cyan = (Hc >= context_cyan_h_lo) & (Hc <= context_cyan_h_hi) & (Sc >= context_cyan_s_min) & (Vc >= context_cyan_v_min)
    dark = Vc <= context_dark_v_max
    hits = int(((cyan | dark) & annulus).sum())
    ratio = hits / float(total)
    return ratio >= (context_ratio_pct / 100.0)


def adaptive_sample_update(frame_hsv, contour):
    if not adaptive_hsv_enabled or contour is None:
        return
    try:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3:
            return
        cx = x + w // 2
        cy = y + h // 2
        r = 3
        y0 = max(0, cy - r)
        y1 = min(frame_hsv.shape[0], cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(frame_hsv.shape[1], cx + r + 1)
        patch = frame_hsv[y0:y1, x0:x1].reshape(-1, 3)
        sel = patch[patch[:, 1] >= adaptive_min_sat]
        if sel.shape[0] > 0:
            adaptive_samples.append(sel.astype(np.float32))
    except Exception:
        pass


def adaptive_cycle_commit():
    global adaptive_samples, adaptive_ranges
    if not adaptive_hsv_enabled:
        adaptive_samples = []
        adaptive_ranges = []
        return
    if not adaptive_samples:
        return
    try:
        all_s = np.vstack(adaptive_samples)
        if all_s.shape[0] < adaptive_min_samples:
            adaptive_samples = []
            return
        h = np.median(all_s[:, 0])
        s_ = np.median(all_s[:, 1])
        v = np.median(all_s[:, 2])
        h_std = max(4.0, float(np.std(all_s[:, 0])))
        s_std = max(20.0, float(np.std(all_s[:, 1])))
        v_std = max(20.0, float(np.std(all_s[:, 2])))
        tol_h = min(15, int(h_std * 2))
        tol_s = min(100, int(s_std * 2))
        tol_v = min(100, int(v_std * 2))
        lo = np.array([max(int(h) - tol_h, 0), max(int(s_) - tol_s, 0), max(int(v) - tol_v, 0)], dtype=np.uint8)
        up = np.array([min(int(h) + tol_h, 179), min(int(s_) + tol_s, 255), min(int(v) + tol_v, 255)], dtype=np.uint8)
        adaptive_ranges = [(lo, up)]
    except Exception:
        pass
    adaptive_samples = []


def prune_tracks(frame_idx):
    global tracks, primary_track_id
    dead = [tid for tid, t in tracks.items() if t.miss > track_miss_limit]
    for tid in dead:
        del tracks[tid]
    if primary_track_id is not None and primary_track_id not in tracks:
        primary_track_id = None


def match_and_update_tracks(candidates, frame_idx):
    global tracks, next_track_id
    for t in tracks.values():
        t.kalman.predict()
    used_tracks = set()
    results = []
    for ci, (cx, cy, area, contour) in enumerate(candidates):
        best_tid = None
        best_d2 = float("inf")
        for tid, t in tracks.items():
            if tid in used_tracks:
                continue
            px, py = t.kalman.pos()
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_d2 and d2 <= (roi_radius * 2) ** 2:
                best_d2 = d2
                best_tid = tid
        if best_tid is not None:
            t = tracks[best_tid]
            t.kalman.update(cx, cy)
            t.hits += 1
            t.miss = 0
            t.last_cx = cx
            t.last_cy = cy
            t.last_seen = frame_idx
            t.confidence = min(1.0, t.confidence + 0.15)
            used_tracks.add(best_tid)
            results.append((t, ci, best_d2, area))
        else:
            if len(tracks) < max_tracks:
                t = Track(next_track_id, cx, cy, frame_idx)
                tracks[next_track_id] = t
                used_tracks.add(next_track_id)
                results.append((t, ci, 0.0, area))
                next_track_id += 1
            else:
                results.append((None, ci, float("inf"), area))
    for tid, t in tracks.items():
        if tid not in used_tracks:
            t.miss += 1
            t.confidence = max(0.0, t.confidence - 0.08)
    return results


def score_candidate(track, cx, cy, area, primary_id, d2_to_crosshair):
    area_n = min(1.0, area / float(max_area))
    dist_n = 1.0 - min(1.0, (d2_to_crosshair**0.5) / max(1.0, float(fov_radius)))
    stick_n = 1.0 if (primary_id is not None and track.id == primary_id) else 0.0
    conf_n = track.confidence if track else 0.3
    return w_area * area_n + max(w_dist, 0.05) * dist_n + w_stick * stick_n + 0.2 * conf_n


pick_pixel_active = [False]
pick_pixel_countdown = [0]


def start_pick_pixel():
    if pick_pixel_active[0]:
        return
    pick_pixel_active[0] = True
    pick_pixel_countdown[0] = 3
    pick_status_lbl.configure(text="Move mouse to marker...")
    _pick_tick()


def _pick_tick():
    if not pick_pixel_active[0]:
        return
    n = pick_pixel_countdown[0]
    if n > 0:
        pick_status_lbl.configure(text=f"PICK in {n}...")
        pick_pixel_countdown[0] = n - 1
        root.after(1000, _pick_tick)
    else:
        _pick_capture()


def _pick_capture():
    pick_pixel_active[0] = False
    try:
        x, y = win32api.GetCursorPos()
    except Exception:
        pick_status_lbl.configure(text="cursor error")
        return
    try:
        with mss.mss() as sct:
            mon = {"top": int(y), "left": int(x), "width": 1, "height": 1}
            img = np.array(sct.grab(mon))
    except Exception as e:
        pick_status_lbl.configure(text=f"grab error: {e}")
        return

    b = int(img[0, 0, 0])
    g = int(img[0, 0, 1])
    r = int(img[0, 0, 2])
    hex_str = f"#{r:02X}{g:02X}{b:02X}"
    pix = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
    h = int(hsv[0])
    s_ = int(hsv[1])
    v = int(hsv[2])

    hex_var.set(hex_str)
    hex_var_pending = hex_str.upper()

    tol_h_var.set(5)
    tol_s_var.set(30)
    tol_v_var.set(30)

    new_s_floor = max(0, min(255, int(s_ * 0.85)))
    new_v_floor = max(0, min(255, int(v * 0.85)))
    s_floor_var.set(new_s_floor)
    v_floor_var.set(new_v_floor)

    global min_saturation_floor, min_value_floor
    min_saturation_floor = new_s_floor
    min_value_floor = new_v_floor

    active_hexes.clear()
    active_hexes.append(hex_var_pending)
    rebuild_active_ranges_from_hexes()
    update_swatch(swatch_canvas, hex_str)
    update_params()

    pick_status_lbl.configure(text=f"OK H={h} S={s_} V={v}")
    try:
        winsound.Beep(1400, 80)
    except Exception:
        pass


def run_loop():
    global running, prev_aim_state, ema_dx, ema_dy, sub_dx, sub_dy
    global prev_err_x, prev_err_y, lock_cx, lock_cy, aim_enabled, lost_frames
    global status_lock_hex, status_ranges, status_contours, status_target
    global status_err, status_hit, status_loop_hz, aim_key, status_tracks
    global status_ctx
    global tracks, primary_track_id, last_switch_time
    global adaptive_last_cycle, adaptive_samples, last_lock_cx, last_lock_cy, lock_velocity

    running = True
    if hasattr(mss, "MSS"):
        s = mss.MSS()
    else:
        s = mss.mss()

    t_last = time.time()
    loop_count = 0
    frame_idx = 0

    while running:
        time.sleep(0.001)
        loop_count += 1
        frame_idx += 1

        now = time.time()
        if now - t_last >= 0.5:
            status_loop_hz = f"{loop_count / (now - t_last):.0f}"
            loop_count = 0
            t_last = now

        status_ranges = str(len(active_ranges) + len(adaptive_ranges))
        status_lock_hex = active_hexes[0] if active_hexes else "-"
        status_tracks = str(len(tracks))

        try:
            GameFrame = np.array(s.grab(regionC))
        except Exception:
            time.sleep(0.01)
            continue

        GameFrame = cv2.cvtColor(GameFrame, cv2.COLOR_BGRA2BGR)

        aim_state = win32api.GetAsyncKeyState(aim_key)

        tk_state = win32api.GetAsyncKeyState(0x77)
        if tk_state < 0 and prev_aim_state >= 0:
            aim_enabled = not aim_enabled
            try:
                winsound.Beep(1200 if aim_enabled else 800, 80)
            except Exception:
                pass
        prev_aim_state = tk_state

        if win32api.GetAsyncKeyState(0x6) < 0:
            break

        if not (aim_enabled and aim_state < 0):
            if not aim_enabled:
                status_target = "aim off (F8)"
            else:
                status_target = f"idle (hold {get_key_name(aim_key)})"
            status_contours = "-"
            status_hit = "-"
            status_err = "-, -"
            status_ctx = "-"
            sub_dx *= 0.5
            sub_dy *= 0.5
            for t in list(tracks.values()):
                t.miss += 3
            prune_tracks(frame_idx)
            continue

        frame_hsv = cv2.cvtColor(GameFrame, cv2.COLOR_BGR2HSV)
        mask = build_mask(frame_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        status_contours = str(len(contours))

        if debug_mask_visible:
            try:
                cv2.imshow("mask", mask)
                cv2.waitKey(1)
            except Exception:
                pass

        if not contours:
            lost_frames += 1
            status_hit = "0"
            status_ctx = "0"
            status_target = "no contour"
            if primary_track_id is not None and primary_track_id in tracks and lost_frames <= track_miss_limit:
                t = tracks[primary_track_id]
                px, py, vx, vy = t.kalman.predict()
                lock_cx, lock_cy = px, py
                lock_velocity = (vx, vy)
            if lost_frames > lost_threshold:
                prev_err_x = prev_err_y = 0.0
                ema_dx = ema_dy = 0.0
                sub_dx = sub_dy = 0.0
            prune_tracks(frame_idx)
            continue

        raw_centroids = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:
                continue
            if not shape_ok(c, a):
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx_i = M["m10"] / M["m00"]
            cy_i = M["m01"] / M["m00"]
            if (cx_i - crosshairU) ** 2 + (cy_i - crosshairU) ** 2 <= fov_radius * fov_radius:
                raw_centroids.append((cx_i, cy_i, a, c))

        centroids = []
        for cx_i, cy_i, a, c in raw_centroids:
            if context_ok(frame_hsv, cx_i, cy_i):
                centroids.append((cx_i, cy_i, a, c))
        status_ctx = f"{len(centroids)}/{len(raw_centroids)}"

        if not centroids:
            lost_frames += 1
            status_hit = "0"
            status_target = "no candidate"
            if primary_track_id is not None and primary_track_id in tracks and lost_frames <= track_miss_limit:
                t = tracks[primary_track_id]
                px, py, vx, vy = t.kalman.predict()
                lock_cx, lock_cy = px, py
                lock_velocity = (vx, vy)
            if lost_frames > lost_threshold:
                prev_err_x = prev_err_y = 0.0
                ema_dx = ema_dy = 0.0
                sub_dx = sub_dy = 0.0
            prune_tracks(frame_idx)
            continue

        status_hit = str(len(centroids))
        lost_frames = 0

        matches = match_and_update_tracks(centroids, frame_idx)

        scored = []
        for t, ci, d2, area in matches:
            if t is None:
                continue
            cx, cy, area2, c = centroids[ci]
            d2c = (cx - crosshairU) ** 2 + (cy - crosshairU) ** 2
            sc = score_candidate(t, cx, cy, area2, primary_track_id, d2c)
            scored.append((sc, t, cx, cy, c))

        if not scored:
            prune_tracks(frame_idx)
            continue

        scored.sort(key=lambda x: -x[0])
        best_sc, best_track, best_cx, best_cy, best_c = scored[0]

        chosen_track = best_track
        if primary_track_id is not None and primary_track_id in tracks:
            current_sc = None
            for sc, t, cx, cy, c in scored:
                if t.id == primary_track_id:
                    current_sc = sc
                    break
            if current_sc is not None:
                if best_track.id != primary_track_id and best_sc <= current_sc * (1.0 + hysteresis_pct):
                    chosen_track = tracks[primary_track_id]
                    for sc, t, cx, cy, c in scored:
                        if t.id == primary_track_id:
                            best_cx, best_cy, best_c = cx, cy, c
                            break
                elif best_track.id != primary_track_id and (now - last_switch_time) * 1000.0 < switch_cooldown_ms:
                    chosen_track = tracks[primary_track_id]
                    for sc, t, cx, cy, c in scored:
                        if t.id == primary_track_id:
                            best_cx, best_cy, best_c = cx, cy, c
                            break
                elif best_track.id != primary_track_id:
                    last_switch_time = now

        primary_track_id = chosen_track.id

        adaptive_sample_update(frame_hsv, best_c)
        if adaptive_hsv_enabled and (frame_idx - adaptive_last_cycle) >= adaptive_cycle_frames:
            adaptive_cycle_commit()
            adaptive_last_cycle = frame_idx

        vx, vy = chosen_track.kalman.vel()
        lock_velocity = (vx, vy)
        cx = best_cx + vx * float(lead_frames)
        cy = best_cy + vy * float(lead_frames)

        track_cx, track_cy = cx, cy
        lock_cx, lock_cy = cx, cy
        last_lock_cx, last_lock_cy = cx, cy

        target_x = cx + offset_x
        target_y = cy + offset_y

        err_x = -(crosshairU - target_x)
        err_y = -(crosshairU - target_y)

        if abs(err_x) < deadzone_px:
            err_x = 0.0
        if abs(err_y) < deadzone_px:
            err_y = 0.0

        cross_x = np.sign(err_x) != np.sign(prev_err_x)
        cross_y = np.sign(err_y) != np.sign(prev_err_y)
        scale_x = np.tanh(abs(err_x) / 10.0)
        scale_y = np.tanh(abs(err_y) / 10.0)

        finalMult = finalComputerSensitivityMultiplier * lock_strength
        dx_raw = (kp * err_x + kd * (err_x - prev_err_x)) * finalMult * scale_x
        dy_raw = (kp * err_y + kd * (err_y - prev_err_y)) * finalMult * scale_y

        if cross_x:
            dx_raw *= 0.5
            ema_dx = 0.0
        if cross_y:
            dy_raw *= 0.5
            ema_dy = 0.0

        dx_raw = float(np.clip(dx_raw, -max_step_px, max_step_px))
        dy_raw = float(np.clip(dy_raw, -max_step_px, max_step_px))

        ema_dx = (1 - smooth_alpha) * ema_dx + smooth_alpha * dx_raw
        ema_dy = (1 - smooth_alpha) * ema_dy + smooth_alpha * dy_raw

        sub_dx += ema_dx
        sub_dy += ema_dy
        move_x = int(sub_dx)
        move_y = int(sub_dy)
        sub_dx -= move_x
        sub_dy -= move_y

        if move_x or move_y:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)

        prev_err_x = err_x
        prev_err_y = err_y

        status_target = f"id{chosen_track.id} {int(cx)},{int(cy)}"
        status_err = f"{err_x:+.1f}, {err_y:+.1f}"

        prune_tracks(frame_idx)

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def update_swatch(canvas, hex_str):
    try:
        canvas.delete("all")
        w = int(canvas.cget("width"))
        h = int(canvas.cget("height"))
        canvas.create_rectangle(0, 0, w - 1, h - 1, fill=hex_str, outline="#000000")
    except Exception:
        pass


def update_palette_canvas(cols):
    try:
        palette_canvas.delete("all")
        h = int(palette_canvas.winfo_height() or 200)
        n = max(1, len(cols))
        bar_h = max(10, h // n)
        for i, hx in enumerate(cols):
            palette_canvas.create_rectangle(0, i * bar_h, 40, (i + 1) * bar_h, fill=hx, outline="")
    except Exception:
        pass


def on_palette_select(event):
    try:
        sel = palette_list.curselection()
        if sel:
            update_swatch(swatch_canvas, palette_list.get(sel[-1]))
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


def run_worker():
    global worker
    if worker and worker.is_alive():
        return
    worker = threading.Thread(target=run_loop, daemon=True)
    worker.start()
    status_var.set("RUNNING")
    top_status_var.set("RUNNING")


def stop_worker():
    global running
    running = False
    status_var.set("STOPPED")
    top_status_var.set("STOPPED")
    hide_overlay()


def on_start():
    root.update_idletasks()
    ensure_overlay()
    run_worker()


def set_from_hex():
    s = hex_var.get()
    r = range_from_hex(s, tol_h_var.get(), tol_s_var.get(), tol_v_var.get(), with_floor=True)
    if r:
        active_hexes.clear()
        active_hexes.append(s.upper())
        rebuild_active_ranges_from_hexes()
        update_swatch(swatch_canvas, s)


def choose_image():
    p = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")])
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
    chosen = []
    for i in sel:
        hx = palette_list.get(i).upper()
        if color_is_marker_grade(hx):
            chosen.append(hx)
    if not chosen:
        chosen = [palette_list.get(i).upper() for i in sel]
    active_hexes.clear()
    active_hexes.extend(chosen)
    rebuild_active_ranges_from_hexes()
    if chosen:
        update_swatch(swatch_canvas, chosen[-1])


def auto_detect_colors_from_folder():
    cols = analyze_folder_colors("images", k=8)
    if not cols:
        return
    filtered = [c for c in cols if color_is_marker_grade(c)]
    if not filtered:
        filtered = cols[:1]
    palette_list.delete(0, tk.END)
    for hx in filtered:
        palette_list.insert(tk.END, hx)
    update_palette_canvas(filtered)


def update_params(*args):
    global lock_strength, smooth_alpha, max_step_px, deadzone_px, fov_radius
    global offset_x, offset_y
    global robloxSensitivity, PF_MouseSensitivity, PF_AimSensitivity
    global PF_sensitivity, finalComputerSensitivityMultiplier
    global lead_frames, w_area, w_dist, w_stick, switch_cooldown_ms, hysteresis_pct
    global shape_filter_enabled, adaptive_hsv_enabled, adaptive_ranges
    global context_check_enabled, context_radius, context_ratio_pct
    global min_saturation_floor, min_value_floor

    lock_strength = round_to_2(strength_var.get())
    smooth_alpha = max(0.01, 1.0 - round_to_2(stability_var.get()))
    max_step_px = int(max_step_var.get())
    deadzone_px = int(deadzone_var.get())
    fov_radius = int(fov_var.get())
    offset_x = int(offset_x_var.get())
    offset_y = int(offset_y_var.get())

    robloxSensitivity = roblox_sens_var.get()
    PF_MouseSensitivity = pf_mouse_var.get()
    PF_AimSensitivity = pf_aim_var.get()
    PF_sensitivity = PF_MouseSensitivity * PF_AimSensitivity
    finalComputerSensitivityMultiplier = ((robloxSensitivity * PF_sensitivity) / 0.55) + movementCompensation

    lead_frames = float(lead_var.get())
    w_area = float(w_area_var.get())
    w_dist = float(w_dist_var.get())
    w_stick = float(w_stick_var.get())
    switch_cooldown_ms = int(cooldown_var.get())
    hysteresis_pct = float(hyst_var.get()) / 100.0

    shape_filter_enabled = bool(shape_filter_var.get())
    new_adaptive = bool(adaptive_var.get())
    if new_adaptive != adaptive_hsv_enabled:
        adaptive_hsv_enabled = new_adaptive
        if not adaptive_hsv_enabled:
            adaptive_ranges = []

    context_check_enabled = bool(context_var.get())
    context_radius = int(context_radius_var.get())
    context_ratio_pct = int(context_ratio_var.get())

    min_saturation_floor = int(s_floor_var.get())
    min_value_floor = int(v_floor_var.get())

    rebuild_active_ranges_from_hexes()
    update_overlay()


def startup_image_scan():
    auto_detect_colors_from_folder()
    if not active_hexes:
        set_from_hex()


COLORKEY_HEX = "#010203"
COLORKEY_REF = 0x030201


def _apply_click_through(hwnd, colorkey_ref):
    ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    ex |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_NOACTIVATE | 0x00000080
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex)
    win32gui.SetLayeredWindowAttributes(hwnd, colorkey_ref, 0, win32con.LWA_COLORKEY)


def _make_overlay():
    global overlay, overlay_canvas
    overlay = tk.Toplevel(root)
    overlay.overrideredirect(True)
    overlay.attributes("-topmost", True)
    try:
        overlay.attributes("-toolwindow", True)
    except Exception:
        pass

    transparent_ok = False
    try:
        overlay.attributes("-transparentcolor", COLORKEY_HEX)
        overlay.configure(bg=COLORKEY_HEX)
        transparent_ok = True
    except Exception:
        overlay.configure(bg="black")

    overlay_canvas = tk.Canvas(overlay, highlightthickness=0, bd=0, bg=COLORKEY_HEX if transparent_ok else "black")
    overlay_canvas.pack(fill="both", expand=True)

    overlay.update_idletasks()
    overlay.deiconify()
    overlay.update_idletasks()
    overlay.update()

    try:
        hwnd = win32gui.GetParent(overlay.winfo_id())
        if hwnd == 0:
            hwnd = overlay.winfo_id()
        _apply_click_through(hwnd, COLORKEY_REF)
    except Exception:
        pass

    overlay.withdraw()


def ensure_overlay():
    global overlay
    if not show_fov_var.get() and not show_dot_var.get():
        hide_overlay()
        return
    if overlay is None or not overlay.winfo_exists():
        _make_overlay()
    update_overlay()


def update_overlay():
    global overlay, overlay_canvas
    if overlay is None or not overlay.winfo_exists():
        return
    show_fov = bool(show_fov_var.get())
    show_dot = bool(show_dot_var.get())
    if not show_fov and not show_dot:
        hide_overlay()
        return

    r = int(fov_var.get())
    d = 2 * r + 6
    x = config.center_x - d // 2
    y = config.center_y - d // 2

    overlay.geometry(f"{d}x{d}+{x}+{y}")
    overlay.deiconify()
    overlay.lift()
    overlay.attributes("-topmost", True)

    try:
        hwnd = win32gui.GetParent(overlay.winfo_id())
        if hwnd == 0:
            hwnd = overlay.winfo_id()
        _apply_click_through(hwnd, COLORKEY_REF)
    except Exception:
        pass

    overlay_canvas.delete("all")
    if show_fov:
        overlay_canvas.create_oval(3, 3, d - 3, d - 3, outline="#E94560", width=2)
    if show_dot:
        cx = d // 2
        cy = d // 2
        overlay_canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill="#39D98A", outline="")
        overlay_canvas.create_line(cx - 9, cy, cx - 3, cy, fill="#39D98A", width=1)
        overlay_canvas.create_line(cx + 3, cy, cx + 9, cy, fill="#39D98A", width=1)
        overlay_canvas.create_line(cx, cy - 9, cx, cy - 3, fill="#39D98A", width=1)
        overlay_canvas.create_line(cx, cy + 3, cx, cy + 9, fill="#39D98A", width=1)


def hide_overlay():
    global overlay
    try:
        if overlay and overlay.winfo_exists():
            overlay.withdraw()
    except Exception:
        pass


BG_ROOT = "#0E0E10"
BG_PANEL = "#16161A"
BG_PANEL_2 = "#1C1C22"
BG_INPUT = "#20202A"
ACCENT = "#E94560"
ACCENT_DIM = "#8A2A3A"
OK_GREEN = "#39D98A"
TXT = "#E0E0E0"
TXT_DIM = "#7A7A82"
BORDER = "#2A2A32"

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("dark-blue")

MONO = ("Consolas", 11)
MONO_B = ("Consolas", 11, "bold")
MONO_L = ("Consolas", 13, "bold")
MONO_S = ("Consolas", 10)

root = customtkinter.CTk()
root.title("Color aimbot")
root.geometry("840x700")
root.resizable(False, False)
root.configure(fg_color=BG_ROOT)

hex_var = customtkinter.StringVar(value="#FDFCB3")
tol_h_var = customtkinter.IntVar(value=8)
tol_s_var = customtkinter.IntVar(value=40)
tol_v_var = customtkinter.IntVar(value=40)
s_floor_var = customtkinter.IntVar(value=min_saturation_floor)
v_floor_var = customtkinter.IntVar(value=min_value_floor)
strength_var = customtkinter.DoubleVar(value=1.5)
stability_var = customtkinter.DoubleVar(value=0.35)
pf_mouse_var = customtkinter.DoubleVar(value=PF_MouseSensitivity)
pf_aim_var = customtkinter.DoubleVar(value=PF_AimSensitivity)
roblox_sens_var = customtkinter.DoubleVar(value=robloxSensitivity)
max_step_var = customtkinter.IntVar(value=max_step_px)
deadzone_var = customtkinter.IntVar(value=deadzone_px)
fov_var = customtkinter.IntVar(value=fov_radius)
offset_x_var = customtkinter.IntVar(value=0)
offset_y_var = customtkinter.IntVar(value=-2)
show_fov_var = customtkinter.BooleanVar(value=True)
show_dot_var = customtkinter.BooleanVar(value=False)
status_var = customtkinter.StringVar(value="STOPPED")
top_status_var = customtkinter.StringVar(value="STOPPED")

lead_var = customtkinter.DoubleVar(value=lead_frames)
w_area_var = customtkinter.DoubleVar(value=w_area)
w_dist_var = customtkinter.DoubleVar(value=w_dist)
w_stick_var = customtkinter.DoubleVar(value=w_stick)
cooldown_var = customtkinter.IntVar(value=switch_cooldown_ms)
hyst_var = customtkinter.IntVar(value=int(hysteresis_pct * 100))
shape_filter_var = customtkinter.BooleanVar(value=shape_filter_enabled)
adaptive_var = customtkinter.BooleanVar(value=adaptive_hsv_enabled)

context_var = customtkinter.BooleanVar(value=context_check_enabled)
context_radius_var = customtkinter.IntVar(value=context_radius)
context_ratio_var = customtkinter.IntVar(value=context_ratio_pct)

aim_key_var = customtkinter.IntVar(value=0x10)
aim_key_display_var = customtkinter.StringVar(value="Left Shift")

live_hex_var = customtkinter.StringVar(value="-")
live_ranges_var = customtkinter.StringVar(value="0")
live_contours_var = customtkinter.StringVar(value="-")
live_target_var = customtkinter.StringVar(value="-")
live_err_var = customtkinter.StringVar(value="-, -")
live_hit_var = customtkinter.StringVar(value="-")
live_hz_var = customtkinter.StringVar(value="0")
live_tracks_var = customtkinter.StringVar(value="0")
live_ctx_var = customtkinter.StringVar(value="-")

strength_display_var = customtkinter.StringVar(value=format_value(strength_var.get()))
stability_display_var = customtkinter.StringVar(value=format_value(stability_var.get()))
pf_mouse_display_var = customtkinter.StringVar(value=format_value(pf_mouse_var.get()))
pf_aim_display_var = customtkinter.StringVar(value=format_value(pf_aim_var.get()))
roblox_display_var = customtkinter.StringVar(value=format_value(roblox_sens_var.get()))
max_step_display_var = customtkinter.StringVar(value=str(int(max_step_var.get())))
deadzone_display_var = customtkinter.StringVar(value=str(int(deadzone_var.get())))
fov_display_var = customtkinter.StringVar(value=str(int(fov_var.get())))
offset_x_display_var = customtkinter.StringVar(value=str(int(offset_x_var.get())))
offset_y_display_var = customtkinter.StringVar(value=str(int(offset_y_var.get())))
lead_display_var = customtkinter.StringVar(value=format_value(lead_var.get()))
w_area_display_var = customtkinter.StringVar(value=format_value(w_area_var.get()))
w_dist_display_var = customtkinter.StringVar(value=format_value(w_dist_var.get()))
w_stick_display_var = customtkinter.StringVar(value=format_value(w_stick_var.get()))
cooldown_display_var = customtkinter.StringVar(value=str(int(cooldown_var.get())))
hyst_display_var = customtkinter.StringVar(value=str(int(hyst_var.get())))
ctx_radius_display_var = customtkinter.StringVar(value=str(int(context_radius_var.get())))
ctx_ratio_display_var = customtkinter.StringVar(value=str(int(context_ratio_var.get())))
s_floor_display_var = customtkinter.StringVar(value=str(int(s_floor_var.get())))
v_floor_display_var = customtkinter.StringVar(value=str(int(v_floor_var.get())))


def _bind_display(var, disp, fmt=".2f"):
    def _upd(*_):
        try:
            if fmt == "d":
                disp.set(str(int(var.get())))
            else:
                disp.set(format(float(var.get()), fmt))
        except Exception:
            pass

    var.trace_add("write", _upd)
    _upd()


_bind_display(strength_var, strength_display_var, ".2f")
_bind_display(stability_var, stability_display_var, ".2f")
_bind_display(pf_mouse_var, pf_mouse_display_var, ".2f")
_bind_display(pf_aim_var, pf_aim_display_var, ".2f")
_bind_display(roblox_sens_var, roblox_display_var, ".2f")
_bind_display(max_step_var, max_step_display_var, "d")
_bind_display(deadzone_var, deadzone_display_var, "d")
_bind_display(fov_var, fov_display_var, "d")
_bind_display(offset_x_var, offset_x_display_var, "d")
_bind_display(offset_y_var, offset_y_display_var, "d")
_bind_display(lead_var, lead_display_var, ".2f")
_bind_display(w_area_var, w_area_display_var, ".2f")
_bind_display(w_dist_var, w_dist_display_var, ".2f")
_bind_display(w_stick_var, w_stick_display_var, ".2f")
_bind_display(cooldown_var, cooldown_display_var, "d")
_bind_display(hyst_var, hyst_display_var, "d")
_bind_display(context_radius_var, ctx_radius_display_var, "d")
_bind_display(context_ratio_var, ctx_ratio_display_var, "d")
_bind_display(s_floor_var, s_floor_display_var, "d")
_bind_display(v_floor_var, v_floor_display_var, "d")

topbar = customtkinter.CTkFrame(root, fg_color=BG_PANEL, height=34, corner_radius=0)
topbar.pack(fill="x", side="top")
topbar.pack_propagate(False)

top_status_lbl = customtkinter.CTkLabel(topbar, textvariable=top_status_var, font=MONO_L, text_color=TXT_DIM)
top_status_lbl.pack(side="left", padx=12)

customtkinter.CTkLabel(topbar, textvariable=live_hz_var, font=MONO_S, text_color=TXT_DIM).pack(side="right", padx=(0, 4))
customtkinter.CTkLabel(topbar, text="hz", font=MONO_S, text_color=TXT_DIM).pack(side="right")
customtkinter.CTkLabel(topbar, text="|", font=MONO_S, text_color=BORDER).pack(side="right", padx=8)
customtkinter.CTkLabel(topbar, textvariable=live_hex_var, font=MONO_B, text_color=ACCENT).pack(side="right")

body = customtkinter.CTkFrame(root, fg_color=BG_ROOT, corner_radius=0)
body.pack(fill="both", expand=True, padx=6, pady=6)

sidebar = customtkinter.CTkFrame(body, fg_color=BG_PANEL, width=210, corner_radius=6)
sidebar.pack(side="left", fill="y", padx=(0, 6))
sidebar.pack_propagate(False)

sb_actions = customtkinter.CTkFrame(sidebar, fg_color=BG_PANEL_2, corner_radius=4)
sb_actions.pack(fill="x", padx=5, pady=(5, 4))

customtkinter.CTkButton(sb_actions, text="START", command=on_start, fg_color=OK_GREEN, hover_color="#2BB070", text_color="#000000", font=MONO_B, height=28).pack(fill="x", padx=5, pady=(5, 3))

customtkinter.CTkButton(sb_actions, text="STOP", command=stop_worker, fg_color=ACCENT, hover_color=ACCENT_DIM, text_color="#FFFFFF", font=MONO_B, height=28).pack(fill="x", padx=5, pady=(0, 5))

sb_live = customtkinter.CTkFrame(sidebar, fg_color=BG_PANEL_2, corner_radius=4)
sb_live.pack(fill="x", padx=5, pady=4)
sb_live.grid_columnconfigure(1, weight=1)

customtkinter.CTkLabel(sb_live, text="LOCK STATUS", font=MONO_B, text_color=ACCENT).grid(row=0, column=0, columnspan=2, sticky="w", padx=9, pady=(6, 3))

live_swatch = tk.Canvas(sb_live, width=190, height=6, highlightthickness=0, bg=BG_PANEL_2)
live_swatch.grid(row=1, column=0, columnspan=2, sticky="ew", padx=9, pady=(0, 6))


def live_row(parent, r_i, label, var):
    customtkinter.CTkLabel(parent, text=label, font=MONO_S, text_color=TXT_DIM, anchor="w").grid(row=r_i, column=0, sticky="w", padx=(9, 5), pady=1)
    customtkinter.CTkLabel(parent, textvariable=var, font=MONO_S, text_color=TXT, anchor="e").grid(row=r_i, column=1, sticky="e", padx=(5, 9), pady=1)


live_row(sb_live, 2, "hex", live_hex_var)
live_row(sb_live, 3, "ranges", live_ranges_var)
live_row(sb_live, 4, "contours", live_contours_var)
live_row(sb_live, 5, "raw/ctx", live_ctx_var)
live_row(sb_live, 6, "candidates", live_hit_var)
live_row(sb_live, 7, "tracks", live_tracks_var)
live_row(sb_live, 8, "target", live_target_var)
live_row(sb_live, 9, "err", live_err_var)

customtkinter.CTkFrame(sb_live, fg_color="transparent", height=4).grid(row=10, column=0, columnspan=2)

sb_vis = customtkinter.CTkFrame(sidebar, fg_color=BG_PANEL_2, corner_radius=4)
sb_vis.pack(fill="x", padx=5, pady=4)

customtkinter.CTkCheckBox(sb_vis, text=" FOV overlay", variable=show_fov_var, command=ensure_overlay, fg_color=ACCENT, hover_color=ACCENT_DIM, font=MONO_S, text_color=TXT, border_color=BORDER, checkmark_color="#FFFFFF").pack(anchor="w", padx=9, pady=(6, 3))

customtkinter.CTkCheckBox(sb_vis, text=" CENTER DOT", variable=show_dot_var, command=ensure_overlay, fg_color=ACCENT, hover_color=ACCENT_DIM, font=MONO_S, text_color=TXT, border_color=BORDER, checkmark_color="#FFFFFF").pack(anchor="w", padx=9, pady=(0, 6))

sb_footer = customtkinter.CTkFrame(sidebar, fg_color="transparent")
sb_footer.pack(side="bottom", fill="x", padx=5, pady=5)
customtkinter.CTkLabel(sb_footer, text="F8 toggle | END exit", font=MONO_S, text_color=TXT_DIM).pack()

tabs_frame = customtkinter.CTkFrame(body, fg_color=BG_PANEL, corner_radius=6)
tabs_frame.pack(side="left", fill="both", expand=True)

tabview = customtkinter.CTkTabview(tabs_frame, fg_color=BG_PANEL, segmented_button_fg_color=BG_PANEL_2, segmented_button_selected_color=ACCENT, segmented_button_selected_hover_color=ACCENT_DIM, segmented_button_unselected_color=BG_PANEL_2, segmented_button_unselected_hover_color=BG_INPUT, text_color=TXT, border_color=BORDER, border_width=1, corner_radius=6)
tabview.pack(fill="both", expand=True, padx=5, pady=5)

tab_aim = tabview.add("AIMBOT")
tab_track = tabview.add("TRACKING")
tab_color = tabview.add("COLOR")
tab_visual = tabview.add("VISUAL")
tab_config = tabview.add("CONFIG")

for tab in (tab_aim, tab_track, tab_color, tab_visual, tab_config):
    tab.configure(fg_color=BG_PANEL)


def make_slider(parent, label, var, frm, to, display_var=None, step=None, label_w=150):
    f = customtkinter.CTkFrame(parent, fg_color=BG_PANEL_2, corner_radius=4, height=30)
    f.pack(fill="x", padx=6, pady=2)
    f.pack_propagate(False)
    customtkinter.CTkLabel(f, text=label, font=MONO_B, text_color=TXT, width=label_w, anchor="w").pack(side="left", padx=(9, 5))
    kwargs = dict(variable=var, command=update_params, button_color=ACCENT, button_hover_color=ACCENT_DIM, progress_color=ACCENT, fg_color=BG_INPUT, height=12)
    if step is not None:
        kwargs["number_of_steps"] = step
    customtkinter.CTkSlider(f, from_=frm, to=to, **kwargs).pack(side="left", fill="x", expand=True, padx=2)
    if display_var is not None:
        customtkinter.CTkLabel(f, textvariable=display_var, font=MONO_S, text_color=ACCENT, width=45, anchor="e").pack(side="right", padx=(5, 9))


make_slider(tab_aim, "LOCK STRENGTH", strength_var, 0.5, 3.0, strength_display_var, step=100)
make_slider(tab_aim, "SMOOTHNESS", stability_var, 0.05, 1.00, stability_display_var, step=100)
make_slider(tab_aim, "AIM SPEED (px/frame)", max_step_var, 1, 20, max_step_display_var, step=19)
make_slider(tab_aim, "AIM DEADZONE (px)", deadzone_var, 0, 15, deadzone_display_var, step=15)
make_slider(tab_aim, "FOV RADIUS", fov_var, 30, 140, fov_display_var, step=110)
make_slider(tab_aim, "OFFSET X", offset_x_var, -100, 100, offset_x_display_var, step=200)
make_slider(tab_aim, "OFFSET Y", offset_y_var, -100, 100, offset_y_display_var, step=200)

make_slider(tab_track, "LEAD (frames)", lead_var, 0.0, 10.0, lead_display_var, step=100)
make_slider(tab_track, "WEIGHT AREA", w_area_var, 0.0, 1.0, w_area_display_var, step=100)
make_slider(tab_track, "WEIGHT DISTANCE", w_dist_var, 0.0, 1.0, w_dist_display_var, step=100)
make_slider(tab_track, "WEIGHT STICKY", w_stick_var, 0.0, 1.0, w_stick_display_var, step=100)
make_slider(tab_track, "SWITCH COOLDOWN (ms)", cooldown_var, 0, 500, cooldown_display_var, step=50)
make_slider(tab_track, "HYSTERESIS (%)", hyst_var, 0, 60, hyst_display_var, step=60)

ttl_vis = customtkinter.CTkFrame(tab_track, fg_color=BG_PANEL_2, corner_radius=4)
ttl_vis.pack(fill="x", padx=6, pady=(4, 3))
customtkinter.CTkLabel(ttl_vis, text="FILTERS", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=10, pady=(5, 3))
customtkinter.CTkCheckBox(ttl_vis, text=" SHAPE FILTER (aspect/solidity/extent)", variable=shape_filter_var, command=update_params, fg_color=ACCENT, hover_color=ACCENT_DIM, font=MONO_S, text_color=TXT, border_color=BORDER, checkmark_color="#FFFFFF").pack(anchor="w", padx=10, pady=(0, 6))

ctx_block = customtkinter.CTkFrame(tab_track, fg_color=BG_PANEL_2, corner_radius=4)
ctx_block.pack(fill="x", padx=6, pady=(3, 6))
customtkinter.CTkLabel(ctx_block, text="CONTEXT FILTER", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=10, pady=(5, 3))
customtkinter.CTkLabel(ctx_block, text="Ring around candidate: cyan (outline) or dark (silhouette).", font=MONO_S, text_color=TXT_DIM, justify="left").pack(anchor="w", padx=10, pady=(0, 5))
customtkinter.CTkCheckBox(ctx_block, text=" ENABLE CONTEXT VERIFY", variable=context_var, command=update_params, fg_color=ACCENT, hover_color=ACCENT_DIM, font=MONO_S, text_color=TXT, border_color=BORDER, checkmark_color="#FFFFFF").pack(anchor="w", padx=10, pady=(0, 6))

ctx_slider_row = customtkinter.CTkFrame(ctx_block, fg_color="transparent")
ctx_slider_row.pack(fill="x", padx=10, pady=(0, 7))
ctx_slider_row.grid_columnconfigure(0, weight=1)
ctx_slider_row.grid_columnconfigure(1, weight=1)

ctx_left = customtkinter.CTkFrame(ctx_slider_row, fg_color="transparent")
ctx_left.grid(row=0, column=0, sticky="ew", padx=(0, 4))
customtkinter.CTkLabel(ctx_left, text="RADIUS (px)", font=MONO_S, text_color=TXT_DIM, anchor="w").pack(fill="x")
customtkinter.CTkSlider(ctx_left, from_=10, to=50, variable=context_radius_var, command=update_params, button_color=ACCENT, button_hover_color=ACCENT_DIM, progress_color=ACCENT, fg_color=BG_INPUT, height=12, number_of_steps=40).pack(fill="x", pady=(2, 1))
customtkinter.CTkLabel(ctx_left, textvariable=ctx_radius_display_var, font=MONO_S, text_color=ACCENT, anchor="e").pack(fill="x")

ctx_right = customtkinter.CTkFrame(ctx_slider_row, fg_color="transparent")
ctx_right.grid(row=0, column=1, sticky="ew", padx=(4, 0))
customtkinter.CTkLabel(ctx_right, text="RATIO (%)", font=MONO_S, text_color=TXT_DIM, anchor="w").pack(fill="x")
customtkinter.CTkSlider(ctx_right, from_=0, to=100, variable=context_ratio_var, command=update_params, button_color=ACCENT, button_hover_color=ACCENT_DIM, progress_color=ACCENT, fg_color=BG_INPUT, height=12, number_of_steps=100).pack(fill="x", pady=(2, 1))
customtkinter.CTkLabel(ctx_right, textvariable=ctx_ratio_display_var, font=MONO_S, text_color=ACCENT, anchor="e").pack(fill="x")

color_top = customtkinter.CTkFrame(tab_color, fg_color=BG_PANEL_2, corner_radius=4)
color_top.pack(fill="x", padx=6, pady=6)

pick_row = customtkinter.CTkFrame(color_top, fg_color="transparent")
pick_row.pack(fill="x", padx=9, pady=(7, 3))
customtkinter.CTkButton(pick_row, text="▶ PICK PIXEL FROM SCREEN", command=start_pick_pixel, height=28, font=MONO_B, fg_color=OK_GREEN, hover_color="#2BB070", text_color="#000000").pack(side="left", fill="x", expand=True)
pick_status_lbl = customtkinter.CTkLabel(pick_row, text="ready", font=MONO_S, text_color=TXT_DIM, width=120, anchor="e")
pick_status_lbl.pack(side="right", padx=(8, 0))

hex_row = customtkinter.CTkFrame(color_top, fg_color="transparent")
hex_row.pack(fill="x", padx=9, pady=(3, 4))
customtkinter.CTkLabel(hex_row, text="HEX", font=MONO_B, text_color=TXT_DIM).pack(side="left", padx=(0, 6))
customtkinter.CTkEntry(hex_row, textvariable=hex_var, width=100, fg_color=BG_INPUT, border_color=BORDER, font=MONO, text_color=TXT, height=26).pack(side="left", padx=(0, 5))
customtkinter.CTkButton(hex_row, text="APPLY", command=set_from_hex, width=60, height=26, font=MONO_B, fg_color=ACCENT, hover_color=ACCENT_DIM).pack(side="left", padx=(0, 8))
swatch_canvas = tk.Canvas(hex_row, width=80, height=26, highlightthickness=1, highlightbackground=BORDER, bg=BG_INPUT)
swatch_canvas.pack(side="left")
update_swatch(swatch_canvas, hex_var.get())

tol_row = customtkinter.CTkFrame(color_top, fg_color="transparent")
tol_row.pack(fill="x", padx=9, pady=(0, 3))


def tol_field(parent, label, var, w=58):
    f = customtkinter.CTkFrame(parent, fg_color="transparent")
    f.pack(side="left", padx=(0, 10))
    customtkinter.CTkLabel(f, text=label, font=MONO_S, text_color=TXT_DIM).pack(side="left", padx=(0, 4))
    customtkinter.CTkEntry(f, textvariable=var, width=w, height=24, fg_color=BG_INPUT, border_color=BORDER, font=MONO_S, text_color=TXT).pack(side="left")
    return f


tol_field(tol_row, "TOL H", tol_h_var)
tol_field(tol_row, "TOL S", tol_s_var)
tol_field(tol_row, "TOL V", tol_v_var)

floors_block = customtkinter.CTkFrame(color_top, fg_color="transparent")
floors_block.pack(fill="x", padx=9, pady=(3, 4))
floors_block.grid_columnconfigure(0, weight=1)
floors_block.grid_columnconfigure(1, weight=1)

sf_frame = customtkinter.CTkFrame(floors_block, fg_color="transparent")
sf_frame.grid(row=0, column=0, sticky="ew", padx=(0, 4))
customtkinter.CTkLabel(sf_frame, text="S FLOOR (cut sky/ground)", font=MONO_S, text_color=TXT_DIM, anchor="w").pack(fill="x")
customtkinter.CTkSlider(sf_frame, from_=0, to=255, variable=s_floor_var, command=update_params, button_color=ACCENT, button_hover_color=ACCENT_DIM, progress_color=ACCENT, fg_color=BG_INPUT, height=12, number_of_steps=51).pack(fill="x", pady=(2, 1))
customtkinter.CTkLabel(sf_frame, textvariable=s_floor_display_var, font=MONO_S, text_color=ACCENT, anchor="e").pack(fill="x")

vf_frame = customtkinter.CTkFrame(floors_block, fg_color="transparent")
vf_frame.grid(row=0, column=1, sticky="ew", padx=(4, 0))
customtkinter.CTkLabel(vf_frame, text="V FLOOR (cut dark edges)", font=MONO_S, text_color=TXT_DIM, anchor="w").pack(fill="x")
customtkinter.CTkSlider(vf_frame, from_=0, to=255, variable=v_floor_var, command=update_params, button_color=ACCENT, button_hover_color=ACCENT_DIM, progress_color=ACCENT, fg_color=BG_INPUT, height=12, number_of_steps=51).pack(fill="x", pady=(2, 1))
customtkinter.CTkLabel(vf_frame, textvariable=v_floor_display_var, font=MONO_S, text_color=ACCENT, anchor="e").pack(fill="x")

customtkinter.CTkCheckBox(color_top, text=" ADAPTIVE HSV (auto-tune, stricter)", variable=adaptive_var, command=update_params, fg_color=ACCENT, hover_color=ACCENT_DIM, font=MONO_S, text_color=TXT, border_color=BORDER, checkmark_color="#FFFFFF").pack(anchor="w", padx=9, pady=(0, 7))

pal_block = customtkinter.CTkFrame(tab_color, fg_color=BG_PANEL_2, corner_radius=4)
pal_block.pack(fill="both", expand=True, padx=6, pady=(0, 6))

customtkinter.CTkLabel(pal_block, text="PALETTE", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=9, pady=(5, 3))

pal_inner = customtkinter.CTkFrame(pal_block, fg_color="transparent")
pal_inner.pack(fill="both", expand=True, padx=9, pady=(0, 7))
pal_inner.grid_columnconfigure(0, weight=1)
pal_inner.grid_rowconfigure(0, weight=1)

palette_list = tk.Listbox(pal_inner, selectmode="multiple", height=5, bg=BG_INPUT, fg=TXT, selectbackground=ACCENT, selectforeground="#FFFFFF", borderwidth=0, highlightthickness=0, font=MONO, activestyle="none", exportselection=False)
palette_list.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

pal_scroll = customtkinter.CTkScrollbar(pal_inner, command=palette_list.yview, fg_color=BG_INPUT, button_color=BORDER, button_hover_color=ACCENT)
pal_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 5))
palette_list.configure(yscrollcommand=pal_scroll.set)
palette_list.bind("<<ListboxSelect>>", on_palette_select)
palette_list.bind("<Double-Button-1>", on_palette_double_click)

palette_canvas = tk.Canvas(pal_inner, width=34, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER)
palette_canvas.grid(row=0, column=2, sticky="ns")

pal_btns = customtkinter.CTkFrame(pal_block, fg_color="transparent")
pal_btns.pack(fill="x", padx=9, pady=(0, 7))

customtkinter.CTkButton(pal_btns, text="PICK IMAGE", command=choose_image, height=24, font=MONO_S, fg_color=BG_INPUT, hover_color=ACCENT_DIM, text_color=TXT, border_width=1, border_color=BORDER).pack(side="left", fill="x", expand=True, padx=(0, 4))

customtkinter.CTkButton(pal_btns, text="SCAN FOLDER", command=auto_detect_colors_from_folder, height=24, font=MONO_S, fg_color=BG_INPUT, hover_color=ACCENT_DIM, text_color=TXT, border_width=1, border_color=BORDER).pack(side="left", fill="x", expand=True, padx=(4, 4))

customtkinter.CTkButton(pal_btns, text="LOCK SELECTED", command=lock_selected_colors, height=24, font=MONO_B, fg_color=ACCENT, hover_color=ACCENT_DIM, text_color="#FFFFFF").pack(side="left", fill="x", expand=True, padx=(4, 0))

vis_block = customtkinter.CTkFrame(tab_visual, fg_color=BG_PANEL_2, corner_radius=4)
vis_block.pack(fill="x", padx=6, pady=6)

customtkinter.CTkLabel(vis_block, text="VISUAL HELPERS", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=9, pady=(5, 3))

customtkinter.CTkLabel(vis_block, text="FOV overlay + center dot are click-through.\n" "Match FOV RADIUS slider with the ring radius.\n" "Debug mask: set debug_mask_visible=True in Python\n" "console to open cv2 mask window.", font=MONO_S, text_color=TXT_DIM, justify="left").pack(anchor="w", padx=9, pady=(0, 9))

status_tbl = customtkinter.CTkFrame(tab_visual, fg_color=BG_PANEL_2, corner_radius=4)
status_tbl.pack(fill="x", padx=6, pady=(0, 6))
customtkinter.CTkLabel(status_tbl, text="RUNTIME", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=9, pady=(5, 3))


def stat_row(parent, label, var):
    f = customtkinter.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=9, pady=1)
    customtkinter.CTkLabel(f, text=label, font=MONO_S, text_color=TXT_DIM, width=115, anchor="w").pack(side="left")
    customtkinter.CTkLabel(f, textvariable=var, font=MONO_S, text_color=TXT, anchor="w").pack(side="left")


stat_row(status_tbl, "state", status_var)
stat_row(status_tbl, "loop hz", live_hz_var)
stat_row(status_tbl, "lock hex", live_hex_var)
stat_row(status_tbl, "ranges", live_ranges_var)
stat_row(status_tbl, "contours", live_contours_var)
stat_row(status_tbl, "raw / ctx-pass", live_ctx_var)
stat_row(status_tbl, "candidates", live_hit_var)
stat_row(status_tbl, "active tracks", live_tracks_var)
stat_row(status_tbl, "target", live_target_var)
stat_row(status_tbl, "err", live_err_var)
customtkinter.CTkFrame(status_tbl, fg_color="transparent", height=7).pack()

sens_block = customtkinter.CTkFrame(tab_config, fg_color=BG_PANEL_2, corner_radius=4)
sens_block.pack(fill="x", padx=6, pady=6)
customtkinter.CTkLabel(sens_block, text="SENSITIVITY", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=9, pady=(5, 3))


def sens_row(parent, label, var, frm, to, disp):
    f = customtkinter.CTkFrame(parent, fg_color="transparent", height=28)
    f.pack(fill="x", padx=9, pady=2)
    f.pack_propagate(False)
    customtkinter.CTkLabel(f, text=label, font=MONO_S, text_color=TXT, width=135, anchor="w").pack(side="left")
    customtkinter.CTkSlider(f, from_=frm, to=to, variable=var, command=update_params, button_color=ACCENT, button_hover_color=ACCENT_DIM, progress_color=ACCENT, fg_color=BG_INPUT, height=12).pack(side="left", fill="x", expand=True, padx=6)
    customtkinter.CTkLabel(f, textvariable=disp, font=MONO_S, text_color=ACCENT, width=44, anchor="e").pack(side="right")


sens_row(sens_block, "in-game mouse sens", pf_mouse_var, 0.1, 5.0, pf_mouse_display_var)
sens_row(sens_block, "in-game aim sens", pf_aim_var, 0.1, 3.0, pf_aim_display_var)
sens_row(sens_block, "roblox sensitivity", roblox_sens_var, 0.1, 2.0, roblox_display_var)
customtkinter.CTkFrame(sens_block, fg_color="transparent", height=5).pack()

key_block = customtkinter.CTkFrame(tab_config, fg_color=BG_PANEL_2, corner_radius=4)
key_block.pack(fill="x", padx=6, pady=6)
customtkinter.CTkLabel(key_block, text="AIM KEY", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=9, pady=(5, 3))

key_inner = customtkinter.CTkFrame(key_block, fg_color="transparent")
key_inner.pack(fill="x", padx=9, pady=(0, 7))

key_display_lbl = customtkinter.CTkLabel(key_inner, textvariable=aim_key_display_var, font=MONO_B, text_color=TXT, fg_color=BG_INPUT, corner_radius=4, width=155, height=26)
key_display_lbl.pack(side="left", padx=(0, 6))

key_capture_active = [False]
key_capture_start = [0.0]
CAPTURE_DEADTIME = 0.30

capture_btn = customtkinter.CTkButton(key_inner, text="SET KEY", command=lambda: None, width=125, height=26, font=MONO_B, fg_color=ACCENT, hover_color=ACCENT_DIM)
capture_btn.pack(side="left")


def _capture_ready():
    if not key_capture_active[0]:
        return False
    if (time.time() - key_capture_start[0]) < CAPTURE_DEADTIME:
        return False
    return True


def _commit_key(key_code):
    global aim_key
    if key_code is None:
        return
    if not (0 < int(key_code) < 256):
        return
    aim_key_var.set(int(key_code))
    aim_key_display_var.set(get_key_name(int(key_code)))
    aim_key = int(key_code)
    key_display_lbl.configure(text_color=TXT)
    capture_btn.configure(text="SET KEY")
    key_capture_active[0] = False


def _cancel_capture():
    key_capture_active[0] = False
    key_display_lbl.configure(text_color=TXT)
    aim_key_display_var.set(get_key_name(aim_key))
    capture_btn.configure(text="SET KEY")


def start_key_capture():
    key_capture_active[0] = True
    key_capture_start[0] = time.time()
    key_display_lbl.configure(text_color=ACCENT)
    aim_key_display_var.set("Press any key...")
    capture_btn.configure(text="...")
    try:
        root.focus_force()
    except Exception:
        pass


capture_btn.configure(command=start_key_capture)


def capture_key(event):
    if not _capture_ready():
        return
    vk = None
    try:
        kc = event.keycode
        if kc is not None and 0 < int(kc) < 256:
            vk = int(kc)
    except Exception:
        vk = None
    if vk is None:
        vk = KEYSYM_TO_VK.get(event.keysym)
    _commit_key(vk)


def capture_mouse(event):
    if not _capture_ready():
        return
    btn_map = {1: 0x01, 2: 0x04, 3: 0x02, 4: 0x05, 5: 0x06}
    _commit_key(btn_map.get(event.num))


def cancel_capture(_event=None):
    if key_capture_active[0]:
        _cancel_capture()


root.bind("<KeyPress>", capture_key)
root.bind("<ButtonPress>", capture_mouse)
root.bind("<Escape>", cancel_capture)

prof_block = customtkinter.CTkFrame(tab_config, fg_color=BG_PANEL_2, corner_radius=4)
prof_block.pack(fill="x", padx=6, pady=6)
customtkinter.CTkLabel(prof_block, text="PROFILE MANAGER", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=9, pady=(5, 3))

prof_list_frame = customtkinter.CTkFrame(prof_block, fg_color="transparent")
prof_list_frame.pack(fill="x", padx=9, pady=3)

profile_listbox = tk.Listbox(prof_list_frame, height=3, bg=BG_INPUT, fg=TXT, selectbackground=ACCENT, selectforeground="#FFFFFF", borderwidth=0, highlightthickness=0, font=MONO_S)
profile_listbox.pack(side="left", fill="x", expand=True)

prof_btn_frame = customtkinter.CTkFrame(prof_block, fg_color="transparent")
prof_btn_frame.pack(fill="x", padx=9, pady=(3, 7))


def refresh_profile_list():
    profile_listbox.delete(0, tk.END)
    profiles = load_profiles()
    for name in profiles:
        profile_listbox.insert(tk.END, name)


def _profile_snapshot():
    return {
        "hex": hex_var.get(),
        "tol_h": tol_h_var.get(),
        "tol_s": tol_s_var.get(),
        "tol_v": tol_v_var.get(),
        "s_floor": s_floor_var.get(),
        "v_floor": v_floor_var.get(),
        "lock_strength": round_to_2(strength_var.get()),
        "smooth_alpha": round_to_2(stability_var.get()),
        "pf_mouse_sensitivity": pf_mouse_var.get(),
        "pf_aim_sensitivity": pf_aim_var.get(),
        "roblox_sensitivity": roblox_sens_var.get(),
        "max_step": max_step_var.get(),
        "deadzone": deadzone_var.get(),
        "fov": fov_var.get(),
        "offset_x": offset_x_var.get(),
        "offset_y": offset_y_var.get(),
        "aim_key": aim_key_var.get(),
        "lead": lead_var.get(),
        "w_area": w_area_var.get(),
        "w_dist": w_dist_var.get(),
        "w_stick": w_stick_var.get(),
        "switch_cooldown": cooldown_var.get(),
        "hysteresis": hyst_var.get(),
        "shape_filter": shape_filter_var.get(),
        "adaptive_hsv": adaptive_var.get(),
        "context_check": context_var.get(),
        "context_radius": context_radius_var.get(),
        "context_ratio": context_ratio_var.get(),
        "show_fov": show_fov_var.get(),
        "show_dot": show_dot_var.get(),
        "palette": list(palette_list.get(0, tk.END)),
    }


def _profile_apply(data):
    hex_var.set(data.get("hex", "#FDFCB3"))
    tol_h_var.set(data.get("tol_h", 8))
    tol_s_var.set(data.get("tol_s", 40))
    tol_v_var.set(data.get("tol_v", 40))
    s_floor_var.set(data.get("s_floor", 180))
    v_floor_var.set(data.get("v_floor", 200))
    strength_var.set(round_to_2(data.get("lock_strength", 1.5)))
    stability_var.set(round_to_2(data.get("smooth_alpha", 0.35)))
    max_step_var.set(data.get("max_step", 5))
    deadzone_var.set(data.get("deadzone", 3))
    fov_var.set(data.get("fov", 70))
    offset_x_var.set(data.get("offset_x", 0))
    offset_y_var.set(data.get("offset_y", -2))
    pf_mouse_var.set(data.get("pf_mouse_sensitivity", 0.5))
    pf_aim_var.set(data.get("pf_aim_sensitivity", 1.0))
    roblox_sens_var.set(data.get("roblox_sensitivity", 0.55))
    lead_var.set(data.get("lead", 1.5))
    w_area_var.set(data.get("w_area", 0.20))
    w_dist_var.set(data.get("w_dist", 0.60))
    w_stick_var.set(data.get("w_stick", 0.40))
    cooldown_var.set(data.get("switch_cooldown", 120))
    hyst_var.set(data.get("hysteresis", 15))
    shape_filter_var.set(data.get("shape_filter", True))
    adaptive_var.set(data.get("adaptive_hsv", False))
    context_var.set(data.get("context_check", True))
    context_radius_var.set(data.get("context_radius", 25))
    context_ratio_var.set(data.get("context_ratio", 25))
    show_fov_var.set(data.get("show_fov", True))
    show_dot_var.set(data.get("show_dot", False))

    saved_key = data.get("aim_key", 0x10)
    aim_key_var.set(saved_key)
    aim_key_display_var.set(get_key_name(saved_key))
    global aim_key
    aim_key = saved_key

    palette_list.delete(0, tk.END)
    for c in data.get("palette", []):
        palette_list.insert(tk.END, c)
    update_palette_canvas(data.get("palette", []))

    active_hexes.clear()
    hx_upper = hex_var.get().upper()
    if hx_upper and hx_upper != "#-":
        active_hexes.append(hx_upper)
    else:
        active_hexes.append("#FDFCB3")

    rebuild_active_ranges_from_hexes()
    update_params()
    update_swatch(swatch_canvas, hex_var.get())


def save_as_profile():
    name = simpledialog.askstring("Save Profile", "Enter profile name:")
    if not name:
        return
    profiles = load_profiles()
    profiles[name] = _profile_snapshot()
    save_profiles(profiles)
    refresh_profile_list()
    messagebox.showinfo("Success", f"Profile '{name}' saved.")


def load_profile():
    sel = profile_listbox.curselection()
    if not sel:
        return
    name = profile_listbox.get(sel[0])
    profiles = load_profiles()
    if name not in profiles:
        return
    _profile_apply(profiles[name])
    messagebox.showinfo("Success", f"Profile '{name}' loaded.")


def delete_profile():
    sel = profile_listbox.curselection()
    if not sel:
        return
    name = profile_listbox.get(sel[0])
    if messagebox.askyesno("Delete", f"Delete profile '{name}'?"):
        profiles = load_profiles()
        if name in profiles:
            del profiles[name]
            save_profiles(profiles)
            refresh_profile_list()


customtkinter.CTkButton(prof_btn_frame, text="SAVE AS", command=save_as_profile, height=22, font=MONO_S, fg_color=OK_GREEN, text_color="#000000").pack(side="left", fill="x", expand=True, padx=(0, 3))
customtkinter.CTkButton(prof_btn_frame, text="LOAD", command=load_profile, height=22, font=MONO_S, fg_color=BG_INPUT, text_color=TXT).pack(side="left", fill="x", expand=True, padx=3)
customtkinter.CTkButton(prof_btn_frame, text="DELETE", command=delete_profile, height=22, font=MONO_S, fg_color=ACCENT, text_color="#FFFFFF").pack(side="left", fill="x", expand=True, padx=(3, 0))

cfg_block = customtkinter.CTkFrame(tab_config, fg_color=BG_PANEL_2, corner_radius=4)
cfg_block.pack(fill="x", padx=6, pady=(0, 6))

customtkinter.CTkLabel(cfg_block, text="GENERAL", font=MONO_B, text_color=ACCENT).pack(anchor="w", padx=9, pady=(5, 3))


def reset_defaults():
    hex_var.set("#FDFCB3")
    tol_h_var.set(5)
    tol_s_var.set(30)
    tol_v_var.set(30)
    s_floor_var.set(180)
    v_floor_var.set(200)
    strength_var.set(1.5)
    stability_var.set(0.35)
    max_step_var.set(5)
    deadzone_var.set(3)
    fov_var.set(70)
    offset_x_var.set(0)
    offset_y_var.set(-2)
    pf_mouse_var.set(0.5)
    pf_aim_var.set(1.0)
    roblox_sens_var.set(0.55)
    lead_var.set(1.5)
    w_area_var.set(0.20)
    w_dist_var.set(0.60)
    w_stick_var.set(0.40)
    cooldown_var.set(120)
    hyst_var.set(15)
    shape_filter_var.set(True)
    adaptive_var.set(False)
    context_var.set(True)
    context_radius_var.set(25)
    context_ratio_var.set(25)
    aim_key_var.set(0x10)
    aim_key_display_var.set("Left Shift")
    global aim_key, adaptive_ranges, tracks, primary_track_id, min_saturation_floor, min_value_floor
    aim_key = 0x10
    adaptive_ranges = []
    tracks = {}
    primary_track_id = None
    min_saturation_floor = 180
    min_value_floor = 200
    active_hexes.clear()
    active_hexes.append("#FDFCB3")
    rebuild_active_ranges_from_hexes()
    update_params()


customtkinter.CTkButton(cfg_block, text="RESET DEFAULTS", command=reset_defaults, height=26, font=MONO_B, fg_color=ACCENT, hover_color=ACCENT_DIM, text_color=TXT, border_width=1, border_color=BORDER).pack(fill="x", padx=9, pady=(3, 7))


def poll_live_status():
    global status_lock_hex, status_ranges
    if status_lock_hex == "-" and active_hexes:
        status_lock_hex = active_hexes[0]
    if status_ranges == "0" and (active_ranges or adaptive_ranges):
        status_ranges = str(len(active_ranges) + len(adaptive_ranges))

    live_hex_var.set(status_lock_hex)
    live_ranges_var.set(status_ranges)
    live_contours_var.set(status_contours)
    live_hit_var.set(status_hit)
    live_target_var.set(status_target)
    live_err_var.set(status_err)
    live_hz_var.set(status_loop_hz)
    live_tracks_var.set(status_tracks)
    live_ctx_var.set(status_ctx)

    try:
        live_swatch.delete("all")
        hx = status_lock_hex
        if hx and hx != "-":
            live_swatch.create_rectangle(0, 0, 190, 6, fill=hx, outline="")
        else:
            live_swatch.create_rectangle(0, 0, 190, 6, fill=BG_INPUT, outline="")
    except Exception:
        pass

    try:
        if status_var.get() == "RUNNING":
            top_status_lbl.configure(text_color=OK_GREEN)
            top_status_var.set("RUNNING")
        else:
            top_status_lbl.configure(text_color=TXT_DIM)
            top_status_var.set("STOPPED")
    except Exception:
        pass

    try:
        if overlay is not None and overlay.winfo_exists():
            if show_fov_var.get() or show_dot_var.get():
                r = int(fov_var.get())
                d = 2 * r + 6
                x = config.center_x - d // 2
                y = config.center_y - d // 2
                overlay.geometry(f"{d}x{d}+{x}+{y}")
    except Exception:
        pass

    root.after(200, poll_live_status)


PROFILES_FILE = "profiles.json"


def load_profiles():
    if not os.path.exists(PROFILES_FILE):
        return {}
    try:
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_profiles(profiles):
    try:
        with open(PROFILES_FILE, "w") as f:
            json.dump(profiles, f, indent=2)
    except Exception as e:
        print(f"Error saving profiles: {e}")


def load_settings():
    try:
        if not os.path.exists("settings.json"):
            return
        if os.path.getsize("settings.json") == 0:
            return
        with open("settings.json", "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Settings file is corrupted. Using defaults.")
                return
        _profile_apply(data)
    except Exception as e:
        print(f"Error loading settings: {e}")


def save_settings():
    try:
        with open("settings.json", "w") as f:
            json.dump(_profile_snapshot(), f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")


def on_close():
    stop_worker()
    save_settings()
    root.destroy()


load_settings()
refresh_profile_list()
update_params()
root.update_idletasks()

root.after(150, startup_image_scan)
root.after(200, poll_live_status)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
