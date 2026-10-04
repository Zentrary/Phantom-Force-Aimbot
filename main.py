import winsound
import win32api
import win32con
import win32gui
import numpy as np
import time
import cv2
import mss
import threading
import tkinter as tk
import os
import json
import colorsys
import customtkinter
import ctypes
from ctypes import wintypes

from tkinter import filedialog, messagebox
from pynput import keyboard as pkeyboard, mouse as pmouse

sigma_user32 = ctypes.windll.user32

SKIBIDY_GWL_EXSTYLE = -20
SKIBIDY_WS_EX_LAYERED = 0x80000
SKIBIDY_WS_EX_TRANSPARENT = 0x20
SKIBIDY_WS_EX_TOOLWINDOW = 0x80
SKIBIDY_SWP_NOMOVE = 0x0002
SKIBIDY_SWP_NOSIZE = 0x0001
SKIBIDY_SWP_FRAMECHANGED = 0x0020
SKIBIDY_HWND_TOPMOST = -1

sigma_user32.GetParent.argtypes = [wintypes.HWND]
sigma_user32.GetParent.restype = wintypes.HWND
sigma_user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
sigma_user32.GetWindowLongW.restype = ctypes.c_long
sigma_user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
sigma_user32.SetWindowLongW.restype = ctypes.c_long
sigma_user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
sigma_user32.SetWindowPos.restype = wintypes.BOOL
sigma_user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
sigma_user32.SetWindowDisplayAffinity.restype = wintypes.BOOL

RIZZ_WDA_NONE = 0x00000000
RIZZ_WDA_EXCLUDEFROMCAPTURE = 0x00000011

def rizz_exclude_from_capture_67(widget, enable=True):
    try:
        hwnd = widget.winfo_id()
        parent = sigma_user32.GetParent(hwnd)
        if parent:
            hwnd = parent
        affinity = RIZZ_WDA_EXCLUDEFROMCAPTURE if enable else RIZZ_WDA_NONE
        return bool(sigma_user32.SetWindowDisplayAffinity(hwnd, affinity))
    except Exception:
        return False

def skibidy_set_titlebar_color_67(hwnd, hex_color):
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        colorref = r | (g << 8) | (b << 16)
        val = ctypes.c_int(colorref)
        DWMWA_CAPTION_COLOR = 35
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:
        pass

def sigma_make_click_through_67(win, transparent_supported):
    try:
        hwnd = win.winfo_id()
        parent = sigma_user32.GetParent(hwnd)
        if parent:
            hwnd = parent
        style = sigma_user32.GetWindowLongW(hwnd, SKIBIDY_GWL_EXSTYLE)
        style |= SKIBIDY_WS_EX_TRANSPARENT | SKIBIDY_WS_EX_TOOLWINDOW
        if transparent_supported:
            style |= SKIBIDY_WS_EX_LAYERED
        sigma_user32.SetWindowLongW(hwnd, SKIBIDY_GWL_EXSTYLE, style)
        sigma_user32.SetWindowPos(hwnd, wintypes.HWND(SKIBIDY_HWND_TOPMOST), 0, 0, 0, 0,
                                  SKIBIDY_SWP_NOMOVE | SKIBIDY_SWP_NOSIZE | SKIBIDY_SWP_FRAMECHANGED)
    except Exception:
        pass

def rizz_beep_sigma_on_67():
    try:
        winsound.Beep(1000, 55)
        winsound.Beep(1500, 70)
    except Exception:
        pass

def rizz_beep_sigma_off_67():
    try:
        winsound.Beep(1500, 55)
        winsound.Beep(900, 70)
    except Exception:
        pass

def rizz_beep_skibidy_start_67():
    try:
        winsound.Beep(900, 40)
        winsound.Beep(1200, 40)
        winsound.Beep(1600, 60)
    except Exception:
        pass

def rizz_beep_skibidy_stop_67():
    try:
        winsound.Beep(1600, 40)
        winsound.Beep(1100, 40)
        winsound.Beep(700, 60)
    except Exception:
        pass

class SigmaSkibidyRizzConfig67:
    def __init__(self):
        try:
            self.sigma_width_67 = win32api.GetSystemMetrics(0)
            self.skibidy_height_67 = win32api.GetSystemMetrics(1)
        except Exception:
            self.sigma_width_67 = 1920
            self.skibidy_height_67 = 1080
        self.rizz_center_x_67 = self.sigma_width_67 // 2
        self.rizz_center_y_67 = self.skibidy_height_67 // 2
        self.skibidy_uniform_capture_size_67 = 240
        self.sigma_crosshair_uniform_67 = self.skibidy_uniform_capture_size_67 // 2
        self.rizz_capture_left_67 = self.rizz_center_x_67 - self.sigma_crosshair_uniform_67
        self.rizz_capture_top_67 = self.rizz_center_y_67 - self.sigma_crosshair_uniform_67
        self.skibidy_region_67 = {
            "top": self.rizz_capture_top_67,
            "left": self.rizz_capture_left_67,
            "width": self.skibidy_uniform_capture_size_67,
            "height": self.skibidy_uniform_capture_size_67,
        }

sigma_config_67 = SigmaSkibidyRizzConfig67()
skibidy_crosshair_u_67 = sigma_config_67.sigma_crosshair_uniform_67
rizz_region_c_67 = sigma_config_67.skibidy_region_67

SIGMA_BASE_DIR_67 = os.path.dirname(os.path.abspath(__file__))
SKIBIDY_CONFIGS_DIR_67 = os.path.join(SIGMA_BASE_DIR_67, "configs")
RIZZ_LAST_SESSION_FILE_67 = os.path.join(SKIBIDY_CONFIGS_DIR_67, "_last_session.json")
SIGMA_LEGACY_SETTINGS_FILE_67 = os.path.join(SIGMA_BASE_DIR_67, "settings.json")

os.makedirs(SKIBIDY_CONFIGS_DIR_67, exist_ok=True)

skibidy_kernel_67 = np.ones((3, 3), np.uint8)
rizz_min_area_67 = 60

sigma_roblox_sensitivity_67 = 0.55
skibidy_pf_mouse_sensitivity_67 = 0.5
rizz_pf_aim_sensitivity_67 = 1.0
sigma_movement_compensation_67 = 0.0

skibidy_deadzone_px_67 = 6
rizz_max_step_px_67 = 6
sigma_smooth_alpha_67 = 0.18
skibidy_ema_dx_67 = 0.0
rizz_ema_dy_67 = 0.0
sigma_kp_67 = 0.45
skibidy_kd_67 = 0.25
rizz_prev_err_x_67 = 0.0
sigma_prev_err_y_67 = 0.0
skibidy_track_cx_67 = None
rizz_track_cy_67 = None
sigma_roi_radius_67 = 50

skibidy_fov_radius_67 = 80
rizz_lock_strength_67 = 1.0
sigma_offset_x_67 = 0
skibidy_offset_y_67 = 0

rizz_current_aim_vk_67 = 0x02
sigma_current_toggle_vk_67 = 0x77
skibidy_prev_toggle_state_67 = 0

rizz_active_ranges_67 = []
sigma_locked_hex_67 = ""
skibidy_locked_hsv_67 = None
rizz_running_67 = False
sigma_worker_67 = None
skibidy_aim_enabled_67 = True

rizz_lower_hsv_67 = np.array([0, 160, 160], dtype=np.uint8)
sigma_upper_hsv_67 = np.array([10, 255, 255], dtype=np.uint8)

skibidy_overlay_67 = None
rizz_overlay_canvas_67 = None
sigma_transparent_supported_67 = False

skibidy_overlay_target_screen_67 = None
rizz_overlay_fps_value_67 = 0

sigma_rainbow_hue_67 = 0

SKIBIDY_VK_NAMES_67 = {
    0x01: "MOUSE L", 0x02: "MOUSE R", 0x04: "MOUSE M",
    0x05: "MOUSE S1", 0x06: "MOUSE S2",
    0x08: "BACKSPACE", 0x09: "TAB", 0x0D: "ENTER",
    0x10: "SHIFT", 0x11: "CTRL", 0x12: "ALT", 0x13: "PAUSE",
    0x14: "CAPS", 0x1B: "ESC", 0x20: "SPACE",
    0x21: "PGUP", 0x22: "PGDN", 0x23: "END", 0x24: "HOME",
    0x25: "LEFT", 0x26: "UP", 0x27: "RIGHT", 0x28: "DOWN",
    0x2D: "INS", 0x2E: "DEL",
    0x70: "F1", 0x71: "F2", 0x72: "F3", 0x73: "F4",
    0x74: "F5", 0x75: "F6", 0x76: "F7", 0x77: "F8",
    0x78: "F9", 0x79: "F10", 0x7A: "F11", 0x7B: "F12",
}

def rizz_vk_display_67(vk):
    if vk == 0:
        return "None"
    if vk in SKIBIDY_VK_NAMES_67:
        return SKIBIDY_VK_NAMES_67[vk]
    if 0x30 <= vk <= 0x39:
        return chr(vk)
    if 0x41 <= vk <= 0x5A:
        return chr(vk)
    return f"0x{vk:02X}"

def sigma_key_event_to_vk_67(key):
    try:
        if hasattr(key, "vk") and key.vk is not None:
            return int(key.vk)
    except Exception:
        pass
    s = str(key).replace("Key.", "").lower()
    return {
        "alt": 0x12, "shift": 0x10, "ctrl": 0x11,
        "space": 0x20, "tab": 0x09, "enter": 0x0D, "esc": 0x1B,
        "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
        "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
        "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
        "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    }.get(s, 0)

def skibidy_mouse_event_to_vk_67(button):
    return {
        pmouse.Button.left: 0x01, pmouse.Button.right: 0x02,
        pmouse.Button.middle: 0x04,
        pmouse.Button.x1: 0x05, pmouse.Button.x2: 0x06,
    }.get(button, 0)

def rizz_hex_to_bgr_67(s):
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6:
        try:
            return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))
        except Exception:
            return None
    return None

def sigma_range_from_hex_67(s, tol_h=10, tol_s=60, tol_v=60):
    bgr = rizz_hex_to_bgr_67(s)
    if bgr is None:
        return None
    pix = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
    h, sa, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    lo = np.array([max(h - tol_h, 0), max(sa - tol_s, 0),
                   max(v - tol_v, 0)], dtype=np.uint8)
    up = np.array([min(h + tol_h, 179), min(sa + tol_s, 255),
                   min(v + tol_v, 255)], dtype=np.uint8)
    return lo, up

def skibidy_hsv_to_hex_67(h, s, v):
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
    return f"#{int(bgr[2]):02X}{int(bgr[1]):02X}{int(bgr[0]):02X}"

def rizz_rainbow_color_67():
    r, g, b = colorsys.hsv_to_rgb(sigma_rainbow_hue_67 / 360.0, 1.0, 1.0)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"

def sigma_set_locked_hex_67(hex_str):
    global sigma_locked_hex_67, skibidy_locked_hsv_67
    sigma_locked_hex_67 = hex_str.strip().upper()
    bgr = rizz_hex_to_bgr_67(sigma_locked_hex_67)
    if bgr is not None:
        pix = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
        skibidy_locked_hsv_67 = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
    else:
        skibidy_locked_hsv_67 = None

def skibidy_build_mask_67(frame_hsv):
    if not rizz_active_ranges_67:
        return cv2.inRange(frame_hsv, rizz_lower_hsv_67, sigma_upper_hsv_67)
    m = None
    for lo, up in rizz_active_ranges_67:
        mm = cv2.inRange(frame_hsv, lo, up)
        m = mm if m is None else cv2.bitwise_or(m, mm)
    return m

def rizz_analyze_files_67(paths, skip_dark=True, skip_gray=True, top_n=12):
    hist = np.zeros(18 * 8 * 8, dtype=np.int64)
    total = 0
    processed = 0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        processed += 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        Hc = (hsv[:, :, 0].astype(np.int32) * 18 // 180).clip(0, 17)
        Sc = (hsv[:, :, 1].astype(np.int32) * 8 // 256).clip(0, 7)
        Vc = (hsv[:, :, 2].astype(np.int32) * 8 // 256).clip(0, 7)
        m = np.ones(Hc.shape, dtype=bool)
        if skip_dark:
            m &= hsv[:, :, 2] > 40
        if skip_gray:
            m &= hsv[:, :, 1] > 40
        idx = (Hc * 64 + Sc * 8 + Vc).ravel()
        mv = m.ravel().astype(np.int64)
        counts = np.bincount(idx, weights=mv, minlength=18 * 8 * 8)
        hist += counts.astype(np.int64)
        total += int(mv.sum())
    if total == 0:
        return [], processed, 0
    order = np.argsort(hist)[::-1]
    rows = []
    for i in order[:top_n]:
        cnt = int(hist[i])
        if cnt == 0:
            continue
        hh = i // 64
        ss = (i % 64) // 8
        vv = i % 8
        H = min(int((hh + 0.5) * 10), 179)
        S = min(int((ss + 0.5) * 32), 255)
        V = min(int((vv + 0.5) * 32), 255)
        rows.append({"hsv": (H, S, V), "hex": skibidy_hsv_to_hex_67(H, S, V),
                     "count": cnt, "pct": cnt / total * 100.0})
    return rows, processed, total

def sigma_analyze_folder_colors_67(folder, skip_dark=True, skip_gray=True, top_n=12):
    try:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith(exts)]
    except Exception:
        return [], 0, 0
    if not files:
        return [], 0, 0
    return rizz_analyze_files_67(files, skip_dark, skip_gray, top_n)

def skibidy_analyze_single_image_67(path, top_n=8):
    return rizz_analyze_files_67([path], top_n=top_n)[0]

rizz_verify_enabled_67 = False
sigma_verify_hex_list_67 = ["#3AA0FF"]
skibidy_verify_tol_h_67 = 12
rizz_verify_tol_s_67 = 70
sigma_verify_tol_v_67 = 70
skibidy_verify_roi_67 = 40
rizz_verify_min_px_67 = 8
sigma_verify_frames_required_67 = 1
skibidy_verify_ranges_67 = []
rizz_verify_ok_frames_67 = 0
sigma_last_verify_px_67 = -1

def skibidy_rebuild_verify_ranges_67():
    global skibidy_verify_ranges_67
    skibidy_verify_ranges_67 = []
    for h in sigma_verify_hex_list_67:
        rng = sigma_range_from_hex_67(h, skibidy_verify_tol_h_67,
                                      rizz_verify_tol_s_67,
                                      sigma_verify_tol_v_67)
        if rng:
            skibidy_verify_ranges_67.append(rng)

def rizz_verify_target_67(frame_hsv, cx, cy):
    global sigma_last_verify_px_67, rizz_verify_ok_frames_67

    if not rizz_verify_enabled_67:
        sigma_last_verify_px_67 = -1
        rizz_verify_ok_frames_67 = 0
        return True
    if not skibidy_verify_ranges_67:
        sigma_last_verify_px_67 = -1
        return True

    h, w = frame_hsv.shape[:2]
    y0 = max(0, int(cy) - skibidy_verify_roi_67)
    y1 = min(h, int(cy) + skibidy_verify_roi_67)
    x0 = max(0, int(cx) - skibidy_verify_roi_67)
    x1 = min(w, int(cx) + skibidy_verify_roi_67)
    if x1 <= x0 or y1 <= y0:
        sigma_last_verify_px_67 = 0
        rizz_verify_ok_frames_67 = 0
        return False
    roi = frame_hsv[y0:y1, x0:x1]

    best_px = 0
    for rng in skibidy_verify_ranges_67:
        m = cv2.inRange(roi, rng[0], rng[1])
        px = int(cv2.countNonZero(m))
        if px > best_px:
            best_px = px
    sigma_last_verify_px_67 = best_px
    marker_present = best_px >= rizz_verify_min_px_67

    if sigma_verify_frames_required_67 > 1:
        if marker_present:
            rizz_verify_ok_frames_67 += 1
            return rizz_verify_ok_frames_67 >= sigma_verify_frames_required_67
        rizz_verify_ok_frames_67 = 0
        return False
    return marker_present

SKIBIDY_LANG_67 = {
    "th": {
        "app_name": "Aimzen",
        "lang_btn": "EN",
        "status_running": "กำลังทำงาน",
        "status_stopped": "หยุดแล้ว",
        "btn_start": "เริ่ม", "btn_stop": "หยุด",

        "nav_aim": "Aimbot", "nav_visual": "การแสดงผล",
        "nav_filter": "กรองสี", "nav_adv": "ขั้นสูง", "nav_prof": "โปรไฟล์",

        "sec_target_color": "สีเป้าหมาย",
        "sec_aim_response": "การตอบสนองการเล็ง",
        "sec_key_bindings": "ปุ่มลัด",
        "sec_overlay": "ออฟเวอร์เลย์",
        "sec_palette_folder": "โฟลเดอร์จานสี",
        "sec_palette_detected": "จานสีที่ตรวจพบ",
        "sec_filter_rule": "เงื่อนไขกรองสี",
        "sec_marker_color": "สีเงื่อนไข",
        "sec_marker_detection": "การตรวจจับสีเงื่อนไข",
        "sec_diag": "ไดแอกโนสติกส์",
        "sec_movement": "ชดเชยการเคลื่อนไหว",
        "sec_gains": "ตัวปรับค่า",
        "sec_tracking": "การติดตามเป้า",
        "sec_profiles": "โปรไฟล์ที่บันทึกไว้",
        "sec_storage": "ที่เก็บข้อมูล",
        "sec_crosshair": "รูปแบบเป้า",

        "lbl_hex": "รหัสสี",
        "lbl_hue_tol": "ค่าความคลาดเคลื่อน Hue",
        "lbl_sat_tol": "ค่าความคลาดเคลื่อน Saturation",
        "lbl_bri_tol": "ค่าความคลาดเคลื่อน Brightness",
        "lbl_lock_strength": "ความแรงล็อค",
        "lbl_smoothing": "ความนุ่มนวล",
        "lbl_mouse_sens": "ความไวเมาส์",
        "lbl_aim_sens": "ความไวการเล็ง",
        "lbl_game_sens": "ความไวในเกม",
        "lbl_max_step": "ก้าวสูงสุดต่อเฟรม",
        "lbl_deadzone": "จุดตาย",
        "lbl_fov": "มุมมอง (FOV)",
        "lbl_offset_x": "ออฟเซ็ตแนว X",
        "lbl_offset_y": "ออฟเซ็ตแนว Y",
        "lbl_aim_key": "ปุ่มเล็ง",
        "lbl_enable_toggle": "ปุ่มเปิด/ปิดระบบ",
        "lbl_panic_key": "ปุ่มหยุดฉุกเฉิน",

        "btn_apply": "ใช้",
        "btn_load_img": "โหลดรูปเดียว",
        "btn_scan": "สแกนโฟลเดอร์",
        "btn_lock_color": "ล็อคสีที่เลือก",
        "btn_clear_palette": "ล้างจานสี",
        "btn_save": "บันทึก",
        "btn_rename": "เปลี่ยนชื่อ",
        "btn_delete": "ลบ",
        "btn_load_sel": "โหลดโปรไฟล์ที่เลือก",
        "btn_reset_all": "รีเซ็ตทั้งหมด",
        "btn_auto_detect": "ตรวจจับสีฟ้าอัตโนมัติ",

        "chk_show_fov": "แสดงวง FOV",
        "chk_show_crosshair": "แสดงเป้ากลางจอ",
        "chk_show_box": "แสดงกรอบเป้า",
        "chk_show_fps": "แสดงตัวนับ FPS",
        "chk_show_aim_line": "แสดงเส้นเล็ง",
        "chk_hide_idle": "ซ่อนออฟเวอร์เลย์เมื่อไม่เล็ง",
        "chk_exclude_capture": "ซ่อนจากการบันทึกหน้าจอ",
        "chk_rainbow": "โหมดสีรุ้ง",
        "chk_ignore_dark": "ข้ามพิกเซลสีมืด",
        "chk_ignore_gray": "ข้ามพิกเซลสีเทา",

        "chk_enable_filter": "เปิดใช้การตรวจสอบสีเงื่อนไข",
        "lbl_marker_hexes": "สีเงื่อนไข (คั่นด้วยจุลภาค)",
        "lbl_sample_radius": "รัศมีสุ่มตัวอย่าง (px)",
        "lbl_required_px": "จำนวนพิกเซลขั้นต่ำ",
        "lbl_required_frames": "จำนวนเฟรมติดกัน",
        "hint_filter": "ติ๊กถูก = ล็อคเฉพาะเมื่อเจอทั้งสีหลักและสีเงื่อนไขพร้อมกัน\nไม่ติ๊ก = ล็อคตามสีหลักอย่างเดียว (ไม่สนใจสีเงื่อนไข)",
        "lbl_marker_diag": "พิกเซลสีเงื่อนไข:",

        "lbl_movement_comp": "ชดเชยการเคลื่อนไหว",
        "lbl_kp": "เกนสัดส่วน (kp)",
        "lbl_kd": "เกนอนุพันธ์ (kd)",
        "lbl_track_radius": "รัศมีการติดตาม",

        "lbl_profile_name": "ชื่อโปรไฟล์",
        "autosave_text": "ทุกการเปลี่ยนแปลงจะถูกบันทึกอัตโนมัติ",
        "confirm_reset_title": "รีเซ็ตทั้งหมด",
        "confirm_reset_msg": "รีเซ็ตค่าทั้งหมดและโปรไฟล์? ไม่สามารถย้อนกลับได้",

        "lbl_overlay_color": "สีออฟเวอร์เลย์",
        "lbl_folder_path": "ที่อยู่โฟลเดอร์",
        "lbl_crosshair_style": "รูปแบบเป้า",
        "lbl_crosshair_size": "ขนาดเป้า",

        "ch_dot": "จุด", "ch_cross": "กากบาท",
        "ch_circle": "วงกลม", "ch_t_cross": "ตัว T",
        "ch_x_cross": "ตัว X", "ch_chevron": "ตัว V",
        "ch_dot_circle": "จุดในวง", "ch_brackets": "วงเล็บมุม",
    },
    "en": {
        "app_name": "Aimzen",
        "lang_btn": "TH",
        "status_running": "Running",
        "status_stopped": "Stopped",
        "btn_start": "Start", "btn_stop": "Stop",

        "nav_aim": "Aimbot", "nav_visual": "Visual",
        "nav_filter": "Color Filter", "nav_adv": "Advanced",
        "nav_prof": "Profiles",

        "sec_target_color": "Target Color",
        "sec_aim_response": "Aim Response",
        "sec_key_bindings": "Key Bindings",
        "sec_overlay": "Overlay",
        "sec_palette_folder": "Palette Folder",
        "sec_palette_detected": "Detected Palette",
        "sec_filter_rule": "Filter Rule",
        "sec_marker_color": "Condition Color",
        "sec_marker_detection": "Condition Color Detection",
        "sec_diag": "Diagnostics",
        "sec_movement": "Movement Compensation",
        "sec_gains": "Controller Gains",
        "sec_tracking": "Target Tracking",
        "sec_profiles": "Saved Profiles",
        "sec_storage": "Storage",
        "sec_crosshair": "Crosshair Style",

        "lbl_hex": "Hex",
        "lbl_hue_tol": "Hue tolerance",
        "lbl_sat_tol": "Saturation tolerance",
        "lbl_bri_tol": "Brightness tolerance",
        "lbl_lock_strength": "Lock strength",
        "lbl_smoothing": "Smoothing",
        "lbl_mouse_sens": "Mouse sensitivity",
        "lbl_aim_sens": "Aim sensitivity",
        "lbl_game_sens": "Game sensitivity",
        "lbl_max_step": "Max step per frame",
        "lbl_deadzone": "Dead zone",
        "lbl_fov": "Field of view",
        "lbl_offset_x": "Aim offset X",
        "lbl_offset_y": "Aim offset Y",
        "lbl_aim_key": "Aim key",
        "lbl_enable_toggle": "Enable toggle",
        "lbl_panic_key": "Panic key",

        "btn_apply": "Apply",
        "btn_load_img": "Load single image",
        "btn_scan": "Scan folder",
        "btn_lock_color": "Lock chosen color",
        "btn_clear_palette": "Clear palette",
        "btn_save": "Save",
        "btn_rename": "Rename",
        "btn_delete": "Delete",
        "btn_load_sel": "Load Selected Profile",
        "btn_reset_all": "Reset All",
        "btn_auto_detect": "Auto-detect blue condition color",

        "chk_show_fov": "Show FOV ring",
        "chk_show_crosshair": "Show center cross",
        "chk_show_box": "Show target box",
        "chk_show_fps": "Show FPS counter",
        "chk_show_aim_line": "Show aim line",
        "chk_hide_idle": "Hide overlay when not aiming",
        "chk_exclude_capture": "Hide from screen capture",
        "chk_rainbow": "Rainbow mode",
        "chk_ignore_dark": "Ignore dark pixels",
        "chk_ignore_gray": "Ignore gray pixels",

        "chk_enable_filter": "Enable condition color check",
        "lbl_marker_hexes": "Condition hexes (comma-separated)",
        "lbl_sample_radius": "Sample radius (px)",
        "lbl_required_px": "Required pixel count",
        "lbl_required_frames": "Required consecutive frames",
        "hint_filter": "Checked   = lock only when BOTH the main color and the condition color are detected.\nUnchecked = lock based on the main color only (condition color is ignored).",
        "lbl_marker_diag": "Condition pixels:",

        "lbl_movement_comp": "Movement comp",
        "lbl_kp": "Proportional gain (kp)",
        "lbl_kd": "Derivative gain (kd)",
        "lbl_track_radius": "Track radius",

        "lbl_profile_name": "Profile name",
        "autosave_text": "Every change is auto-saved to the configs folder.",
        "confirm_reset_title": "Reset All",
        "confirm_reset_msg": "Reset all settings and profiles? This cannot be undone.",

        "lbl_overlay_color": "Overlay color",
        "lbl_folder_path": "Folder path",
        "lbl_crosshair_style": "Crosshair style",
        "lbl_crosshair_size": "Crosshair size",

        "ch_dot": "Dot", "ch_cross": "Cross",
        "ch_circle": "Circle", "ch_t_cross": "T-Cross",
        "ch_x_cross": "X-Cross", "ch_chevron": "Chevron",
        "ch_dot_circle": "Dot + Circle", "ch_brackets": "Corner Brackets",
    },
}

sigma_current_lang_67 = "th"

def rizz_T_67(key):
    return SKIBIDY_LANG_67[sigma_current_lang_67].get(key, key)

skibidy_i18n_widgets_67 = []
rizz_i18n_menus_67 = []

def sigma_reg_67(widget, key, prop="text", uppercase=False):
    skibidy_i18n_widgets_67.append((widget, key, prop, uppercase))

def skibidy_reg_menu_67(menu, keys):
    rizz_i18n_menus_67.append((menu, keys))

def rizz_apply_lang_67():
    for w, key, prop, up in skibidy_i18n_widgets_67:
        try:
            txt = rizz_T_67(key)
            if up:
                txt = txt.upper()
            w.configure(**{prop: txt})
        except Exception:
            pass
    for menu, keys in rizz_i18n_menus_67:
        try:
            menu.configure(values=[rizz_T_67(k) for k in keys])
        except Exception:
            pass
    try:
        skibidy_ch_style_var_67.set(rizz_T_67(f"ch_{sigma_ch_style_internal_67['v']}"))
    except Exception:
        pass
    try:
        if rizz_running_67:
            skibidy_start_stop_btn_67.configure(text=rizz_T_67("btn_stop"))
        else:
            skibidy_start_stop_btn_67.configure(text=rizz_T_67("btn_start"))
    except Exception:
        pass
    try:
        sigma_lang_btn_67.configure(text=rizz_T_67("lang_btn"))
    except Exception:
        pass
    try:
        sigma_status_var_67.set(rizz_T_67("status_running") if rizz_running_67 else rizz_T_67("status_stopped"))
    except Exception:
        pass
    try:
        skibidy_switch_tab_67(sigma_active_tab_67["name"])
    except Exception:
        pass

def sigma_toggle_lang_67():
    global sigma_current_lang_67
    sigma_current_lang_67 = "en" if sigma_current_lang_67 == "th" else "th"
    rizz_apply_lang_67()
    rizz_auto_save_67()

RIZZ_BG_DARK_67      = "#0d0d0d"
SKIBIDY_BG_SIDEBAR_67 = "#111111"
SIGMA_BG_PANEL_67    = "#151515"
RIZZ_BG_CARD_67      = "#1a1a1a"
SKIBIDY_BG_INPUT_67   = "#202020"
SIGMA_TITLEBAR_BG_67 = "#282828"
RIZZ_PRIMARY_67      = "#2a2a2a"
SKIBIDY_PRIMARY_HOV_67 = "#3a3a3a"
SIGMA_ACCENT_67      = "#2c5cff"
RIZZ_ACCENT_HOV_67   = "#4a75ff"
SKIBIDY_ACCENT_DIM_67 = "#1e3f9c"
SIGMA_DANGER_67      = "#c9304a"
RIZZ_DANGER_HOV_67   = "#e04a63"
SKIBIDY_TEXT_LIGHT_67 = "#e6e6e6"
SIGMA_TEXT_DIM_67    = "#7a7a7a"
RIZZ_BORDER_67       = "#242424"

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("dark-blue")

SKIBIDY_WIN_W_67 = 780
RIZZ_WIN_H_67 = 560
SIGMA_SIDEBAR_W_67 = 160

rizz_root_67 = customtkinter.CTk()
rizz_root_67.title("Aimzen")
rizz_root_67.geometry(f"{SKIBIDY_WIN_W_67}x{RIZZ_WIN_H_67}")
rizz_root_67.resizable(False, False)
rizz_root_67.configure(fg_color=RIZZ_BG_DARK_67)
sigma_icon_path_67 = os.path.join(SKIBIDY_CONFIGS_DIR_67, "configs/icon.ico")
if os.path.exists(sigma_icon_path_67):
    try:
        rizz_root_67.iconbitmap(sigma_icon_path_67)
    except Exception as e:
        print(f"ไม่สามารถโหลด icon ได้: {e}")

skibidy_hex_var_67 = customtkinter.StringVar(value="#feffb2")
rizz_tol_h_var_67 = customtkinter.IntVar(value=10)
sigma_tol_s_var_67 = customtkinter.IntVar(value=60)
skibidy_tol_v_var_67 = customtkinter.IntVar(value=60)
rizz_strength_var_67 = customtkinter.DoubleVar(value=1.0)
sigma_stability_var_67 = customtkinter.DoubleVar(value=0.82)
skibidy_pf_mouse_var_67 = customtkinter.DoubleVar(value=0.5)
rizz_pf_aim_var_67 = customtkinter.DoubleVar(value=1.0)
sigma_roblox_sens_var_67 = customtkinter.DoubleVar(value=0.55)
skibidy_max_step_var_67 = customtkinter.IntVar(value=6)
rizz_deadzone_var_67 = customtkinter.IntVar(value=6)
sigma_fov_var_67 = customtkinter.IntVar(value=80)
skibidy_offset_x_var_67 = customtkinter.IntVar(value=0)
rizz_offset_y_var_67 = customtkinter.IntVar(value=0)
sigma_show_fov_var_67 = customtkinter.BooleanVar(value=False)
skibidy_show_crosshair_var_67 = customtkinter.BooleanVar(value=False)
rizz_exclude_capture_var_67 = customtkinter.BooleanVar(value=False)
sigma_status_var_67 = customtkinter.StringVar(value="Stopped")
skibidy_folder_var_67 = customtkinter.StringVar(value="images")
rizz_skip_dark_var_67 = customtkinter.BooleanVar(value=True)
sigma_skip_gray_var_67 = customtkinter.BooleanVar(value=True)
skibidy_selected_palette_index_67 = customtkinter.IntVar(value=-1)
rizz_config_name_var_67 = customtkinter.StringVar(value="")

sigma_ov_show_box_var_67 = customtkinter.BooleanVar(value=True)
skibidy_ov_show_fps_var_67 = customtkinter.BooleanVar(value=True)
rizz_ov_show_aim_line_var_67 = customtkinter.BooleanVar(value=False)
sigma_ov_hide_idle_var_67 = customtkinter.BooleanVar(value=False)
skibidy_ov_rainbow_var_67 = customtkinter.BooleanVar(value=False)
rizz_ov_color_var_67 = customtkinter.StringVar(value="#ff4040")

sigma_verify_enable_var_67 = customtkinter.BooleanVar(value=rizz_verify_enabled_67)
skibidy_verify_hexes_var_67 = customtkinter.StringVar(value=", ".join(sigma_verify_hex_list_67))
rizz_verify_tol_h_var_67 = customtkinter.IntVar(value=skibidy_verify_tol_h_67)
sigma_verify_tol_s_var_67 = customtkinter.IntVar(value=rizz_verify_tol_s_67)
skibidy_verify_tol_v_var_67 = customtkinter.IntVar(value=sigma_verify_tol_v_67)
rizz_verify_roi_var_67 = customtkinter.IntVar(value=skibidy_verify_roi_67)
sigma_verify_minpx_var_67 = customtkinter.IntVar(value=rizz_verify_min_px_67)
skibidy_verify_frames_var_67 = customtkinter.IntVar(value=sigma_verify_frames_required_67)

rizz_movement_compensation_var_67 = customtkinter.DoubleVar(value=sigma_movement_compensation_67)
sigma_kp_var_67 = customtkinter.DoubleVar(value=sigma_kp_67)
skibidy_kd_var_67 = customtkinter.DoubleVar(value=skibidy_kd_67)
rizz_roi_radius_var_67 = customtkinter.IntVar(value=sigma_roi_radius_67)

SIGMA_CROSSHAIR_STYLES_67 = ["dot", "cross", "circle", "t_cross",
                              "x_cross", "chevron", "dot_circle", "brackets"]
sigma_ch_style_internal_67 = {"v": "cross"}
skibidy_ch_style_var_67 = customtkinter.StringVar(value=rizz_T_67("ch_cross"))
rizz_ch_size_var_67 = customtkinter.IntVar(value=12)

skibidy_strength_display_67 = customtkinter.StringVar(value=f"{rizz_strength_var_67.get():.2f}")
rizz_stability_display_67 = customtkinter.StringVar(value=f"{sigma_stability_var_67.get():.2f}")
sigma_pf_mouse_display_67 = customtkinter.StringVar(value=f"{skibidy_pf_mouse_var_67.get():.2f}")
skibidy_pf_aim_display_67 = customtkinter.StringVar(value=f"{rizz_pf_aim_var_67.get():.2f}")
rizz_roblox_display_67 = customtkinter.StringVar(value=f"{sigma_roblox_sens_var_67.get():.2f}")
sigma_max_step_display_67 = customtkinter.StringVar(value=str(int(skibidy_max_step_var_67.get())))
skibidy_deadzone_display_67 = customtkinter.StringVar(value=str(int(rizz_deadzone_var_67.get())))
rizz_fov_display_67 = customtkinter.StringVar(value=str(int(sigma_fov_var_67.get())))
sigma_offset_x_display_67 = customtkinter.StringVar(value=str(int(skibidy_offset_x_var_67.get())))
skibidy_offset_y_display_67 = customtkinter.StringVar(value=str(int(rizz_offset_y_var_67.get())))
rizz_tol_h_display_67 = customtkinter.StringVar(value=str(int(rizz_tol_h_var_67.get())))
sigma_tol_s_display_67 = customtkinter.StringVar(value=str(int(sigma_tol_s_var_67.get())))
skibidy_tol_v_display_67 = customtkinter.StringVar(value=str(int(skibidy_tol_v_var_67.get())))
rizz_movement_display_67 = customtkinter.StringVar(value=f"{rizz_movement_compensation_var_67.get():.2f}")
sigma_kp_display_67 = customtkinter.StringVar(value=f"{sigma_kp_var_67.get():.2f}")
skibidy_kd_display_67 = customtkinter.StringVar(value=f"{skibidy_kd_var_67.get():.2f}")
rizz_roi_radius_display_67 = customtkinter.StringVar(value=str(int(rizz_roi_radius_var_67.get())))
sigma_ch_size_display_67 = customtkinter.StringVar(value=str(int(rizz_ch_size_var_67.get())))

skibidy_value_labels_67 = {}

def sigma_round_to_2_67(value):
    return round(float(value), 2)

def skibidy_format_value_67(value, digits=2):
    return f"{sigma_round_to_2_67(value):.{digits}f}"

def rizz_update_swatch_67(canvas, hex_str):
    try:
        canvas.delete("all")
        canvas.create_rectangle(0, 0, 60, 26, fill=hex_str, outline="#000000")
    except Exception:
        pass

def sigma_find_scrollable_67(w):
    while w is not None:
        if isinstance(w, customtkinter.CTkScrollableFrame):
            return w
        w = getattr(w, "master", None)
    return None

def skibidy_neutralize_slider_wheel_67(slider, scrollable):
    if scrollable is None:
        return
    def wheel(event):
        try:
            scrollable._parent_canvas.yview_scroll(int(-event.delta / 120), "units")
        except Exception:
            pass
        return "break"
    for attr in ("_canvas", "_button"):
        try:
            child = getattr(slider, attr, None)
            if child is None:
                continue
            child.unbind("<MouseWheel>")
            child.bind("<MouseWheel>", wheel)
        except Exception:
            pass

def rizz_section_label_67(parent, key):
    f = customtkinter.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=14, pady=(10, 4))
    lbl = customtkinter.CTkLabel(f, text=rizz_T_67(key).upper(),
                                 font=("Segoe UI", 10, "bold"),
                                 text_color=SIGMA_TEXT_DIM_67, anchor="w")
    lbl.pack(side="left")
    sigma_reg_67(lbl, key, uppercase=True)
    return f

def sigma_add_slider_row_67(parent, label_key, var, display_var, key,
                            frm, to, is_int=False, on_change=None):
    card = customtkinter.CTkFrame(parent, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
    card.pack(fill="x", padx=14, pady=3)
    card.grid_columnconfigure(1, weight=1)
    lbl = customtkinter.CTkLabel(card, text=rizz_T_67(label_key),
                                 font=("Segoe UI", 11),
                                 text_color=SKIBIDY_TEXT_LIGHT_67, anchor="w", width=170)
    lbl.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=8)
    sigma_reg_67(lbl, label_key)
    s = customtkinter.CTkSlider(card, from_=frm, to=to, variable=var,
                                command=on_change if on_change else skibidy_update_params_67,
                                button_color=SIGMA_ACCENT_67,
                                button_hover_color=RIZZ_ACCENT_HOV_67,
                                progress_color=SKIBIDY_ACCENT_DIM_67,
                                fg_color=SKIBIDY_BG_INPUT_67, height=14)
    s.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
    val = customtkinter.CTkLabel(card, textvariable=display_var,
                                 font=("Consolas", 11),
                                 text_color=SIGMA_ACCENT_67, width=52, anchor="e")
    val.grid(row=0, column=2, sticky="e", padx=(6, 14), pady=8)
    if key:
        skibidy_value_labels_67[key] = display_var
    skibidy_neutralize_slider_wheel_67(s, sigma_find_scrollable_67(parent))
    return lbl

def rizz_add_checkbox_row_67(parent, text_key, var, command=None):
    card = customtkinter.CTkFrame(parent, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
    card.pack(fill="x", padx=14, pady=3)
    chk = customtkinter.CTkCheckBox(
        card, text=rizz_T_67(text_key), variable=var, command=command,
        font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
        fg_color=SIGMA_ACCENT_67, hover_color=RIZZ_ACCENT_HOV_67,
        border_color=RIZZ_BORDER_67, checkmark_color="#ffffff")
    chk.pack(anchor="w", padx=12, pady=8)
    sigma_reg_67(chk, text_key)
    return chk

def sigma_add_entry_row_67(parent, label_key, var, button_text_key=None, button_cmd=None):
    card = customtkinter.CTkFrame(parent, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
    card.pack(fill="x", padx=14, pady=3)
    card.grid_columnconfigure(1, weight=1)
    lbl = customtkinter.CTkLabel(card, text=rizz_T_67(label_key),
                                 font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                 width=170, anchor="w")
    lbl.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=8)
    sigma_reg_67(lbl, label_key)
    ent = customtkinter.CTkEntry(card, textvariable=var, height=28,
                                 fg_color=SKIBIDY_BG_INPUT_67, border_color=RIZZ_BORDER_67,
                                 text_color=SKIBIDY_TEXT_LIGHT_67)
    ent.grid(row=0, column=1, sticky="ew", padx=6, pady=8)
    btn = None
    if button_text_key:
        btn = customtkinter.CTkButton(
            card, text=rizz_T_67(button_text_key), command=button_cmd,
            width=60, height=28, fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67,
            text_color=SKIBIDY_TEXT_LIGHT_67, corner_radius=4, font=("Segoe UI", 11))
        btn.grid(row=0, column=2, padx=(0, 12), pady=8)
        sigma_reg_67(btn, button_text_key)
    return ent, btn

def skibidy_refresh_value_labels_67():
    for key, var in skibidy_value_labels_67.items():
        try:
            if key in ("strength", "stability", "pf_mouse", "pf_aim",
                       "roblox_sens", "movement_comp", "kp_gain", "kd_gain"):
                src = {
                    "strength": rizz_strength_var_67, "stability": sigma_stability_var_67,
                    "pf_mouse": skibidy_pf_mouse_var_67, "pf_aim": rizz_pf_aim_var_67,
                    "roblox_sens": sigma_roblox_sens_var_67,
                    "movement_comp": rizz_movement_compensation_var_67,
                    "kp_gain": sigma_kp_var_67, "kd_gain": skibidy_kd_var_67,
                }[key]
                var.set(skibidy_format_value_67(src.get()))
            elif key in ("max_step", "deadzone", "fov", "offset_x", "offset_y",
                         "tol_h", "tol_s", "tol_v", "roi_radius",
                         "ch_size"):
                src = {
                    "max_step": skibidy_max_step_var_67, "deadzone": rizz_deadzone_var_67,
                    "fov": sigma_fov_var_67, "offset_x": skibidy_offset_x_var_67,
                    "offset_y": rizz_offset_y_var_67, "tol_h": rizz_tol_h_var_67,
                    "tol_s": sigma_tol_s_var_67, "tol_v": skibidy_tol_v_var_67,
                    "roi_radius": rizz_roi_radius_var_67, "ch_size": rizz_ch_size_var_67,
                }[key]
                var.set(str(int(src.get())))
        except Exception:
            pass

def skibidy_update_params_67(*args):
    global rizz_lock_strength_67, sigma_smooth_alpha_67
    global rizz_max_step_px_67, skibidy_deadzone_px_67
    global skibidy_fov_radius_67, sigma_offset_x_67, skibidy_offset_y_67
    global sigma_roblox_sensitivity_67, skibidy_pf_mouse_sensitivity_67
    global rizz_pf_aim_sensitivity_67
    global sigma_movement_compensation_67, sigma_kp_67, skibidy_kd_67, sigma_roi_radius_67

    rizz_lock_strength_67 = sigma_round_to_2_67(rizz_strength_var_67.get())
    sigma_smooth_alpha_67 = max(0.01, 1.0 - sigma_round_to_2_67(sigma_stability_var_67.get()))
    rizz_max_step_px_67 = int(skibidy_max_step_var_67.get())
    skibidy_deadzone_px_67 = int(rizz_deadzone_var_67.get())
    skibidy_fov_radius_67 = int(sigma_fov_var_67.get())
    sigma_offset_x_67 = int(skibidy_offset_x_var_67.get())
    skibidy_offset_y_67 = int(rizz_offset_y_var_67.get())
    sigma_roblox_sensitivity_67 = float(sigma_roblox_sens_var_67.get())
    skibidy_pf_mouse_sensitivity_67 = float(skibidy_pf_mouse_var_67.get())
    rizz_pf_aim_sensitivity_67 = float(rizz_pf_aim_var_67.get())
    try:
        sigma_movement_compensation_67 = sigma_round_to_2_67(rizz_movement_compensation_var_67.get())
        sigma_kp_67 = sigma_round_to_2_67(sigma_kp_var_67.get())
        skibidy_kd_67 = sigma_round_to_2_67(skibidy_kd_var_67.get())
        sigma_roi_radius_67 = int(rizz_roi_radius_var_67.get())
    except Exception:
        pass
    skibidy_refresh_value_labels_67()
    rizz_auto_save_67()

def rizz_update_verify_params_67():
    global rizz_verify_enabled_67, skibidy_verify_tol_h_67, rizz_verify_tol_s_67
    global sigma_verify_tol_v_67, skibidy_verify_roi_67, rizz_verify_min_px_67
    global sigma_verify_frames_required_67, sigma_verify_hex_list_67

    rizz_verify_enabled_67 = bool(sigma_verify_enable_var_67.get())
    skibidy_verify_tol_h_67 = int(rizz_verify_tol_h_var_67.get())
    rizz_verify_tol_s_67 = int(sigma_verify_tol_s_var_67.get())
    sigma_verify_tol_v_67 = int(skibidy_verify_tol_v_var_67.get())
    skibidy_verify_roi_67 = int(rizz_verify_roi_var_67.get())
    rizz_verify_min_px_67 = int(sigma_verify_minpx_var_67.get())
    sigma_verify_frames_required_67 = max(1, int(skibidy_verify_frames_var_67.get()))

    def parse_list(s):
        return [x.strip() for x in s.split(",")
                if x.strip().startswith("#") and len(x.strip()) == 7]

    sigma_verify_hex_list_67 = parse_list(skibidy_verify_hexes_var_67.get()) or ["#3AA0FF"]
    skibidy_rebuild_verify_ranges_67()
    rizz_auto_save_67()

def sigma_on_ch_style_change_67(choice):
    for k in SIGMA_CROSSHAIR_STYLES_67:
        if rizz_T_67(f"ch_{k}") == choice:
            sigma_ch_style_internal_67["v"] = k
            break
    rizz_auto_save_67()

_rizz_auto_save_ready_67 = False

def rizz_auto_save_67():
    if not _rizz_auto_save_ready_67:
        return
    sigma_save_settings_67()

def skibidy_draw_crosshair_67(canvas, cx, cy, color, size):
    style = sigma_ch_style_internal_67["v"]
    h = size
    t = 2

    if style == "dot":
        r = max(1, size // 6)
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                           fill=color, outline="")
    elif style == "cross":
        canvas.create_line(cx - h, cy, cx + h, cy, fill=color, width=t)
        canvas.create_line(cx, cy - h, cx, cy + h, fill=color, width=t)
    elif style == "circle":
        canvas.create_oval(cx - h, cy - h, cx + h, cy + h,
                           outline=color, width=t)
    elif style == "t_cross":
        canvas.create_line(cx - h, cy, cx + h, cy, fill=color, width=t)
        canvas.create_line(cx, cy - h, cx, cy, fill=color, width=t)
    elif style == "x_cross":
        d = int(h * 0.7)
        canvas.create_line(cx - d, cy - d, cx + d, cy + d, fill=color, width=t)
        canvas.create_line(cx - d, cy + d, cx + d, cy - d, fill=color, width=t)
    elif style == "chevron":
        d = int(h * 0.7)
        canvas.create_line(cx - d, cy + d // 2, cx, cy - d, fill=color, width=t)
        canvas.create_line(cx, cy - d, cx + d, cy + d // 2, fill=color, width=t)
    elif style == "dot_circle":
        canvas.create_oval(cx - h, cy - h, cx + h, cy + h,
                           outline=color, width=t)
        r = max(1, size // 8)
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                           fill=color, outline="")
    elif style == "brackets":
        g = max(3, size // 3)
        L = h
        canvas.create_line(cx - L, cy - L, cx - L + g, cy - L, fill=color, width=t)
        canvas.create_line(cx - L, cy - L, cx - L, cy - L + g, fill=color, width=t)
        canvas.create_line(cx + L, cy - L, cx + L - g, cy - L, fill=color, width=t)
        canvas.create_line(cx + L, cy - L, cx + L, cy - L + g, fill=color, width=t)
        canvas.create_line(cx - L, cy + L, cx - L + g, cy + L, fill=color, width=t)
        canvas.create_line(cx - L, cy + L, cx - L, cy + L - g, fill=color, width=t)
        canvas.create_line(cx + L, cy + L, cx + L - g, cy + L, fill=color, width=t)
        canvas.create_line(cx + L, cy + L, cx + L, cy + L - g, fill=color, width=t)

def rizz_ensure_overlay_67():
    global skibidy_overlay_67, rizz_overlay_canvas_67, sigma_transparent_supported_67
    if skibidy_overlay_67 is None or not skibidy_overlay_67.winfo_exists():
        skibidy_overlay_67 = tk.Toplevel(rizz_root_67)
        skibidy_overlay_67.overrideredirect(True)
        skibidy_overlay_67.attributes("-topmost", True)
        try:
            skibidy_overlay_67.attributes("-transparentcolor", "magenta")
            bgc = "magenta"
            sigma_transparent_supported_67 = True
        except Exception:
            bgc = "black"
            sigma_transparent_supported_67 = False
        skibidy_overlay_67.configure(bg=bgc)
        rizz_overlay_canvas_67 = tk.Canvas(skibidy_overlay_67, bg=bgc,
                                           highlightthickness=0, width=100, height=100)
        rizz_overlay_canvas_67.pack(fill="both", expand=True)
        skibidy_overlay_67.update_idletasks()
        sigma_make_click_through_67(skibidy_overlay_67, sigma_transparent_supported_67)
        if rizz_exclude_capture_var_67.get():
            rizz_exclude_from_capture_67(skibidy_overlay_67, True)
    skibidy_update_overlay_67()

def skibidy_update_overlay_67():
    global skibidy_overlay_67, rizz_overlay_canvas_67
    if skibidy_overlay_67 is None or not skibidy_overlay_67.winfo_exists():
        return

    aiming = win32api.GetAsyncKeyState(rizz_current_aim_vk_67) < 0
    hide_idle = sigma_ov_hide_idle_var_67.get()

    want_fov = sigma_show_fov_var_67.get() and not (hide_idle and not aiming)
    want_cross = skibidy_show_crosshair_var_67.get() and not (hide_idle and not aiming)
    want_box = sigma_ov_show_box_var_67.get() and not (hide_idle and not aiming)
    want_line = rizz_ov_show_aim_line_var_67.get() and not (hide_idle and not aiming)
    want_fps = skibidy_ov_show_fps_var_67.get()

    if not (want_fov or want_cross or want_box or want_line or want_fps):
        rizz_hide_overlay_67()
        return

    skibidy_overlay_67.geometry(f"{sigma_config_67.sigma_width_67}x{sigma_config_67.skibidy_height_67}+0+0")
    skibidy_overlay_67.deiconify()
    skibidy_overlay_67.lift()
    rizz_overlay_canvas_67.delete("all")

    color = rizz_rainbow_color_67() if skibidy_ov_rainbow_var_67.get() else (rizz_ov_color_var_67.get() or "#ff4040")

    if want_fov:
        r = int(sigma_fov_var_67.get())
        cx = sigma_config_67.rizz_center_x_67
        cy = sigma_config_67.rizz_center_y_67
        rizz_overlay_canvas_67.create_oval(cx - r, cy - r, cx + r, cy + r,
                                           outline=color, width=2)

    if want_cross:
        cx = sigma_config_67.rizz_center_x_67
        cy = sigma_config_67.rizz_center_y_67
        skibidy_draw_crosshair_67(rizz_overlay_canvas_67, cx, cy, color, int(rizz_ch_size_var_67.get()))

    if want_box and skibidy_overlay_target_screen_67 is not None:
        tx, ty = skibidy_overlay_target_screen_67
        box = int(rizz_verify_roi_67)
        rizz_overlay_canvas_67.create_rectangle(tx - box, ty - box,
                                                tx + box, ty + box,
                                                outline=color, width=2)

    if want_line and skibidy_overlay_target_screen_67 is not None:
        tx, ty = skibidy_overlay_target_screen_67
        rizz_overlay_canvas_67.create_line(sigma_config_67.rizz_center_x_67,
                                           sigma_config_67.rizz_center_y_67,
                                           tx, ty, fill=color, width=1)

    if want_fps:
        rizz_overlay_canvas_67.create_text(20, 20, anchor="nw",
                                           text=f"FPS {rizz_overlay_fps_value_67}",
                                           fill=color, font=("Consolas", 12, "bold"))

def rizz_hide_overlay_67():
    global skibidy_overlay_67
    try:
        if skibidy_overlay_67 and skibidy_overlay_67.winfo_exists():
            skibidy_overlay_67.withdraw()
    except Exception:
        pass

def sigma_on_capture_exclude_toggle_67():
    en = rizz_exclude_capture_var_67.get()
    rizz_exclude_from_capture_67(rizz_root_67, en)
    if skibidy_overlay_67 and skibidy_overlay_67.winfo_exists():
        rizz_exclude_from_capture_67(skibidy_overlay_67, en)
    rizz_auto_save_67()

def skibidy_run_loop_67():
    global rizz_running_67, skibidy_track_cx_67, rizz_track_cy_67
    global skibidy_prev_toggle_state_67
    global skibidy_ema_dx_67, rizz_ema_dy_67
    global rizz_prev_err_x_67, sigma_prev_err_y_67
    global skibidy_aim_enabled_67, rizz_current_aim_vk_67
    global sigma_current_toggle_vk_67
    global skibidy_overlay_target_screen_67, rizz_overlay_fps_value_67

    rizz_running_67 = True
    s = mss.MSS()
    fps_frames = 0
    fps_t0 = time.time()
    while rizz_running_67:
        time.sleep(0.001)
        try:
            GameFrame = np.array(s.grab(rizz_region_c_67))
            GameFrame = cv2.cvtColor(GameFrame, cv2.COLOR_BGRA2BGR)
        except Exception:
            continue

        skibidy_overlay_target_screen_67 = None

        tk_state = win32api.GetAsyncKeyState(sigma_current_toggle_vk_67)
        if tk_state < 0 and skibidy_prev_toggle_state_67 >= 0:
            skibidy_aim_enabled_67 = not skibidy_aim_enabled_67
            if skibidy_aim_enabled_67:
                rizz_beep_sigma_on_67()
            else:
                rizz_beep_sigma_off_67()
        skibidy_prev_toggle_state_67 = tk_state

        if win32api.GetAsyncKeyState(0x75) < 0:
            break

        if skibidy_aim_enabled_67 and win32api.GetAsyncKeyState(rizz_current_aim_vk_67) < 0:
            frame_hsv = cv2.cvtColor(GameFrame, cv2.COLOR_BGR2HSV)
            mask = skibidy_build_mask_67(frame_hsv)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, skibidy_kernel_67, iterations=1)
            mask = cv2.dilate(mask, skibidy_kernel_67, iterations=1)
            mask = cv2.medianBlur(mask, 5)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                centroids = []
                for c in contours:
                    a = cv2.contourArea(c)
                    if a < rizz_min_area_67:
                        continue
                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        continue
                    cx_i = M["m10"] / M["m00"]
                    cy_i = M["m01"] / M["m00"]
                    if (cx_i - skibidy_crosshair_u_67) ** 2 + (cy_i - skibidy_crosshair_u_67) ** 2 <= skibidy_fov_radius_67 ** 2:
                        centroids.append((cx_i, cy_i, a, c))
                if centroids:
                    if skibidy_track_cx_67 is not None:
                        near = [t for t in centroids
                                if (t[0] - skibidy_track_cx_67) ** 2 +
                                   (t[1] - rizz_track_cy_67) ** 2 <= sigma_roi_radius_67 ** 2]
                        chosen = min(near, key=lambda t: (t[0] - skibidy_track_cx_67) ** 2 +
                                                          (t[1] - rizz_track_cy_67) ** 2) \
                                 if near else max(centroids, key=lambda t: t[2])
                    else:
                        chosen = max(centroids, key=lambda t: t[2])
                    cx, cy, area, _ = chosen

                    if not rizz_verify_target_67(frame_hsv, cx, cy):
                        skibidy_track_cx_67, rizz_track_cy_67 = None, None
                        rizz_prev_err_x_67 = 0.0
                        sigma_prev_err_y_67 = 0.0
                        skibidy_ema_dx_67 = 0.0
                        rizz_ema_dy_67 = 0.0
                        continue

                    skibidy_track_cx_67, rizz_track_cy_67 = cx, cy
                    skibidy_overlay_target_screen_67 = (rizz_region_c_67["left"] + int(cx),
                                                        rizz_region_c_67["top"] + int(cy))

                    target_x = cx + sigma_offset_x_67
                    target_y = cy + skibidy_offset_y_67
                    err_x = -(skibidy_crosshair_u_67 - target_x)
                    err_y = -(skibidy_crosshair_u_67 - target_y)
                    if abs(err_x) < skibidy_deadzone_px_67:
                        err_x = 0.0
                    if abs(err_y) < skibidy_deadzone_px_67:
                        err_y = 0.0

                    cross_x = np.sign(err_x) != np.sign(rizz_prev_err_x_67)
                    cross_y = np.sign(err_y) != np.sign(sigma_prev_err_y_67)
                    scale_x = np.tanh(abs(err_x) / 10.0)
                    scale_y = np.tanh(abs(err_y) / 10.0)

                    PF_sensitivity = skibidy_pf_mouse_sensitivity_67 * rizz_pf_aim_sensitivity_67
                    finalMult = ((sigma_roblox_sensitivity_67 * PF_sensitivity) / 0.55 +
                                 sigma_movement_compensation_67) * rizz_lock_strength_67

                    dx_raw = (sigma_kp_67 * err_x + skibidy_kd_67 * (err_x - rizz_prev_err_x_67)) * finalMult * scale_x
                    dy_raw = (sigma_kp_67 * err_y + skibidy_kd_67 * (err_y - sigma_prev_err_y_67)) * finalMult * scale_y
                    if cross_x:
                        dx_raw *= 0.5
                    if cross_y:
                        dy_raw *= 0.5
                    dx_raw = float(np.clip(dx_raw, -rizz_max_step_px_67, rizz_max_step_px_67))
                    dy_raw = float(np.clip(dy_raw, -rizz_max_step_px_67, rizz_max_step_px_67))
                    skibidy_ema_dx_67 = (1 - sigma_smooth_alpha_67) * skibidy_ema_dx_67 + sigma_smooth_alpha_67 * dx_raw
                    rizz_ema_dy_67 = (1 - sigma_smooth_alpha_67) * rizz_ema_dy_67 + sigma_smooth_alpha_67 * dy_raw
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,
                                         int(skibidy_ema_dx_67), int(rizz_ema_dy_67), 0, 0)
                    rizz_prev_err_x_67 = err_x
                    sigma_prev_err_y_67 = err_y

        fps_frames += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            rizz_overlay_fps_value_67 = fps_frames
            fps_frames = 0
            fps_t0 = now

    cv2.destroyAllWindows()

def sigma_run_worker_67():
    global sigma_worker_67
    if sigma_worker_67 and sigma_worker_67.is_alive():
        return
    sigma_worker_67 = threading.Thread(target=skibidy_run_loop_67, daemon=True)
    sigma_worker_67.start()
    sigma_status_var_67.set(rizz_T_67("status_running"))

def skibidy_stop_worker_67():
    global rizz_running_67
    rizz_running_67 = False
    sigma_status_var_67.set(rizz_T_67("status_stopped"))
    rizz_hide_overlay_67()

_rizz_capturing_67 = None
_skibidy_capture_start_67 = 0.0

def sigma_clear_capture_67():
    global _rizz_capturing_67
    _rizz_capturing_67 = None
    try:
        skibidy_aim_vk_btn_67.configure(text=rizz_vk_display_67(rizz_current_aim_vk_67))
        rizz_toggle_vk_btn_67.configure(text=rizz_vk_display_67(sigma_current_toggle_vk_67))
    except Exception:
        pass

def skibidy_cancel_capture_67():
    sigma_clear_capture_67()

def rizz_start_capture_67(which):
    global _rizz_capturing_67, _skibidy_capture_start_67
    if _rizz_capturing_67:
        skibidy_cancel_capture_67()
        return
    _rizz_capturing_67 = which
    _skibidy_capture_start_67 = time.time()
    btn = {"aim": skibidy_aim_vk_btn_67, "toggle": rizz_toggle_vk_btn_67}.get(which)
    if btn is not None:
        btn.configure(text="...")

    def finish(vk):
        global rizz_current_aim_vk_67, sigma_current_toggle_vk_67
        if which == "aim":
            rizz_current_aim_vk_67 = vk
            skibidy_aim_vk_btn_67.configure(text=rizz_vk_display_67(vk))
        elif which == "toggle":
            sigma_current_toggle_vk_67 = vk
            rizz_toggle_vk_btn_67.configure(text=rizz_vk_display_67(vk))
        sigma_clear_capture_67()
        rizz_auto_save_67()

    def on_key(key):
        if _rizz_capturing_67 != which:
            return False
        if time.time() - _skibidy_capture_start_67 < 0.25:
            return
        vk = sigma_key_event_to_vk_67(key)
        if vk == 0x1B:
            skibidy_cancel_capture_67()
            return False
        if vk:
            finish(vk)
        return False

    def on_mouse(x, y, button, pressed):
        if _rizz_capturing_67 != which:
            return False
        if not pressed:
            return
        if time.time() - _skibidy_capture_start_67 < 0.25:
            return
        vk = skibidy_mouse_event_to_vk_67(button)
        if vk:
            finish(vk)
        return False

    kl = pkeyboard.Listener(on_press=on_key)
    ml = pmouse.Listener(on_click=on_mouse)
    kl.daemon = True
    ml.daemon = True
    kl.start()
    ml.start()

sigma_palette_data_67 = []
skibidy_palette_row_widgets_67 = []

def rizz_refresh_palette_ui_67():
    for w in skibidy_palette_row_widgets_67:
        w.destroy()
    skibidy_palette_row_widgets_67.clear()

    for idx, item in enumerate(sigma_palette_data_67):
        row = customtkinter.CTkFrame(sigma_palette_inner_67, fg_color="transparent", height=22)
        row.pack(fill="x", pady=1)
        row.pack_propagate(False)

        rb = customtkinter.CTkRadioButton(
            row, text="", variable=skibidy_selected_palette_index_67, value=idx,
            width=18, radiobutton_width=14, radiobutton_height=14,
            fg_color=SIGMA_ACCENT_67, hover_color=RIZZ_ACCENT_HOV_67, border_color=RIZZ_BORDER_67,
            command=sigma_on_palette_radio_67)
        rb.pack(side="left", padx=(2, 6))

        sw = tk.Canvas(row, width=28, height=16, bg=SKIBIDY_BG_INPUT_67,
                       highlightthickness=1, highlightbackground=RIZZ_BORDER_67)
        sw.create_rectangle(0, 0, 28, 16, outline="", fill=item["hex"])
        sw.pack(side="left", padx=(0, 6))

        customtkinter.CTkLabel(row, text=item["hex"], font=("Consolas", 10),
                               text_color=SKIBIDY_TEXT_LIGHT_67, width=72,
                               anchor="w").pack(side="left")
        customtkinter.CTkLabel(row, text=f"{item['pct']:5.2f}%",
                               font=("Consolas", 10), text_color=SIGMA_TEXT_DIM_67,
                               width=60, anchor="e").pack(side="right", padx=(0, 6))
        skibidy_palette_row_widgets_67.append(row)

    if sigma_palette_data_67 and skibidy_selected_palette_index_67.get() < 0:
        skibidy_selected_palette_index_67.set(0)
        sigma_on_palette_radio_67(skip_hex=True)

def sigma_on_palette_radio_67(skip_hex=False):
    idx = skibidy_selected_palette_index_67.get()
    if 0 <= idx < len(sigma_palette_data_67):
        item = sigma_palette_data_67[idx]
        skibidy_hex_var_67.set(item["hex"])
        rizz_update_swatch_67(skibidy_swatch_canvas_67, item["hex"])

def skibidy_set_from_hex_67():
    global rizz_active_ranges_67
    s = skibidy_hex_var_67.get()
    rng = sigma_range_from_hex_67(s, int(rizz_tol_h_var_67.get()),
                                  int(sigma_tol_s_var_67.get()), int(skibidy_tol_v_var_67.get()))
    if rng:
        rizz_active_ranges_67 = [rng]
        sigma_set_locked_hex_67(s)
        rizz_update_swatch_67(skibidy_swatch_canvas_67, s)
        skibidy_refresh_locked_label_67()
    rizz_auto_save_67()

def rizz_choose_image_67():
    global sigma_palette_data_67
    p = filedialog.askopenfilename(
        filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
    if not p:
        return
    rows = skibidy_analyze_single_image_67(p, top_n=8)
    sigma_palette_data_67 = rows
    skibidy_selected_palette_index_67.set(-1)
    rizz_refresh_palette_ui_67()
    if rows:
        skibidy_hex_var_67.set(rows[0]["hex"])
        rizz_update_swatch_67(skibidy_swatch_canvas_67, rows[0]["hex"])

def sigma_analyze_images_folder_67():
    global sigma_palette_data_67
    folder = skibidy_folder_var_67.get().strip() or "images"
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception:
            return
    rows, processed, total = sigma_analyze_folder_colors_67(
        folder, skip_dark=rizz_skip_dark_var_67.get(),
        skip_gray=sigma_skip_gray_var_67.get(), top_n=12)
    sigma_palette_data_67 = rows
    skibidy_selected_palette_index_67.set(-1)
    rizz_refresh_palette_ui_67()
    if rows:
        skibidy_hex_var_67.set(rows[0]["hex"])
        rizz_update_swatch_67(skibidy_swatch_canvas_67, rows[0]["hex"])
    else:
        fallback = [
            {"hex": "#FEFFB2", "hsv": (30, 240, 255), "count": 0, "pct": 100.0},
            {"hex": "#FF0000", "hsv": (0, 255, 255), "count": 0, "pct": 0.0},
            {"hex": "#00FF00", "hsv": (60, 255, 255), "count": 0, "pct": 0.0},
            {"hex": "#0000FF", "hsv": (120, 255, 255), "count": 0, "pct": 0.0},
        ]
        sigma_palette_data_67 = fallback
        rizz_refresh_palette_ui_67()
        skibidy_hex_var_67.set(fallback[0]["hex"])
        rizz_update_swatch_67(skibidy_swatch_canvas_67, fallback[0]["hex"])

def rizz_lock_selected_color_67():
    global rizz_active_ranges_67
    idx = skibidy_selected_palette_index_67.get()
    if idx < 0 or idx >= len(sigma_palette_data_67):
        return
    item = sigma_palette_data_67[idx]
    rng = sigma_range_from_hex_67(item["hex"], int(rizz_tol_h_var_67.get()),
                                  int(sigma_tol_s_var_67.get()),
                                  int(skibidy_tol_v_var_67.get()))
    if rng:
        rizz_active_ranges_67 = [rng]
        sigma_set_locked_hex_67(item["hex"])
        rizz_update_swatch_67(skibidy_swatch_canvas_67, item["hex"])
        skibidy_hex_var_67.set(item["hex"])
        skibidy_refresh_locked_label_67()
    rizz_auto_save_67()

def sigma_clear_palette_67():
    global sigma_palette_data_67
    sigma_palette_data_67 = []
    skibidy_selected_palette_index_67.set(-1)
    rizz_refresh_palette_ui_67()

def skibidy_pick_folder_67():
    d = filedialog.askdirectory(initialdir=skibidy_folder_var_67.get() or ".")
    if d:
        skibidy_folder_var_67.set(d)
        rizz_auto_save_67()

def skibidy_refresh_locked_label_67():
    if not rizz_active_ranges_67 or not sigma_locked_hex_67:
        skibidy_locked_label_67.configure(text="—", text_color=SIGMA_TEXT_DIM_67)
        return
    if skibidy_locked_hsv_67 is not None:
        h, s, v = skibidy_locked_hsv_67
        skibidy_locked_label_67.configure(text=f"{sigma_locked_hex_67}   HSV[{h},{s},{v}]",
                                          text_color=SIGMA_ACCENT_67)
    else:
        skibidy_locked_label_67.configure(text=sigma_locked_hex_67, text_color=SIGMA_ACCENT_67)

rizz_root_67.grid_columnconfigure(0, minsize=SIGMA_SIDEBAR_W_67, weight=0)
rizz_root_67.grid_columnconfigure(1, weight=1)
rizz_root_67.grid_rowconfigure(0, weight=1)

skibidy_sidebar_67 = customtkinter.CTkFrame(rizz_root_67, fg_color=SKIBIDY_BG_SIDEBAR_67, corner_radius=0,
                                            width=SIGMA_SIDEBAR_W_67)
skibidy_sidebar_67.grid(row=0, column=0, sticky="nsew")
skibidy_sidebar_67.grid_propagate(False)

sigma_content_67 = customtkinter.CTkFrame(rizz_root_67, fg_color=SIGMA_BG_PANEL_67, corner_radius=0)
sigma_content_67.grid(row=0, column=1, sticky="nsew")
sigma_content_67.grid_propagate(False)
sigma_content_67.grid_rowconfigure(1, weight=1)
sigma_content_67.grid_columnconfigure(0, weight=1)

skibidy_sb_header_67 = customtkinter.CTkFrame(skibidy_sidebar_67, fg_color="transparent", height=54)
skibidy_sb_header_67.pack(fill="x", padx=10, pady=(12, 6))
skibidy_sb_header_67.pack_propagate(False)

skibidy_sb_logo_67 = tk.Canvas(skibidy_sb_header_67, width=26, height=26, bg=SKIBIDY_BG_SIDEBAR_67,
                               highlightthickness=0)
skibidy_sb_logo_67.pack(side="left", padx=(2, 8))
skibidy_sb_logo_67.create_oval(2, 2, 24, 24, outline=SIGMA_ACCENT_67, width=3)
skibidy_sb_logo_67.create_oval(9, 9, 17, 17, outline=SIGMA_ACCENT_67, width=2)

customtkinter.CTkLabel(skibidy_sb_header_67, text="Aimzen",
                       font=("Segoe UI", 15, "bold"),
                       text_color=SKIBIDY_TEXT_LIGHT_67, anchor="w").pack(side="left")

sigma_lang_btn_67 = customtkinter.CTkButton(
    skibidy_sb_header_67, text=rizz_T_67("lang_btn"), command=sigma_toggle_lang_67,
    width=36, height=24, fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67,
    text_color=SIGMA_ACCENT_67, font=("Segoe UI", 11, "bold"), corner_radius=4)
sigma_lang_btn_67.pack(side="right")

skibidy_sb_nav_67 = customtkinter.CTkFrame(skibidy_sidebar_67, fg_color="transparent")
skibidy_sb_nav_67.pack(fill="x", padx=10, pady=(0, 6))

rizz_nav_buttons_67 = {}
sigma_active_tab_67 = {"name": "aim"}

def skibidy_make_nav_button_67(parent, key, label_key):
    btn = customtkinter.CTkButton(
        parent, text=rizz_T_67(label_key), anchor="w",
        font=("Segoe UI", 12), fg_color="transparent",
        hover_color=RIZZ_PRIMARY_67, text_color=SIGMA_TEXT_DIM_67,
        corner_radius=6, height=34)
    btn.pack(fill="x", pady=2)
    rizz_nav_buttons_67[key] = btn
    sigma_reg_67(btn, label_key)
    btn.configure(command=lambda k=key: skibidy_switch_tab_67(k))

def sigma_paint_nav_67():
    for k, btn in rizz_nav_buttons_67.items():
        if k == sigma_active_tab_67["name"]:
            btn.configure(fg_color=SIGMA_ACCENT_67, text_color="#ffffff",
                          hover_color=RIZZ_ACCENT_HOV_67)
        else:
            btn.configure(fg_color="transparent", text_color=SIGMA_TEXT_DIM_67,
                          hover_color=RIZZ_PRIMARY_67)

skibidy_make_nav_button_67(skibidy_sb_nav_67, "aim", "nav_aim")
skibidy_make_nav_button_67(skibidy_sb_nav_67, "visual", "nav_visual")
skibidy_make_nav_button_67(skibidy_sb_nav_67, "enemy", "nav_filter")
skibidy_make_nav_button_67(skibidy_sb_nav_67, "misc", "nav_adv")
skibidy_make_nav_button_67(skibidy_sb_nav_67, "config", "nav_prof")

skibidy_sb_footer_67 = customtkinter.CTkFrame(skibidy_sidebar_67, fg_color=RIZZ_BG_CARD_67,
                                              corner_radius=6, height=58)
skibidy_sb_footer_67.pack(side="bottom", fill="x", padx=10, pady=10)
skibidy_sb_footer_67.pack_propagate(False)

sigma_footer_inner_67 = customtkinter.CTkFrame(skibidy_sb_footer_67, fg_color="transparent")
sigma_footer_inner_67.pack(fill="both", expand=True, padx=8, pady=8)
sigma_footer_inner_67.grid_columnconfigure(1, weight=1)

skibidy_sb_status_dot_67 = tk.Canvas(sigma_footer_inner_67, width=10, height=10, bg=RIZZ_BG_CARD_67,
                                     highlightthickness=0)
skibidy_sb_status_dot_67.grid(row=0, column=0, padx=(2, 6))
rizz_sb_status_dot_id_67 = skibidy_sb_status_dot_67.create_oval(1, 1, 9, 9,
                                                                fill=SIGMA_TEXT_DIM_67, outline="")

skibidy_sb_status_lbl_67 = customtkinter.CTkLabel(sigma_footer_inner_67, textvariable=sigma_status_var_67,
                                                  font=("Segoe UI", 11),
                                                  text_color=SKIBIDY_TEXT_LIGHT_67, anchor="w")
skibidy_sb_status_lbl_67.grid(row=0, column=1, sticky="w")

skibidy_start_stop_btn_67 = customtkinter.CTkButton(
    sigma_footer_inner_67, text=rizz_T_67("btn_start"), width=62, height=26,
    fg_color=SIGMA_ACCENT_67, hover_color=RIZZ_ACCENT_HOV_67, text_color="#ffffff",
    font=("Segoe UI", 11, "bold"), corner_radius=4)

def sigma_toggle_running_67():
    if rizz_running_67:
        skibidy_stop_worker_67()
        rizz_beep_skibidy_stop_67()
    else:
        rizz_beep_skibidy_start_67()
        rizz_ensure_overlay_67()
        sigma_run_worker_67()
    rizz_refresh_start_stop_btn_67()

def rizz_refresh_start_stop_btn_67():
    try:
        if rizz_running_67:
            skibidy_start_stop_btn_67.configure(text=rizz_T_67("btn_stop"), fg_color=SIGMA_DANGER_67,
                                                hover_color=RIZZ_DANGER_HOV_67)
            skibidy_sb_status_dot_67.itemconfig(rizz_sb_status_dot_id_67, fill=SIGMA_ACCENT_67)
        else:
            skibidy_start_stop_btn_67.configure(text=rizz_T_67("btn_start"), fg_color=SIGMA_ACCENT_67,
                                                hover_color=RIZZ_ACCENT_HOV_67)
            skibidy_sb_status_dot_67.itemconfig(rizz_sb_status_dot_id_67, fill=SIGMA_TEXT_DIM_67)
    except Exception:
        pass

skibidy_start_stop_btn_67.configure(command=sigma_toggle_running_67)
skibidy_start_stop_btn_67.grid(row=0, column=2, padx=(6, 2))

skibidy_top_bar_67 = customtkinter.CTkFrame(sigma_content_67, fg_color=RIZZ_BG_CARD_67, height=42,
                                            corner_radius=0)
skibidy_top_bar_67.grid(row=0, column=0, sticky="ew")
skibidy_top_bar_67.grid_propagate(False)
skibidy_top_bar_67.grid_columnconfigure(0, weight=1)

skibidy_top_title_67 = customtkinter.CTkLabel(skibidy_top_bar_67, text="Aimbot",
                                              font=("Segoe UI", 14, "bold"),
                                              text_color=SKIBIDY_TEXT_LIGHT_67, anchor="w")
skibidy_top_title_67.grid(row=0, column=0, sticky="w", padx=16, pady=10)

rizz_page_host_67 = customtkinter.CTkFrame(sigma_content_67, fg_color=SIGMA_BG_PANEL_67, corner_radius=0)
rizz_page_host_67.grid(row=1, column=0, sticky="nsew")
rizz_page_host_67.grid_rowconfigure(0, weight=1)
rizz_page_host_67.grid_columnconfigure(0, weight=1)

rizz_pages_67 = {}

def sigma_make_page_67(key):
    scroll = customtkinter.CTkScrollableFrame(
        rizz_page_host_67, fg_color=SIGMA_BG_PANEL_67,
        scrollbar_button_color=RIZZ_PRIMARY_67,
        scrollbar_button_hover_color=SKIBIDY_PRIMARY_HOV_67,
        corner_radius=0)
    rizz_pages_67[key] = scroll
    return scroll

skibidy_page_aim_67 = sigma_make_page_67("aim")

rizz_section_label_67(skibidy_page_aim_67, "sec_target_color")

skibidy_card_hex_67 = customtkinter.CTkFrame(skibidy_page_aim_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_hex_67.pack(fill="x", padx=14, pady=3)
skibidy_card_hex_67.grid_columnconfigure(1, weight=1)
_rizz_lbl_67 = customtkinter.CTkLabel(skibidy_card_hex_67, text=rizz_T_67("lbl_hex"),
                                      font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                      width=170, anchor="w")
_rizz_lbl_67.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=8)
sigma_reg_67(_rizz_lbl_67, "lbl_hex")
customtkinter.CTkEntry(skibidy_card_hex_67, textvariable=skibidy_hex_var_67, height=28,
                       fg_color=SKIBIDY_BG_INPUT_67, border_color=RIZZ_BORDER_67,
                       text_color=SKIBIDY_TEXT_LIGHT_67).grid(row=0, column=1,
                                                              sticky="ew", padx=6, pady=8)
rizz_btn_apply_hex_67 = customtkinter.CTkButton(
    skibidy_card_hex_67, text=rizz_T_67("btn_apply"), command=skibidy_set_from_hex_67,
    width=60, height=28, fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67,
    text_color=SKIBIDY_TEXT_LIGHT_67, corner_radius=4, font=("Segoe UI", 11))
rizz_btn_apply_hex_67.grid(row=0, column=2, padx=(0, 6), pady=8)
sigma_reg_67(rizz_btn_apply_hex_67, "btn_apply")
skibidy_swatch_canvas_67 = tk.Canvas(skibidy_card_hex_67, width=44, height=28,
                                     highlightthickness=1, highlightbackground=RIZZ_BORDER_67,
                                     bg=SKIBIDY_BG_INPUT_67)
skibidy_swatch_canvas_67.grid(row=0, column=3, padx=(0, 12), pady=8)

sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_hue_tol", rizz_tol_h_var_67, rizz_tol_h_display_67,
                        "tol_h", 0, 90, True)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_sat_tol", sigma_tol_s_var_67, sigma_tol_s_display_67,
                        "tol_s", 0, 255, True)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_bri_tol", skibidy_tol_v_var_67, skibidy_tol_v_display_67,
                        "tol_v", 0, 255, True)

skibidy_locked_label_67 = customtkinter.CTkLabel(
    skibidy_page_aim_67, text="—", font=("Consolas", 10, "bold"),
    text_color=SIGMA_TEXT_DIM_67, anchor="w")
skibidy_locked_label_67.pack(fill="x", padx=18, pady=(0, 6))

rizz_section_label_67(skibidy_page_aim_67, "sec_aim_response")
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_lock_strength", rizz_strength_var_67, skibidy_strength_display_67,
                        "strength", 0.5, 3.0)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_smoothing", sigma_stability_var_67, rizz_stability_display_67,
                        "stability", 0.05, 0.99)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_mouse_sens", skibidy_pf_mouse_var_67, sigma_pf_mouse_display_67,
                        "pf_mouse", 0.1, 5.0)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_aim_sens", rizz_pf_aim_var_67, skibidy_pf_aim_display_67,
                        "pf_aim", 0.1, 3.0)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_game_sens", sigma_roblox_sens_var_67, rizz_roblox_display_67,
                        "roblox_sens", 0.1, 2.0)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_max_step", skibidy_max_step_var_67, sigma_max_step_display_67,
                        "max_step", 1, 20, True)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_deadzone", rizz_deadzone_var_67, skibidy_deadzone_display_67,
                        "deadzone", 0, 15, True)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_fov", sigma_fov_var_67, rizz_fov_display_67,
                        "fov", 30, 140, True)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_offset_x", skibidy_offset_x_var_67, sigma_offset_x_display_67,
                        "offset_x", -100, 100, True)
sigma_add_slider_row_67(skibidy_page_aim_67, "lbl_offset_y", rizz_offset_y_var_67, skibidy_offset_y_display_67,
                        "offset_y", -100, 100, True)

rizz_section_label_67(skibidy_page_aim_67, "sec_key_bindings")
skibidy_card_hk_67 = customtkinter.CTkFrame(skibidy_page_aim_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_hk_67.pack(fill="x", padx=14, pady=3)
skibidy_card_hk_67.grid_columnconfigure(1, weight=1)

def _rizz_hk_row_67(parent, row, label_key, init_text, cmd):
    lbl = customtkinter.CTkLabel(parent, text=rizz_T_67(label_key),
                                 font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                 width=170, anchor="w")
    lbl.grid(row=row, column=0, sticky="w", padx=(12, 6), pady=8)
    sigma_reg_67(lbl, label_key)
    btn = customtkinter.CTkButton(
        parent, text=init_text, command=cmd,
        fg_color=SKIBIDY_BG_INPUT_67, hover_color=SKIBIDY_PRIMARY_HOV_67,
        text_color=SIGMA_ACCENT_67, font=("Consolas", 11, "bold"),
        height=28, corner_radius=4)
    btn.grid(row=row, column=1, sticky="ew", padx=6, pady=8)
    return btn

skibidy_aim_vk_btn_67 = _rizz_hk_row_67(skibidy_card_hk_67, 0, "lbl_aim_key",
                                        rizz_vk_display_67(rizz_current_aim_vk_67),
                                        lambda: rizz_start_capture_67("aim"))
rizz_toggle_vk_btn_67 = _rizz_hk_row_67(skibidy_card_hk_67, 1, "lbl_enable_toggle",
                                        rizz_vk_display_67(sigma_current_toggle_vk_67),
                                        lambda: rizz_start_capture_67("toggle"))

_rizz_lbl_67 = customtkinter.CTkLabel(skibidy_card_hk_67, text=rizz_T_67("lbl_panic_key"),
                                      font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                      width=170, anchor="w")
_rizz_lbl_67.grid(row=2, column=0, sticky="w", padx=(12, 6), pady=8)
sigma_reg_67(_rizz_lbl_67, "lbl_panic_key")
customtkinter.CTkLabel(skibidy_card_hk_67, text="F6", font=("Consolas", 11, "bold"),
                       text_color=SIGMA_TEXT_DIM_67).grid(row=2, column=1, sticky="w",
                                                          padx=12, pady=8)

skibidy_page_vis_67 = sigma_make_page_67("visual")

rizz_section_label_67(skibidy_page_vis_67, "sec_crosshair")

skibidy_card_cs_67 = customtkinter.CTkFrame(skibidy_page_vis_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_cs_67.pack(fill="x", padx=14, pady=3)
skibidy_card_cs_67.grid_columnconfigure(1, weight=1)

_rizz_lbl_67 = customtkinter.CTkLabel(skibidy_card_cs_67, text=rizz_T_67("lbl_crosshair_style"),
                                      font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                      width=170, anchor="w")
_rizz_lbl_67.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=8)
sigma_reg_67(_rizz_lbl_67, "lbl_crosshair_style")

skibidy_ch_style_menu_67 = customtkinter.CTkOptionMenu(
    skibidy_card_cs_67, variable=skibidy_ch_style_var_67,
    values=[rizz_T_67(f"ch_{k}") for k in SIGMA_CROSSHAIR_STYLES_67],
    command=sigma_on_ch_style_change_67,
    fg_color=SKIBIDY_BG_INPUT_67, button_color=RIZZ_PRIMARY_67, button_hover_color=SKIBIDY_PRIMARY_HOV_67,
    text_color=SKIBIDY_TEXT_LIGHT_67, font=("Segoe UI", 11),
    dropdown_fg_color=RIZZ_BG_CARD_67, dropdown_text_color=SKIBIDY_TEXT_LIGHT_67,
    dropdown_hover_color=RIZZ_PRIMARY_67)
skibidy_ch_style_menu_67.grid(row=0, column=1, sticky="ew", padx=6, pady=8)
skibidy_reg_menu_67(skibidy_ch_style_menu_67, [f"ch_{k}" for k in SIGMA_CROSSHAIR_STYLES_67])

sigma_add_slider_row_67(skibidy_page_vis_67, "lbl_crosshair_size", rizz_ch_size_var_67, sigma_ch_size_display_67,
                        "ch_size", 4, 40, True)

rizz_section_label_67(skibidy_page_vis_67, "sec_overlay")

rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_show_fov", sigma_show_fov_var_67, command=rizz_ensure_overlay_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_show_crosshair", skibidy_show_crosshair_var_67, command=rizz_ensure_overlay_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_show_box", sigma_ov_show_box_var_67, command=rizz_ensure_overlay_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_show_fps", skibidy_ov_show_fps_var_67, command=rizz_ensure_overlay_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_show_aim_line", rizz_ov_show_aim_line_var_67, command=rizz_ensure_overlay_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_hide_idle", sigma_ov_hide_idle_var_67, command=rizz_ensure_overlay_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_rainbow", skibidy_ov_rainbow_var_67, command=rizz_ensure_overlay_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_exclude_capture", rizz_exclude_capture_var_67,
                         command=sigma_on_capture_exclude_toggle_67)

skibidy_card_ovc_67 = customtkinter.CTkFrame(skibidy_page_vis_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_ovc_67.pack(fill="x", padx=14, pady=3)
skibidy_card_ovc_67.grid_columnconfigure(1, weight=1)
_rizz_lbl_67 = customtkinter.CTkLabel(skibidy_card_ovc_67, text=rizz_T_67("lbl_overlay_color"),
                                      font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                      width=170, anchor="w")
_rizz_lbl_67.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=8)
sigma_reg_67(_rizz_lbl_67, "lbl_overlay_color")
customtkinter.CTkEntry(skibidy_card_ovc_67, textvariable=rizz_ov_color_var_67, height=28,
                       fg_color=SKIBIDY_BG_INPUT_67, border_color=RIZZ_BORDER_67,
                       text_color=SKIBIDY_TEXT_LIGHT_67).grid(row=0, column=1,
                                                              sticky="ew", padx=6, pady=8)
skibidy_ov_swatch_67 = tk.Canvas(skibidy_card_ovc_67, width=44, height=28,
                                 highlightthickness=1, highlightbackground=RIZZ_BORDER_67,
                                 bg=SKIBIDY_BG_INPUT_67)
skibidy_ov_swatch_67.grid(row=0, column=2, padx=(0, 12), pady=8)

def _skibidy_update_ov_swatch_67(*_):
    try:
        skibidy_ov_swatch_67.delete("all")
        skibidy_ov_swatch_67.create_rectangle(0, 0, 44, 28, outline="",
                                              fill=rizz_ov_color_var_67.get())
    except Exception:
        pass
    rizz_auto_save_67()
rizz_ov_color_var_67.trace_add("write", _skibidy_update_ov_swatch_67)
_skibidy_update_ov_swatch_67()

rizz_section_label_67(skibidy_page_vis_67, "sec_palette_folder")

skibidy_card_folder_67 = customtkinter.CTkFrame(skibidy_page_vis_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_folder_67.pack(fill="x", padx=14, pady=3)
skibidy_card_folder_67.grid_columnconfigure(1, weight=1)
_rizz_lbl_67 = customtkinter.CTkLabel(skibidy_card_folder_67, text=rizz_T_67("lbl_folder_path"),
                                      font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                      width=170, anchor="w")
_rizz_lbl_67.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=8)
sigma_reg_67(_rizz_lbl_67, "lbl_folder_path")
customtkinter.CTkEntry(skibidy_card_folder_67, textvariable=skibidy_folder_var_67, height=28,
                       fg_color=SKIBIDY_BG_INPUT_67, border_color=RIZZ_BORDER_67,
                       text_color=SKIBIDY_TEXT_LIGHT_67).grid(row=0, column=1,
                                                              sticky="ew", padx=6, pady=8)
customtkinter.CTkButton(skibidy_card_folder_67, text="...", command=skibidy_pick_folder_67,
                        width=34, height=28, fg_color=RIZZ_PRIMARY_67,
                        hover_color=SKIBIDY_PRIMARY_HOV_67, text_color=SKIBIDY_TEXT_LIGHT_67,
                        corner_radius=4).grid(row=0, column=2,
                                              padx=(0, 12), pady=8)

rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_ignore_dark", rizz_skip_dark_var_67)
rizz_add_checkbox_row_67(skibidy_page_vis_67, "chk_ignore_gray", sigma_skip_gray_var_67)

rizz_section_label_67(skibidy_page_vis_67, "sec_palette_detected")

sigma_palette_wrap_67 = customtkinter.CTkFrame(skibidy_page_vis_67, fg_color=SKIBIDY_BG_INPUT_67,
                                               corner_radius=6, height=180)
sigma_palette_wrap_67.pack(fill="x", padx=14, pady=3)
sigma_palette_wrap_67.pack_propagate(False)
sigma_palette_wrap_67.grid_columnconfigure(0, weight=1)
sigma_palette_wrap_67.grid_rowconfigure(0, weight=1)
sigma_palette_canvas_67 = tk.Canvas(sigma_palette_wrap_67, bg=SKIBIDY_BG_INPUT_67,
                                    highlightthickness=0, bd=0)
sigma_palette_canvas_67.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
skibidy_palette_scroll_67 = customtkinter.CTkScrollbar(
    sigma_palette_wrap_67, orientation="vertical", command=sigma_palette_canvas_67.yview,
    button_color=RIZZ_PRIMARY_67, button_hover_color=SKIBIDY_PRIMARY_HOV_67,
    fg_color=SKIBIDY_BG_INPUT_67, width=12)
skibidy_palette_scroll_67.grid(row=0, column=1, sticky="ns", pady=4, padx=(0, 4))
sigma_palette_canvas_67.configure(yscrollcommand=skibidy_palette_scroll_67.set)
sigma_palette_inner_67 = customtkinter.CTkFrame(sigma_palette_canvas_67, fg_color=SKIBIDY_BG_INPUT_67,
                                                corner_radius=0)
rizz_palette_inner_id_67 = sigma_palette_canvas_67.create_window((0, 0), window=sigma_palette_inner_67,
                                                                 anchor="nw")

def _rizz_on_palette_inner_cfg_67(event):
    sigma_palette_canvas_67.configure(scrollregion=sigma_palette_canvas_67.bbox("all"))

def _rizz_on_palette_canvas_cfg_67(event):
    sigma_palette_canvas_67.itemconfig(rizz_palette_inner_id_67, width=event.width)

sigma_palette_inner_67.bind("<Configure>", _rizz_on_palette_inner_cfg_67)
sigma_palette_canvas_67.bind("<Configure>", _rizz_on_palette_canvas_cfg_67)

def _rizz_palette_wheel_67(event):
    sigma_palette_canvas_67.yview_scroll(int(-1 * (event.delta / 120)), "units")
sigma_palette_canvas_67.bind("<MouseWheel>", _rizz_palette_wheel_67)
sigma_palette_inner_67.bind("<MouseWheel>", _rizz_palette_wheel_67)

skibidy_card_pal_btns_67 = customtkinter.CTkFrame(skibidy_page_vis_67, fg_color="transparent")
skibidy_card_pal_btns_67.pack(fill="x", padx=14, pady=6)
skibidy_card_pal_btns_67.grid_columnconfigure(0, weight=1)
skibidy_card_pal_btns_67.grid_columnconfigure(1, weight=1)

skibidy_btn_pick_image_67 = customtkinter.CTkButton(
    skibidy_card_pal_btns_67, text=rizz_T_67("btn_load_img"), command=rizz_choose_image_67,
    fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67, text_color=SKIBIDY_TEXT_LIGHT_67,
    height=30, corner_radius=4, font=("Segoe UI", 11))
skibidy_btn_pick_image_67.grid(row=0, column=0, sticky="ew", padx=(0, 3))
sigma_reg_67(skibidy_btn_pick_image_67, "btn_load_img")

skibidy_btn_analyze_67 = customtkinter.CTkButton(
    skibidy_card_pal_btns_67, text=rizz_T_67("btn_scan"), command=sigma_analyze_images_folder_67,
    fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67, text_color=SKIBIDY_TEXT_LIGHT_67,
    height=30, corner_radius=4, font=("Segoe UI", 11))
skibidy_btn_analyze_67.grid(row=0, column=1, sticky="ew", padx=(3, 0))
sigma_reg_67(skibidy_btn_analyze_67, "btn_scan")

skibidy_btn_lock_colors_67 = customtkinter.CTkButton(
    skibidy_page_vis_67, text=rizz_T_67("btn_lock_color"), command=rizz_lock_selected_color_67,
    fg_color=SIGMA_ACCENT_67, hover_color=RIZZ_ACCENT_HOV_67, text_color="#ffffff",
    font=("Segoe UI", 11, "bold"), height=32, corner_radius=4)
skibidy_btn_lock_colors_67.pack(fill="x", padx=14, pady=(4, 3))
sigma_reg_67(skibidy_btn_lock_colors_67, "btn_lock_color")

skibidy_btn_clear_67 = customtkinter.CTkButton(
    skibidy_page_vis_67, text=rizz_T_67("btn_clear_palette"), command=sigma_clear_palette_67,
    fg_color="transparent", border_width=1, border_color=RIZZ_BORDER_67,
    hover_color=RIZZ_PRIMARY_67, text_color=SIGMA_TEXT_DIM_67, height=28, corner_radius=4,
    font=("Segoe UI", 11))
skibidy_btn_clear_67.pack(fill="x", padx=14, pady=(0, 10))
sigma_reg_67(skibidy_btn_clear_67, "btn_clear_palette")

skibidy_page_enemy_67 = sigma_make_page_67("enemy")

rizz_section_label_67(skibidy_page_enemy_67, "sec_filter_rule")

rizz_add_checkbox_row_67(skibidy_page_enemy_67, "chk_enable_filter", sigma_verify_enable_var_67,
                         command=rizz_update_verify_params_67)

skibidy_card_hint_67 = customtkinter.CTkFrame(skibidy_page_enemy_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_hint_67.pack(fill="x", padx=14, pady=3)
skibidy_hint_lbl_67 = customtkinter.CTkLabel(skibidy_card_hint_67, text=rizz_T_67("hint_filter"),
                                             font=("Segoe UI", 10), text_color=SIGMA_TEXT_DIM_67,
                                             justify="left", anchor="w")
skibidy_hint_lbl_67.pack(fill="x", padx=12, pady=8)
sigma_reg_67(skibidy_hint_lbl_67, "hint_filter")

rizz_section_label_67(skibidy_page_enemy_67, "sec_marker_color")

sigma_add_entry_row_67(skibidy_page_enemy_67, "lbl_marker_hexes", skibidy_verify_hexes_var_67,
                       button_text_key="btn_apply", button_cmd=rizz_update_verify_params_67)

skibidy_card_auto_67 = customtkinter.CTkFrame(skibidy_page_enemy_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_auto_67.pack(fill="x", padx=14, pady=3)

def skibidy_auto_detect_marker_67():
    folder = skibidy_folder_var_67.get().strip() or "images"
    if not os.path.isdir(folder):
        return
    rows, _, _ = sigma_analyze_folder_colors_67(folder, True, True, 40)
    for r in rows:
        h, s, v = r["hsv"]
        if 95 <= h <= 135 and s > 80:
            skibidy_verify_hexes_var_67.set(r["hex"])
            rizz_update_verify_params_67()
            break

skibidy_btn_auto_67 = customtkinter.CTkButton(
    skibidy_card_auto_67, text=rizz_T_67("btn_auto_detect"), command=skibidy_auto_detect_marker_67,
    fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67, text_color=SKIBIDY_TEXT_LIGHT_67,
    height=28, corner_radius=4, font=("Segoe UI", 11))
skibidy_btn_auto_67.pack(fill="x", padx=12, pady=8)
sigma_reg_67(skibidy_btn_auto_67, "btn_auto_detect")

rizz_section_label_67(skibidy_page_enemy_67, "sec_marker_detection")

def _rizz_vslider_67(parent, label_key, var, frm, to, on_change=None):
    card = customtkinter.CTkFrame(parent, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
    card.pack(fill="x", padx=14, pady=3)
    card.grid_columnconfigure(1, weight=1)
    lbl = customtkinter.CTkLabel(card, text=rizz_T_67(label_key),
                                 font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                 width=170, anchor="w")
    lbl.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=8)
    sigma_reg_67(lbl, label_key)
    s = customtkinter.CTkSlider(card, from_=frm, to=to, variable=var,
                                command=on_change if on_change else (
                                    lambda _=0: rizz_update_verify_params_67()),
                                button_color=SIGMA_ACCENT_67,
                                button_hover_color=RIZZ_ACCENT_HOV_67,
                                progress_color=SKIBIDY_ACCENT_DIM_67,
                                fg_color=SKIBIDY_BG_INPUT_67, height=14)
    s.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
    customtkinter.CTkLabel(card, textvariable=var, font=("Consolas", 11),
                           text_color=SIGMA_ACCENT_67, width=52,
                           anchor="e").grid(row=0, column=2, sticky="e",
                                            padx=(6, 14), pady=8)
    skibidy_neutralize_slider_wheel_67(s, sigma_find_scrollable_67(parent))

_rizz_vslider_67(skibidy_page_enemy_67, "lbl_hue_tol", rizz_verify_tol_h_var_67, 0, 90)
_rizz_vslider_67(skibidy_page_enemy_67, "lbl_sat_tol", sigma_verify_tol_s_var_67, 0, 255)
_rizz_vslider_67(skibidy_page_enemy_67, "lbl_bri_tol", skibidy_verify_tol_v_var_67, 0, 255)
_rizz_vslider_67(skibidy_page_enemy_67, "lbl_sample_radius", rizz_verify_roi_var_67, 6, 120)
_rizz_vslider_67(skibidy_page_enemy_67, "lbl_required_px", sigma_verify_minpx_var_67, 1, 120)
_rizz_vslider_67(skibidy_page_enemy_67, "lbl_required_frames", skibidy_verify_frames_var_67, 1, 10)

rizz_section_label_67(skibidy_page_enemy_67, "sec_diag")
skibidy_card_diag_67 = customtkinter.CTkFrame(skibidy_page_enemy_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_diag_67.pack(fill="x", padx=14, pady=3)
skibidy_diag_lbl_67 = customtkinter.CTkLabel(skibidy_card_diag_67, text=rizz_T_67("lbl_marker_diag"),
                                             font=("Consolas", 10), text_color=SIGMA_TEXT_DIM_67,
                                             anchor="w")
skibidy_diag_lbl_67.pack(fill="x", padx=12, pady=8)

skibidy_page_misc_67 = sigma_make_page_67("misc")

rizz_section_label_67(skibidy_page_misc_67, "sec_movement")
sigma_add_slider_row_67(skibidy_page_misc_67, "lbl_movement_comp", rizz_movement_compensation_var_67,
                        rizz_movement_display_67, "movement_comp", -0.5, 0.5)

rizz_section_label_67(skibidy_page_misc_67, "sec_gains")
sigma_add_slider_row_67(skibidy_page_misc_67, "lbl_kp", sigma_kp_var_67, sigma_kp_display_67, "kp_gain", 0.05, 1.5)
sigma_add_slider_row_67(skibidy_page_misc_67, "lbl_kd", skibidy_kd_var_67, skibidy_kd_display_67, "kd_gain", 0.0, 1.0)

rizz_section_label_67(skibidy_page_misc_67, "sec_tracking")
sigma_add_slider_row_67(skibidy_page_misc_67, "lbl_track_radius", rizz_roi_radius_var_67,
                        rizz_roi_radius_display_67, "roi_radius", 10, 120, True)

skibidy_page_cfg_67 = sigma_make_page_67("config")

def _sigma_safe_filename_67(name):
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name).strip()
    return out[:60] or "profile"

def _rizz_profile_path_67(name):
    return os.path.join(SKIBIDY_CONFIGS_DIR_67, _sigma_safe_filename_67(name) + ".json")

def _skibidy_read_configs_db_67():
    db = {}
    try:
        for fn in os.listdir(SKIBIDY_CONFIGS_DIR_67):
            if fn.startswith("_") or not fn.endswith(".json"):
                continue
            name = fn[:-5]
            try:
                with open(os.path.join(SKIBIDY_CONFIGS_DIR_67, fn), "r",
                          encoding="utf-8") as f:
                    db[name] = json.load(f)
            except Exception:
                pass
    except Exception:
        pass
    return db

rizz_section_label_67(skibidy_page_cfg_67, "sec_profiles")

skibidy_card_profile_67 = customtkinter.CTkFrame(skibidy_page_cfg_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_profile_67.pack(fill="x", padx=14, pady=3)
skibidy_card_profile_67.grid_columnconfigure(0, weight=1)
skibidy_card_profile_67.grid_columnconfigure(1, weight=0)

skibidy_name_row_67 = customtkinter.CTkFrame(skibidy_card_profile_67, fg_color="transparent")
skibidy_name_row_67.grid(row=0, column=0, columnspan=2, sticky="ew",
                         padx=12, pady=(12, 6))
skibidy_name_row_67.grid_columnconfigure(1, weight=1)

_rizz_lbl_67 = customtkinter.CTkLabel(skibidy_name_row_67, text=rizz_T_67("lbl_profile_name"),
                                      font=("Segoe UI", 11), text_color=SKIBIDY_TEXT_LIGHT_67,
                                      anchor="w")
_rizz_lbl_67.grid(row=0, column=0, sticky="w", padx=(0, 8))
sigma_reg_67(_rizz_lbl_67, "lbl_profile_name")

customtkinter.CTkEntry(skibidy_name_row_67, textvariable=rizz_config_name_var_67, height=28,
                       fg_color=SKIBIDY_BG_INPUT_67, border_color=RIZZ_BORDER_67,
                       text_color=SKIBIDY_TEXT_LIGHT_67).grid(row=0, column=1, sticky="ew")

def skibidy_save_named_profile_67():
    name = rizz_config_name_var_67.get().strip()
    if not name:
        return
    try:
        with open(_rizz_profile_path_67(name), "w", encoding="utf-8") as f:
            json.dump(_sigma_collect_settings_67(), f, indent=2)
    except Exception as e:
        print(f"[profile] save failed: {e}")
    rizz_refresh_profile_list_67()

skibidy_btn_save_named_67 = customtkinter.CTkButton(
    skibidy_name_row_67, text=rizz_T_67("btn_save"), width=64, height=28,
    fg_color=SIGMA_ACCENT_67, hover_color=RIZZ_ACCENT_HOV_67, text_color="#ffffff",
    font=("Segoe UI", 11, "bold"), corner_radius=4,
    command=skibidy_save_named_profile_67)
skibidy_btn_save_named_67.grid(row=0, column=2, padx=(8, 0))
sigma_reg_67(skibidy_btn_save_named_67, "btn_save")

skibidy_profile_list_wrap_67 = customtkinter.CTkFrame(skibidy_card_profile_67, fg_color=SKIBIDY_BG_INPUT_67,
                                                      corner_radius=4, height=150)
skibidy_profile_list_wrap_67.grid(row=1, column=0, columnspan=2, sticky="ew",
                                  padx=12, pady=(0, 8))
skibidy_profile_list_wrap_67.grid_propagate(False)

skibidy_profile_listbox_67 = tk.Listbox(
    skibidy_profile_list_wrap_67, bg=SKIBIDY_BG_INPUT_67, fg=SKIBIDY_TEXT_LIGHT_67,
    selectbackground=SIGMA_ACCENT_67, selectforeground="#ffffff",
    borderwidth=0, highlightthickness=0, activestyle="none",
    font=("Segoe UI", 11))
skibidy_profile_listbox_67.pack(fill="both", expand=True, padx=6, pady=6)

skibidy_profile_btn_row_67 = customtkinter.CTkFrame(skibidy_card_profile_67, fg_color="transparent")
skibidy_profile_btn_row_67.grid(row=2, column=0, columnspan=2, sticky="ew",
                                padx=12, pady=(0, 12))
skibidy_profile_btn_row_67.grid_columnconfigure(0, weight=1)
skibidy_profile_btn_row_67.grid_columnconfigure(1, weight=1)

def rizz_refresh_profile_list_67():
    skibidy_profile_listbox_67.delete(0, "end")
    db = _skibidy_read_configs_db_67()
    for name in sorted(db.keys()):
        skibidy_profile_listbox_67.insert("end", name)

def sigma_load_named_profile_67():
    sel = skibidy_profile_listbox_67.curselection()
    if not sel:
        return
    name = skibidy_profile_listbox_67.get(sel[0])
    db = _skibidy_read_configs_db_67()
    if name not in db:
        return
    _skibidy_apply_settings_67(db[name])
    rizz_config_name_var_67.set(name)
    skibidy_refresh_locked_label_67()
    skibidy_refresh_value_labels_67()
    rizz_update_verify_params_67()

def skibidy_delete_named_profile_67():
    sel = skibidy_profile_listbox_67.curselection()
    if not sel:
        return
    name = skibidy_profile_listbox_67.get(sel[0])
    try:
        os.remove(_rizz_profile_path_67(name))
    except Exception:
        pass
    rizz_refresh_profile_list_67()

def sigma_rename_named_profile_67():
    sel = skibidy_profile_listbox_67.curselection()
    if not sel:
        return
    old = skibidy_profile_listbox_67.get(sel[0])
    new = rizz_config_name_var_67.get().strip()
    if not new or new == old:
        return
    try:
        os.replace(_rizz_profile_path_67(old), _rizz_profile_path_67(new))
    except Exception:
        pass
    rizz_refresh_profile_list_67()

skibidy_btn_rename_67 = customtkinter.CTkButton(
    skibidy_profile_btn_row_67, text=rizz_T_67("btn_rename"), command=sigma_rename_named_profile_67,
    fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67, text_color=SKIBIDY_TEXT_LIGHT_67,
    height=30, corner_radius=4, font=("Segoe UI", 11))
skibidy_btn_rename_67.grid(row=0, column=0, sticky="ew", padx=(0, 3))
sigma_reg_67(skibidy_btn_rename_67, "btn_rename")

skibidy_btn_delete_67 = customtkinter.CTkButton(
    skibidy_profile_btn_row_67, text=rizz_T_67("btn_delete"), command=skibidy_delete_named_profile_67,
    fg_color="transparent", border_width=1, border_color=SIGMA_DANGER_67,
    hover_color=RIZZ_PRIMARY_67, text_color=SIGMA_DANGER_67, height=30, corner_radius=4,
    font=("Segoe UI", 11))
skibidy_btn_delete_67.grid(row=0, column=1, sticky="ew", padx=(3, 0))
sigma_reg_67(skibidy_btn_delete_67, "btn_delete")

skibidy_btn_load_named_67 = customtkinter.CTkButton(
    skibidy_card_profile_67, text=rizz_T_67("btn_load_sel"), command=sigma_load_named_profile_67,
    fg_color=SIGMA_ACCENT_67, hover_color=RIZZ_ACCENT_HOV_67, text_color="#ffffff",
    height=32, corner_radius=4, font=("Segoe UI", 11, "bold"))
skibidy_btn_load_named_67.grid(row=3, column=0, columnspan=2, sticky="ew",
                               padx=12, pady=(0, 12))
sigma_reg_67(skibidy_btn_load_named_67, "btn_load_sel")

rizz_section_label_67(skibidy_page_cfg_67, "sec_storage")

skibidy_card_auto2_67 = customtkinter.CTkFrame(skibidy_page_cfg_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_auto2_67.pack(fill="x", padx=14, pady=3)
skibidy_autosave_lbl_67 = customtkinter.CTkLabel(skibidy_card_auto2_67, text=rizz_T_67("autosave_text"),
                                                 font=("Segoe UI", 10),
                                                 text_color=SIGMA_TEXT_DIM_67, anchor="w",
                                                 justify="left")
skibidy_autosave_lbl_67.pack(fill="x", padx=12, pady=8)
sigma_reg_67(skibidy_autosave_lbl_67, "autosave_text")

skibidy_card_path_67 = customtkinter.CTkFrame(skibidy_page_cfg_67, fg_color=RIZZ_BG_CARD_67, corner_radius=6)
skibidy_card_path_67.pack(fill="x", padx=14, pady=3)
customtkinter.CTkLabel(skibidy_card_path_67, text=SKIBIDY_CONFIGS_DIR_67,
                       font=("Consolas", 10), text_color=SIGMA_TEXT_DIM_67,
                       anchor="w").pack(fill="x", padx=12, pady=8)

def skibidy_open_configs_folder_67():
    try:
        os.startfile(SKIBIDY_CONFIGS_DIR_67)
    except Exception:
        pass

customtkinter.CTkButton(
    skibidy_page_cfg_67, text="Open configs folder", command=skibidy_open_configs_folder_67,
    fg_color=RIZZ_PRIMARY_67, hover_color=SKIBIDY_PRIMARY_HOV_67, text_color=SKIBIDY_TEXT_LIGHT_67,
    height=28, corner_radius=4, font=("Segoe UI", 11)).pack(
        fill="x", padx=14, pady=(3, 3))

def skibidy_do_reset_all_67():
    if not messagebox.askyesno(rizz_T_67("confirm_reset_title"),
                               rizz_T_67("confirm_reset_msg")):
        return
    global rizz_active_ranges_67, sigma_palette_data_67
    global sigma_locked_hex_67, skibidy_locked_hsv_67
    rizz_active_ranges_67 = []
    sigma_palette_data_67 = []
    sigma_locked_hex_67 = ""
    skibidy_locked_hsv_67 = None
    try:
        for fn in os.listdir(SKIBIDY_CONFIGS_DIR_67):
            if fn.endswith(".json"):
                os.remove(os.path.join(SKIBIDY_CONFIGS_DIR_67, fn))
    except Exception:
        pass
    rizz_refresh_palette_ui_67()
    skibidy_refresh_locked_label_67()
    rizz_refresh_profile_list_67()

skibidy_btn_reset_67 = customtkinter.CTkButton(
    skibidy_page_cfg_67, text=rizz_T_67("btn_reset_all"), command=skibidy_do_reset_all_67,
    fg_color="transparent", border_width=1, border_color=SIGMA_DANGER_67,
    hover_color=RIZZ_PRIMARY_67, text_color=SIGMA_DANGER_67, height=30, corner_radius=4,
    font=("Segoe UI", 11))
skibidy_btn_reset_67.pack(fill="x", padx=14, pady=(6, 12))
sigma_reg_67(skibidy_btn_reset_67, "btn_reset_all")

def skibidy_switch_tab_67(key):
    sigma_active_tab_67["name"] = key
    for k, p in rizz_pages_67.items():
        if k == key:
            p.pack(fill="both", expand=True)
        else:
            p.pack_forget()
    titles = {
        "aim": rizz_T_67("nav_aim"),
        "visual": rizz_T_67("nav_visual"),
        "enemy": rizz_T_67("nav_filter"),
        "misc": rizz_T_67("nav_adv"),
        "config": rizz_T_67("nav_prof"),
    }
    skibidy_top_title_67.configure(text=titles.get(key, key.title()))
    sigma_paint_nav_67()

def _sigma_collect_settings_67():
    return {
        "sigma_skibidy_rizz_67_hex": skibidy_hex_var_67.get(),
        "sigma_skibidy_rizz_67_tol_h": int(rizz_tol_h_var_67.get()),
        "sigma_skibidy_rizz_67_tol_s": int(sigma_tol_s_var_67.get()),
        "sigma_skibidy_rizz_67_tol_v": int(skibidy_tol_v_var_67.get()),
        "sigma_skibidy_rizz_67_lock_strength": sigma_round_to_2_67(rizz_strength_var_67.get()),
        "sigma_skibidy_rizz_67_stability": sigma_round_to_2_67(sigma_stability_var_67.get()),
        "sigma_skibidy_rizz_67_pf_mouse_sensitivity": float(skibidy_pf_mouse_var_67.get()),
        "sigma_skibidy_rizz_67_pf_aim_sensitivity": float(rizz_pf_aim_var_67.get()),
        "sigma_skibidy_rizz_67_roblox_sensitivity": float(sigma_roblox_sens_var_67.get()),
        "sigma_skibidy_rizz_67_max_step": int(skibidy_max_step_var_67.get()),
        "sigma_skibidy_rizz_67_deadzone": int(rizz_deadzone_var_67.get()),
        "sigma_skibidy_rizz_67_fov": int(sigma_fov_var_67.get()),
        "sigma_skibidy_rizz_67_offset_x": int(skibidy_offset_x_var_67.get()),
        "sigma_skibidy_rizz_67_offset_y": int(rizz_offset_y_var_67.get()),
        "sigma_skibidy_rizz_67_show_fov": bool(sigma_show_fov_var_67.get()),
        "sigma_skibidy_rizz_67_show_crosshair": bool(skibidy_show_crosshair_var_67.get()),
        "sigma_skibidy_rizz_67_exclude_capture": bool(rizz_exclude_capture_var_67.get()),
        "sigma_skibidy_rizz_67_folder": skibidy_folder_var_67.get(),
        "sigma_skibidy_rizz_67_skip_dark": bool(rizz_skip_dark_var_67.get()),
        "sigma_skibidy_rizz_67_skip_gray": bool(sigma_skip_gray_var_67.get()),
        "sigma_skibidy_rizz_67_aim_vk": int(rizz_current_aim_vk_67),
        "sigma_skibidy_rizz_67_toggle_vk": int(sigma_current_toggle_vk_67),
        "sigma_skibidy_rizz_67_palette": sigma_palette_data_67,
        "sigma_skibidy_rizz_67_verify_enabled": bool(rizz_verify_enabled_67),
        "sigma_skibidy_rizz_67_verify_hexes": list(sigma_verify_hex_list_67),
        "sigma_skibidy_rizz_67_verify_tol_h": int(skibidy_verify_tol_h_67),
        "sigma_skibidy_rizz_67_verify_tol_s": int(rizz_verify_tol_s_67),
        "sigma_skibidy_rizz_67_verify_tol_v": int(sigma_verify_tol_v_67),
        "sigma_skibidy_rizz_67_verify_roi": int(skibidy_verify_roi_67),
        "sigma_skibidy_rizz_67_verify_min_px": int(rizz_verify_min_px_67),
        "sigma_skibidy_rizz_67_verify_frames": int(sigma_verify_frames_required_67),
        "sigma_skibidy_rizz_67_movement_comp": float(sigma_movement_compensation_67),
        "sigma_skibidy_rizz_67_kp": float(sigma_kp_67),
        "sigma_skibidy_rizz_67_kd": float(skibidy_kd_67),
        "sigma_skibidy_rizz_67_roi_radius": int(sigma_roi_radius_67),
        "sigma_skibidy_rizz_67_locked_hex": sigma_locked_hex_67,
        "sigma_skibidy_rizz_67_locked_hsv": list(skibidy_locked_hsv_67) if skibidy_locked_hsv_67 else None,
        "sigma_skibidy_rizz_67_ov_show_box": bool(sigma_ov_show_box_var_67.get()),
        "sigma_skibidy_rizz_67_ov_show_fps": bool(skibidy_ov_show_fps_var_67.get()),
        "sigma_skibidy_rizz_67_ov_show_aim_line": bool(rizz_ov_show_aim_line_var_67.get()),
        "sigma_skibidy_rizz_67_ov_hide_idle": bool(sigma_ov_hide_idle_var_67.get()),
        "sigma_skibidy_rizz_67_ov_rainbow": bool(skibidy_ov_rainbow_var_67.get()),
        "sigma_skibidy_rizz_67_ov_color": rizz_ov_color_var_67.get(),
        "sigma_skibidy_rizz_67_ch_style": sigma_ch_style_internal_67["v"],
        "sigma_skibidy_rizz_67_ch_size": int(rizz_ch_size_var_67.get()),
        "sigma_skibidy_rizz_67_lang": sigma_current_lang_67,
    }

def _skibidy_apply_settings_67(data):
    global rizz_current_aim_vk_67, sigma_current_toggle_vk_67
    global rizz_active_ranges_67, sigma_palette_data_67
    global rizz_verify_enabled_67, skibidy_verify_tol_h_67, rizz_verify_tol_s_67
    global sigma_verify_tol_v_67, skibidy_verify_roi_67, rizz_verify_min_px_67
    global sigma_verify_frames_required_67, sigma_verify_hex_list_67
    global sigma_locked_hex_67, skibidy_locked_hsv_67
    global sigma_movement_compensation_67, sigma_kp_67, skibidy_kd_67, sigma_roi_radius_67
    global sigma_current_lang_67

    if not isinstance(data, dict):
        return

    p = "sigma_skibidy_rizz_67_"

    skibidy_hex_var_67.set(data.get(p + "hex", "#feffb2"))
    rizz_tol_h_var_67.set(data.get(p + "tol_h", 10))
    sigma_tol_s_var_67.set(data.get(p + "tol_s", 60))
    skibidy_tol_v_var_67.set(data.get(p + "tol_v", 60))
    rizz_strength_var_67.set(sigma_round_to_2_67(data.get(p + "lock_strength", 1.0)))
    sigma_stability_var_67.set(sigma_round_to_2_67(data.get(p + "stability", 0.82)))
    skibidy_max_step_var_67.set(data.get(p + "max_step", 6))
    rizz_deadzone_var_67.set(data.get(p + "deadzone", 6))
    sigma_fov_var_67.set(data.get(p + "fov", 80))
    skibidy_offset_x_var_67.set(data.get(p + "offset_x", 0))
    rizz_offset_y_var_67.set(data.get(p + "offset_y", 0))
    skibidy_pf_mouse_var_67.set(data.get(p + "pf_mouse_sensitivity", 0.5))
    rizz_pf_aim_var_67.set(data.get(p + "pf_aim_sensitivity", 1.0))
    sigma_roblox_sens_var_67.set(data.get(p + "roblox_sensitivity", 0.55))
    sigma_show_fov_var_67.set(data.get(p + "show_fov", False))
    skibidy_show_crosshair_var_67.set(data.get(p + "show_crosshair", False))
    rizz_exclude_capture_var_67.set(data.get(p + "exclude_capture", False))
    skibidy_folder_var_67.set(data.get(p + "folder", "images"))
    rizz_skip_dark_var_67.set(data.get(p + "skip_dark", True))
    sigma_skip_gray_var_67.set(data.get(p + "skip_gray", True))

    rizz_current_aim_vk_67 = int(data.get(p + "aim_vk", 0x02))
    sigma_current_toggle_vk_67 = int(data.get(p + "toggle_vk", 0x77))
    try:
        skibidy_aim_vk_btn_67.configure(text=rizz_vk_display_67(rizz_current_aim_vk_67))
        rizz_toggle_vk_btn_67.configure(text=rizz_vk_display_67(sigma_current_toggle_vk_67))
    except Exception:
        pass

    pal = data.get(p + "palette", [])
    sigma_palette_data_67 = pal if isinstance(pal, list) else []

    rizz_verify_enabled_67 = bool(data.get(p + "verify_enabled", False))
    sigma_verify_hex_list_67 = data.get(p + "verify_hexes", ["#3AA0FF"]) or ["#3AA0FF"]
    skibidy_verify_tol_h_67 = int(data.get(p + "verify_tol_h", 12))
    rizz_verify_tol_s_67 = int(data.get(p + "verify_tol_s", 70))
    sigma_verify_tol_v_67 = int(data.get(p + "verify_tol_v", 70))
    skibidy_verify_roi_67 = int(data.get(p + "verify_roi", 40))
    rizz_verify_min_px_67 = int(data.get(p + "verify_min_px", 8))
    sigma_verify_frames_required_67 = max(1, int(data.get(p + "verify_frames", 1)))

    sigma_movement_compensation_67 = float(data.get(p + "movement_comp", 0.0))
    sigma_kp_67 = float(data.get(p + "kp", 0.45))
    skibidy_kd_67 = float(data.get(p + "kd", 0.25))
    sigma_roi_radius_67 = int(data.get(p + "roi_radius", 50))

    sigma_ov_show_box_var_67.set(data.get(p + "ov_show_box", True))
    skibidy_ov_show_fps_var_67.set(data.get(p + "ov_show_fps", True))
    rizz_ov_show_aim_line_var_67.set(data.get(p + "ov_show_aim_line", False))
    sigma_ov_hide_idle_var_67.set(data.get(p + "ov_hide_idle", False))
    skibidy_ov_rainbow_var_67.set(data.get(p + "ov_rainbow", False))
    rizz_ov_color_var_67.set(data.get(p + "ov_color", "#ff4040"))

    sigma_ch_style_internal_67["v"] = data.get(p + "ch_style", "cross")
    rizz_ch_size_var_67.set(int(data.get(p + "ch_size", 12)))

    new_lang = data.get(p + "lang", sigma_current_lang_67)
    if new_lang in ("th", "en") and new_lang != sigma_current_lang_67:
        sigma_current_lang_67 = new_lang

    try:
        sigma_verify_enable_var_67.set(rizz_verify_enabled_67)
        skibidy_verify_hexes_var_67.set(", ".join(sigma_verify_hex_list_67))
        rizz_verify_tol_h_var_67.set(skibidy_verify_tol_h_67)
        sigma_verify_tol_s_var_67.set(rizz_verify_tol_s_67)
        skibidy_verify_tol_v_var_67.set(sigma_verify_tol_v_67)
        rizz_verify_roi_var_67.set(skibidy_verify_roi_67)
        sigma_verify_minpx_var_67.set(rizz_verify_min_px_67)
        skibidy_verify_frames_var_67.set(sigma_verify_frames_required_67)
        rizz_movement_compensation_var_67.set(sigma_movement_compensation_67)
        sigma_kp_var_67.set(sigma_kp_67)
        skibidy_kd_var_67.set(skibidy_kd_67)
        rizz_roi_radius_var_67.set(sigma_roi_radius_67)
        skibidy_ch_style_var_67.set(rizz_T_67(f"ch_{sigma_ch_style_internal_67['v']}"))
    except Exception:
        pass

    sigma_locked_hex_67 = data.get(p + "locked_hex", "")
    lh = data.get(p + "locked_hsv", None)
    skibidy_locked_hsv_67 = tuple(lh) if isinstance(lh, (list, tuple)) and len(lh) == 3 else None
    if not sigma_locked_hex_67 and skibidy_hex_var_67.get():
        sigma_set_locked_hex_67(skibidy_hex_var_67.get())

    skibidy_rebuild_verify_ranges_67()
    rizz_update_swatch_67(skibidy_swatch_canvas_67, skibidy_hex_var_67.get())
    skibidy_update_params_67()
    _skibidy_update_ov_swatch_67()
    rizz_apply_lang_67()

def sigma_load_settings_67():
    data = None
    for path in (RIZZ_LAST_SESSION_FILE_67, SIGMA_LEGACY_SETTINGS_FILE_67):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                break
            except Exception:
                continue
    if data:
        _skibidy_apply_settings_67(data)

def sigma_save_settings_67():
    try:
        with open(RIZZ_LAST_SESSION_FILE_67, "w", encoding="utf-8") as f:
            json.dump(_sigma_collect_settings_67(), f, indent=2)
    except Exception as e:
        print(f"[settings] save failed: {e}")

def skibidy_on_close_67():
    skibidy_stop_worker_67()
    sigma_save_settings_67()
    try:
        if skibidy_overlay_67 and skibidy_overlay_67.winfo_exists():
            skibidy_overlay_67.destroy()
    except Exception:
        pass
    rizz_root_67.destroy()

_rizz_last_running_67 = False

def sigma_poll_state_67():
    global _rizz_last_running_67, sigma_rainbow_hue_67
    if rizz_running_67 != _rizz_last_running_67:
        _rizz_last_running_67 = rizz_running_67
        rizz_refresh_start_stop_btn_67()

    if skibidy_ov_rainbow_var_67.get():
        sigma_rainbow_hue_67 = (sigma_rainbow_hue_67 + 6) % 360

    try:
        if (sigma_show_fov_var_67.get() or skibidy_show_crosshair_var_67.get()
                or sigma_ov_show_box_var_67.get() or skibidy_ov_show_fps_var_67.get()
                or rizz_ov_show_aim_line_var_67.get()):
            skibidy_update_overlay_67()
    except Exception:
        pass

    try:
        if not rizz_verify_enabled_67:
            skibidy_diag_lbl_67.configure(
                text=f"{rizz_T_67('lbl_marker_diag')} filter OFF -> lock by main color only",
                text_color=SIGMA_TEXT_DIM_67)
        elif sigma_last_verify_px_67 < 0:
            skibidy_diag_lbl_67.configure(text=rizz_T_67("lbl_marker_diag"),
                                          text_color=SIGMA_TEXT_DIM_67)
        else:
            thr = int(sigma_verify_minpx_var_67.get())
            marker_present = sigma_last_verify_px_67 >= thr
            if marker_present:
                verdict = "CONDITION FOUND -> LOCK"
                color = SIGMA_ACCENT_67
            else:
                verdict = "CONDITION MISSING -> SKIP"
                color = SIGMA_DANGER_67
            skibidy_diag_lbl_67.configure(
                text=f"{rizz_T_67('lbl_marker_diag')} {sigma_last_verify_px_67}  "
                     f"(thr {thr}) -> {verdict}",
                text_color=color)
    except Exception:
        pass

    rizz_root_67.after(50, sigma_poll_state_67)

def sigma_startup_scan_67(retries=3):
    try:
        sigma_load_settings_67()
    except Exception as e:
        print(f"[startup load] {e}")
    try:
        sigma_analyze_images_folder_67()
    except Exception as e:
        print(f"[startup scan] {e}")
    if not sigma_palette_data_67 and retries > 0:
        rizz_root_67.after(500, lambda: sigma_startup_scan_67(retries - 1))
        return
    if not rizz_active_ranges_67 and sigma_palette_data_67:
        skibidy_selected_palette_index_67.set(0)
        rizz_lock_selected_color_67()

skibidy_switch_tab_67("aim")
sigma_load_settings_67()
skibidy_refresh_value_labels_67()
skibidy_update_params_67()
rizz_refresh_palette_ui_67()
skibidy_refresh_locked_label_67()
skibidy_rebuild_verify_ranges_67()
rizz_update_verify_params_67()
rizz_refresh_profile_list_67()
rizz_refresh_start_stop_btn_67()
rizz_apply_lang_67()

rizz_root_67.after(600, lambda: globals().update({"_rizz_auto_save_ready_67": True}))

if rizz_exclude_capture_var_67.get():
    rizz_root_67.after(150, sigma_on_capture_exclude_toggle_67)

def _rizz_color_titlebar_67():
    try:
        rizz_root_67.update_idletasks()
        hwnd = sigma_user32.GetParent(rizz_root_67.winfo_id()) or rizz_root_67.winfo_id()
        skibidy_set_titlebar_color_67(hwnd, SIGMA_TITLEBAR_BG_67)
    except Exception:
        pass

rizz_root_67.after(60, _rizz_color_titlebar_67)
rizz_root_67.after(400, sigma_startup_scan_67)
rizz_root_67.after(400, sigma_poll_state_67)
rizz_root_67.protocol("WM_DELETE_WINDOW", skibidy_on_close_67)
rizz_root_67.mainloop()
