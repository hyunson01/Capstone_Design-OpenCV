import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import glob
import os

# 이미지 경로 설정
IMG_PATH = r"D:\git\Capstone_Design-OpenCV\img\calibration\3-2\*.jpg"
image_paths = sorted(glob.glob(IMG_PATH))
if not image_paths:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

# 예시 K_f, D_f 설정 (실제 캘리브레이션 값으로 교체)
# h, w는 첫 이미지 기준
sample_img = cv2.imread(image_paths[0])
h, w = sample_img.shape[:2]
K_f = np.array([[300.0, 0.0, w / 2],
                [0.0, 300.0, h / 2],
                [0.0, 0.0, 1.0]])
D_f = np.array([[-0.1], [0.01], [0.0], [0.0]])

# 현재 이미지 인덱스
img_idx = 0

# Tkinter GUI
root = tk.Tk()
root.title("Fisheye Undistortion Viewer (A/D to navigate)")

# Tk Label 영역
panel = tk.Label(root)
panel.pack()

label_info = tk.Label(root, text="", font=("Arial", 12))
label_info.pack()

slider_label = tk.Label(root, text="balance = 0.30", font=("Arial", 12))
slider_label.pack()

slider = ttk.Scale(root, from_=0, to=100, orient='horizontal')
slider.set(30)
slider.pack()

def load_image(idx):
    img_path = image_paths[idx]
    return cv2.imread(img_path), os.path.basename(img_path)

def show_image(balance_val=None):
    global img_idx

    if balance_val is None:
        balance = float(slider.get()) / 100
    else:
        balance = float(balance_val) / 100
        slider.set(balance * 100)

    img, name = load_image(img_idx)

    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_f, D_f, (w, h), np.eye(3), balance=balance)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_f, D_f, np.eye(3), newK, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Convert to PIL Image
    img_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((int(w * 0.6), int(h * 0.6)))
    img_tk = ImageTk.PhotoImage(image=img_pil)

    panel.config(image=img_tk)
    panel.image = img_tk

    label_info.config(text=f"[{img_idx + 1}/{len(image_paths)}] {name}")
    slider_label.config(text=f"balance = {balance:.2f}")

def on_key(event):
    global img_idx
    key = event.keysym.lower()
    if key == 'a':
        img_idx = (img_idx - 1) % len(image_paths)
        show_image()
    elif key == 'd':
        img_idx = (img_idx + 1) % len(image_paths)
        show_image()

slider.configure(command=show_image)
root.bind('<Key>', on_key)

# 초기 표시
show_image()

root.mainloop()
