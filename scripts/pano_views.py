# -*- coding: utf-8 -*-
"""Perspective views from an equirectangular panorama (numpy + scipy, no OpenCV).

`perspective(equi, yaw, pitch=0, fov=90, size=768)` renders a pinhole view looking at
`yaw` degrees measured clockwise from the panorama's centre column (yaw 0 = centre,
90 = right, 180 = back, 270 = left). Bilinear sampling; the horizontal wrap is handled.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def perspective(equi: Image.Image, yaw: float, pitch: float = 0.0, fov: float = 90.0, size: int = 768) -> Image.Image:
    arr = np.asarray(equi.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    f = (size / 2) / np.tan(np.radians(fov) / 2)
    xs, ys = np.meshgrid(np.arange(size) - size / 2 + 0.5, np.arange(size) - size / 2 + 0.5)
    # camera rays: x right, y down, z forward
    dirs = np.stack([xs, ys, np.full_like(xs, f)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    p, yw = np.radians(pitch), np.radians(yaw)
    rot_x = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rot_y = np.array([[np.cos(yw), 0, np.sin(yw)], [0, 1, 0], [-np.sin(yw), 0, np.cos(yw)]])
    d = dirs @ (rot_y @ rot_x).T
    lon = np.arctan2(d[..., 0], d[..., 2])          # -pi..pi, 0 = centre column
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))       # -pi/2 (up) .. pi/2 (down)
    u = (lon / (2 * np.pi) + 0.5) * w - 0.5
    v = (lat / np.pi + 0.5) * h - 0.5
    out = np.empty((size, size, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(arr[..., c], [v, u], order=1, mode="wrap")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
