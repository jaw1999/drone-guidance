#!/usr/bin/env python3
"""Fake RTSP camera server for testing terminal guidance.

This script creates an RTSP server that streams either:
  1. A real video file (recommended for realistic testing)
  2. Synthetic ocean scene with ships (fallback)

For best YOLO detection results, use real maritime footage.
You can download sample videos from YouTube or use your own drone footage.

Usage:
  # Stream a real video file (recommended):
  python scripts/fake_camera.py --video /path/to/boats.mp4

  # Stream synthetic scene (fallback):
  python scripts/fake_camera.py

  # Options:
  python scripts/fake_camera.py --video boats.mp4 --port 8554 --loop

Connect with:
  rtsp://localhost:8554/stream
  or http://localhost:8554/stream (MJPEG fallback)

Sample maritime videos (search YouTube for):
  - "drone footage boats ocean"
  - "aerial ship tracking"
  - "maritime surveillance footage"
"""

import argparse
import logging
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class VideoFileSource:
    """Streams frames from a video file with looping support."""

    def __init__(self, video_path: str, width: int = 1280, height: int = 720,
                 fps: int = 30, loop: bool = True):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.fps = fps
        self.loop = loop
        self.frame_count = 0
        self.start_time = time.time()

        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Loaded video: {video_path}")
        logger.info(f"  Source: {self.source_width}x{self.source_height} @ {self.source_fps:.1f}fps")
        logger.info(f"  Output: {self.width}x{self.height} @ {self.fps}fps")
        logger.info(f"  Duration: {self.total_frames / self.source_fps:.1f}s ({self.total_frames} frames)")
        logger.info(f"  Loop: {self.loop}")

    def generate_frame(self) -> np.ndarray:
        """Get next frame from video file."""
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                # Restart from beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    # Video is broken, return black frame
                    return np.zeros((self.height, self.width, 3), dtype=np.uint8)
                logger.debug("Video looped")
            else:
                # Return last frame or black
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.frame_count += 1

        # Resize to target resolution if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Add minimal timestamp overlay
        elapsed = time.time() - self.start_time
        fps_actual = self.frame_count / max(0.1, elapsed)
        timestamp = time.strftime("%H:%M:%S")

        cv2.putText(frame, timestamp, (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{fps_actual:.0f}fps", (self.width - 50, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        return frame

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


def generate_perlin_noise_2d(shape, res, seed=None):
    """Generate 2D Perlin noise for realistic water texture."""
    if seed is not None:
        np.random.seed(seed)

    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)

    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11

    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


@dataclass
class Ship3D:
    """A ship in 3D world space (meters from drone)."""
    world_x: float
    world_z: float
    heading: float
    speed: float
    length: float
    width: float
    height: float
    ship_type: str
    wake_phase: float = 0.0

    def update(self, dt: float, world_bounds: float = 2000) -> None:
        """Update ship position in world space."""
        rad = math.radians(self.heading)
        self.world_x += math.sin(rad) * self.speed * dt
        self.world_z += math.cos(rad) * self.speed * dt
        self.wake_phase += dt * 2

        if self.world_z < 100:
            self.world_z = world_bounds
        elif self.world_z > world_bounds:
            self.world_z = 100
        if abs(self.world_x) > world_bounds:
            self.world_x = -self.world_x * 0.5


class TestPatternGenerator:
    """Generates photorealistic drone camera view of ocean with ships."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update = self.start_time

        # Camera/drone parameters
        self.drone_altitude = 100.0
        self.camera_pitch = -15.0
        self.camera_fov_h = 70.0

        # Horizon at ~20% from top for more ocean view
        self.horizon_ratio = 0.20

        # Animation
        self.wave_time = 0.0

        # Perspective
        self.focal_length = self.width / (2 * math.tan(math.radians(self.camera_fov_h / 2)))

        # Create ships - mix of sizes for detection testing
        self.ships: List[Ship3D] = [
            # Large cargo ship
            Ship3D(
                world_x=50, world_z=200,
                heading=70, speed=5,
                length=180, width=28, height=22,
                ship_type="cargo",
            ),
            # Tanker - medium distance
            Ship3D(
                world_x=-120, world_z=380,
                heading=40, speed=4,
                length=220, width=38, height=16,
                ship_type="tanker",
            ),
            # Fishing boat - close (YOLO trained on these)
            Ship3D(
                world_x=-40, world_z=100,
                heading=95, speed=7,
                length=18, width=5, height=3,
                ship_type="fishing",
            ),
            # Another fishing boat
            Ship3D(
                world_x=100, world_z=150,
                heading=130, speed=6,
                length=22, width=6, height=4,
                ship_type="fishing",
            ),
            # Small speedboat/yacht (YOLO sees these as "boat")
            Ship3D(
                world_x=20, world_z=80,
                heading=85, speed=12,
                length=12, width=4, height=2,
                ship_type="speedboat",
            ),
            # Distant cargo
            Ship3D(
                world_x=-200, world_z=650,
                heading=25, speed=5,
                length=160, width=25, height=20,
                ship_type="cargo",
            ),
        ]

        # Pre-generate textures
        self._generate_ocean_texture()
        self._generate_sky_texture()
        self._generate_ship_sprites()

    def _generate_sky_texture(self) -> None:
        """Generate realistic sky with gradient and clouds."""
        horizon_y = int(self.height * self.horizon_ratio)
        self.sky = np.zeros((horizon_y, self.width, 3), dtype=np.uint8)

        for y in range(horizon_y):
            t = y / max(1, horizon_y - 1)
            # Sky gradient: deep blue at top -> pale hazy blue at horizon
            r = int(135 + 85 * t)
            g = int(175 + 60 * t)
            b = int(220 + 30 * t)
            self.sky[y, :] = (b, g, r)

        # Add some haze/clouds near horizon
        cloud_band = self.sky[int(horizon_y*0.7):, :].astype(np.float32)
        cloud_band += 15
        self.sky[int(horizon_y*0.7):, :] = np.clip(cloud_band, 0, 255).astype(np.uint8)

    def _generate_ocean_texture(self) -> None:
        """Generate base ocean texture with Perlin noise."""
        ocean_height = self.height - int(self.height * self.horizon_ratio)

        # Generate multiple octaves of noise for realistic water
        try:
            noise1 = generate_perlin_noise_2d((ocean_height, self.width), (4, 8), seed=42)
            noise2 = generate_perlin_noise_2d((ocean_height, self.width), (8, 16), seed=43)
            noise3 = generate_perlin_noise_2d((ocean_height, self.width), (16, 32), seed=44)
            combined = noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2
        except:
            # Fallback if Perlin fails
            combined = np.random.rand(ocean_height, self.width) * 0.3

        self.ocean_noise = ((combined + 1) * 0.5 * 30).astype(np.int16)

    def _generate_ship_sprites(self) -> None:
        """Pre-render ship sprites at various angles for realistic look."""
        self.ship_sprites = {}

        for ship_type in ["cargo", "tanker", "fishing", "speedboat"]:
            sprites = {}

            # Generate sprites for different view angles
            for angle in range(0, 360, 15):
                sprite = self._render_ship_sprite(ship_type, angle)
                sprites[angle] = sprite

            self.ship_sprites[ship_type] = sprites

    def _render_ship_sprite(self, ship_type: str, angle: int) -> np.ndarray:
        """Render a photorealistic ship sprite with soft edges and natural shading."""
        # Larger sprite size for more detail
        if ship_type == "cargo":
            w, h = 280, 70
        elif ship_type == "tanker":
            w, h = 320, 80
        elif ship_type == "speedboat":
            w, h = 70, 25
        else:  # fishing
            w, h = 100, 35

        # Create with extra padding for blur effects
        pad = 10
        sprite = np.zeros((h * 2 + pad * 2, w + pad * 2, 4), dtype=np.uint8)

        # Offset for padding
        ox, oy = pad, pad

        # Ship hull shape
        hull_pts = self._get_hull_shape(ship_type, w, h)
        hull_pts = hull_pts + np.array([ox, oy])

        # Realistic ship colors (from real ship photos)
        if ship_type == "cargo":
            # Container ships often have dark blue/black hulls, red below waterline
            hull_color = (35, 35, 45)  # Dark gray-blue
            deck_color = (85, 90, 100)  # Lighter gray deck
            superstructure_color = (200, 205, 210)  # White-ish bridge
        elif ship_type == "tanker":
            # Tankers often dark with white/cream superstructure
            hull_color = (25, 30, 40)  # Very dark
            deck_color = (70, 75, 85)  # Gray deck
            superstructure_color = (195, 200, 205)  # Off-white
        elif ship_type == "speedboat":
            # Speedboats/yachts - white fiberglass hull
            hull_color = (220, 215, 210)  # White
            deck_color = (200, 195, 185)  # Off-white deck
            superstructure_color = (235, 235, 230)  # Bright white
        else:
            # Fishing boats - often white/blue hull
            hull_color = (130, 100, 85)  # Blue-ish
            deck_color = (180, 175, 170)  # Light gray
            superstructure_color = (220, 220, 215)  # White

        # Draw main hull body
        cv2.fillPoly(sprite, [hull_pts], (*hull_color, 255))

        # Add texture/weathering to hull (makes it look less flat)
        hull_mask = np.zeros((h * 2 + pad * 2, w + pad * 2), dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull_pts], 255)

        # Add random streaks for weathering/rust
        for _ in range(15):
            sx = np.random.randint(ox, ox + w)
            sy = np.random.randint(oy + h - 10, oy + h + 10)
            streak_len = np.random.randint(5, 20)
            streak_color = tuple(min(255, c + np.random.randint(-20, 30)) for c in hull_color)
            cv2.line(sprite, (sx, sy), (sx + np.random.randint(-3, 3), sy + streak_len),
                    (*streak_color, 200), 1)

        # Add waterline - red anti-fouling paint (realistic detail)
        waterline_pts = hull_pts.copy()
        waterline_pts[:, 1] = np.clip(waterline_pts[:, 1], oy + h - 5, oy + h + 8)
        cv2.fillPoly(sprite, [waterline_pts], (40, 35, 120, 255))  # Dark red

        # Deck area (inset from hull)
        deck_pts = self._get_hull_shape(ship_type, int(w * 0.88), int(h * 0.7))
        deck_pts = deck_pts + np.array([ox + int(w * 0.06), oy + int(h * 0.15)])
        cv2.fillPoly(sprite, [deck_pts], (*deck_color, 255))

        # Add deck texture - small variations
        deck_mask = np.zeros((h * 2 + pad * 2, w + pad * 2), dtype=np.uint8)
        cv2.fillPoly(deck_mask, [deck_pts], 255)
        deck_noise = np.random.randint(-8, 8, (h * 2 + pad * 2, w + pad * 2, 3), dtype=np.int16)
        deck_region = sprite[:, :, :3].astype(np.int16)
        deck_region = np.where(deck_mask[:, :, np.newaxis] > 0,
                               np.clip(deck_region + deck_noise, 0, 255), deck_region)
        sprite[:, :, :3] = deck_region.astype(np.uint8)

        # Add superstructure based on type
        if ship_type == "cargo":
            # Bridge/accommodation block at stern
            bx = ox + int(w * 0.08)
            by = oy + h - int(h * 0.55)
            bw = int(w * 0.14)
            bh = int(h * 1.1)

            # Multi-level bridge
            for level in range(4):
                lw = bw - level * 3
                lh = bh // 4
                ly = by + level * lh
                lx = bx + level * 1
                color_fade = tuple(max(0, c - level * 10) for c in superstructure_color)
                cv2.rectangle(sprite, (lx, ly), (lx + lw, ly + lh), (*color_fade, 255), -1)

            # Windows on bridge
            for wy in range(by + 5, by + bh - 10, 8):
                cv2.rectangle(sprite, (bx + 2, wy), (bx + bw - 2, wy + 4), (80, 60, 40, 255), -1)

            # Container stacks - varied colors like real container ships
            container_colors = [
                (35, 45, 170),   # Red (Maersk-ish)
                (45, 130, 45),   # Green (Evergreen-ish)
                (160, 50, 40),   # Blue
                (30, 170, 180),  # Yellow/Orange
                (120, 120, 125), # Gray (Hapag-Lloyd-ish)
                (50, 40, 35),    # Dark (MSC-ish)
            ]

            # Container block
            container_start = ox + int(w * 0.28)
            container_end = ox + int(w * 0.92)
            container_rows = 3
            container_cols = 8
            cw = (container_end - container_start) // container_cols
            ch = int(h * 0.18)

            for row in range(container_rows):
                for col in range(container_cols):
                    cx = container_start + col * cw
                    cy = oy + h - int(h * 0.35) + row * ch
                    color = container_colors[(row * 3 + col) % len(container_colors)]
                    cv2.rectangle(sprite, (cx, cy), (cx + cw - 1, cy + ch - 1), (*color, 255), -1)
                    # Subtle edge
                    cv2.rectangle(sprite, (cx, cy), (cx + cw - 1, cy + ch - 1),
                                 (color[0]//2, color[1]//2, color[2]//2, 255), 1)

            # Cargo cranes
            for crane_x in [int(w * 0.4), int(w * 0.6)]:
                cx = ox + crane_x
                cv2.line(sprite, (cx, oy + h - int(h*0.4)), (cx, oy + int(h*0.2)), (100, 105, 115, 255), 3)
                cv2.line(sprite, (cx, oy + int(h*0.2)), (cx + int(w*0.12), oy + int(h*0.4)), (100, 105, 115, 255), 2)

        elif ship_type == "tanker":
            # Bridge at stern
            bx = ox + int(w * 0.05)
            bw = int(w * 0.12)
            bh = int(h * 1.0)
            by = oy + h - bh // 2

            for level in range(3):
                lw = bw - level * 4
                lh = bh // 3
                cv2.rectangle(sprite, (bx + level * 2, by + level * lh),
                             (bx + level * 2 + lw, by + (level + 1) * lh),
                             (*superstructure_color, 255), -1)

            # Spherical LNG tanks or cylindrical oil tanks
            tank_color = (160, 165, 175)  # Silvery gray
            for i in range(5):
                tx = ox + int(w * 0.22 + i * w * 0.145)
                ty = oy + h
                rx, ry = int(w * 0.055), int(h * 0.35)
                cv2.ellipse(sprite, (tx, ty), (rx, ry), 0, 180, 360, (*tank_color, 255), -1)
                # Tank highlight
                cv2.ellipse(sprite, (tx - 3, ty - 2), (rx - 5, ry - 5), 0, 200, 320,
                           (tank_color[0] + 30, tank_color[1] + 30, tank_color[2] + 25, 255), 2)

            # Pipes running along deck
            pipe_y = oy + h - 2
            cv2.line(sprite, (ox + int(w * 0.18), pipe_y), (ox + int(w * 0.95), pipe_y), (75, 80, 90, 255), 3)

        elif ship_type == "speedboat":
            # Small cabin/windshield
            bx = ox + int(w * 0.25)
            bw = int(w * 0.3)
            bh = int(h * 0.6)
            by = oy + h - bh // 2

            # Windshield (dark tinted)
            cv2.rectangle(sprite, (bx, by), (bx + bw, by + bh), (50, 40, 35, 255), -1)

            # Seats/cockpit area
            cv2.rectangle(sprite, (ox + int(w * 0.55), oy + h - int(h * 0.25)),
                         (ox + int(w * 0.8), oy + h + int(h * 0.25)), (60, 55, 50, 255), -1)

            # Engine area at stern
            cv2.rectangle(sprite, (ox + int(w * 0.05), oy + h - int(h * 0.2)),
                         (ox + int(w * 0.2), oy + h + int(h * 0.2)), (40, 40, 45, 255), -1)

        else:  # fishing
            # Wheelhouse/bridge
            bx = ox + int(w * 0.25)
            bw = int(w * 0.35)
            bh = int(h * 0.9)
            by = oy + h - bh // 2
            cv2.rectangle(sprite, (bx, by), (bx + bw, by + bh), (*superstructure_color, 255), -1)

            # Windows
            cv2.rectangle(sprite, (bx + 3, by + 3), (bx + bw - 3, by + int(bh * 0.4)), (70, 50, 35, 255), -1)

            # Mast
            mx = ox + int(w * 0.4)
            cv2.line(sprite, (mx, by), (mx, oy + int(h * 0.1)), (60, 60, 70, 255), 2)

            # Boom/crane
            cv2.line(sprite, (mx, oy + int(h * 0.3)), (ox + int(w * 0.7), oy + h - 5), (70, 70, 80, 255), 2)

            # Fishing gear at stern
            cv2.rectangle(sprite, (ox + int(w * 0.7), oy + h - int(h * 0.3)),
                         (ox + int(w * 0.9), oy + h + int(h * 0.3)), (90, 95, 100, 255), -1)

        # Apply realistic lighting - sun from upper left
        # Gradient shadow on right side
        for x in range(w + pad * 2):
            shadow_factor = 0.7 + 0.3 * (1 - x / (w + pad * 2))
            sprite[:, x, :3] = (sprite[:, x, :3] * shadow_factor).astype(np.uint8)

        # Slight vertical gradient (top lighter - sun reflection)
        for y in range(h * 2 + pad * 2):
            t = y / (h * 2 + pad * 2)
            light_factor = 1.1 - 0.2 * t
            sprite[y, :, :3] = np.clip(sprite[y, :, :3] * light_factor, 0, 255).astype(np.uint8)

        # CRITICAL: Apply Gaussian blur to soften hard edges
        # This makes it look more like a real photo, not a drawing
        rgb = sprite[:, :, :3]
        alpha = sprite[:, :, 3]

        # Blur the RGB slightly
        rgb_blurred = cv2.GaussianBlur(rgb, (3, 3), 0.8)

        # Blur alpha more to create soft edges
        alpha_blurred = cv2.GaussianBlur(alpha, (5, 5), 1.2)

        sprite = np.dstack([rgb_blurred, alpha_blurred])

        # Add subtle noise to simulate camera sensor (very important for realism)
        noise = np.random.normal(0, 3, sprite[:, :, :3].shape).astype(np.int16)
        sprite[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return sprite

    def _get_hull_shape(self, ship_type: str, w: int, h: int) -> np.ndarray:
        """Get hull polygon points for ship type."""
        if ship_type == "cargo":
            # Cargo ship - boxy with tapered bow
            pts = np.array([
                [w - 5, h],  # Bow point
                [int(w * 0.85), h - int(h * 0.4)],
                [int(w * 0.15), h - int(h * 0.45)],
                [5, h - int(h * 0.3)],  # Stern
                [5, h + int(h * 0.3)],
                [int(w * 0.15), h + int(h * 0.45)],
                [int(w * 0.85), h + int(h * 0.4)],
            ], dtype=np.int32)
        elif ship_type == "tanker":
            # Tanker - wider, flatter
            pts = np.array([
                [w - 5, h],
                [int(w * 0.9), h - int(h * 0.45)],
                [int(w * 0.1), h - int(h * 0.48)],
                [5, h - int(h * 0.35)],
                [5, h + int(h * 0.35)],
                [int(w * 0.1), h + int(h * 0.48)],
                [int(w * 0.9), h + int(h * 0.45)],
            ], dtype=np.int32)
        elif ship_type == "speedboat":
            # Speedboat - sleek, pointed bow
            pts = np.array([
                [w - 2, h],  # Sharp bow
                [int(w * 0.6), h - int(h * 0.4)],
                [int(w * 0.1), h - int(h * 0.35)],
                [3, h - int(h * 0.2)],  # Squared stern
                [3, h + int(h * 0.2)],
                [int(w * 0.1), h + int(h * 0.35)],
                [int(w * 0.6), h + int(h * 0.4)],
            ], dtype=np.int32)
        else:  # fishing
            # Fishing boat - smaller, more curved
            pts = np.array([
                [w - 3, h],
                [int(w * 0.7), h - int(h * 0.4)],
                [int(w * 0.2), h - int(h * 0.45)],
                [3, h],
                [int(w * 0.2), h + int(h * 0.45)],
                [int(w * 0.7), h + int(h * 0.4)],
            ], dtype=np.int32)

        return pts

    def _world_to_screen(self, world_x: float, world_y: float, world_z: float) -> Optional[Tuple[int, int, float]]:
        """Project 3D world point to 2D screen coordinates."""
        if world_z <= 10:
            return None

        pitch_rad = math.radians(self.camera_pitch)
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)

        y_rot = world_y * cos_p - world_z * sin_p
        z_rot = world_y * sin_p + world_z * cos_p

        if z_rot <= 10:
            return None

        screen_x = int(self.width / 2 + (world_x * self.focal_length) / z_rot)
        screen_y = int(self.height / 2 - (y_rot * self.focal_length) / z_rot)

        return screen_x, screen_y, world_z

    def _draw_ocean(self, frame: np.ndarray) -> None:
        """Draw photorealistic ocean."""
        horizon_y = int(self.height * self.horizon_ratio)
        ocean = frame[horizon_y:, :]

        # Base ocean color with depth gradient
        for y in range(ocean.shape[0]):
            # Distance factor (0 at horizon, 1 at bottom/near camera)
            t = y / max(1, ocean.shape[0] - 1)

            # Deep blue near horizon, darker greenish-blue nearby
            b = int(160 - 80 * t + 10 * math.sin(self.wave_time + y * 0.02))
            g = int(130 - 70 * t + 5 * math.sin(self.wave_time * 1.3 + y * 0.015))
            r = int(80 - 50 * t)

            ocean[y, :] = (max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r)))

        # Add wave texture from pre-generated noise
        if hasattr(self, 'ocean_noise'):
            # Animate the noise
            offset = int(self.wave_time * 30) % self.width
            noise_shifted = np.roll(self.ocean_noise, offset, axis=1)

            # Apply noise as brightness variation
            ocean_float = ocean.astype(np.float32)
            for c in range(3):
                ocean_float[:, :, c] += noise_shifted[:ocean.shape[0], :self.width] * (0.5 + 0.5 * c)

            ocean[:] = np.clip(ocean_float, 0, 255).astype(np.uint8)

        # Add wave highlights (white caps)
        for y in range(10, ocean.shape[0], 3):
            t = y / ocean.shape[0]
            wave_density = 0.02 + 0.08 * t  # More waves closer

            for x in range(0, self.width, 4):
                wave = math.sin(x * (0.02 + 0.03 * t) + self.wave_time * 2 + y * 0.1)
                wave += 0.5 * math.sin(x * 0.03 + self.wave_time * 1.5 - y * 0.05)

                if wave > 0.6:
                    # White cap
                    intensity = int((wave - 0.6) * 150 * t)
                    oy = min(y + 2, ocean.shape[0])
                    ox = min(x + 4, self.width)
                    ocean[y:oy, x:ox] = np.clip(
                        ocean[y:oy, x:ox].astype(np.int16) + intensity, 0, 255
                    ).astype(np.uint8)

        # Sun reflection path (glitter)
        sun_x = self.width // 2 + int(100 * math.sin(self.wave_time * 0.1))
        for _ in range(80):
            gx = sun_x + int(np.random.normal(0, self.width * 0.12))
            gy = np.random.randint(5, int(ocean.shape[0] * 0.5))
            if 0 <= gx < self.width:
                intensity = np.random.randint(30, 100)
                size = np.random.randint(1, 3)
                cv2.circle(ocean, (gx, gy), size,
                          (min(255, 100 + intensity), min(255, 80 + intensity), min(255, 60 + intensity)), -1)

    def _draw_ship(self, frame: np.ndarray, ship: Ship3D) -> None:
        """Draw ship with perspective transformation and atmospheric effects."""
        water_y = -self.drone_altitude

        # Get ship center position on screen
        result = self._world_to_screen(ship.world_x, water_y + ship.height / 2, ship.world_z)
        if result is None:
            return

        screen_x, screen_y, depth = result

        # Calculate apparent size based on distance
        scale = self.focal_length / depth
        apparent_width = int(ship.length * scale)
        apparent_height = int(ship.width * scale * 1.5)  # Perspective compression

        # Minimum size for YOLO detection (needs reasonable pixel coverage)
        if apparent_width < 15 or apparent_height < 8:
            return  # Too small to detect reliably

        # Get sprite for ship's heading
        relative_heading = (ship.heading - 0) % 360
        sprite_angle = round(relative_heading / 15) * 15 % 360

        if ship.ship_type in self.ship_sprites and sprite_angle in self.ship_sprites[ship.ship_type]:
            sprite = self.ship_sprites[ship.ship_type][sprite_angle].copy()
        else:
            sprite = self.ship_sprites.get(ship.ship_type, {}).get(0)
            if sprite is None:
                return
            sprite = sprite.copy()

        # Apply atmospheric haze based on distance (distant ships appear hazier)
        max_visible_distance = 800
        haze_factor = min(0.6, depth / max_visible_distance * 0.5)
        if haze_factor > 0.05:
            haze_color = np.array([200, 195, 185], dtype=np.float32)  # Blueish atmospheric haze
            sprite_rgb = sprite[:, :, :3].astype(np.float32)
            sprite[:, :, :3] = (sprite_rgb * (1 - haze_factor) +
                               haze_color * haze_factor).astype(np.uint8)
            # Also reduce contrast for distant ships
            sprite[:, :, :3] = np.clip(
                sprite[:, :, :3].astype(np.float32) * (1 - haze_factor * 0.3) + 40 * haze_factor,
                0, 255
            ).astype(np.uint8)

        # Resize sprite based on distance
        resized = cv2.resize(sprite, (apparent_width, apparent_height), interpolation=cv2.INTER_AREA)

        # Apply perspective transform (skew for viewing angle)
        src_pts = np.float32([
            [0, 0],
            [apparent_width, 0],
            [apparent_width, apparent_height],
            [0, apparent_height]
        ])

        # Skew based on position relative to center (perspective effect)
        skew = (screen_x - self.width / 2) / self.width * 0.15
        horizon_y = int(self.height * self.horizon_ratio)
        vert_skew = max(0, (screen_y - horizon_y) / (self.height - horizon_y)) * 0.25

        dst_pts = np.float32([
            [apparent_width * skew, -apparent_height * vert_skew * 0.3],
            [apparent_width * (1 + skew * 0.4), -apparent_height * vert_skew * 0.2],
            [apparent_width * (1 - skew * 0.2), apparent_height * (1 + vert_skew * 0.8)],
            [-apparent_width * skew * 0.3, apparent_height * (1 + vert_skew * 0.6)]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(resized, M, (apparent_width * 2, apparent_height * 2),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # Draw wake first (behind ship)
        self._draw_wake(frame, ship, water_y, scale)

        # Composite ship onto frame
        self._composite_sprite(frame, warped,
                              screen_x - apparent_width // 2,
                              screen_y - apparent_height // 2)

    def _draw_wake(self, frame: np.ndarray, ship: Ship3D, water_y: float, scale: float) -> None:
        """Draw realistic V-shaped ship wake with foam."""
        heading_rad = math.radians(ship.heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        wake_length = int(ship.length * 2.5)

        # Draw stern turbulence (white water directly behind ship)
        for i in range(5, 40, 2):
            for _ in range(3):
                spread = np.random.normal(0, ship.width * 0.3)
                local_z = -ship.length / 2 - i + np.random.normal(0, 3)

                wx = ship.world_x + spread * cos_h - local_z * sin_h
                wz = ship.world_z + spread * sin_h + local_z * cos_h

                result = self._world_to_screen(wx, water_y, wz)
                if result:
                    sx, sy, depth = result
                    if 0 <= sx < self.width and 0 <= sy < self.height:
                        intensity = int(200 - i * 3)
                        size = max(1, int(4 * scale))
                        color = (min(255, 180 + intensity // 4),
                                min(255, 175 + intensity // 4),
                                min(255, 165 + intensity // 4))
                        cv2.circle(frame, (sx, sy), size, color, -1)

        # Draw V-shaped wake arms
        for i in range(15, wake_length, 3):
            # V-wake spreads at ~19 degrees (Kelvin wake angle)
            spread = i * 0.35
            wave_off = math.sin(ship.wake_phase + i * 0.1) * 2

            for side in [-1, 1]:
                local_x = side * (spread + wave_off)
                local_z = -ship.length / 2 - i

                wx = ship.world_x + local_x * cos_h - local_z * sin_h
                wz = ship.world_z + local_x * sin_h + local_z * cos_h

                result = self._world_to_screen(wx, water_y, wz)
                if result:
                    sx, sy, depth = result

                    # Fade with distance
                    fade = max(0, 1 - i / wake_length)
                    intensity = int(120 * fade)
                    size = max(1, int(3 * scale * fade))

                    # Slightly blue-white foam
                    color = (min(255, 160 + intensity),
                            min(255, 155 + intensity),
                            min(255, 145 + intensity))

                    if 0 <= sx < self.width and 0 <= sy < self.height:
                        cv2.circle(frame, (sx, sy), size, color, -1)

    def _composite_sprite(self, frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        """Composite BGRA sprite onto frame with alpha blending."""
        if sprite.shape[2] != 4:
            return

        sh, sw = sprite.shape[:2]
        fh, fw = frame.shape[:2]

        # Clip to frame bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + sw), min(fh, y + sh)

        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return

        sprite_region = sprite[sy1:sy2, sx1:sx2]
        frame_region = frame[y1:y2, x1:x2]

        alpha = sprite_region[:, :, 3:4].astype(np.float32) / 255.0

        blended = (sprite_region[:, :, :3].astype(np.float32) * alpha +
                   frame_region.astype(np.float32) * (1 - alpha))

        frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    def generate_frame(self) -> np.ndarray:
        """Generate next frame."""
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        self.frame_count += 1
        self.wave_time += dt

        # Create frame with sky
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        horizon_y = int(self.height * self.horizon_ratio)
        frame[:horizon_y, :] = self.sky

        # Draw ocean
        self._draw_ocean(frame)

        # Atmospheric horizon haze - crucial for realism
        haze_height = 30
        for y in range(max(0, horizon_y - haze_height), min(self.height, horizon_y + haze_height)):
            dist = abs(y - horizon_y)
            haze_strength = 0.4 * (1 - dist / haze_height)
            if haze_strength > 0:
                haze_color = np.array([210, 205, 195], dtype=np.float32)
                frame[y] = (frame[y].astype(np.float32) * (1 - haze_strength) +
                           haze_color * haze_strength).astype(np.uint8)

        # Update and draw ships
        for ship in self.ships:
            ship.update(dt)

        sorted_ships = sorted(self.ships, key=lambda s: -s.world_z)
        for ship in sorted_ships:
            self._draw_ship(frame, ship)

        # Apply photographic post-processing
        frame = self._apply_camera_effects(frame)

        # Minimal HUD (after effects so it stays crisp)
        self._draw_hud(frame, now)

        return frame

    def _apply_camera_effects(self, frame: np.ndarray) -> np.ndarray:
        """Apply realistic camera/lens effects."""
        # Slight overall blur to simulate lens softness
        frame = cv2.GaussianBlur(frame, (3, 3), 0.5)

        # Add subtle vignette (darker corners like real camera)
        rows, cols = frame.shape[:2]
        X = np.arange(0, cols)
        Y = np.arange(0, rows)
        X, Y = np.meshgrid(X, Y)
        center_x, center_y = cols // 2, rows // 2
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - 0.3 * (dist / max_dist) ** 2
        vignette = vignette[:, :, np.newaxis]
        frame = (frame * vignette).astype(np.uint8)

        # Add very subtle sensor noise
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Slight color temperature adjustment (warmer, like afternoon sun)
        frame[:, :, 2] = np.clip(frame[:, :, 2].astype(np.int16) + 5, 0, 255).astype(np.uint8)  # More red
        frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) - 3, 0, 255).astype(np.uint8)  # Less blue

        return frame

    def _draw_hud(self, frame: np.ndarray, now: float) -> None:
        """Draw minimal HUD overlay."""
        elapsed = now - self.start_time
        fps_actual = self.frame_count / max(0.1, elapsed)

        # Timestamp only - keep it minimal so YOLO focuses on ships
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Small FPS indicator
        cv2.putText(frame, f"{fps_actual:.0f}fps", (self.width - 50, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)


def run_rtsp_server_gstreamer(
    width: int,
    height: int,
    fps: int,
    port: int,
) -> None:
    """Run RTSP server using GStreamer (preferred method)."""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        gi.require_version('GstRtspServer', '1.0')
        from gi.repository import Gst, GstRtspServer, GLib
    except ImportError:
        logger.error("GStreamer Python bindings not found.")
        logger.error("Install with: pip install PyGObject")
        logger.error("On macOS: brew install pygobject3 gst-rtsp-server")
        logger.error("On Linux: sudo apt install python3-gi gstreamer1.0-rtsp")
        logger.error("")
        logger.error("Falling back to FFmpeg method...")
        run_rtsp_server_ffmpeg(width, height, fps, port)
        return

    Gst.init(None)

    generator = TestPatternGenerator(width, height, fps)

    class VideoFactory(GstRtspServer.RTSPMediaFactory):
        def __init__(self, gen: TestPatternGenerator):
            super().__init__()
            self.gen = gen
            self.frame_duration = 1.0 / fps
            self.set_shared(True)

        def do_create_element(self, url):
            pipeline_str = (
                f"appsrc name=source is-live=true format=time "
                f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
                f"! videoconvert "
                f"! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast "
                f"! rtph264pay name=pay0 pt=96 config-interval=1"
            )
            return Gst.parse_launch(pipeline_str)

    class RTSPServer:
        def __init__(self):
            self.server = GstRtspServer.RTSPServer()
            self.server.set_service(str(port))

            self.factory = VideoFactory(generator)
            self.factory.set_shared(True)

            mount_points = self.server.get_mount_points()
            mount_points.add_factory("/stream", self.factory)

            self.server.attach(None)
            logger.info(f"RTSP server started: rtsp://localhost:{port}/stream")

    # For GStreamer RTSP, we need a different approach - use FFmpeg instead
    logger.info("GStreamer RTSP factory setup is complex, using FFmpeg approach...")
    run_rtsp_server_ffmpeg(width, height, fps, port)


def run_rtsp_server_ffmpeg(
    width: int,
    height: int,
    fps: int,
    port: int,
    video_path: str = None,
    loop: bool = True,
) -> None:
    """Run RTSP server using FFmpeg + mediamtx."""
    import shutil
    import subprocess
    import tempfile
    import os

    # Choose frame source
    if video_path:
        generator = VideoFileSource(video_path, width, height, fps, loop)
    else:
        generator = TestPatternGenerator(width, height, fps)
    frame_time = 1.0 / fps

    # Check for mediamtx or rtsp-simple-server
    mediamtx_path = shutil.which("mediamtx") or shutil.which("rtsp-simple-server")

    if not mediamtx_path:
        logger.warning("mediamtx/rtsp-simple-server not found.")
        logger.warning("Install mediamtx for proper RTSP server:")
        logger.warning("  macOS: brew install mediamtx")
        logger.warning("  Linux: Download from https://github.com/bluenviron/mediamtx/releases")
        logger.warning("")
        logger.warning("Falling back to HTTP MJPEG stream instead...")
        run_http_mjpeg_server(width, height, fps, port, video_path, loop)
        return

    # FFmpeg to push to RTSP server
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.error("FFmpeg not found. Install with: brew install ffmpeg")
        return

    # Create a minimal mediamtx config that allows publishing
    # RTP port must be even, RTCP = RTP + 1
    rtp_port = port + 2 if (port + 2) % 2 == 0 else port + 3
    rtcp_port = rtp_port + 1
    config_content = f"""
logLevel: warn
logDestinations: [stdout]

rtspAddress: :{port}
rtpAddress: :{rtp_port}
rtcpAddress: :{rtcp_port}

paths:
  stream:
    source: publisher
"""

    # Write temp config
    config_fd, config_path = tempfile.mkstemp(suffix=".yml", prefix="mediamtx_")
    try:
        os.write(config_fd, config_content.encode())
        os.close(config_fd)

        # Start mediamtx with our config
        logger.info(f"Starting mediamtx RTSP server on port {port}...")
        mediamtx_proc = subprocess.Popen(
            [mediamtx_path, config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait for mediamtx to be ready
        time.sleep(2)

        # Check if mediamtx started successfully
        if mediamtx_proc.poll() is not None:
            output = mediamtx_proc.stdout.read().decode() if mediamtx_proc.stdout else ""
            logger.error(f"mediamtx failed to start: {output[:300]}")
            logger.warning("Falling back to HTTP MJPEG stream...")
            run_http_mjpeg_server(width, height, fps, port, video_path, loop)
            return

        # FFmpeg command to push RTSP stream
        ffmpeg_cmd = [
            ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", "2000k",
            "-g", str(fps),
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            f"rtsp://127.0.0.1:{port}/stream",
        ]

        logger.info("Starting FFmpeg encoder...")

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give FFmpeg a moment to connect
        time.sleep(1)

        # Check if FFmpeg started successfully
        if ffmpeg_proc.poll() is not None:
            stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
            logger.error(f"FFmpeg failed to start: {stderr[-300:]}")
            mediamtx_proc.terminate()
            logger.warning("Falling back to HTTP MJPEG stream...")
            run_http_mjpeg_server(width, height, fps, port, video_path, loop)
            return

        running = True

        def shutdown(sig, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        logger.info(f"Fake camera streaming at: rtsp://localhost:{port}/stream")
        logger.info("Press Ctrl+C to stop")

        try:
            frames_sent = 0
            while running:
                loop_start = time.time()

                # Check if mediamtx is still running
                if mediamtx_proc.poll() is not None:
                    output = mediamtx_proc.stdout.read().decode() if mediamtx_proc.stdout else ""
                    logger.error(f"mediamtx died: {output[-300:]}")
                    break

                frame = generator.generate_frame()

                try:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                    ffmpeg_proc.stdin.flush()
                    frames_sent += 1
                    if frames_sent == 1:
                        logger.info("First frame sent successfully")
                except BrokenPipeError:
                    # Check why FFmpeg died
                    if ffmpeg_proc.poll() is not None:
                        stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
                        logger.error(f"FFmpeg exited after {frames_sent} frames: {stderr[-300:]}")
                    else:
                        logger.error(f"FFmpeg pipe broken after {frames_sent} frames")
                    break

                # Rate limiting
                elapsed = time.time() - loop_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        finally:
            logger.info("Shutting down...")
            if ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.close()
                except Exception:
                    pass
            ffmpeg_proc.terminate()
            mediamtx_proc.terminate()
            try:
                ffmpeg_proc.wait(timeout=2)
                mediamtx_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                ffmpeg_proc.kill()
                mediamtx_proc.kill()

    finally:
        # Clean up temp config
        try:
            os.unlink(config_path)
        except Exception:
            pass


def run_http_mjpeg_server(
    width: int,
    height: int,
    fps: int,
    port: int,
    video_path: str = None,
    loop: bool = True,
) -> None:
    """Fallback: Run HTTP MJPEG server (works without extra dependencies)."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import socketserver

    # Choose frame source
    if video_path:
        generator = VideoFileSource(video_path, width, height, fps, loop)
    else:
        generator = TestPatternGenerator(width, height, fps)
    frame_time = 1.0 / fps

    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress HTTP logs

        def do_GET(self):
            if self.path == "/stream" or self.path == "/":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()

                try:
                    while True:
                        loop_start = time.time()

                        frame = generator.generate_frame()
                        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b"\r\n")

                        elapsed = time.time() - loop_start
                        if elapsed < frame_time:
                            time.sleep(frame_time - elapsed)

                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", port), MJPEGHandler)

    logger.info(f"HTTP MJPEG server started: http://localhost:{port}/stream")
    logger.info("Note: This is MJPEG, not RTSP. Update camera config to use HTTP URL.")
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Fake RTSP camera server for testing Terminal Guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream a real video file (best for YOLO detection):
  python scripts/fake_camera.py --video boats.mp4

  # Stream with looping disabled:
  python scripts/fake_camera.py --video boats.mp4 --no-loop

  # Use synthetic ships (fallback):
  python scripts/fake_camera.py

  # HTTP MJPEG mode (simpler, works without mediamtx):
  python scripts/fake_camera.py --video boats.mp4 --http

Sample maritime videos - search YouTube/Pexels for:
  - "drone footage boats ocean 4k"
  - "aerial ship tracking"
  - "sailing yacht drone view"
        """
    )
    parser.add_argument(
        "--video", "-v", type=str, default=None,
        help="Path to video file to stream (recommended for realistic testing)"
    )
    parser.add_argument(
        "--port", type=int, default=8554,
        help="RTSP server port (default: 8554)"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Video width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Video height (default: 720)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--loop", action="store_true", default=True,
        help="Loop video file (default: True)"
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="Don't loop video file"
    )
    parser.add_argument(
        "--http", action="store_true",
        help="Use HTTP MJPEG instead of RTSP (simpler, no dependencies)"
    )
    args = parser.parse_args()

    # Handle loop flag
    loop = args.loop and not args.no_loop

    if args.video:
        logger.info(f"Starting video stream: {args.video}")
        logger.info(f"Output: {args.width}x{args.height}@{args.fps}fps, loop={loop}")
    else:
        logger.info(f"Starting synthetic camera: {args.width}x{args.height}@{args.fps}fps")
        logger.info("Tip: Use --video <file> for better YOLO detection with real footage")

    if args.http:
        run_http_mjpeg_server(args.width, args.height, args.fps, args.port, args.video, loop)
    else:
        run_rtsp_server_ffmpeg(args.width, args.height, args.fps, args.port, args.video, loop)


if __name__ == "__main__":
    main()
