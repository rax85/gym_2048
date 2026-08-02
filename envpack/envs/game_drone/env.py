"""A Gymnasium environment for a physically accurate fixed-wing drone flight simulator with 3D wireframe rendering."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Simulation Constants
DT = 0.05                         # 20 Hz simulation rate
MAP_SIZE = 2000.0                 # 2000m x 2000m area
G = 9.81                          # Gravity (m/s^2)
RHO = 1.225                       # Air density at sea level (kg/m^3)

# Drone Physical Parameters
MASS = 5.0                        # kg
WING_AREA = 0.8                   # m^2
C_L0 = 0.2                        # Lift coefficient at zero AoA
C_L_ALPHA = 4.0                   # Lift slope (per radian)
C_D0 = 0.04                       # Parasitic drag coefficient
K_INDUCED = 0.05                  # Induced drag factor (K * C_L^2)
MAX_THRUST = 45.0                 # N (thrust at 100% throttle)
FUEL_CAPACITY = 200.0             # fuel units

# Control Limits (Fly-by-wire rates)
MAX_PITCH_RATE = 0.8              # rad/s
MAX_ROLL_RATE = 1.2               # rad/s
MAX_YAW_RATE = 0.4                # rad/s

# Airstrips/Runways Configuration
RUNWAY_1 = {"x": 200.0, "z": 200.0, "heading": math.pi / 4, "length": 120.0, "width": 15.0} # Start Runway
RUNWAY_2 = {"x": 1600.0, "z": 1600.0, "heading": math.pi / 4, "length": 120.0, "width": 15.0} # Target Runway

# Canvas Dimensions
WIDTH = 400
HEIGHT = 300
HEADER_PX = 30
FOOTER_PX = 20

# Colors
COLOR_BG = (15, 23, 42)           # Dark slate blue
COLOR_HEADER = (30, 41, 59)
COLOR_TEXT = (248, 250, 252)
COLOR_SKY = (30, 41, 59)
COLOR_GROUND = (15, 118, 110)      # Teal/Cyan ground lines
COLOR_RUNWAY = (234, 179, 8)       # Yellow runway markings
COLOR_DRONE = (244, 63, 94)        # Rose/Red drone color
COLOR_HUD = (34, 197, 94)          # Green HUD markings


class GymDroneEnv(gym.Env):
    """A Gymnasium environment for a physically accurate fixed-wing drone simulator."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        # Font setup
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._title_font = ImageFont.truetype(font_file, 12)
            self._stats_font = ImageFont.truetype(font_file, 10)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._title_font = ImageFont.load_default()
            self._stats_font = ImageFont.load_default()

        # Action space: [throttle, elevator, aileron, rudder]
        # throttle: [0.0, 1.0]
        # elevator/aileron/rudder: [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Observation space
        # [x, y, z, vx, vy, vz, pitch, roll, yaw, fuel, dist_to_r2, heading_err]
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                ),
                "valid_mask": spaces.Box(
                    low=1, high=1, shape=(4,), dtype=np.int8
                ),
            }
        )

        # Background base setup
        self._background = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[HEIGHT - FOOTER_PX :, :] = COLOR_HEADER

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset flight state variables."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._pos = np.array(state["pos"], dtype=np.float64)
            self._vel = np.array(state["vel"], dtype=np.float64)
            self._pitch = state["pitch"]
            self._roll = state["roll"]
            self._yaw = state["yaw"]
            self._fuel = state["fuel"]
            self._step_count = state["step_count"]
            self._crashed = state["crashed"]
            self._landed = state["landed"]
            return self._create_observation(), {}

        # Reset drone at Runway 1 starting position
        self._pos = np.array([RUNWAY_1["x"], 0.0, RUNWAY_1["z"]], dtype=np.float64)
        # Pointing along the runway heading
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = RUNWAY_1["heading"]

        # Initially stationary
        self._vel = np.zeros(3, dtype=np.float64)
        self._fuel = FUEL_CAPACITY
        self._step_count = 0
        self._crashed = False
        self._landed = False

        return self._create_observation(), {}

    def _get_terrain_height(self, x: float, z: float) -> float:
        """Procedurally generate landscape height using superposition of harmonic waves."""
        # Ensure flat runway pads
        # Start Runway weight
        d1 = math.hypot(x - RUNWAY_1["x"], z - RUNWAY_1["z"])
        w1 = max(0.0, min(1.0, (150.0 - d1) / 50.0)) if d1 < 150.0 else 0.0

        # Target Runway weight
        d2 = math.hypot(x - RUNWAY_2["x"], z - RUNWAY_2["z"])
        w2 = max(0.0, min(1.0, (150.0 - d2) / 50.0)) if d2 < 150.0 else 0.0

        # Procedural mountain/hill formulas
        h = 80.0 * (
            0.45 * math.sin(0.0035 * x) * math.cos(0.003 * z)
            + 0.25 * math.sin(0.009 * x + 0.004 * z) * math.cos(0.008 * z)
            + 0.15 * math.sin(0.018 * x) * math.sin(0.015 * z)
        )
        # Scale height down to zero near runways
        h_final = h * (1.0 - w1) * (1.0 - w2)
        return max(0.0, h_final)

    def _create_observation(self) -> Dict[str, Any]:
        """Generate observation vector and action valid mask."""
        # Calculate target metrics
        dx = RUNWAY_2["x"] - self._pos[0]
        dz = RUNWAY_2["z"] - self._pos[2]
        dist = math.hypot(dx, dz)
        
        target_yaw = math.atan2(dx, dz)
        heading_err = (target_yaw - self._yaw + math.pi) % (2.0 * math.pi) - math.pi

        obs = np.array(
            [
                self._pos[0],            # X coordinate
                self._pos[1],            # Altitude (Y)
                self._pos[2],            # Z coordinate
                self._vel[0],            # Vx
                self._vel[1],            # Vy
                self._vel[2],            # Vz
                self._pitch,             # Pitch
                self._roll,              # Roll
                self._yaw,               # Yaw
                self._fuel,              # Fuel
                dist,                    # Distance to R2
                heading_err,             # Heading error to R2
            ],
            dtype=np.float32,
        )

        valid_mask = np.ones((4,), dtype=np.int8)

        return {
            "observation": obs,
            "valid_mask": valid_mask,
        }

    def _on_runway(self, x: float, z: float) -> bool:
        """Check if coordinates are within Runway 1 or Runway 2 dimensions."""
        for rwy in [RUNWAY_1, RUNWAY_2]:
            dx = x - rwy["x"]
            dz = z - rwy["z"]
            angle = rwy["heading"]
            # Rotate offset back to runway local frame
            local_x = dx * math.cos(angle) + dz * math.sin(angle)
            local_z = -dx * math.sin(angle) + dz * math.cos(angle)
            if abs(local_x) < rwy["length"] / 2.0 and abs(local_z) < rwy["width"] / 2.0:
                return True
        return False

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Run one physics update step."""
        self._step_count += 1
        throttle, elevator, aileron, rudder = action

        # Fuel consumption
        fuel_consumption = (0.05 + 0.45 * throttle) * DT
        self._fuel = max(0.0, self._fuel - fuel_consumption)

        if self._crashed or self._landed:
            return self._create_observation(), 0.0, True, False, {"state": self._get_state()}

        # 1. Update Attitude kinematics (Fly-by-wire gyroscope rates)
        # Clamped pitch and roll if on the ground rolling
        is_on_rwy = self._on_runway(self._pos[0], self._pos[2])
        ground_y = self._get_terrain_height(self._pos[0], self._pos[2])
        is_grounded = (self._pos[1] - ground_y) <= 0.05

        if is_grounded and is_on_rwy:
            # Allow pitching up to lift off, but keep roll flat and pitch controlled
            self._roll = 0.0
            self._pitch = max(0.0, self._pitch + elevator * MAX_PITCH_RATE * DT)
            self._yaw += rudder * MAX_YAW_RATE * DT
        else:
            self._pitch += elevator * MAX_PITCH_RATE * DT
            self._roll += aileron * MAX_ROLL_RATE * DT
            self._yaw += rudder * MAX_YAW_RATE * DT

        # Normalize angles
        self._pitch = max(-math.pi / 3, min(math.pi / 3, self._pitch))
        self._roll = max(-math.pi / 2, min(math.pi / 2, self._roll))
        self._yaw = (self._yaw + math.pi) % (2.0 * math.pi) - math.pi

        # 2. Flight Dynamics Physics
        # Compute airspeed vector and magnitude
        airspeed = np.linalg.norm(self._vel)

        # Get orientation unit vectors
        # Heading vector (forward nose)
        hx = math.cos(self._pitch) * math.sin(self._yaw)
        hy = math.sin(self._pitch)
        hz = math.cos(self._pitch) * math.cos(self._yaw)
        heading_vec = np.array([hx, hy, hz])

        # Lift vector (perpendicular to wings, tilting with roll)
        fwd = heading_vec
        right = np.array([math.cos(self._yaw), 0.0, -math.sin(self._yaw)])
        up = np.cross(right, fwd)
        
        # Apply roll rotation to the lift direction
        lift_dir = up * math.cos(self._roll) + right * math.sin(self._roll)
        lift_dir /= np.linalg.norm(lift_dir)

        # Thrust Force
        thrust_force = throttle * MAX_THRUST * heading_vec if self._fuel > 0 else np.zeros(3)

        # Compute Angle of Attack (AoA)
        if airspeed > 1.0:
            vel_dir = self._vel / airspeed
            aoa = math.acos(max(-1.0, min(1.0, np.dot(heading_vec, vel_dir))))
            if np.dot(heading_vec, vel_dir) < 0:
                aoa = math.pi - aoa
        else:
            aoa = 0.0
            vel_dir = heading_vec

        # Stall condition: lift breaks down at high AoA
        is_stalled = abs(aoa) > 0.28  # ~16 degrees

        # Lift Force
        c_l = (C_L0 + C_L_ALPHA * aoa) if not is_stalled else 0.05
        lift_mag = 0.5 * RHO * (airspeed**2) * WING_AREA * c_l
        lift_force = lift_mag * lift_dir

        # Drag Force
        c_d = C_D0 + K_INDUCED * (c_l**2)
        drag_mag = 0.5 * RHO * (airspeed**2) * WING_AREA * c_d
        drag_force = -drag_mag * vel_dir

        # Gravity Force
        gravity_force = np.array([0.0, -MASS * G, 0.0])

        # Combine Forces
        total_force = thrust_force + drag_force + lift_force + gravity_force
        
        # Ground reaction and friction when grounded on runway
        if is_grounded and is_on_rwy:
            # Cancel gravity and downward velocity
            if total_force[1] < 0.0:
                total_force[1] = 0.0
            # Apply rolling friction
            friction = -0.15 * self._vel
            total_force += friction

        accel = total_force / MASS

        # 3. Update position and velocity
        self._vel += accel * DT
        
        # Prevent sinking into the runway ground when rolling
        if is_grounded and is_on_rwy:
            self._vel[1] = max(0.0, self._vel[1])
            
        self._pos += self._vel * DT

        # Keep within map bounds
        self._pos[0] = max(0.0, min(MAP_SIZE, self._pos[0]))
        self._pos[2] = max(0.0, min(MAP_SIZE, self._pos[2]))

        # Calculate heights
        ground_y = self._get_terrain_height(self._pos[0], self._pos[2])
        agl = self._pos[1] - ground_y

        # 4. Check Ground Collision & Landing
        reward = 0.1  # Survival reward
        terminated = False

        # Reward lower fuel usage
        reward -= 0.05 * fuel_consumption

        # Terrain proximity penalty (Minimal AGL requirement = 15m)
        # Ignore this penalty when within takeoff/landing range of the runways
        d_runway1 = math.hypot(self._pos[0] - RUNWAY_1["x"], self._pos[2] - RUNWAY_1["z"])
        d_runway2 = math.hypot(self._pos[0] - RUNWAY_2["x"], self._pos[2] - RUNWAY_2["z"])
        
        near_runway = (d_runway1 < 150.0) or (d_runway2 < 150.0)

        if agl < 15.0 and not near_runway:
            # Penalty proportional to terrain closeness
            reward -= 0.2 * (15.0 - agl)

        # Ground collision check
        if agl <= 0.0:
            is_currently_on_rwy = self._on_runway(self._pos[0], self._pos[2])
            if is_currently_on_rwy:
                # Safe touchdown check
                sink_ok = self._vel[1] >= -3.0  # soft landing sink rate
                pitch_ok = abs(self._pitch) < 0.15
                roll_ok = abs(self._roll) < 0.15
                
                if sink_ok and pitch_ok and roll_ok:
                    # Grounded on runway: clamp altitude and vertical velocity
                    self._pos[1] = ground_y
                    self._vel[1] = 0.0
                    self._roll = 0.0
                    
                    # If we touched down on target Runway 2 at safe speed, we won!
                    if d_runway2 < 60.0 and airspeed > 2.0:
                        self._landed = True
                        reward += 100.0  # Success!
                        terminated = True
                else:
                    self._crashed = True
                    reward -= 50.0
                    terminated = True
            else:
                # Crashed elsewhere in the landscape
                self._crashed = True
                reward -= 50.0
                terminated = True


        # Timeout / Max steps check handled by Gymnasium wrapper or max steps limit
        return self._create_observation(), float(reward), terminated, False, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        return {
            "pos": self._pos.tolist(),
            "vel": self._vel.tolist(),
            "pitch": self._pitch,
            "roll": self._roll,
            "yaw": self._yaw,
            "fuel": self._fuel,
            "step_count": self._step_count,
            "crashed": self._crashed,
            "landed": self._landed,
        }

    def render(self) -> np.ndarray:
        """Produce 3D flat shaded render and 2D overview map onto canvas."""
        image = Image.fromarray(self._background.copy())
        draw = ImageDraw.Draw(image)

        # 3D Viewport Drawing
        self._render_3d_viewport(draw)

        # Redraw Header on top of 3D rendering to mask overflow
        draw.rectangle([0, 0, WIDTH, HEADER_PX], fill=COLOR_HEADER)
        draw.text((10, 8), "FIXED WING AUTOPILOT SIMULATOR", fill=COLOR_TEXT, font=self._title_font)

        # Redraw Footer on top of 3D rendering to mask overflow
        draw.rectangle([0, HEIGHT - FOOTER_PX, WIDTH, HEIGHT], fill=COLOR_HEADER)
        airspeed = np.linalg.norm(self._vel)
        agl = self._pos[1] - self._get_terrain_height(self._pos[0], self._pos[2])
        draw.text(
            (10, HEIGHT - 15),
            f"SPD: {airspeed:4.1f}m/s  |  ALT: {self._pos[1]:4.0f}m (AGL:{agl:3.0f}m)  |  FUEL: {self._fuel:3.0f}u",
            fill=COLOR_TEXT,
            font=self._stats_font,
        )

        # 2D Map Overlap Drawing (Top Right)
        self._render_2d_map(draw)

        return np.array(image, dtype=np.uint8)

    def _render_3d_viewport(self, draw: ImageDraw.Draw) -> None:
        """Render a 3D flat shaded representation of the landscape, runway, and drone."""
        # Define Chase Camera Position (Behind the drone)
        hx = math.cos(self._pitch) * math.sin(self._yaw)
        hy = math.sin(self._pitch)
        hz = math.cos(self._pitch) * math.cos(self._yaw)
        heading_vec = np.array([hx, hy, hz])

        # Put camera 35m behind and 8m above drone
        cam_pos = self._pos - 35.0 * heading_vec + np.array([0.0, 8.0, 0.0])
        look_at = self._pos + 15.0 * heading_vec

        # Build View Matrix vectors
        fwd_cam = look_at - cam_pos
        fwd_cam /= np.linalg.norm(fwd_cam)

        right_cam = np.cross(np.array([0.0, 1.0, 0.0]), fwd_cam)
        right_cam /= np.linalg.norm(right_cam)

        up_cam = np.cross(fwd_cam, right_cam)

        # Projection Focal Length
        focal = 220.0
        cx, cy = WIDTH // 2, (HEIGHT - HEADER_PX - FOOTER_PX) // 2 + HEADER_PX

        def to_cam_space(pt_world: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            rel = pt_world - cam_pos
            return np.array([
                np.dot(rel, right_cam),
                np.dot(rel, up_cam),
                np.dot(rel, fwd_cam)
            ])

        def clip_polygon_z(pts_cam: List[npt.NDArray[np.float64]], min_z: float = 0.5) -> List[npt.NDArray[np.float64]]:
            clipped = []
            if not pts_cam:
                return clipped
            for i in range(len(pts_cam)):
                p1 = pts_cam[i]
                p2 = pts_cam[(i + 1) % len(pts_cam)]
                
                p1_in = p1[2] >= min_z
                p2_in = p2[2] >= min_z
                
                if p1_in:
                    if p2_in:
                        clipped.append(p2)
                    else:
                        t = (min_z - p1[2]) / (p2[2] - p1[2])
                        intersect = p1 + t * (p2 - p1)
                        clipped.append(intersect)
                else:
                    if p2_in:
                        t = (min_z - p1[2]) / (p2[2] - p1[2])
                        intersect = p1 + t * (p2 - p1)
                        clipped.append(intersect)
                        clipped.append(p2)
            return clipped

        def project_cam(pt_cam: npt.NDArray[np.float64]) -> Tuple[float, float]:
            x_cam, y_cam, z_cam = pt_cam
            sx = cx + (x_cam / z_cam) * focal
            sy = cy - (y_cam / z_cam) * focal
            return sx, sy

        # Generate a list of elements to render
        render_list = []

        # 1. Generate Terrain Grid
        grid_res = 16
        grid_spacing = 50.0
        center_x = round(self._pos[0] / grid_spacing) * grid_spacing
        center_z = round(self._pos[2] / grid_spacing) * grid_spacing

        world_grid = {}
        camera_grid = {}
        for u in range(-8, 9):
            for v in range(-8, 9):
                gx = max(0.0, min(MAP_SIZE, center_x + u * grid_spacing))
                gz = max(0.0, min(MAP_SIZE, center_z + v * grid_spacing))
                gy = self._get_terrain_height(gx, gz)
                w_pt = np.array([gx, gy, gz])
                world_grid[(u, v)] = w_pt
                camera_grid[(u, v)] = to_cam_space(w_pt)

        # Build terrain triangles
        for u in range(-8, 8):
            for v in range(-8, 8):
                w00 = world_grid[(u, v)]
                w10 = world_grid[(u + 1, v)]
                w11 = world_grid[(u + 1, v + 1)]
                w01 = world_grid[(u, v + 1)]

                c00 = camera_grid[(u, v)]
                c10 = camera_grid[(u + 1, v)]
                c11 = camera_grid[(u + 1, v + 1)]
                c01 = camera_grid[(u, v + 1)]

                # Triangle 1
                avg_z1 = (c00[2] + c10[2] + c01[2]) / 3.0
                if avg_z1 >= 0.5:
                    render_list.append({
                        'type': 'terrain',
                        'w0': w00, 'w1': w10, 'w2': w01,
                        'c0': c00, 'c1': c10, 'c2': c01,
                        'avg_z': avg_z1
                    })

                # Triangle 2
                avg_z2 = (c10[2] + c11[2] + c01[2]) / 3.0
                if avg_z2 >= 0.5:
                    render_list.append({
                        'type': 'terrain',
                        'w0': w10, 'w1': w11, 'w2': w01,
                        'c0': c10, 'c1': c11, 'c2': c01,
                        'avg_z': avg_z2
                    })

        # 2. Generate Runways
        for rwy in (RUNWAY_1, RUNWAY_2):
            rx, rz = rwy["x"], rwy["z"]
            r_head = rwy["heading"]
            r_len = rwy["length"]
            r_wid = rwy["width"]

            r_fwd = np.array([math.sin(r_head), 0.0, math.cos(r_head)])
            r_right = np.cross(np.array([0.0, 1.0, 0.0]), r_fwd)

            y = self._get_terrain_height(rx, rz) + 0.1  # slightly above terrain to prevent z-fighting

            w1 = np.array([rx, y, rz]) + r_fwd * (r_len / 2) - r_right * (r_wid / 2)
            w2 = np.array([rx, y, rz]) + r_fwd * (r_len / 2) + r_right * (r_wid / 2)
            w3 = np.array([rx, y, rz]) - r_fwd * (r_len / 2) + r_right * (r_wid / 2)
            w4 = np.array([rx, y, rz]) - r_fwd * (r_len / 2) - r_right * (r_wid / 2)

            c1 = to_cam_space(w1)
            c2 = to_cam_space(w2)
            c3 = to_cam_space(w3)
            c4 = to_cam_space(w4)

            # Tri 1: w1, w2, w4
            avg_z1 = (c1[2] + c2[2] + c4[2]) / 3.0
            if avg_z1 >= 0.5:
                render_list.append({
                    'type': 'runway',
                    'w0': w1, 'w1': w2, 'w2': w4,
                    'c0': c1, 'c1': c2, 'c2': c4,
                    'avg_z': avg_z1,
                    'fill_color': (50, 55, 65),
                    'outline_color': COLOR_RUNWAY
                })

            # Tri 2: w2, w3, w4
            avg_z2 = (c2[2] + c3[2] + c4[2]) / 3.0
            if avg_z2 >= 0.5:
                render_list.append({
                    'type': 'runway',
                    'w0': w2, 'w1': w3, 'w2': w4,
                    'c0': c2, 'c1': c3, 'c2': c4,
                    'avg_z': avg_z2,
                    'fill_color': (50, 55, 65),
                    'outline_color': COLOR_RUNWAY
                })

        # Sort all elements back-to-front (descending depth)
        render_list.sort(key=lambda item: item['avg_z'], reverse=True)

        # Light source for flat shading
        light_dir = np.array([0.3, 0.9, 0.3])
        light_dir /= np.linalg.norm(light_dir)

        # Render sorted elements
        for item in render_list:
            c0, c1, c2 = item['c0'], item['c1'], item['c2']
            clipped_cam = clip_polygon_z([c0, c1, c2])
            if len(clipped_cam) < 3:
                continue

            pts_scr = [project_cam(p) for p in clipped_cam]
            pts_scr = [(int(x), int(y)) for x, y in pts_scr]

            if item['type'] == 'terrain':
                w0, w1, w2 = item['w0'], item['w1'], item['w2']
                # Calculate face normal in world space
                v1 = w1 - w0
                v2 = w2 - w0
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
                else:
                    normal = np.array([0.0, 1.0, 0.0])
                if normal[1] < 0:
                    normal = -normal

                # Simple diffuse lighting
                dot = np.dot(normal, light_dir)
                dot = max(0.1, min(1.0, dot))

                # Altitude-based color gradient
                avg_y = (w0[1] + w1[1] + w2[1]) / 3.0
                height_factor = max(0.0, min(1.0, avg_y / 80.0))

                r_base = 15 + 30 * height_factor
                g_base = 80 + 80 * height_factor
                b_base = 80 + 50 * height_factor

                lit_r = int(r_base * (0.3 + 0.7 * dot))
                lit_g = int(g_base * (0.3 + 0.7 * dot))
                lit_b = int(b_base * (0.3 + 0.7 * dot))
                color = (lit_r, lit_g, lit_b)

                draw.polygon(pts_scr, fill=color, outline=color)

            elif item['type'] == 'runway':
                fill_color = item['fill_color']
                outline_color = item['outline_color']
                draw.polygon(pts_scr, fill=fill_color, outline=outline_color)

        # 3. Draw a HUD Flight Ladder (Drawn on top of 3D rendering)
        # Center indicator
        draw.line([cx - 8, cy, cx - 2, cy], fill=COLOR_HUD)
        draw.line([cx + 2, cy, cx + 8, cy], fill=COLOR_HUD)
        draw.line([cx, cy - 2, cx, cy + 2], fill=COLOR_HUD)

        # Pitch indicators (moving with roll and pitch)
        pitch_deg = math.degrees(self._pitch)
        roll_deg = math.degrees(self._roll)
        
        # Simple pitch bar draw
        bar_y_offset = (pitch_deg / 20.0) * 80.0
        bx = cx
        by = cy + bar_y_offset

        # Draw a line rotated by the roll angle
        cos_r = math.cos(-self._roll)
        sin_r = math.sin(-self._roll)
        
        rx1, ry1 = -20 * cos_r, -20 * sin_r
        rx2, ry2 = 20 * cos_r, 20 * sin_r
        
        draw.line([bx + rx1, by + ry1, bx + rx2, by + ry2], fill=COLOR_HUD)

    def _render_2d_map(self, draw: ImageDraw.Draw) -> None:
        """Render a 2D overview map showing the drone position and targets."""
        map_size_px = 70
        offset_x = WIDTH - map_size_px - 8
        offset_y = HEADER_PX + 8

        # Draw map outline
        draw.rectangle(
            [offset_x - 1, offset_y - 1, offset_x + map_size_px + 1, offset_y + map_size_px + 1],
            outline=COLOR_TEXT,
            fill=(10, 15, 30),
        )

        def map_coords(wx: float, wz: float) -> Tuple[float, float]:
            sx = offset_x + (wx / MAP_SIZE) * map_size_px
            sy = offset_y + (wz / MAP_SIZE) * map_size_px
            return sx, sy

        # Draw Runway 1 (Green)
        r1x, r1y = map_coords(RUNWAY_1["x"], RUNWAY_1["z"])
        draw.rectangle([r1x - 2, r1y - 2, r1x + 2, r1y + 2], fill=(74, 222, 128))

        # Draw Runway 2 (Red)
        r2x, r2y = map_coords(RUNWAY_2["x"], RUNWAY_2["z"])
        draw.rectangle([r2x - 2, r2y - 2, r2x + 2, r2y + 2], fill=(248, 113, 113))

        # Draw Drone indicator (arrow based on yaw)
        dx, dy = map_coords(self._pos[0], self._pos[2])
        # Simple triangle arrow pointing along yaw heading
        yaw_vec = np.array([math.sin(self._yaw), -math.cos(self._yaw)])  # top-down 2D vector
        yaw_perp = np.array([-yaw_vec[1], yaw_vec[0]])
        
        p1 = np.array([dx, dy]) + 4.0 * yaw_vec
        p2 = np.array([dx, dy]) - 3.0 * yaw_vec + 2.0 * yaw_perp
        p3 = np.array([dx, dy]) - 3.0 * yaw_vec - 2.0 * yaw_perp

        draw.polygon([tuple(p1), tuple(p2), tuple(p3)], fill=COLOR_DRONE)

    def close(self) -> None:
        pass
