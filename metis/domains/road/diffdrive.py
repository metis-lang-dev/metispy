"""Differential-drive tracker — wheelchair item (g).

A wheelchair is not a bicycle: it turns in place and has no steering
geometry. This tracker maps the planner's setpoint (anchor +
target_speed, the same vocabulary the MPC consumes) to wheel speeds
(v_left, v_right) for the hardware layer. CARLA has no diff-drive
actor, so live CARLA trials keep SimplePilot on a car stand-in; this
module is the A72/hardware path, validated by the kinematic unit sim
below (drive_to in tests).

Control law: heading error -> yaw rate (P), forward speed scaled down
by heading error (drive where you face), full turn-in-place above
TURN_IN_PLACE_RAD. Wheel speeds clamp to the profile top speed; the
bucket contract is the caller's business (command target_speed one
margin below the bucket edge, as everywhere)."""

import math
from dataclasses import dataclass

from .profile import WHEELCHAIR, EnvProfile


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0    # rad


class DiffDrivePilot:
    K_HEADING = 2.0            # yaw-rate P gain [1/s]
    TURN_IN_PLACE_RAD = 1.0    # ~57 deg: stop and pivot beyond this
    TRACK_W = 0.6              # wheel track [m]

    def __init__(self, profile: EnvProfile = WHEELCHAIR):
        self.profile = profile

    def step(self, pose: Pose, target_xy, target_speed: float):
        """-> (v_left, v_right) wheel speeds [m/s]."""
        if target_xy is None or target_speed <= 0.0:
            return 0.0, 0.0
        dx, dy = target_xy[0] - pose.x, target_xy[1] - pose.y
        if dx * dx + dy * dy < 0.01:
            return 0.0, 0.0
        err = math.atan2(dy, dx) - pose.yaw
        err = (err + math.pi) % (2 * math.pi) - math.pi
        omega = max(-2.0, min(2.0, self.K_HEADING * err))
        if abs(err) > self.TURN_IN_PLACE_RAD:
            v = 0.0                       # pivot first
        else:
            v = target_speed * max(0.0, math.cos(err))
        cap = self.profile.cells_to_speed(self.profile.v_max)
        v = min(v, cap)
        half = omega * self.TRACK_W / 2.0
        return v - half, v + half

    def integrate(self, pose: Pose, v_l: float, v_r: float,
                  dt: float) -> Pose:
        """Kinematic model for tests: exact unicycle step."""
        v = (v_l + v_r) / 2.0
        omega = (v_r - v_l) / self.TRACK_W
        return Pose(pose.x + v * math.cos(pose.yaw) * dt,
                    pose.y + v * math.sin(pose.yaw) * dt,
                    pose.yaw + omega * dt)
