"""Streams every motor's torque and speed, normalized to its limits, to the live-plot window.

The sim publishes over zmq (localhost); visualization/live_plot.py runs as a separate
process with its own Qt window, so it doesn't fight mjpython's viewer for the main thread.
Sending never blocks: if the plot window is closed, data is simply dropped.

All values are signed and at the JOINT (output) side:
  torque = applied torque / stall torque     (stall = motor stall x gear, capped by the actuator)
  speed  = joint speed / no-load speed       (no-load = motor no-load / gear)
MotorModel's limit is |torque| <= 1 - |speed|: a diamond in these units.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zmq

PORT = 5556
SEND_HZ = 30
# Order of the `motors` list in Simulation.py
NAMES = ['fr hip', 'fl hip', 'br hip', 'bl hip',
         'fr knee', 'fl knee', 'br knee', 'bl knee',
         'fr wheel1', 'fr wheel2', 'fl wheel1', 'fl wheel2',
         'br wheel1', 'br wheel2', 'bl wheel1', 'bl wheel2']
GROUPS = {'hip': [0, 1, 2, 3], 'knee': [4, 5, 6, 7], 'wheel': list(range(8, 16))}


def launch_plotter():
    """Start the plot window as its own Python process (same venv)."""
    script = Path(__file__).resolve().parent.parent / 'visualization' / 'live_plot.py'
    python = Path(sys.prefix) / 'bin' / 'python'      # plain python, not mjpython
    return subprocess.Popen([str(python), str(script), str(PORT)])


class LivePublisher:
    def __init__(self, m, motors):
        self.motors = motors
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 50)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://127.0.0.1:{PORT}")
        self.dof = [int(m.jnt_dofadr[m.joint(mot.motor_name).id]) for mot in motors]
        stall, no_load = [], []
        for mot in motors:
            s = mot.t_stall * mot.gear_ratio
            i = mot.ctrl_index
            if m.actuator_ctrllimited[i]:
                s = min(s, float(m.actuator_ctrlrange[i][1] * m.actuator_gear[i][0]))
            stall.append(s)
            no_load.append(mot.w_no_load / mot.gear_ratio)
        self.stall, self.no_load = np.array(stall), np.array(no_load)
        # Rated (continuous) torque/speed per joint type, as fractions of stall / no-load
        self.rated = {}
        for g, idx in GROUPS.items():
            mot, i = motors[idx[0]], idx[0]
            p = mot.motor_params
            if 'rated_torque' in p and 'rated_speed' in p:
                self.rated[g] = [p['rated_torque'] * mot.gear_ratio / self.stall[i],
                                 p['rated_speed'] / mot.gear_ratio / self.no_load[i]]
        self.limits = {g: [float(self.stall[idx[0]]), float(self.no_load[idx[0]])] for g, idx in GROUPS.items()}
        print("Live plot: joint limits (stall Nm / no-load rad/s): " + ", ".join(
            f"{g} {v[0]:.0f} / {v[1]:.1f}" for g, v in self.limits.items()))
        self.pk_tau = None
        self.last_send = 0.0

    def update(self, d, status=""):
        tau = d.qfrc_actuator[self.dof] / self.stall
        # Keep the largest-magnitude torque between sends so short spikes still show up
        if self.pk_tau is None:
            self.pk_tau = tau.copy()
        else:
            bigger = np.abs(tau) > np.abs(self.pk_tau)
            self.pk_tau[bigger] = tau[bigger]
        now = time.time()
        if now - self.last_send < 1.0 / SEND_HZ:
            return
        self.last_send = now
        w = d.qvel[self.dof] / self.no_load
        msg = {'t': float(d.time), 'status': status, 'names': NAMES, 'groups': GROUPS,
               'rated': self.rated, 'limits': self.limits,
               'tau': [round(float(x), 4) for x in self.pk_tau],
               'w': [round(float(x), 4) for x in w]}
        try:
            self.sock.send_string(json.dumps(msg), zmq.NOBLOCK)
        except zmq.Again:
            pass
        self.pk_tau = None
