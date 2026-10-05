import time
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import lib.MotorModel as motor

class JoystickController:
    def __init__(self, controller_type: str, m: mujoco.MjModel, d: mujoco.MjData, motors: motor.MotorModel):
        self.controller_type = controller_type
        self.m = m
        self.d = d
        self.motors = motors

        # Initializing joystick with pygame
        pygame.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(self.js.get_name())

        # User control variables:
        self.joystick_deadzone = 0.2
        self.max_wheel_vel = 100
        self.wheel_accel = 200     # how fast wheel speed targets may ramp UP (rad/s^2); slowing down is instant
        self.max_knee_vel = 0.01
        self.max_knee_lead = 0.3   # anti-windup: knee target may lead the actual knee by at most this (rad)

        # Desired setpoints
        self.fr_knee_des_pos = 0
        self.fl_knee_des_pos = 0
        self.br_knee_des_pos = 0
        self.bl_knee_des_pos = 0

        self.fr_knee_des_vel = 0
        self.fl_knee_des_vel = 0
        self.br_knee_des_vel = 0
        self.bl_knee_des_vel = 0

        self.fr_hip_des_pos = 0
        self.fl_hip_des_pos = 0
        self.br_hip_des_pos = 0
        self.bl_hip_des_pos = 0

        self.fr_hip_des_vel = 0
        self.fl_hip_des_vel = 0
        self.br_hip_des_vel = 0
        self.bl_hip_des_vel = 0

        self.right_wheel_vel_des = 0
        self.left_wheel_vel_des = 0

        self.d_up = 0
        self.d_down = 0
        self.d_left = 0
        self.d_right = 0

        self.hip_splay = 0
        self.dt = m.opt.timestep

        # Controller modes (toggled with the Share button on PS4)
        self.modes = ["Basic", "Smart Gen-0"]
        self.mode_index = 0
        self.aux_button = 0        # default so non-PS4 mappings don't crash
        self.prev_aux_button = 0   # for edge detection (toggle once per press)

        # Smart stair climbing (touchpad starts; touchpad again or Options stops).
        # Tuned for the full-scale (FS) robot. Wheel speeds in rad/s, angles in rad.
        self.touchpad = 0
        self.prev_touchpad = 0
        self.climb_phase = None                   # None, SETUP, APPROACH, TUMBLE, ROLL, SYNC
        self.climb_creep_speed = 3.0              # gentle forward wheel speed (FS: ~0.3 m/s)
        self.climb_hold_speed = 1.0               # light forward push while pressed against a riser
        self.climb_bump_frac = 0.5                # bump = hip torque drops this fraction of the expected
                                                  # riser load below its running baseline...
        self.climb_stall_time = 0.05              # ...for this long (s)
        self.climb_baseline_tau = 1.0             # time constant of each hip's torque baseline (s)
        self.climb_torque_filter = 0.05           # low-pass on hip torque before comparing (s)
        self.climb_pair_wait = 0.4                # synced: wait this long for the other end to bump (s)
        self.climb_settle_time = 0.3              # ignore stalls this soon after the wheels start (s)
        self.climb_tumble_rate = 1.5              # bogie tumble speed (rad/s)
        self.climb_knee_lead = 0.3                # tumble target may lead the bogie by at most this
        self.climb_land_min = np.deg2rad(100)     # a bogie stopping only counts as a landing past this rotation
        self.climb_stuck_speed = np.deg2rad(5)    # bogie turning slower than this = not progressing (rad/s)
        self.climb_land_time = 0.1                # stopped this long past land_min = landed on the tread (s)
        self.climb_stuck_time = 1.5               # not progressing this long before land_min = can't climb (s)
        self.climb_hip_rate = 1.0                 # hips to straight-down at the start (rad/s)
        self.climb_splay_rate = 0.4               # SYNC: hip splay change speed (rad/s)
        self.climb_max_splay = np.deg2rad(60)     # SYNC: give up past this splay
        self.climb_hip_block = 0.15               # SYNC: hip lagging its command this much = that end is blocked
        self.climb_riser_tol = 0.04               # wheel within this of a riser = against it (m)
        self.climb_run_guess = 0.30               # stair run assumed until measured (m)
        self.climb_top_margin = 0.5               # roll this many runs past the expected riser = top reached
        self.climb_max_roll = np.deg2rad(20)      # stop if the body rolls more than this
        self.climb_timeout = 12.0                 # stop if any phase takes longer than this (s)
        self.wheel_leg = None                     # per-leg wheel speeds {leg: rad/s} overriding left/right
        self.climb_verbose = True                 # print phase changes and a status snapshot every 0.5 s
        self.climb_torque_frac = 0.5              # wheel torque cap while climbing, as a fraction of the slip
                                                  # limit (mu=1 x per-wheel load x radius), so a blocked wheel
                                                  # stalls instead of spinning in place
        self.climb_wheel_cap = None               # output torque cap (Nm), computed at climb start
        # Stance and pitch limiting while climbing
        self.climb_base_splay = np.deg2rad(30)    # pre-splay (from APPROACH on): room to lengthen AND shorten legs
        self.climb_pitch_cap = np.deg2rad(25)     # keep body pitch (nose-up) at or below this...
        self.climb_pitch_deadband = np.deg2rad(2) # ...acting only when this far past the target
        self.climb_pitch_gain = 2.0               # pitch error (rad) -> differential splay speed (rad/s)
        self.climb_pitch_rate = 0.5               # max differential splay speed (rad/s)
        self.climb_hip_track_rate = 1.0           # how fast hip commands follow their targets (rad/s)
        # ROLL: fixed wheel torque that rolls on a tread but can't climb or spin against a riser
        self.climb_roll_torque_frac = 0.3         # fraction of the slip limit (FS: ~3.3 Nm per wheel)
        self.climb_roll_stall_speed = 0.15        # every wheel below this = robot blocked (rad/s)
        self.climb_roll_stall_time = 0.1          # ...for this long (s)
        self.wheel_leg_torque = None              # per-leg wheel torques {leg: Nm}, overriding speed control
        self.climb_advance_margin = 0.05          # a front tumble needs the rear to have a wheelbase + this of room (m)
        # Buttons: R2 starts the ascent, L2 the descent; either trigger cancels a running one
        self.prev_r2 = False
        self.prev_l2 = False
        self.climb_mode = 'up'                    # 'up' (ascent) or 'down' (descent)
        # Descent, phase 1 (edge alignment): front bogies tip back so their trailing wheel
        # hangs a few cm up and spins slowly backward; when the leading wheel rolls off the
        # edge, that corner sags, the trailing wheel touches down and is forced to roll
        # forward: a sign flip on its encoder marks the edge for that leg
        self.desc_lift = 0.025                    # trailing wheel lift (m)
        self.desc_reverse_frac = 0.1              # trailing wheel reverse torque, fraction of the slip limit
        self.desc_detect_speed = 0.5              # (wheel-flip threshold, kept for reference)
        self.desc_arm_speed = 0.25                # trailing wheel counts as airborne once back-spinning faster than this (rad/s)...
        self.desc_arm_time = 0.2                  # ...steadily for this long (s)
        self.desc_kill_frac = 0.25                # edge = backspin collapses below this fraction of its airborne speed...
        self.desc_detect_time = 0.05              # ...for this long (s)
        self.desc_settle_time = 0.6               # ignore the trailing wheels this long after starting (s)
        self.desc_max_nose_down = np.deg2rad(35)  # stop if the body stays nose-down past this...
        self.desc_nose_down_time = 0.75           # ...for longer than this (s); brief excursions are fine
        self.desc_hard_nose_down = np.deg2rad(55) # stop immediately past this
        # Descent tumble (slide-down): pivot creeps forward, swing wheel limp so it can ride
        # over the corner and roll down the riser face; knee rotates forward as in the ascent
        self.desc_pivot_frac = 0.3                # pivot (trailing) wheel forward torque, fraction of slip
        self.desc_tumble_rate = 0.6               # descent bogie rotation speed (rad/s)
        self.desc_pivot_max_past = 0.02           # pivot wheel center may roll at most this far past the edge (m)
        self.desc_land_log_min = np.deg2rad(30)   # log knee-torque reversal (landing) only past this rotation
        self.desc_follow_frac = 0.15              # wheels following the body during a tumble (fraction of slip)
        self.desc_timeout = 25.0                  # edge alignment may creep a while (s)
        self.wheel_each = None                    # per-wheel commands {motor index: ('torque', Nm) | ('vel', rad/s)}
        self.climb_bogie_offset = {}              # per-leg bogie tilt added to "flat" (rad)

        # Smart Gen-0 parameters: hips hold the bodies level at a target ride height.
        # Pure feedforward geometry + PD trim on tilt (no integral terms).
        self.level_min_splay = np.deg2rad(5)      # hip splay limits (rad)
        self.level_max_splay = np.deg2rad(85)
        self.level_kp = 1.0                       # PD trim: tilt (rad) -> corner height via lever arm
        self.level_kd = 0.05                      # PD trim on tilt rate (s)
        self.level_filter_tau = 0.05              # low-pass time constant on measured tilt (s)
        self.level_max_hip_rate = 2.0             # limit on how fast hip targets move (rad/s)
        self.level_compensate_knees = True        # keep bogies flat as the hips swing
        self.default_ride_splay = np.deg2rad(30)  # sets the default ride height
        self.height_rate = 0.05                   # D-pad up/down changes target height (m/s)
        self.pitch_target = 0.0                   # desired body pitch, + = nose UP (rad); L1 up, R1 down
        self.pitch_target_rate = np.deg2rad(10)   # how fast L1/R1 change it (rad/s)
        self.pitch_target_limit = np.deg2rad(30)  # max |desired pitch| (rad)
        self.knee_offsets = np.zeros(4)           # added to knee targets [fr, fl, br, bl]
        self.tilt = np.zeros(3)                   # latest [pitch, torso roll, head roll], rad
        self.height = 0.0                         # latest measured ride height (m)
        self.height_target = None                 # set on first use / when entering Smart mode
        self.level_geom = None                    # robot geometry, filled in on first use
        self.reset_leveling()


    # Function that controls the wheels
    def control_wheels(self):
        # Control the wheels with basic joystick control
        left_cmd = self.max_wheel_vel*(-self.left_stick_ud + self.left_stick_lr)
        right_cmd = self.max_wheel_vel*(-self.left_stick_ud - self.left_stick_lr)
        self.left_wheel_vel_des = self.ramp_up_only(self.left_wheel_vel_des, left_cmd)
        self.right_wheel_vel_des = self.ramp_up_only(self.right_wheel_vel_des, right_cmd)

    # Speed-up is rate-limited ("windup"); slowing down or stopping is immediate.
    # A direction reversal drops to zero instantly, then ramps up the other way.
    def ramp_up_only(self, current, command):
        if current * command < 0:
            current = 0.0
        if abs(command) <= abs(current):
            return command
        step = self.wheel_accel * self.dt
        return current + np.sign(command) * min(step, abs(command) - abs(current))

    def control_knees(self):
        self.left_knee_des_vel = self.max_knee_vel*(-self.right_stick_ud + self.right_stick_lr)
        self.right_knee_des_vel = self.max_knee_vel*(-self.right_stick_ud - self.right_stick_lr)
        # Anti-windup: while the stick is driving a side, keep each knee target
        # within max_knee_lead of where that knee actually is, so the bogies stop
        # promptly when the stick is released. (Snap buttons aren't affected.)
        # In Smart mode with knee compensation, targets are bogie angles (knee + hip).
        smart = self.mode == "Smart Gen-0" and self.level_compensate_knees
        def step(leg, vel):
            if vel == 0:
                return
            actual = getattr(self, f'{leg}_knee_pos')
            if smart:
                actual += getattr(self, f'{leg}_hip_pos')
            target = getattr(self, f'{leg}_knee_des_pos') + vel
            setattr(self, f'{leg}_knee_des_pos',
                    float(np.clip(target, actual - self.max_knee_lead, actual + self.max_knee_lead)))
        step('fl', self.left_knee_des_vel)
        step('bl', self.left_knee_des_vel)
        step('fr', self.right_knee_des_vel)
        step('br', self.right_knee_des_vel)
        


    # Function that handles button inputs
    def button_controls(self):
        if(self.a_button):
            self.fr_hip_des_pos = 0
            self.fl_hip_des_pos = 0
            self.br_hip_des_pos = 0
            self.bl_hip_des_pos = 0
            self.fr_knee_des_pos = self.nearest_pi(self.fr_knee_pos)
            self.fl_knee_des_pos = self.nearest_pi(self.fl_knee_pos)
            self.br_knee_des_pos = self.nearest_pi(self.br_knee_pos)
            self.bl_knee_des_pos = self.nearest_pi(self.bl_knee_pos)
        elif(self.b_button):
            self.fr_hip_des_pos = -np.pi/2
            self.fl_hip_des_pos = -np.pi/2
            self.br_hip_des_pos = np.pi/2
            self.bl_hip_des_pos = np.pi/2
            self.fr_knee_des_pos = self.nearest_pi(self.fr_knee_pos)
            self.fl_knee_des_pos = self.nearest_pi(self.fl_knee_pos)
            self.br_knee_des_pos = self.nearest_pi(self.br_knee_pos)
            self.bl_knee_des_pos = self.nearest_pi(self.bl_knee_pos)
        elif(self.y_button):
            self.fr_hip_des_pos = -np.pi/3
            self.fl_hip_des_pos = -np.pi/3
            self.br_hip_des_pos = np.pi/3
            self.bl_hip_des_pos = np.pi/3
        elif(self.x_button):
            self.fr_hip_des_pos = -np.pi/12
            self.fl_hip_des_pos = -np.pi/12
            self.br_hip_des_pos = np.pi/12
            self.bl_hip_des_pos = np.pi/12
            self.fr_knee_des_pos = self.nearest_pi(self.fr_knee_pos) - np.pi/4
            self.fl_knee_des_pos = self.nearest_pi(self.fl_knee_pos) - np.pi/4
            self.br_knee_des_pos = self.nearest_pi(self.br_knee_pos) + np.pi/4
            self.bl_knee_des_pos = self.nearest_pi(self.bl_knee_pos) + np.pi/4


    def update_hip_splay(self):
        self.hip_splay = 0.001*self.d_up
        self.front_splay = -0.001*self.left_bumper
        self.fr_hip_des_pos -= self.hip_splay + self.front_splay
        self.fl_hip_des_pos -= self.hip_splay + self.front_splay
        self.br_hip_des_pos += self.hip_splay
        self.bl_hip_des_pos += self.hip_splay


    # Current controller mode name
    @property
    def mode(self):
        return self.modes[self.mode_index]

    # Toggle mode on the rising edge of the Share button (press, not hold)
    def update_mode(self):
        if self.aux_button and not self.prev_aux_button:
            # Keep the physical knee targets continuous across the switch
            hips = [self.fr_hip_des_pos, self.fl_hip_des_pos, self.br_hip_des_pos, self.bl_hip_des_pos]
            knees = ['fr_knee_des_pos', 'fl_knee_des_pos', 'br_knee_des_pos', 'bl_knee_des_pos']
            for i, k in enumerate(knees):
                # Fold any active offset back in (raw knee angle) ...
                setattr(self, k, getattr(self, k) + self.knee_offsets[i])
            self.knee_offsets[:] = 0
            self.mode_index = (self.mode_index + 1) % len(self.modes)
            if self.mode == "Smart Gen-0" and self.level_compensate_knees:
                # ... then express it as a bogie angle (knee + hip) for Smart mode
                for i, k in enumerate(knees):
                    setattr(self, k, getattr(self, k) + hips[i])
            print(f"Controller mode: {self.mode}")
            self.reset_leveling()
            if self.mode == "Smart Gen-0" and self.level_geom is not None:
                self.height_target = self.clamp_height(self.height)
        self.prev_aux_button = self.aux_button

    # Reset the simulation if requested (shared by all modes)
    def handle_reset(self, m, d):
        if(self.start_button):
            mujoco.mj_resetData(m, d)
            if self.climb_phase is not None:
                print("Climb: cancelled (reset)")
            self.climb_phase = None
            self.wheel_leg = None
            self.wheel_leg_torque = None
            self.wheel_each = None
            self.left_wheel_vel_des = 0
            self.right_wheel_vel_des = 0
            self.reset_leveling()
            self.height_target = None
            self.fr_knee_des_pos = 0
            self.fl_knee_des_pos = 0
            self.br_knee_des_pos = 0
            self.bl_knee_des_pos = 0
            self.fr_hip_des_pos = 0
            self.fl_hip_des_pos = 0
            self.br_hip_des_pos = 0
            self.bl_hip_des_pos = 0

    # Main control function
    def control(self,m,d):
        # Get current joint angles
        self.get_mujoco_state(m,d)

        # Get joystick state
        self.get_joystick_state()

        # Check for a mode toggle
        self.update_mode()

        # Measure body tilt in every mode (shown on screen for comparison)
        self.measure_state(m, d)

        # Stair climbing takes over all joints while active
        self.update_climb_trigger(m, d)
        if self.climb_phase is not None:
            self.control_climb(m, d)
            return m,d

        # Run the active controller
        if self.mode == "Basic":
            self.control_basic(m, d)
        elif self.mode == "Smart Gen-0":
            self.control_smart_gen0(m, d)

        return m,d

    # "Basic" mode: original joystick controller
    def control_basic(self, m, d):
        self.knee_offsets[:] = 0
        self.handle_reset(m, d)
        self.update_hip_splay()
        self.control_wheels()
        self.control_knees()
        self.button_controls()
        self.send_commands()

    # "Smart Gen-0" mode: wheels and knees as in Basic; hips automatically
    # level the torso and head in pitch and roll
    def control_smart_gen0(self, m, d):
        self.handle_reset(m, d)
        self.control_wheels()
        self.control_knees()
        if self.level_compensate_knees:
            # Knee targets now mean "bogie angle relative to the body", so the
            # snap buttons (X, O) should snap the bogie flat, not the raw knee
            self.fr_knee_pos += self.fr_hip_pos
            self.fl_knee_pos += self.fl_hip_pos
            self.br_knee_pos += self.br_hip_pos
            self.bl_knee_pos += self.bl_hip_pos
        self.button_controls()   # knee snaps still apply; hip presets get overridden below
        self.level_hips(m, d)
        if self.level_compensate_knees:
            # Subtract each hip angle so the bogie stays at its commanded angle
            self.knee_offsets[:] = -np.array([self.fr_hip_des_pos, self.fl_hip_des_pos,
                                              self.br_hip_des_pos, self.bl_hip_des_pos])
        else:
            self.knee_offsets[:] = 0
        self.send_commands()

    # ---------------- Body leveling + ride height (Smart Gen-0) ----------------
    def reset_leveling(self):
        self.prev_tilt = None
        self.tilt_filt = None
        self.pitch_target = 0.0

    # Pitch (+ = nose down) and roll (+ = left side up) of a body, independent of yaw
    def body_tilt(self, d, body_name):
        R = d.body(body_name).xmat.reshape(3, 3)
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        roll = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
        return pitch, roll

    # Look up joint/body ids and dimensions once from the model
    def init_level_geometry(self, m, d):
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        R_t = d.body('torso').xmat.reshape(3, 3)
        p_t = d.body('torso').xpos
        legs = {}
        for key, body, side in [('fr', 'head', 'right'), ('fl', 'head', 'left'),
                                ('br', 'torso', 'right'), ('bl', 'torso', 'left')]:
            hip = jid(f'{body}_{side}_thigh_joint')
            legs[key] = {
                'body': body,
                'hip': hip,
                'knee': jid(f'{body}_{side}_thigh_shin_joint'),
                'wheels': [bid(f'{body}_{side}_front_wheel'), bid(f'{body}_{side}_rear_wheel')],
                'p': R_t.T @ (d.xanchor[hip] - p_t),   # hip position in torso frame
            }
        x_mid = np.mean([leg['p'][0] for leg in legs.values()])
        for leg in legs.values():
            leg['x'] = leg['p'][0] - x_mid                # lever arm for pitch
            leg['y'] = leg['p'][1]                        # lever arm for roll
        L = abs(m.body_pos[m.jnt_bodyid[legs['bl']['knee']]][2])   # thigh length
        wheel_geom = m.body_geomadr[legs['bl']['wheels'][0]]
        r = m.geom_size[wheel_geom][0]                              # wheel radius
        self.level_geom = {'legs': legs, 'L': L, 'r': r,
                           'e_min': L * np.cos(self.level_max_splay),
                           'e_max': L * np.cos(self.level_min_splay)}
        print(f"Leveling geometry: thigh {L:.3f} m, wheel radius {r:.3f} m, "
              f"pitch arms {[round(leg['x'], 3) for leg in legs.values()]}")

    def default_height(self):
        g = self.level_geom
        return g['r'] + g['L'] * np.cos(self.default_ride_splay)

    # Ride height = hip height above the ground under the wheels (bogies flat)
    def clamp_height(self, h):
        g = self.level_geom
        return float(np.clip(h, g['r'] + g['e_min'], g['r'] + g['e_max']))

    # Measure tilt, ground under each leg, and ride height (every mode)
    def measure_state(self, m, d):
        if self.level_geom is None:
            self.init_level_geometry(m, d)
        g = self.level_geom
        pitch, roll_t = self.body_tilt(d, 'torso')
        _, roll_h = self.body_tilt(d, 'head')
        self.tilt = np.array([pitch, roll_t, roll_h])
        for leg in g['legs'].values():
            # Ground under a leg = its lower wheel center minus the wheel radius
            leg['ground'] = min(d.xpos[w][2] for w in leg['wheels']) - g['r']
            leg['knee_z'] = d.xanchor[leg['knee']][2]
            leg['hip_z'] = d.xanchor[leg['hip']][2]
        legs = g['legs'].values()
        self.height = np.mean([l['hip_z'] for l in legs]) - np.mean([l['ground'] for l in legs])

    def level_hips(self, m, d):
        g = self.level_geom
        legs = g['legs']
        if self.height_target is None:
            self.height_target = self.default_height()

        # D-pad up/down adjusts the target ride height
        self.height_target = self.clamp_height(
            self.height_target + self.height_rate * self.dt * (self.d_up - self.d_down))

        # L1 pitches the nose up, R1 pitches it down; both together reset to flat
        if self.left_bumper and self.right_bumper:
            self.pitch_target = 0.0
        else:
            self.pitch_target = float(np.clip(
                self.pitch_target + self.pitch_target_rate * self.dt * (self.left_bumper - self.right_bumper),
                -self.pitch_target_limit, self.pitch_target_limit))

        # Low-pass filter the tilt; its rate feeds the D term
        if self.tilt_filt is None:
            self.tilt_filt = self.tilt.copy()
        alpha = self.dt / (self.level_filter_tau + self.dt)
        self.tilt_filt = self.tilt_filt + alpha * (self.tilt - self.tilt_filt)
        rate = np.zeros(3) if self.prev_tilt is None else (self.tilt_filt - self.prev_tilt) / self.dt
        self.prev_tilt = self.tilt_filt.copy()

        # 1) Feedforward geometry: put the hips on a plane through height H (at the
        #    center) at the desired pitch, with H at the target height above the
        #    average ground under the wheels. Front hips sit x*sin(pitch) higher.
        #    Leg extension needed: e = H + offset - knee height.
        ground = np.mean([leg['ground'] for leg in legs.values()])
        offset = {k: leg['x'] * np.sin(self.pitch_target) for k, leg in legs.items()}
        knee_eff = np.array([leg['knee_z'] - offset[k] for k, leg in legs.items()])
        H = ground + self.height_target
        # Attitude first: shift H (height drifts) so every leg stays within its range
        H_lo = np.max(knee_eff) + g['e_min']
        H_hi = np.min(knee_eff) + g['e_max']
        H = np.clip(H, H_lo, H_hi) if H_lo <= H_hi else 0.5 * (H_lo + H_hi)

        # 2) PD trim on the remaining attitude error. Measured pitch is + nose DOWN,
        #    the target is + nose UP, so the pitch error is tilt + target.
        err = self.tilt_filt + np.array([self.pitch_target, 0.0, 0.0])
        trim = self.level_kp * err + self.level_kd * rate
        des = {}
        for key, leg in legs.items():
            roll_trim = trim[1] if leg['body'] == 'torso' else trim[2]
            e = H + offset[key] - leg['knee_z'] + leg['x'] * trim[0] - leg['y'] * roll_trim
            e = np.clip(e, g['e_min'], g['e_max'])
            splay = np.arccos(e / g['L'])
            # Front (head) hips splay negative, rear (torso) hips positive, as in the presets
            des[key] = -splay if leg['body'] == 'head' else splay

        # Rate-limit hip targets so they ramp instead of jumping
        step = self.level_max_hip_rate * self.dt
        def ramp(cur, target):
            return cur + np.clip(target - cur, -step, step)
        self.fr_hip_des_pos = ramp(self.fr_hip_des_pos, des['fr'])
        self.fl_hip_des_pos = ramp(self.fl_hip_des_pos, des['fl'])
        self.br_hip_des_pos = ramp(self.br_hip_des_pos, des['br'])
        self.bl_hip_des_pos = ramp(self.bl_hip_des_pos, des['bl'])


    # ---------------- Smart stair climbing (touchpad) ----------------
    # APPROACH: creep forward until the front wheels bump the first riser.
    # Front-only loop: TUMBLE front bogies up onto the tread, ROLL gently to the
    # next riser, repeat, mapping each riser. When the rear legs reach the first
    # riser, SYNC: splay the hips until front and rear wheels both press risers.
    # Synced loop: TUMBLE all four, ROLL all four, re-SYNC if needed.
    # Stops when no riser appears where expected (top), or on any fault.
    CL_LEGS = ('fr', 'fl', 'br', 'bl')
    CL_FRONT = ('fr', 'fl')
    CL_REAR = ('br', 'bl')
    CL_NAMES = {'fr': ('head', 'right'), 'fl': ('head', 'left'),
                'br': ('torso', 'right'), 'bl': ('torso', 'left')}

    # Mode plus climb phase, for the on-screen readout
    @property
    def status(self):
        if self.climb_phase is None:
            return self.mode
        if self.climb_mode == 'down':
            stage = "front done" if getattr(self, 'desc_front_done', False) else \
                    ("synced" if self.climb_synced else "front only")
            return f"{self.mode} | Descend: {self.climb_phase[2:]} ({stage}, edges {len(getattr(self, 'desc_edges', []))})"
        stage = "synced" if self.climb_synced else "front only"
        return f"{self.mode} | Climb: {self.climb_phase} ({stage}, step {len(self.climb_risers)})"

    def update_climb_trigger(self, m, d):
        r2 = getattr(self, 'right_trigger', -1.0) > 0.5
        l2 = getattr(self, 'left_trigger', -1.0) > 0.5
        up, down = r2 and not self.prev_r2, l2 and not self.prev_l2
        self.prev_r2, self.prev_l2 = r2, l2
        if not (up or down):
            return
        if self.climb_phase is not None:
            self.stop_climb("cancelled")
            return
        if d.body('torso').xmat.reshape(3, 3)[2, 2] < 0.7:
            print("Stairs: robot isn't upright; can't start")
            return
        self.start_climb(m, d, mode='up' if up else 'down')

    def start_climb(self, m, d, mode='up'):
        self.climb_mode = mode
        R = d.body('torso').xmat.reshape(3, 3)
        h = np.array([R[0, 0], R[1, 0], 0.0])
        self.climb_heading = h / np.linalg.norm(h)        # straight ahead, fixed for the climb
        self.climb_wheel_bodies = {leg: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{b}_{s}_{w}_wheel')
                                         for w in ('front', 'rear')]
                                   for leg, (b, s) in self.CL_NAMES.items()}
        self.climb_wheel_joints = {leg: [f'{b}_{s}_shin_{w}_wheel_joint' for w in ('front', 'rear')]
                                   for leg, (b, s) in self.CL_NAMES.items()}
        self.climb_r = m.geom_size[m.body_geomadr[self.climb_wheel_bodies['fr'][0]]][0]
        self.climb_wheelbase = 2 * abs(float(m.body_pos[self.climb_wheel_bodies['fr'][0]][0]))   # bogie wheelbase
        torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        mass = float(m.body_subtreemass[torso_id])                # the robot only, not the scenery
        self.climb_slip_torque = 1.0 * (mass * 9.81 / 8) * self.climb_r      # mu=1 x per-wheel load x radius
        self.climb_wheel_cap = self.climb_torque_frac * self.climb_slip_torque
        knee_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'torso_left_thigh_shin_joint')
        self.climb_L = abs(m.body_pos[m.jnt_bodyid[knee_j]][2])                  # thigh length
        leg_push = 2 * self.climb_wheel_cap / self.climb_r                       # one leg's wheels (N)
        self.climb_bump_thr = self.climb_bump_frac * leg_push * self.climb_L     # hip torque (Nm)
        print(f"Stairs: robot {mass:.0f} kg, wheel torque capped at {self.climb_wheel_cap:.1f} Nm, "
              f"bump = hip torque {self.climb_bump_thr:.1f} Nm below baseline")
        self.climb_risers = []          # riser positions along the heading (m), mapped by the front legs
        self.climb_treads = []          # tread heights (m), measured by the front legs
        self.climb_synced = False
        self.climb_splay = 0.0          # SYNC splay offset, shared by front and rear (rad)
        self.climb_dpitch = 0.0         # pitch limiter: front splays more, rear less, by this (rad)
        self.knee_offsets[:] = 0
        # Bogie command = hip + knee (relative to body); start from where they are
        self.climb_bogie_cmd = {leg: getattr(self, f'{leg}_knee_pos') + getattr(self, f'{leg}_hip_pos')
                                for leg in self.CL_LEGS}
        self.climb_bogie_offset = {}
        self.wheel_each = None
        if mode == 'down':
            self.desc_start(d)
            return
        print("Climb: started")
        self.enter_climb('SETUP')

    def enter_climb(self, phase, tumble_legs=(), moving=()):
        if self.climb_verbose:
            extra = f" legs={','.join(tumble_legs)}" if tumble_legs else ""
            print(f"Climb: -> {phase}{extra}")
        self.climb_phase = phase
        self.climb_t = 0.0
        self.climb_stall_t = {leg: 0.0 for leg in self.CL_LEGS}
        self.climb_bumped = {leg: False for leg in self.CL_LEGS}
        self.climb_tau_base = {leg: None for leg in self.CL_LEGS}
        self.climb_tau_filt = {leg: None for leg in self.CL_LEGS}
        self.climb_dev_win = {}
        self.climb_wait = 0.0
        self.climb_blocked_t = 0.0
        if phase == 'SYNC':
            self.climb_moving = tuple(moving)
        if phase == 'TUMBLE':
            self.climb_tumbling = tuple(tumble_legs)
            self.climb_beta0 = {leg: self.climb_beta(leg) for leg in tumble_legs}
            # Target: the next flat angle forward (one half-turn, rear wheel over the front)
            self.climb_k_end = {leg: np.round((self.climb_beta0[leg] + self.climb_phi) / np.pi) + 1
                                for leg in tumble_legs}
            self.climb_landed = {leg: False for leg in tumble_legs}
            self.climb_pinned_t = {leg: 0.0 for leg in tumble_legs}       # time not progressing
            self.climb_beta_prev = {leg: self.climb_beta0[leg] for leg in tumble_legs}
            self.climb_beta_rate = {leg: 0.0 for leg in tumble_legs}      # filtered bogie speed (rad/s)

    def stop_climb(self, reason):
        print(f"{'Descend' if self.climb_mode == 'down' else 'Climb'}: stopped ({reason})")
        self.climb_phase = None
        self.wheel_leg = None
        self.wheel_leg_torque = None
        self.wheel_each = None
        self.left_wheel_vel_des = 0
        self.right_wheel_vel_des = 0
        # Hand the joints back where they are; Smart mode's knee targets are bogie angles
        for leg in self.CL_LEGS:
            if self.mode == "Smart Gen-0" and self.level_compensate_knees:
                setattr(self, f'{leg}_knee_des_pos', self.climb_bogie_cmd[leg])
            else:
                setattr(self, f'{leg}_knee_des_pos', self.climb_bogie_cmd[leg] - getattr(self, f'{leg}_hip_des_pos'))
        self.reset_leveling()
        self.height_target = None

    # ---- measurements ----
    def climb_beta(self, leg):          # bogie angle relative to the body
        return getattr(self, f'{leg}_knee_pos') + getattr(self, f'{leg}_hip_pos')

    def climb_lead(self, d, leg):       # leading wheel: distance along heading (m) and height (m)
        best = None
        for b in self.climb_wheel_bodies[leg]:
            p = d.xpos[b]
            sd = float(np.dot(p, self.climb_heading))
            if best is None or sd > best[0]:
                best = (sd, float(p[2]))
        return best

    def climb_wheel_speed(self, d, leg):
        return float(np.mean([d.jnt(j).qvel[0] for j in self.climb_wheel_joints[leg]]))

    climb_max_splay_k = np.deg2rad(85)            # never splay a hip past this

    # Pitch target: the stair slope once measured, but never above the cap
    def climb_pitch_target(self):
        rise = self.climb_rise()
        if rise is None:
            return self.climb_pitch_cap
        return min(self.climb_pitch_cap, float(np.arctan2(rise, self.climb_run())))

    def climb_run(self):
        if len(self.climb_risers) >= 2:
            return float(np.mean(np.diff(self.climb_risers)))
        return self.climb_run_guess

    def climb_rise(self):
        if len(self.climb_treads) >= 2:
            return float(np.mean(np.diff(self.climb_treads)))
        return None

    # Nearest mapped riser ahead of a leg's leading wheel, and the gap to it (m)
    def climb_gap(self, d, leg, predicted=False):
        s_face = self.climb_lead(d, leg)[0] + self.climb_r
        ahead = [x for x in self.climb_risers if x >= s_face - self.climb_riser_tol]
        if ahead:
            return ahead[0] - s_face
        if predicted and self.climb_risers:
            return self.climb_risers[-1] + self.climb_run() - s_face
        return None

    def climb_record_riser(self, d):
        s_face = float(np.mean([self.climb_lead(d, leg)[0] for leg in self.CL_FRONT])) + self.climb_r
        if not self.climb_risers or s_face > self.climb_risers[-1] + 0.5 * self.climb_run():
            self.climb_risers.append(s_face)
            rise = self.climb_rise()
            print(f"Climb: riser {len(self.climb_risers)} at {s_face - self.climb_risers[0]:+.2f} m"
                  f" | run ~{self.climb_run():.2f} m" + (f", rise ~{rise:.2f} m" if rise else ""))

    # Hip motor output torque (Nm) from the last command; on hardware, motor current
    CL_HIP_MOTOR = {'fr': 0, 'fl': 1, 'br': 2, 'bl': 3}
    def climb_hip_torque(self, leg):
        mot = self.motors[self.CL_HIP_MOTOR[leg]]
        return float(mot.limited_torque * mot.gear_ratio)

    # Per-leg bump detection: a riser pushes its leg backward at about knee height, so
    # the hip motor's torque drops well below its running baseline. (A leg driving the
    # robot against something else gets pushed forward, which raises it instead.)
    CL_PARTNER = {'fr': 'br', 'br': 'fr', 'fl': 'bl', 'bl': 'fl'}   # same side, other end
    def climb_update_bumps(self):
        a_base = self.dt / (self.climb_baseline_tau + self.dt)
        a_filt = self.dt / (self.climb_torque_filter + self.dt)
        dev = {}
        for leg in self.CL_LEGS:
            raw = self.climb_hip_torque(leg)
            if self.climb_tau_filt[leg] is None:
                self.climb_tau_filt[leg] = raw
                self.climb_tau_base[leg] = raw
            tau = self.climb_tau_filt[leg] = self.climb_tau_filt[leg] + a_filt * (raw - self.climb_tau_filt[leg])
            dev[leg] = tau - self.climb_tau_base[leg]
            lo, hi = self.climb_dev_win.get(leg, (dev[leg], dev[leg]))
            self.climb_dev_win[leg] = (min(lo, dev[leg]), max(hi, dev[leg]))
        for leg in self.CL_LEGS:
            # Against a riser: this leg is pushed back (torque well below baseline) while the
            # other end of the same side is pushed forward (it's driving into the block)
            hit = (dev[leg] < -self.climb_bump_thr and dev[self.CL_PARTNER[leg]] > 0.25 * self.climb_bump_thr)
            if self.climb_t > self.climb_settle_time and hit:
                self.climb_stall_t[leg] += self.dt
            else:
                self.climb_stall_t[leg] = 0.0
            # Track slow changes, but freeze whenever the torque is clearly loaded either way
            if abs(dev[leg]) < 0.5 * self.climb_bump_thr:
                self.climb_tau_base[leg] += a_base * (self.climb_tau_filt[leg] - self.climb_tau_base[leg])
            if self.climb_stall_t[leg] >= self.climb_stall_time:
                self.climb_bumped[leg] = True

    # ---------------- Stair descent: phase 1, edge alignment (L2) ----------------
    CL_WHEEL_MOTORS = {'fr': (8, 9), 'fl': (10, 11), 'br': (12, 13), 'bl': (14, 15)}   # (front-named, rear-named)
    CL_SIDE = {'fr': 'br', 'fl': 'bl', 'br': 'fr', 'bl': 'fl'}

    def desc_start(self, d):
        tilt = float(np.arcsin(min(1.0, self.desc_lift / self.climb_wheelbase)))
        self.desc_tilt = tilt
        self.desc_edges = []            # edge positions along the heading (m), mapped by the front legs
        self.desc_treads = []           # tread heights (m), measured by the front legs
        self.desc_front_done = False    # front legs have reached the bottom
        self.desc_flag = {leg: False for leg in self.CL_LEGS}
        self.desc_flag_t = {leg: 0.0 for leg in self.CL_LEGS}
        self.desc_flag_s = {}
        self.climb_synced = False
        self.desc_assign_roles(d)
        self.desc_treads.append(float(np.mean([self.desc_lead_z(d, leg) for leg in self.CL_FRONT])))
        print(f"Descend: started (bogies tip back {np.degrees(tilt):.0f} deg to lift their trailing wheels"
              f" {100*self.desc_lift:.1f} cm)")
        self.enter_climb('D_SETUP')

    # Which wheel of each bogie is physically leading (bogies flip with every tumble)
    def desc_assign_roles(self, d):
        self.desc_lead, self.desc_trail = {}, {}
        for leg in self.CL_LEGS:
            s_pos = [float(np.dot(d.xpos[b], self.climb_heading)) for b in self.climb_wheel_bodies[leg]]
            lead = 0 if s_pos[0] >= s_pos[1] else 1
            idx, jnt, bod = self.CL_WHEEL_MOTORS[leg], self.climb_wheel_joints[leg], self.climb_wheel_bodies[leg]
            self.desc_lead[leg] = (idx[lead], jnt[lead], bod[lead])
            self.desc_trail[leg] = (idx[1 - lead], jnt[1 - lead], bod[1 - lead])

    def desc_lead_s(self, d, leg):
        return float(np.dot(d.xpos[self.desc_lead[leg][2]], self.climb_heading))

    def desc_lead_z(self, d, leg):
        return float(d.xpos[self.desc_lead[leg][2]][2]) - self.climb_r

    def desc_run(self):
        return float(np.mean(np.diff(self.desc_edges))) if len(self.desc_edges) >= 2 else self.climb_run_guess

    def desc_rise(self):
        return float(-np.mean(np.diff(self.desc_treads))) if len(self.desc_treads) >= 2 else None

    # The edge this leg is going over: front legs, the latest mapped edge; rear legs, the
    # next mapped edge ahead of their trailing wheel
    def desc_leg_edge(self, d, leg):
        if not self.desc_edges:
            return None
        if leg in self.CL_FRONT and not self.desc_front_done:
            return self.desc_edges[-1]
        trail_s = float(np.dot(d.xpos[self.desc_trail[leg][2]], self.climb_heading))
        ahead = [e for e in self.desc_edges if e >= trail_s - 0.1]
        return ahead[0] if ahead else None

    def desc_record_edge(self):
        s_edge = float(np.mean([self.desc_flag_s[leg] for leg in self.CL_FRONT]))
        if not self.desc_edges or s_edge > self.desc_edges[-1] + 0.5 * self.desc_run():
            self.desc_edges.append(s_edge)
            rise = self.desc_rise()
            print(f"Descend: edge {len(self.desc_edges)} at {s_edge - self.desc_edges[0]:+.2f} m"
                  f" | run ~{self.desc_run():.2f} m" + (f", rise ~{rise:.2f} m" if rise else ""))

    # Legs whose bogies are tipped back (trailing wheel airborne) in this phase
    def desc_tipped_legs(self):
        return self.CL_REAR if self.desc_front_done else self.CL_LEGS

    def desc_enter_edge(self, d):
        self.desc_assign_roles(d)
        self.climb_bogie_offset = {leg: self.desc_tilt for leg in self.desc_tipped_legs()}
        self.desc_flag = {leg: False for leg in self.CL_LEGS}
        self.desc_flag_t = {leg: 0.0 for leg in self.CL_LEGS}
        self.desc_flag_s = {}
        self.desc_arm = {}
        self.enter_climb('D_EDGE')

    def desc_enter_tumble(self, d, legs):
        self.climb_bogie_offset = {leg: self.desc_tilt for leg in self.desc_tipped_legs() if leg not in legs}
        self.enter_climb('TUMBLE', legs)          # sets beta0 / k_end / progress tracking
        self.climb_phase = 'D_TUMBLE'
        # Roles: the trailing wheel (on the upper tread) pivots; the leading one swings down
        self.desc_pivot = {leg: self.desc_trail[leg] for leg in legs}
        self.desc_swing = {leg: self.desc_lead[leg] for leg in legs}
        self.desc_landed = {leg: False for leg in legs}
        self.desc_land_t = {leg: 0.0 for leg in legs}
        self.desc_tumbling = tuple(legs)
        self.desc_knee_sign = {leg: 0 for leg in self.CL_LEGS}
        if self.climb_verbose:
            print(f"Descend: -> D_TUMBLE legs={','.join(legs)}")

    # Trailing-wheel event for the given legs. The trailing wheel "arms" once it spins
    # steadily in one direction: backward if airborne (small reverse torque), or forward
    # if lightly dragging on the tread. When the leading wheel goes over the edge, that
    # corner sags and the trailing wheel's spin collapses or reverses; that marks the edge.
    def desc_update_edges(self, d, legs):
        if not hasattr(self, 'desc_arm'):
            self.desc_arm = {}
        for leg in legs:
            if self.desc_flag[leg]:
                continue
            st = self.desc_arm.setdefault(leg, {'t': 0.0, 'v_ref': None, 'armed': False})
            v = d.jnt(self.desc_trail[leg][1]).qvel[0]
            if not st['armed']:
                if abs(v) > self.desc_arm_speed and (st['v_ref'] is None or np.sign(v) == np.sign(st['v_ref'])):
                    st['t'] += self.dt
                    st['v_ref'] = v if st['v_ref'] is None else st['v_ref'] + 0.05 * (v - st['v_ref'])
                    if st['t'] >= self.desc_arm_time:
                        st['armed'] = True
                else:
                    st['t'], st['v_ref'] = 0.0, None
                continue
            ref = st['v_ref']
            if np.sign(v) == np.sign(ref) and abs(v) > self.desc_arm_speed:
                st['v_ref'] += 0.01 * (v - ref)              # keep tracking the steady speed
            changed = (v / ref) < self.desc_kill_frac        # collapsed below 25% of it, or reversed
            self.desc_flag_t[leg] = self.desc_flag_t[leg] + self.dt if changed else 0.0
            if self.desc_flag_t[leg] >= self.desc_detect_time:
                self.desc_flag[leg] = True
                self.desc_flag_s[leg] = self.desc_lead_s(d, leg)
                if self.climb_verbose:
                    kind = "backspin" if ref < 0 else "forward drag"
                    print(f"Descend: {leg} found an edge ({kind} {ref:+.2f} -> {v:+.2f} rad/s)")

    def desc_enter_sync(self, d, moving):
        self.desc_moving = tuple(moving)
        for leg in moving:
            self.desc_flag[leg], self.desc_flag_t[leg] = False, 0.0
            getattr(self, 'desc_arm', {}).pop(leg, None)
        self.enter_climb('SYNC', moving=moving)
        self.climb_phase = 'D_SYNC'
        if self.climb_verbose:
            print(f"Descend: -> D_SYNC moving={','.join(moving)}")

    def desc_step(self, d):
        nose_down = self.climb_phi
        self.desc_steep_t = getattr(self, 'desc_steep_t', 0.0) + self.dt if nose_down > self.desc_max_nose_down else 0.0
        if nose_down > self.desc_hard_nose_down or self.desc_steep_t > self.desc_nose_down_time:
            self.stop_climb(f"nose-down {np.degrees(nose_down):.0f} deg"
                            + ("" if nose_down > self.desc_hard_nose_down else f" for over {self.desc_nose_down_time:.2f} s"))
            return
        drive = self.climb_roll_torque_frac * self.climb_slip_torque
        reverse = self.desc_reverse_frac * self.climb_slip_torque
        follow = self.desc_follow_frac * self.climb_slip_torque
        pivot = self.desc_pivot_frac * self.climb_slip_torque
        each = {}
        def tipped(leg, stopped=False):
            # Leading wheel drives (or holds); airborne trailing wheel spins slowly back
            if stopped:
                each[self.desc_lead[leg][0]] = each[self.desc_trail[leg][0]] = ('brake', 0.0)
            else:
                each[self.desc_lead[leg][0]] = ('torque', drive)
                each[self.desc_trail[leg][0]] = ('torque', -reverse)
        def flat_drive(leg, torque):
            for i in self.CL_WHEEL_MOTORS[leg]:
                each[i] = ('torque', torque)
        phase = self.climb_phase

        if phase == 'D_SETUP':
            for leg in self.CL_LEGS:
                flat_drive(leg, 0.0)
            setup = max(self.climb_base_splay / self.climb_hip_rate, self.desc_tilt / self.climb_tumble_rate) + 0.4
            if self.climb_t > setup:
                self.desc_enter_edge(d)

        elif phase == 'D_EDGE':
            watch = self.desc_tipped_legs()
            self.desc_update_edges(d, watch)
            front_found = all(self.desc_flag[leg] for leg in self.CL_FRONT)
            rear_found = all(self.desc_flag[leg] for leg in self.CL_REAR)
            for leg in self.CL_LEGS:
                if leg in watch:
                    side_stop = self.desc_flag[leg] or self.desc_flag[self.CL_SIDE[leg]]
                    tipped(leg, stopped=side_stop)
                else:
                    flat_drive(leg, drive)          # front legs on the floor after the bottom
            self.climb_wait = self.climb_wait + self.dt if (front_found or rear_found) else 0.0

            if self.desc_front_done:
                # Rear-only: tumble down each mapped edge until past the last one
                last = self.desc_edges[-1] if self.desc_edges else None
                rear_s = min(self.desc_lead_s(d, leg) for leg in self.CL_REAR)
                if last is not None and rear_s > last + self.climb_top_margin * self.desc_run() + self.climb_wheelbase:
                    self.stop_climb("descent complete: rear legs past the last edge")
                    return
                if rear_found:
                    self.desc_enter_tumble(d, self.CL_REAR)
            else:
                # Bottom: front crept well past where the next edge should be
                if self.desc_edges and not front_found:
                    expected = self.desc_edges[-1] + self.desc_run()
                    s_front = min(self.desc_lead_s(d, leg) for leg in self.CL_FRONT)
                    if s_front > expected + self.climb_top_margin * self.desc_run():
                        print("Descend: front reached the bottom; rear legs continue")
                        self.desc_front_done = True
                        self.desc_enter_edge(d)
                        self.desc_each = each
                        return
                if not self.climb_synced:
                    if rear_found:
                        print("Descend: rear legs reached the top edge; syncing leg spacing")
                        self.climb_synced = True
                        if front_found:
                            self.desc_record_edge()
                            self.desc_enter_tumble(d, self.CL_LEGS)
                        else:
                            self.desc_enter_sync(d, moving=self.CL_FRONT)
                    elif front_found:
                        self.desc_record_edge()
                        # A front tumble moves the body forward ~one wheelbase; if the rear is
                        # closer than that to the top edge, bring it to the edge first
                        room = (self.desc_edges[0] - max(self.desc_lead_s(d, leg) for leg in self.CL_REAR)
                                if self.desc_edges else None)
                        needed = self.climb_wheelbase + self.climb_advance_margin
                        if room is not None and room < needed:
                            print(f"Descend: rear only {room:.2f} m from the top edge (front tumble needs"
                                  f" {needed:.2f} m); bringing the rear up to the edge")
                            self.climb_synced = True
                            self.desc_enter_sync(d, moving=self.CL_REAR)
                        else:
                            self.desc_enter_tumble(d, self.CL_FRONT)
                elif front_found and rear_found:
                    self.desc_record_edge()
                    self.desc_enter_tumble(d, self.CL_LEGS)
                elif self.climb_wait > self.climb_pair_wait:
                    if front_found:
                        self.desc_record_edge()
                    self.desc_enter_sync(d, moving=self.CL_REAR if front_found else self.CL_FRONT)

        elif phase == 'D_SYNC':
            moving = self.desc_moving
            direction = +1.0 if moving == self.CL_FRONT else -1.0
            self.climb_splay += direction * self.climb_splay_rate * self.dt
            if abs(self.climb_splay) > self.climb_max_splay or \
                    abs(self.climb_base_splay + self.climb_splay) > self.climb_max_splay_k:
                self.stop_climb("can't match leg spacing to the stairs")
                return
            self.desc_update_edges(d, moving)
            for leg in self.CL_LEGS:
                tipped(leg, stopped=leg not in moving)
            if all(self.desc_flag[leg] for leg in moving):
                if moving == self.CL_FRONT:
                    self.desc_record_edge()
                self.desc_enter_tumble(d, self.CL_LEGS)

        elif phase == 'D_TUMBLE':
            tumbling = self.climb_tumbling
            # Rear reaches the top edge during a front-only tumble: rear legs join in
            if tumbling == self.CL_FRONT and not self.desc_front_done:
                self.desc_update_edges(d, self.CL_REAR)
                if all(self.desc_flag[leg] for leg in self.CL_REAR):
                    print("Descend: rear reached the top edge during the front tumble; rear legs joining in")
                    for leg in self.CL_REAR:
                        self.climb_beta0[leg] = self.climb_beta(leg)
                        self.climb_k_end[leg] = np.round((self.climb_beta0[leg] + self.climb_phi) / np.pi) + 1
                        self.climb_landed[leg] = False
                        self.climb_pinned_t[leg] = 0.0
                        self.climb_beta_prev[leg] = self.climb_beta0[leg]
                        self.climb_beta_rate[leg] = 0.0
                        self.desc_pivot[leg], self.desc_swing[leg] = self.desc_trail[leg], self.desc_lead[leg]
                        self.desc_landed[leg], self.desc_land_t[leg] = False, 0.0
                        self.desc_knee_sign[leg] = 0
                        self.climb_bogie_offset.pop(leg, None)
                    self.climb_tumbling = tumbling = self.CL_LEGS
                    self.climb_synced = True
            for leg in self.CL_LEGS:
                if leg in tumbling:
                    rotated = self.climb_beta(leg) - self.climb_beta0[leg]
                    # Landing (logged only): knee torque reverses as the lower wheel takes the load
                    knee = self.motors[{'fr': 4, 'fl': 5, 'br': 6, 'bl': 7}[leg]]
                    tq = knee.limited_torque * knee.gear_ratio
                    if self.desc_knee_sign[leg] == 0 and rotated > np.deg2rad(5) and abs(tq) > 1.0:
                        self.desc_knee_sign[leg] = np.sign(tq)
                    if (not self.desc_landed[leg] and rotated > self.desc_land_log_min
                            and self.desc_knee_sign[leg] != 0 and np.sign(tq) == -self.desc_knee_sign[leg] and abs(tq) > 1.0):
                        self.desc_landed[leg] = True
                        if self.climb_verbose:
                            print(f"Descend: {leg} knee torque reversed at {np.degrees(rotated):.0f} deg"
                                  f" (lower wheel likely took the load)")
                    # Pivot creeps forward only until it sits on the corner, then holds, so it
                    # can't roll off the edge before the lower wheel has landed
                    edge = self.desc_leg_edge(d, leg)
                    trail_s = float(np.dot(d.xpos[self.desc_trail[leg][2]], self.climb_heading))
                    on_corner = edge is not None and trail_s >= edge + self.desc_pivot_max_past
                    each[self.desc_trail[leg][0]] = ('vel', 0.0) if on_corner else ('torque', pivot)
                    each[self.desc_lead[leg][0]] = ('torque', 0.0)             # swing wheel limp
                elif leg in self.desc_tipped_legs():
                    each[self.desc_lead[leg][0]] = ('torque', follow)          # follow the body forward
                    each[self.desc_trail[leg][0]] = ('torque', -reverse)
                else:
                    flat_drive(leg, follow)
            # Tumble progress / stuck / completion (shared with the ascent's TUMBLE logic)
            done = True
            for leg in tumbling:
                beta = self.climb_beta(leg)
                end = self.climb_k_end[leg] * np.pi - self.climb_phi
                cmd = min(self.climb_bogie_cmd[leg] + self.desc_tumble_rate * self.dt, end)
                self.climb_bogie_cmd[leg] = min(cmd, beta + self.climb_knee_lead)
                a = self.dt / (0.05 + self.dt)
                rate = (beta - self.climb_beta_prev[leg]) / self.dt
                self.climb_beta_prev[leg] = beta
                self.climb_beta_rate[leg] += a * (rate - self.climb_beta_rate[leg])
                stopped = self.climb_beta_rate[leg] < self.climb_stuck_speed and self.climb_t > self.climb_settle_time
                self.climb_pinned_t[leg] = self.climb_pinned_t[leg] + self.dt if stopped else 0.0
                rotated = beta - self.climb_beta0[leg]
                if self.climb_pinned_t[leg] > self.climb_stuck_time and rotated < self.climb_land_min:
                    self.stop_climb(f"{leg} bogie stopped at {np.degrees(rotated):.0f} deg; can't tumble down")
                    return
                settled = rotated > self.climb_land_min and self.climb_pinned_t[leg] > 0.5
                if abs(end - beta) > 0.1 and not settled:
                    done = False
            self.desc_tumbling = tumbling
            if done:
                if self.CL_FRONT[0] in tumbling and not self.desc_front_done:
                    self.desc_treads.append(float(np.mean([self.desc_lead_z(d, leg) for leg in self.CL_FRONT])))
                self.desc_enter_edge(d)

        self.desc_each = each

    def desc_armed_label(self, leg):
        st = getattr(self, 'desc_arm', {}).get(leg)
        if not st or not st['armed']:
            return '-'
        return 'back' if st['v_ref'] < 0 else 'fwd'

    def desc_log(self, d):
        print(f"  [{self.climb_phase} t={self.climb_t:.1f}s] pitch(nose-up) {np.degrees(-self.climb_phi):+.1f} deg")
        for leg in self.CL_LEGS:
            vl = d.jnt(self.desc_lead[leg][1]).qvel[0]
            vt = d.jnt(self.desc_trail[leg][1]).qvel[0]
            knee = self.motors[{'fr': 4, 'fl': 5, 'br': 6, 'bl': 7}[leg]]
            print(f"    {leg}: lead wheel {vl:+5.2f} rad/s, trail wheel {vt:+5.2f} rad/s"
                  f" | knee torque {knee.limited_torque * knee.gear_ratio:+6.1f} Nm"
                  + f" | armed={self.desc_armed_label(leg)}"
                  + f" | edge={int(self.desc_flag.get(leg, False))}"
                  + (f" | landed={int(self.desc_landed[leg])}" if self.climb_phase == 'D_TUMBLE' and leg in getattr(self, 'desc_landed', {}) else ""))

    # One-line-per-leg status snapshot for debugging
    def climb_log(self, d):
        if self.climb_mode == 'down':
            self.desc_log(d)
            return
        cmd = self.wheel_leg or {}
        print(f"  [{self.climb_phase} t={self.climb_t:.1f}s] pitch(nose-up) {np.degrees(-self.climb_phi):+.0f} deg, "
              f"splay {np.degrees(self.climb_base_splay + self.climb_splay):+.0f} deg"
              f" (pitch limiter {np.degrees(self.climb_dpitch):+.0f}, target {np.degrees(self.climb_pitch_target()):.0f}),"
              f" risers {len(self.climb_risers)}")
        for leg in self.CL_LEGS:
            s_lead, z_lead = self.climb_lead(d, leg)
            gap = self.climb_gap(d, leg, predicted=(leg[0] == 'f'))
            print(f"    {leg}: wheel {self.climb_wheel_speed(d, leg):+5.1f}/{cmd.get(leg, 0):+4.1f} rad/s"
                  f" bumped={int(self.climb_bumped[leg])}"
                  f" bogie {np.degrees(self.climb_beta(leg)):+5.0f}->{np.degrees(self.climb_bogie_cmd[leg]):+5.0f} deg"
                  f" hip {np.degrees(getattr(self, f'{leg}_hip_pos')):+4.0f}->{np.degrees(getattr(self, f'{leg}_hip_des_pos')):+4.0f}"
                  f" lead s={s_lead:.2f} z={z_lead:.2f}" + (f" gap={gap:+.3f}" if gap is not None else "")
                  + (f" hipT {self.climb_tau_filt[leg]:+6.1f} (base {self.climb_tau_base[leg]:+6.1f})"
                     f" dev[{self.climb_dev_win.get(leg, (0, 0))[0]:+6.1f}..{self.climb_dev_win.get(leg, (0, 0))[1]:+6.1f}]"
                     f" bump<-{self.climb_bump_thr:.1f}"
                     if self.climb_tau_base.get(leg) is not None else ""))
        self.climb_dev_win = {}

    # ---- main loop ----
    def control_climb(self, m, d):
        self.handle_reset(m, d)
        if self.climb_phase is None:
            return
        self.climb_t += self.dt
        self.climb_phi = self.tilt[0]                 # body pitch, + = nose down
        if abs(self.tilt[1]) > self.climb_max_roll:
            self.stop_climb("body rolled too far")
            return
        if self.climb_t > (self.desc_timeout if self.climb_phase.startswith('D_') else self.climb_timeout):
            self.stop_climb(f"{self.climb_phase} timed out")
            return

        if self.climb_verbose:
            self.climb_log_t = getattr(self, 'climb_log_t', 0.0) + self.dt
            if self.climb_log_t >= 0.5:
                self.climb_log_t = 0.0
                self.climb_log(d)

        creep, hold = self.climb_creep_speed, self.climb_hold_speed
        self.climb_cap_now = self.climb_wheel_cap
        wheels = {leg: creep for leg in self.CL_LEGS}
        tumbling = ()
        phase = self.climb_phase

        roll_torque = None
        if phase in ('APPROACH', 'ROLL'):
            # Top reached: front rolled well past where the next riser should be
            if self.climb_risers:
                expected = self.climb_risers[-1] + self.climb_run()
                s_front = np.mean([self.climb_lead(d, leg)[0] for leg in self.CL_FRONT]) + self.climb_r
                if s_front > expected + self.climb_top_margin * self.climb_run():
                    self.stop_climb(f"no riser where expected: top reached after {len(self.climb_risers)} steps")
                    return

        if phase in ('D_SETUP', 'D_EDGE', 'D_TUMBLE', 'D_SYNC'):
            self.desc_step(d)
            if self.climb_phase is None:
                return
        elif phase == 'SETUP':
            # Wheels limp while the hips splay out (if already against the riser, the
            # robot just eases back instead of fighting it)
            roll_torque = 0.0
            if self.climb_t > self.climb_base_splay / self.climb_hip_rate + 0.3:
                self.enter_climb('APPROACH')

        elif phase in ('APPROACH', 'ROLL'):
            # Fixed torque: rolls on a tread, stalls (without slipping) against a riser
            roll_torque = self.climb_roll_torque_frac * self.climb_slip_torque
            # Blocked if EITHER end fully stops: the ends are rigidly coupled, so one end
            # stalled means the robot is stuck even if lightly loaded wheels still creep/slip
            def stalled(legs):
                return max(abs(self.climb_wheel_speed(d, leg)) for leg in legs) < self.climb_roll_stall_speed
            blocked = (stalled(self.CL_FRONT) or stalled(self.CL_REAR)) and self.climb_t > self.climb_settle_time
            self.climb_blocked_t = self.climb_blocked_t + self.dt if blocked else 0.0
            if self.climb_blocked_t >= self.climb_roll_stall_time and phase == 'APPROACH':
                # Nothing is behind the robot yet, so the front found the first riser
                self.climb_record_riser(d)
                self.enter_climb('TUMBLE', self.CL_FRONT)
            elif self.climb_blocked_t >= self.climb_roll_stall_time:
                # Robot is blocked; the riser map says which end hit
                rear_gaps = [self.climb_gap(d, leg) for leg in self.CL_REAR]
                rear_at = all(g is not None and abs(g) < self.climb_riser_tol for g in rear_gaps)
                front_gaps = [self.climb_gap(d, leg, predicted=True) for leg in self.CL_FRONT]
                front_at = all(g is not None and abs(g) < self.climb_riser_tol for g in front_gaps)
                if self.climb_verbose:
                    print(f"Climb: blocked | rear gaps {[round(g, 3) if g is not None else None for g in rear_gaps]}"
                          f" front gaps {[round(g, 3) if g is not None else None for g in front_gaps]}")
                if not self.climb_synced:
                    if rear_at:
                        print("Climb: rear legs reached the first riser; syncing leg spacing")
                        self.climb_synced = True
                        self.enter_climb('SYNC', moving=self.CL_FRONT)
                    else:
                        self.climb_record_riser(d)                 # front legs found the next riser
                        # A front tumble moves the knees forward ~one wheelbase, so the rear
                        # must be able to roll that far; if not, bring the rear in and climb together
                        room = [g for g in rear_gaps if g is not None]
                        needed = self.climb_wheelbase + self.climb_advance_margin
                        if room and min(room) < needed:
                            print(f"Climb: rear only {min(room):.2f} m from its riser (front tumble needs"
                                  f" {needed:.2f} m); bringing the rear in to climb together")
                            self.climb_synced = True
                            self.enter_climb('SYNC', moving=self.CL_REAR)
                        else:
                            self.enter_climb('TUMBLE', self.CL_FRONT)
                elif rear_at and front_at:
                    self.climb_record_riser(d)
                    self.enter_climb('TUMBLE', self.CL_LEGS)
                elif rear_at:
                    self.enter_climb('SYNC', moving=self.CL_FRONT)  # front still short of its riser
                else:
                    self.climb_record_riser(d)                     # front hit; rear short of its riser
                    self.enter_climb('SYNC', moving=self.CL_REAR)

        elif phase == 'TUMBLE':
            if self.climb_tumbling == self.CL_FRONT:
                rear_gaps = [self.climb_gap(d, leg) for leg in self.CL_REAR]
                if all(g is not None and g < self.climb_riser_tol for g in rear_gaps):
                    # Rear hit its riser mid-tumble: the body can't advance, so the rear
                    # legs tumble too. Front legs keep their original target angle.
                    print("Climb: rear reached its riser during the front tumble; rear legs joining in")
                    for leg in self.CL_REAR:
                        self.climb_beta0[leg] = self.climb_beta(leg)
                        self.climb_k_end[leg] = np.round((self.climb_beta0[leg] + self.climb_phi) / np.pi) + 1
                        self.climb_landed[leg] = False
                        self.climb_pinned_t[leg] = 0.0
                        self.climb_beta_prev[leg] = self.climb_beta0[leg]
                        self.climb_beta_rate[leg] = 0.0
                    self.climb_tumbling = self.CL_LEGS
                    self.climb_synced = True
            tumbling = self.climb_tumbling
            wheels = {leg: hold for leg in self.CL_LEGS}  # keep the pivot wheels pressed into the risers
            done = True
            for leg in tumbling:
                beta = self.climb_beta(leg)
                end = self.climb_k_end[leg] * np.pi - self.climb_phi
                cmd = min(self.climb_bogie_cmd[leg] + self.climb_tumble_rate * self.dt, end)
                cmd = min(cmd, beta + self.climb_knee_lead)  # anti-windup
                self.climb_bogie_cmd[leg] = cmd
                rotated = beta - self.climb_beta0[leg]
                # Judge by progress, not lag: a loaded knee may trail its command and still climb
                a = self.dt / (0.05 + self.dt)
                rate = (beta - self.climb_beta_prev[leg]) / self.dt
                self.climb_beta_prev[leg] = beta
                self.climb_beta_rate[leg] += a * (rate - self.climb_beta_rate[leg])
                stopped = self.climb_beta_rate[leg] < self.climb_stuck_speed and self.climb_t > self.climb_settle_time
                self.climb_pinned_t[leg] = self.climb_pinned_t[leg] + self.dt if stopped else 0.0
                if stopped and rotated > self.climb_land_min and self.climb_pinned_t[leg] > self.climb_land_time \
                        and not self.climb_landed[leg]:
                    self.climb_landed[leg] = True       # swinging wheel came down on the tread
                    if self.climb_verbose:
                        print(f"Climb: {leg} landed on the tread at {np.degrees(rotated):.0f} deg of tumble")
                if rotated < self.climb_land_min and self.climb_pinned_t[leg] > self.climb_stuck_time:
                    self.stop_climb(f"{leg} bogie stopped at {np.degrees(rotated):.0f} deg; can't tumble up the step")
                    return
                # Done when flat on the tread, or landed and the knee can't push further
                # (ROLL keeps pulling the bogie toward flat as the robot moves on)
                settled = self.climb_landed[leg] and self.climb_pinned_t[leg] > 0.5
                if abs(end - beta) > 0.1 and not settled:
                    done = False
            if done:
                if self.CL_FRONT[0] in tumbling:
                    z = np.mean([self.climb_lead(d, leg)[1] for leg in self.CL_FRONT]) - self.climb_r
                    self.climb_treads.append(float(z))
                self.enter_climb('ROLL')

        elif phase == 'SYNC':
            # The end that isn't against a riser closes the gap: both ends splay outward
            # (spacing grows, front moves) or inward (spacing shrinks, rear moves)
            moving = self.climb_moving
            direction = +1.0 if moving == self.CL_FRONT else -1.0
            self.climb_splay += direction * self.climb_splay_rate * self.dt
            if abs(self.climb_splay) > self.climb_max_splay or abs(self.climb_base_splay + self.climb_splay) > self.climb_max_splay_k:
                self.stop_climb("can't match leg spacing to the stairs")
                return
            for leg in self.CL_LEGS:
                wheels[leg] = creep if leg in moving else hold
            self.climb_update_bumps()
            # Done when the moving end bumps its riser (hip torque), or its hips get blocked
            lag = max(abs(getattr(self, f'{leg}_hip_des_pos') - getattr(self, f'{leg}_hip_pos')) for leg in moving)
            blocked = lag > self.climb_hip_block and self.climb_t > self.climb_settle_time
            if all(self.climb_bumped[leg] for leg in moving) or blocked:
                self.climb_record_riser(d)
                self.enter_climb('TUMBLE', self.CL_LEGS)

        # ---- joint commands ----
        # Pitch limiter: if the body pitches nose-up past the target, shorten the front legs
        # (more splay) and lengthen the rear (less splay); relax back when below it
        # Works both ways: nose-up past the target shortens the front (dpitch > 0); nose-down
        # past it (descent) shortens the rear (dpitch < 0); inside the band it relaxes to 0
        nose_up = -self.climb_phi
        up_lim = self.climb_pitch_target() if self.climb_mode == 'up' else self.climb_pitch_cap
        dn_lim = self.climb_pitch_cap
        db, g, r = self.climb_pitch_deadband, self.climb_pitch_gain, self.climb_pitch_rate
        if nose_up > up_lim + db:
            self.climb_dpitch += min(g * (nose_up - up_lim), r) * self.dt
        elif -nose_up > dn_lim + db:
            self.climb_dpitch -= min(g * (-nose_up - dn_lim), r) * self.dt
        elif self.climb_dpitch > 0 and nose_up < up_lim - db:
            self.climb_dpitch = max(0.0, self.climb_dpitch - min(g * (up_lim - nose_up), r) * self.dt)
        elif self.climb_dpitch < 0 and -nose_up < dn_lim - db:
            self.climb_dpitch = min(0.0, self.climb_dpitch + min(g * (dn_lim + nose_up), r) * self.dt)
        common = self.climb_base_splay + self.climb_splay
        span = max(0.0, self.climb_max_splay_k - common)
        self.climb_dpitch = float(np.clip(self.climb_dpitch, -span, span))
        lo = min(0.0, common)                          # a leg lengthens until straight
        k_front = float(np.clip(common + self.climb_dpitch, lo, self.climb_max_splay_k))
        k_rear = float(np.clip(common - self.climb_dpitch, lo, self.climb_max_splay_k))
        # Hips: outward splay k (front negative-outward, rear positive)
        for leg in self.CL_LEGS:
            target = -k_front if leg[0] == 'f' else k_rear
            cur = getattr(self, f'{leg}_hip_des_pos')
            rate = self.climb_hip_rate if self.climb_phase == 'SETUP' else self.climb_hip_track_rate
            step = rate * self.dt
            setattr(self, f'{leg}_hip_des_pos', cur + float(np.clip(target - cur, -step, step)))
        # Bogies: tumbling legs set above; the rest stay flat in the world
        if self.climb_phase == 'D_TUMBLE':
            tumbling = self.desc_tumbling
        for leg in self.CL_LEGS:
            if leg not in tumbling:
                off = self.climb_bogie_offset.get(leg, 0.0)
                flat = np.round((self.climb_bogie_cmd[leg] + self.climb_phi - off) / np.pi) * np.pi - self.climb_phi + off
                step = self.climb_tumble_rate * self.dt
                self.climb_bogie_cmd[leg] += float(np.clip(flat - self.climb_bogie_cmd[leg], -step, step))
            setattr(self, f'{leg}_knee_des_pos', self.climb_bogie_cmd[leg] - getattr(self, f'{leg}_hip_des_pos'))
        self.knee_offsets[:] = 0
        self.wheel_leg = wheels
        self.wheel_leg_torque = ({leg: roll_torque for leg in self.CL_LEGS}
                                 if (roll_torque is not None and self.climb_phase in ('SETUP', 'APPROACH', 'ROLL')) else None)
        self.wheel_each = self.desc_each if (self.climb_phase or '').startswith('D_') else None
        self.send_commands()


    # Wheel velocity control with the motor's usual gain, but output torque capped at cap (Nm)
    def capped_wheel_vel(self, motor, target_vel, cap):
        qdot = motor.d.jnt(motor.motor_name).qvel[0]
        tau = motor.Kp * (target_vel - qdot)                     # motor-side torque, as in vel_control
        tau = float(np.clip(tau, -cap / motor.gear_ratio, cap / motor.gear_ratio))
        motor.torque_control(tau)

    # Function that sends the commands to the motors
    def send_commands(self):
        self.motors[0].pos_control(self.fr_hip_des_pos)
        self.motors[1].pos_control(self.fl_hip_des_pos)
        self.motors[2].pos_control(self.br_hip_des_pos)
        self.motors[3].pos_control(self.bl_hip_des_pos)

        self.motors[4].pos_control(self.fr_knee_des_pos + self.knee_offsets[0])
        self.motors[5].pos_control(self.fl_knee_des_pos + self.knee_offsets[1])
        self.motors[6].pos_control(self.br_knee_des_pos + self.knee_offsets[2])
        self.motors[7].pos_control(self.bl_knee_des_pos + self.knee_offsets[3])

        # Wheels by side normally; per leg while stair climbing
        if self.wheel_leg is None:
            fr = br = self.right_wheel_vel_des
            fl = bl = self.left_wheel_vel_des
        elif self.wheel_each is not None:
            # Descent: each wheel gets its own torque, or a torque-capped velocity hold
            for i, (kind, val) in self.wheel_each.items():
                mot = self.motors[i]
                if kind == 'torque':
                    mot.torque_control(val / mot.gear_ratio)
                elif kind == 'brake':                         # hold still, up to the full slip torque
                    self.capped_wheel_vel(mot, val, self.climb_slip_torque)
                else:
                    self.capped_wheel_vel(mot, val, self.climb_wheel_cap)
            return
        else:
            # Climbing: fixed torque (ROLL), or velocity control with the torque capped below slip
            legs = {'fr': (8, 9), 'fl': (10, 11), 'br': (12, 13), 'bl': (14, 15)}
            for leg, idx in legs.items():
                for i in idx:
                    if self.wheel_leg_torque is not None:
                        mot = self.motors[i]
                        mot.torque_control(self.wheel_leg_torque[leg] / mot.gear_ratio)
                    else:
                        self.capped_wheel_vel(self.motors[i], self.wheel_leg[leg], self.climb_cap_now)
            return
        self.motors[9].vel_control(fr)
        self.motors[8].vel_control(fr)
        self.motors[10].vel_control(fl)
        self.motors[11].vel_control(fl)
        self.motors[12].vel_control(br)
        self.motors[13].vel_control(br)
        self.motors[14].vel_control(bl)
        self.motors[15].vel_control(bl)

            

    # Calculates nearest pi value to the current angle
    def nearest_pi(self, current_angle):
        near_pi = np.round(current_angle/np.pi)*np.pi
        return near_pi
    
    # Get joystick state
    def get_joystick_state(self):
        pygame.event.get()
        if(self.controller_type == "logitech"):
            self.left_stick_lr = self.js.get_axis(0)
            self.left_stick_ud = self.js.get_axis(1)
            self.left_trigger = self.js.get_axis(2)
            self.right_stick_lr = self.js.get_axis(3)
            self.right_stick_ud = self.js.get_axis(4)
            self.right_trigger = self.js.get_axis(5)
            self.a_button = self.js.get_button(0)
            self.b_button = self.js.get_button(1)
            self.x_button = self.js.get_button(2)
            self.y_button = self.js.get_button(3)
            self.left_bumper = self.js.get_button(4)
            self.right_bumper = self.js.get_button(5)
            self.start_button = self.js.get_button(7)
            self.back_button = self.js.get_button(6)
            self.home_button = self.js.get_button(8)
            self.d_up = self.js.get_hat(0)[1]
            self.d_down = self.js.get_hat(0)[1]
            self.d_left = self.js.get_hat(0)[0]
            self.d_right = self.js.get_hat(0)[0]
            
        elif(self.controller_type == "logitech2"):
            self.left_stick_lr = self.js.get_axis(0)
            self.left_stick_ud = self.js.get_axis(1)
            # self.left_trigger = self.js.get_axis(2)
            self.right_stick_lr = self.js.get_axis(2)
            self.right_stick_ud = self.js.get_axis(3)+0.16
            # self.right_trigger = self.js.get_axis(5)
            self.a_button = self.js.get_button(0)
            self.b_button = self.js.get_button(1)
            self.x_button = self.js.get_button(2)
            self.y_button = self.js.get_button(3)
            self.start_button = self.js.get_button(7)
            self.back_button = self.js.get_button(6)
            self.home_button = self.js.get_button(8)
            self.d_up = self.js.get_hat(0)[1]
            self.d_down = self.js.get_hat(0)[1]
            self.d_left = self.js.get_hat(0)[0]
            self.d_right = self.js.get_hat(0)[0]
            self.left_bumper = self.js.get_button(4)
            self.right_bumper = self.js.get_button(5)

        elif(self.controller_type == "ps4"):
            self.left_stick_lr = self.js.get_axis(0)
            self.left_stick_ud = self.js.get_axis(1)
            self.left_trigger = self.js.get_axis(4)
            self.right_stick_lr = self.js.get_axis(2)
            self.right_stick_ud = self.js.get_axis(3)
            self.right_trigger = self.js.get_axis(5)
            self.a_button = self.js.get_button(0)
            self.b_button = self.js.get_button(1)
            self.x_button = self.js.get_button(2)
            self.y_button = self.js.get_button(3)

            self.aux_button = self.js.get_button(4)
            self.start_button = self.js.get_button(6)

            self.d_up = self.js.get_button(11)
            self.d_down = self.js.get_button(12)
            # self.d_left = self.js.get_button(13)
            # self.d_right = self.js.get_button(14)

            self.left_bumper = self.js.get_button(9)
            self.right_bumper = self.js.get_button(10)
            self.touchpad = self.js.get_button(15)

        # Filter joystick values based on deadzone:
        if(abs(self.left_stick_lr) < self.joystick_deadzone):
            self.left_stick_lr = 0
        if(abs(self.left_stick_ud) < self.joystick_deadzone):
            self.left_stick_ud = 0
        if(abs(self.right_stick_lr) < self.joystick_deadzone):
            self.right_stick_lr = 0
        if(abs(self.right_stick_ud) < self.joystick_deadzone):
            self.right_stick_ud = 0

    # Get current joint angles
    def get_mujoco_state(self,m,d):
        # Get current joint angles
        self.fr_hip_pos = self.d.jnt('head_right_thigh_joint').qpos[0]
        self.fl_hip_pos = self.d.jnt('head_left_thigh_joint').qpos[0]
        self.br_hip_pos = self.d.jnt('torso_right_thigh_joint').qpos[0]
        self.bl_hip_pos = self.d.jnt('torso_left_thigh_joint').qpos[0]

        self.fr_knee_pos = self.d.jnt('head_right_thigh_shin_joint').qpos[0]
        self.fl_knee_pos = self.d.jnt('head_left_thigh_shin_joint').qpos[0]
        self.br_knee_pos = self.d.jnt('torso_right_thigh_shin_joint').qpos[0]
        self.bl_knee_pos = self.d.jnt('torso_left_thigh_shin_joint').qpos[0]



    def print_all_joystick_states(self):
        self.get_joystick_state()
        # Print all joystick button states:

        print("A Button:", self.a_button)
        print("B Button:", self.b_button)
        print("X Button:", self.x_button)
        print("Y Button:", self.y_button)

        # print("Left Bumper:", self.left_bumper)
        # print("Right Bumper:", self.right_bumper)
        print("Index 1: ", self.js.get_button(1))
        print("Index 2: ", self.js.get_button(2))
        print("Index 3: ", self.js.get_button(3))
        
        
        print("Index 4: ", self.js.get_button(4))
        print("Index 5: ", self.js.get_button(5))
        print("Index 6: ", self.js.get_button(6))
        print("Index 7: ", self.js.get_button(7))
        print("Index 8: ", self.js.get_button(8))
        print("Index 9: ", self.js.get_button(9))
        print("Index 10: ", self.js.get_button(10))
        # print("Index 11: ", self.js.get_button(11))
        # print("Index 12: ", self.js.get_button(12))
        # print("Index 13: ", self.js.get_button(13))
        # print("Index 14: ", self.js.get_button(14))