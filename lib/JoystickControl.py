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
        self.knee_offsets = np.zeros(4)           # added to knee targets [fr, fl, br, bl]
        self.tilt = np.zeros(3)                   # latest [pitch, torso roll, head roll], rad
        self.height = 0.0                         # latest measured ride height (m)
        self.height_target = None                 # set on first use / when entering Smart mode
        self.level_geom = None                    # robot geometry, filled in on first use
        self.reset_leveling()


    # Function that controls the wheels
    def control_wheels(self):
        # Control the wheels with basic joystick control
        self.left_wheel_vel_des = self.max_wheel_vel*(-self.left_stick_ud + self.left_stick_lr)
        self.right_wheel_vel_des = self.max_wheel_vel*(-self.left_stick_ud - self.left_stick_lr)

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

        # Low-pass filter the tilt; its rate feeds the D term
        if self.tilt_filt is None:
            self.tilt_filt = self.tilt.copy()
        alpha = self.dt / (self.level_filter_tau + self.dt)
        self.tilt_filt = self.tilt_filt + alpha * (self.tilt - self.tilt_filt)
        rate = np.zeros(3) if self.prev_tilt is None else (self.tilt_filt - self.prev_tilt) / self.dt
        self.prev_tilt = self.tilt_filt.copy()

        # 1) Feedforward geometry: put every hip at one world height H (level body),
        #    with H at the target height above the average ground under the wheels.
        #    Leg extension needed: e = H - knee height.
        ground = np.mean([leg['ground'] for leg in legs.values()])
        knee_z = np.array([leg['knee_z'] for leg in legs.values()])
        H = ground + self.height_target
        # Level first: shift H (height drifts) so every leg stays within its range
        H_lo = np.max(knee_z) + g['e_min']
        H_hi = np.min(knee_z) + g['e_max']
        H = np.clip(H, H_lo, H_hi) if H_lo <= H_hi else 0.5 * (H_lo + H_hi)

        # 2) PD trim on the remaining tilt (raises low corners, lowers high ones)
        trim = self.level_kp * self.tilt_filt + self.level_kd * rate
        des = {}
        for key, leg in legs.items():
            roll_trim = trim[1] if leg['body'] == 'torso' else trim[2]
            e = H - leg['knee_z'] + leg['x'] * trim[0] - leg['y'] * roll_trim
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

        self.motors[9].vel_control(self.right_wheel_vel_des)
        self.motors[8].vel_control(self.right_wheel_vel_des)
        self.motors[10].vel_control(self.left_wheel_vel_des)
        self.motors[11].vel_control(self.left_wheel_vel_des)
        self.motors[12].vel_control(self.right_wheel_vel_des)
        self.motors[13].vel_control(self.right_wheel_vel_des)
        self.motors[14].vel_control(self.left_wheel_vel_des)
        self.motors[15].vel_control(self.left_wheel_vel_des)

            

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