import time
import mujoco
import mujoco.viewer
import numpy as np
import yaml
from pathlib import Path
import os
import sys
print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir,)))
import lib.MotorModel as motor
import lib.JoystickControl as js_ctrl
import AutoSim


# Call AutoSim to generate the new robot spec:
model_config_path = 'model_configs/WS_Scale/model_config.yaml'
motor_config_path = 'model_configs/WS_Scale/motor_config.yaml'

# Load motor params for later access
motor_config = yaml.safe_load(Path(motor_config_path).read_text())

# Generate the new robot spec:
walter = AutoSim.GenerateModel(model_config_path=model_config_path, motor_config_path=motor_config_path)

#Add payload to walter body
# walter.add_payload(mass = 32, body_loc = [0,0,0.2], size = [0.2, 0.1, 0.1])

# Generate the scene around the robot (groundplane and sky)
walter.gen_scene()

# Add some obstacles:

walter.add_stairs(rise=0.2,run=0.3,num_steps=15)
walter.add_log(d=0.4,length = 2)
walter.add_incline(angle_deg=40, pos = [3, 5, 0], width = 1.5, length = 4 )
walter.add_box(pos = [5.5, 5, 2.5], size = [1, 2, 0.1])
walter.add_incline(angle_deg=-40, pos = [8, 5, 0], width = 1.5, length = 4 )
# ---- Wheel visuals: tire-colored cylinders with a spin marker (physics unchanged) ----
TIRE_RGBA   = [0.10, 0.10, 0.10, 1]   # rubber
HUB_RGBA    = [0.60, 0.60, 0.62, 1]   # metal hub
MARKER_RGBA = [0.95, 0.75, 0.10, 1]   # bright spoke so rotation is visible
TREAD_RGBA  = [0.30, 0.30, 0.30, 1]   # tread blocks, lighter than the tire so they read
N_TREAD     = 12                      # tread blocks per wheel (one is drawn in MARKER_RGBA)
TREAD_DEPTH = 0.008                   # how far the blocks stand out from the tire (m)

def add_wheel_visuals(spec, model_config):
    """Add visual-only geoms to every wheel body. Returns the names of the
    original collision geoms so they can be hidden after compiling."""
    r = model_config['wheel_params']['radius']
    w = model_config['wheel_params']['width']
    axis_y = [1, 1, 0, 0]              # rotates a cylinder's z-axis onto the wheel's spin axis (y)
    visual = dict(contype=0, conaffinity=0, density=0, group=1)   # no contacts, no mass
    collision_geoms = []
    for body in ['torso', 'head']:
        for side in ['left', 'right']:
            for wheel in ['front', 'rear']:
                name = f'{body}_{side}_{wheel}_wheel'
                b = spec.worldbody.find_child(name)
                collision_geoms.append(f'{name}_geom')
                # Tire, inset so the tread's outer face sits exactly at the
                # collision radius (wheels touch the ground visually where they do physically)
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r - TREAD_DEPTH + 0.002, w/2, 0],
                           quat=axis_y, rgba=TIRE_RGBA, **visual)
                # Tread blocks around the circumference
                arc = 2*np.pi*r / N_TREAD
                for k in range(N_TREAD):
                    phi = 2*np.pi*k / N_TREAD
                    rc = r - TREAD_DEPTH/2
                    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=[rc*np.cos(phi), 0, rc*np.sin(phi)],
                               quat=[np.cos(-phi/2), 0, np.sin(-phi/2), 0],   # radial along box x
                               size=[TREAD_DEPTH/2, 0.85*w/2, 0.3*arc],
                               rgba=MARKER_RGBA if k == 0 else TREAD_RGBA, **visual)
                # Hub, slightly proud of both faces
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.45*r, w/2 + 0.002, 0],
                           quat=axis_y, rgba=HUB_RGBA, **visual)
                # One radial spoke on each face (single spoke = no 180-degree ambiguity)
                for face in [1, -1]:
                    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=[0.45*r, face*(w/2 + 0.004), 0],
                               size=[0.42*r, 0.002, 0.008],
                               rgba=MARKER_RGBA, **visual)
    return collision_geoms

model_config = yaml.safe_load(Path(model_config_path).read_text())
wheel_collision_geoms = add_wheel_visuals(walter.spec, model_config)

# Compile the model:
m = walter.spec.compile()
d = mujoco.MjData(m)

# Hide the original ellipsoid collision geoms (group 3 is hidden by default;
# press 3 in the viewer to show them). They still handle all wheel contact.
for g in wheel_collision_geoms:
    m.geom_group[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)] = 3

# Initializing motor models (ignore this part)
fr_hip = motor.MotorModel(m, d, 'head_right_thigh_joint', motor_config['hip_params'], 12)
fl_hip = motor.MotorModel(m, d,'head_left_thigh_joint',  motor_config['hip_params'], 8)
br_hip = motor.MotorModel(m, d,'torso_right_thigh_joint',  motor_config['hip_params'], 4)
bl_hip = motor.MotorModel(m, d,'torso_left_thigh_joint',  motor_config['hip_params'], 0)

fr_knee = motor.MotorModel(m, d, 'head_right_thigh_shin_joint', motor_config['knee_params'], 13)
fl_knee = motor.MotorModel(m, d, 'head_left_thigh_shin_joint', motor_config['knee_params'], 9)
br_knee = motor.MotorModel(m, d, 'torso_right_thigh_shin_joint', motor_config['knee_params'], 5)
bl_knee = motor.MotorModel(m, d, 'torso_left_thigh_shin_joint', motor_config['knee_params'], 1)

fr_wheel1_joint = motor.MotorModel(m, d, 'head_right_shin_front_wheel_joint', motor_config['wheel_params'], 14)
fr_wheel2_joint = motor.MotorModel(m, d, 'head_right_shin_rear_wheel_joint', motor_config['wheel_params'], 15)
fl_wheel1_joint = motor.MotorModel(m, d, 'head_left_shin_front_wheel_joint', motor_config['wheel_params'], 10)
fl_wheel2_joint = motor.MotorModel(m, d, 'head_left_shin_rear_wheel_joint', motor_config['wheel_params'], 11)
br_wheel1_joint = motor.MotorModel(m, d, 'torso_right_shin_front_wheel_joint', motor_config['wheel_params'], 6)
br_wheel2_joint = motor.MotorModel(m, d, 'torso_right_shin_rear_wheel_joint', motor_config['wheel_params'], 7)
bl_wheel1_joint = motor.MotorModel(m, d, 'torso_left_shin_front_wheel_joint', motor_config['wheel_params'], 2)
bl_wheel2_joint = motor.MotorModel(m, d, 'torso_left_shin_rear_wheel_joint', motor_config['wheel_params'], 3)

motors = [fr_hip, fl_hip, br_hip, bl_hip, 
          fr_knee, fl_knee, br_knee, bl_knee, 
          fr_wheel1_joint, fr_wheel2_joint, fl_wheel1_joint, fl_wheel2_joint, br_wheel1_joint, br_wheel2_joint, bl_wheel1_joint, bl_wheel2_joint]


# Initialize joystick controller
controller = js_ctrl.JoystickController("ps4", m, d, motors)

# Main simulation loop:
with mujoco.viewer.launch_passive(m,d,show_left_ui=False,show_right_ui=False) as viewer:
    start = time.time()
    shown_mode = None
    last_overlay = 0.0
    while viewer.is_running():
        step_start = time.time()

        # Step the simulation forward
        mujoco.mj_step(m, d)

        # Call joystick controller:
        controller.control(m,d)


        # Log motor data to plot later:
        br_wheel1_joint.log_data() 
        br_knee.log_data()
        br_hip.log_data()

        # On-screen readout of mode and body tilt, refreshed ~10x per second
        # (updating every physics step slows the sim down)
        if controller.mode != shown_mode or time.time() - last_overlay > 0.1:
            p, rt, rh = np.degrees(controller.tilt)
            h = f"{100*controller.height:.1f} cm"
            if controller.mode == "Smart Gen-0" and controller.height_target is not None:
                h += f" (target {100*controller.height_target:.1f})"
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               "Mode\nPitch\nRoll (torso)\nRoll (head)\nRide height",
                               f"{controller.mode}\n{p:+5.1f} deg\n{rt:+5.1f} deg\n{rh:+5.1f} deg\n{h}")])
            shown_mode = controller.mode
            last_overlay = time.time()

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# br_wheel1_joint.plot_data_output_rpms()
# br_knee.plot_data_output_rpms()
# br_hip.plot_data_output_rpms()