import pystk


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    import numpy as np

    # Initialize an action object
    action = pystk.Action()

    # PID parameters for steering
    # Steering PID
    Kp_steer = 5.07233064855576
    Ki_steer = 0.01
    Kd_steer = 0.09217618229759023

    # Velocity PID
    Kp_vel = 7.948198073336594
    Ki_vel = 0.5475394791973792
    Kd_vel = 0.7194280724990274
    # State variables for PID
    if not hasattr(control, "steer_error_prev"):
        # Initialize PID memory for steering
        control.steer_error_prev = 0
        control.steer_integral = 0
        control.vel_error_prev = 0
        control.vel_integral = 0

    dt = 0.1  # Assuming a constant timestep; you might calculate this dynamically.

    # Steering PID
    steer_error = aim_point[0]  # Error is the horizontal deviation
    control.steer_integral += steer_error * dt
    steer_derivative = (steer_error - control.steer_error_prev) / dt
    steering_output = (
        Kp_steer * steer_error +
        Ki_steer * control.steer_integral +
        Kd_steer * steer_derivative
    )
    control.steer_error_prev = steer_error

    # Velocity PID
    vel_error = target_vel - current_vel  # Error is the difference from target velocity
    control.vel_integral += vel_error * dt
    vel_derivative = (vel_error - control.vel_error_prev) / dt
    acceleration_output = (
        Kp_vel * vel_error +
        Ki_vel * control.vel_integral +
        Kd_vel * vel_derivative
    )
    control.vel_error_prev = vel_error

    # Apply steering output
    action.steer = np.clip(steering_output, -1, 1)

    # Drift if steering exceeds skid threshold
    action.drift = abs(aim_point[0]) > skid_thresh

    # Apply acceleration/braking logic
    if acceleration_output > 0:
        action.acceleration = np.clip(acceleration_output, 0, 1)
        action.brake = False
        # Activate nitro for straight sections
        action.nitro = abs(aim_point[0]) < 0.1 and current_vel >= 0.9 * target_vel
    else:
        action.acceleration = 0
        action.brake = True

    return action
    




if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser
    import numpy as np
    
    def test_controller(args):
        
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
