import pystk


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, base_target_vel=25, kart_id=0):
    """
    aim_point: Normalized aim point (-1 to 1)
    current_vel: Current velocity of the kart
    steer_gain: Steering sense
    skid_thresh: Threshold for drifting
    base_target_vel: Base target velocity
    kart_id: Kart identifier (for multi-agent support)
    return: Action object for PySuperTux
    """
    import numpy as np
    action = pystk.Action()

    dynamic_steer_gain = steer_gain / (1 + 0.1 * current_vel)
    steering = dynamic_steer_gain * aim_point[0]
    action.steer = np.clip(steering, -1, 1)

    action.drift = abs(aim_point[0]) > skid_thresh

    target_vel = base_target_vel * (1 - 0.5 * abs(aim_point[0]))
    vel_error = target_vel - current_vel

    if vel_error > 0:
        action.acceleration = np.clip(vel_error / target_vel, 0, 1)
        action.brake = False
        action.nitro = abs(aim_point[0]) < 0.1 and current_vel >= 0.9 * target_vel
    else:
        action.acceleration = 0
        action.brake = True

    return action


if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        """
        Test the controller on specified tracks.
        :param args: Command line arguments
        """
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(
                t, 
                lambda aim, vel: control(aim, vel, kart_id=0),  # Single kart control
                max_frames=1000, 
                verbose=args.verbose
            )
            print(f"Track: {t}, Steps: {steps}, Distance: {how_far}")
        pytux.close()

    parser = ArgumentParser("Test Controller")
    parser.add_argument('track', nargs='+', help="Tracks to test the controller on")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    args = parser.parse_args()
    test_controller(args)