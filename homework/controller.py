import pystk


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    import numpy as np
    #this seems to initialize an object
    action = pystk.Action()
    

    #compute acceleration
    action.acceleration = 0.1

    steering = steer_gain * aim_point[0]

    action.steer = np.clip(steering, -1, 1)
    
    action.drift = abs(aim_point[0]) > skid_thresh
    
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
        import numpy as np
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
