from skopt import gp_minimize
from skopt.space import Real
from utils import PyTux
from controller import control
# Global PyTux instance (singleton)
pytux = None

# List of tracks to evaluate
TRACKS = [
    "zengarden", "lighthouse", "hacienda",
    "snowtuxpeak", "cornfield_crossing", "scotland"
]

def objective_function(params):
    """
    Evaluate the performance of PID parameters on multiple tracks.
    
    params: [Kp_steer, Ki_steer, Kd_steer, Kp_vel, Ki_vel, Kd_vel]
    """
    global pytux  # Use the global PyTux instance
    Kp_steer, Ki_steer, Kd_steer, Kp_vel, Ki_vel, Kd_vel = params

    # Override PID parameters in the control function
    def control_with_params(aim_point, current_vel):
        return control(
            aim_point, 
            current_vel, 
            steer_gain=Kp_steer, 
            skid_thresh=0.2, 
            target_vel=25
        )

    total_score = 0  # Sum of scores across all tracks

    for track in TRACKS:
        steps, how_far = pytux.rollout(track, control_with_params, max_frames=1000, verbose=False)
        print(steps, how_far)
        # Combine metrics to compute a score (adjust weights as needed)
        score = how_far - 0.01 * steps  # Prefer greater distance with fewer steps
        total_score += score

    return -total_score  # Negative because Scikit-Optimize minimizes the objective

if __name__ == "__main__":
    # Initialize the global PyTux instance
    pytux = PyTux()

    # Define the search space for each parameter
    search_space = [
        Real(0.1, 10.0, name="Kp_steer"),  # Steering proportional gain
        Real(0.01, 1.0, name="Ki_steer"),  # Steering integral gain
        Real(0.01, 1.0, name="Kd_steer"),  # Steering derivative gain
        Real(0.1, 10.0, name="Kp_vel"),    # Velocity proportional gain
        Real(0.01, 1.0, name="Ki_vel"),    # Velocity integral gain
        Real(0.01, 1.0, name="Kd_vel")     # Velocity derivative gain
    ]

    # Run the optimization
    try:
        result = gp_minimize(objective_function, search_space, n_calls=50, random_state=42)
    finally:
        # Ensure PyTux is closed properly
        pytux.close()

    # Best parameters and score
    print("Best parameters found:", result.x)
    print("Best total score:", -result.fun)  # Convert back to positive score

    # Save the best parameters for future use
    best_params = result.x
