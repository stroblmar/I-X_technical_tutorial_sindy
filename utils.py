import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ============================================================================================================
### Constants
LV_PALETTE = {'x': "#3B8638", 'y': "#8F18E7", 'x_sindy': "#B5CF22", 'y_sindy': "#BE9FD5"}

# ============================================================================================================
### Helper functions
# ============================================================================================================
def generate_lv_data(params_dict, ic_dict, sampling_interval=1, t_end=100, num_observations=None,integrator_keywords={}):
    '''
    Generate synthetic data for the Lotka-Volterra system using the provided parameters and initial conditions.
    '''
    # Set default integrator keywords if not provided
    integrator_keywords['method'] = integrator_keywords.get('method', 'RK45')
    integrator_keywords['rtol'] = integrator_keywords.get('rtol', 1e-12)
    integrator_keywords['atol'] = integrator_keywords.get('atol', 1e-12)
    # Define the time span and evaluation points for the simulation
    t_span = (0, t_end)
    if num_observations is not None:
        sampling_interval = (t_end - t_span[0]) / (num_observations - 1)
    t_eval = np.arange(t_span[0], t_span[1] + sampling_interval, sampling_interval)
    # Define the initial conditions based on the provided parameters
    if 'n0' in ic_dict and 'f_y' in ic_dict:
        x0 = [ic_dict['n0'] * (1 - ic_dict['f_y']), ic_dict['n0'] * ic_dict['f_y']]
    elif 'x0' in ic_dict and 'y0' in ic_dict:
        x0 = [ic_dict['x0'], ic_dict['y0']]
    else:
        raise ValueError("Initial conditions must be provided as either 'n0' and 'f_y' or 'x0' and 'y0' in ic_dict.")
    # L-V system right-hand side function
    def lotka_volterra_rhs(t, x, params_dict):
        dx = params_dict['r_x'] * x[0] - params_dict['c_xx'] * x[0] * x[0] - params_dict['c_yx'] * x[0] * x[1]
        dy = params_dict['r_y'] * x[1] - params_dict['c_yy'] * x[1] * x[1] - params_dict['c_xy'] * x[0] * x[1]
        return [dx, dy]

    # Solve the initial value problem: Runga-Kutta method
    solution = solve_ivp(
        fun=lotka_volterra_rhs,
        t_span=t_span,
        y0=x0,
        args=(params_dict,),
        t_eval=t_eval,
        **integrator_keywords
    )

    results_df = pd.DataFrame(data=solution.y.T, columns=['x', 'y'], index=solution.t)
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 't'}, inplace=True)
    return results_df
