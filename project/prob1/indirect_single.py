""" Script that solves the optimal control problem for the project using indirect single shooting
    Author: Andres Pulido
    Date: April 2022
"""

from re import U
import time
from scipy.optimize import root
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def indirect_single_shooting(initial_states, final_states, guesses, nx, states_str, nu, control_str, is_print=False):

    def objective(obj0):
        """ Objective of the root finder function. The equality is the transversality conditions
        """
        if is_print:
            print("\nDecision vector: ", obj0)

        global u_array, t_array
        tf = obj0[nx]
        t_eval = np.linspace(0, tf, 1000)
        in0 = [x0, y0, theta0, obj0[0], obj0[1], obj0[2]]

        # solve_ivp solver
        sol = solve_ivp(dynamics, [0, tf], in0, t_eval=t_eval, method='Radau')
        x = sol.y[0]
        y = sol.y[1]
        theta = sol.y[2]
        lambx = sol.y[3]
        lamby = sol.y[4]
        lambtheta = sol.y[5]

        q = np.array([x[-1], y[-1], theta[-1], theta[-1]])
        lamb_tf = np.array(
            [lambx[-1], lamby[-1], lambtheta[-1]])
        comb = np.concatenate((q, lamb_tf), axis=0)
        H = np.matmul(lamb_tf, dynamics(tf, comb)[nx:])

        u_array = []
        t_array = []

        eq1 = x[-1] - xf
        eq2 = y[-1] - yf
        eq3 = theta[-1] - thetaf
        eq4 = H + 1
        eqs = [eq1, eq2, eq3, eq4]
        print('objective eqs: ', eqs)
        return eqs

    def dynamics(t, s):
        """Dynamics of the mobile robot"""

        global u_array, t_array

        x_t = s[0]
        y_t = s[1]
        theta_t = s[2]
        lambx_t = s[3]
        lamby_t = s[4]
        lambtheta_t = s[5]

        u = 1 if lambtheta_t < 0 else -1

        u_array.append(u)
        t_array.append(t)

        x_dot = np.cos(theta_t)
        y_dot = np.sin(theta_t)
        theta_dot = u
        lambx_dot = 0
        lamby_dot = 0
        lambtheta_dot = -lambx_t*np.sin(theta_t) + lamby_t*np.cos(theta_t)

        return np.array([x_dot, y_dot, theta_dot, lambx_dot, lamby_dot, lambtheta_dot])

    global u_array, t_array

    u_array = []
    t_array = []

    xf = final_states[0]
    yf = final_states[1]
    thetaf = final_states[2]
    x0 = initial_states[0]
    y0 = initial_states[1]
    theta0 = initial_states[2]

    start = time.time()

    obj_sol = root(objective, guesses[0:-1], method="hybr", tol=1e-3,)
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)

    obj_sol = obj_sol.x

    _ = objective(obj_sol)

    sol_initial_states = [x0, y0, theta0, obj_sol[0], obj_sol[1], obj_sol[2]]

    tf = obj_sol[nx]
    t_eval = np.linspace(0, tf, 1000)

    # euler forward solver
    soly = np.zeros((1000+1, 2*nx))
    soly[0, :] = sol_initial_states
    dt = tf/1000
    for ii, t in enumerate(t_eval):
        soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii])*dt

    print("u array shape: ", len(u_array))
    control_val = u_array
    control_time = np.array(t_array)

    #states_val = np.transpose(sol.y)
    states_val = soly[0:-1, :]

    end = time.time()
    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", t_eval[-1])

    plot_shooting(time=t_eval, states_val=states_val, states_str=states_str,
                  nx=nx, control_val=control_val, control_str=control_str, nu=nu, control_time=control_time, is_costates=True)


def plot_shooting(time, states_val, states_str, nx, control_val, control_str, nu, control_time, is_costates=False):
    """ Plots all states, all costates (if is_costate == True), the beta control and the orbit transfer in polar coord.
        nx - number of states
        nu - number of controls
    """
    fig1, axs1 = plt.subplots(nx)
    fig1.suptitle("State evolution of {} ".format(states_str[0:nx]))
    for jj in range(nx):
        axs1[jj].plot(time, states_val[:, jj])
        axs1[jj].set_ylabel(states_str[jj])
    if is_costates:
        fig3, axs3 = plt.subplots(nx)
        fig3.suptitle("Co-state evolution of {} ".format(states_str[nx:]))
        for jj in range(nx):
            axs3[jj].plot(time, states_val[:, nx+jj])
            axs3[jj].set_ylabel(states_str[nx+jj])
    plt.xlabel("time [s]")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(control_time, control_val)
    plt.ylabel(control_str[0])
    plt.xlabel("time [s]")
    plt.title(f'Control evolution')
    plt.show()


def main():
    nx = 3  # number of states
    nu = 1  # number of controls

    s_str = ['$x$', '$y$', '$theta$']
    cs_str = ['$lambda_x$', '$lambda_y$', '$lambda_theta$']
    states_str = s_str + cs_str
    control_str = ['$u$']

    # Initial conditions
    x0 = 0
    y0 = 0
    theta0 = -np.pi
    xf = 0
    yf = 0
    thetaf = np.pi

    initial_states = [x0, y0, theta0]
    final_states = [xf, yf, thetaf]

    lamb0_guess = [0.1, 0.1, 0.2]
    tf_guess = [2]
    u_guess = [0.5]

    guesses = lamb0_guess + tf_guess + u_guess  # list of element guesses

    indirect_single_shooting(
        initial_states, final_states, guesses, nx, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
