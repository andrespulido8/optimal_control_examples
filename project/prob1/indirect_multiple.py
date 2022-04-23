""" Script that solves the optimal control problem for the project using indirect multiple shooting
    Author: Andres Pulido
    Date: April 2022
"""

import time
from scipy.optimize import root
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def multiple_shooting(initial_states, final_states, guesses, nx, K, states_str, nu, control_str, is_print=False):

    def objective(obj0):
        """ Objective of the root finder function. The equality is the difference between the states at the end of 
            a partition and the initial states at the beginning of the next partition, and the transversality conditions
        """
        if is_print:
            print("\nDecision vector: ", obj0)

        global u_array, t_array
        tf = obj0[nx]

        ptot0 = np.reshape(obj0[nx+1:], [2*nx, K-1])
        E_array = np.empty([K-1, nx*2])

        for k in range(K):
            if k == 0:
                p0 = [x0, y0, theta0, obj0[0], obj0[1], obj0[2]]
            else:
                p0 = ptot0[:, k-1]

            sol = solve_ivp(dynamics, [tau[k], tau[k+1]], p0,
                            args=(t0, tf), method='Radau')

            x = sol.y[0]
            y = sol.y[1]
            theta = sol.y[2]
            lambx = sol.y[3]
            lamby = sol.y[4]
            lambtheta = sol.y[5]

            if k < K-1:
                ptots_end = ptot0[:, k]
                eq1k = x[-1] - ptots_end[0]
                eq2k = y[-1] - ptots_end[1]
                eq3k = theta[-1] - ptots_end[2]
                eq4k = lambx[-1] - ptots_end[3]
                eq5k = lamby[-1] - ptots_end[4]
                eq6k = lambtheta[-1] - ptots_end[5]
                E_array[k, :] = [eq1k, eq2k, eq3k, eq4k, eq5k, eq6k]
                #print([eq1k, eq2k, eq3k, eq4k, eq5k, eq6k])

        q_tf = np.array([x[-1], y[-1], theta[-1]])
        lamb_tf = np.array(
            [lambx[-1], lamby[-1], lambtheta[-1]])
        comb = np.concatenate((q_tf, lamb_tf), axis=0)
        H = np.matmul(lamb_tf, dynamics(tf, comb, 0, tf)[nx:])

        u_array = []
        t_array = []

        eq1 = x[-1] - xf
        eq2 = y[-1] - yf
        eq3 = theta[-1] - thetaf
        eq4 = H + 1
        print("Final eqs: ", [eq1, eq2, eq3, eq4])

        E_array = np.reshape(E_array, 2*nx*(K-1))
        E_array = np.concatenate(
            (E_array, [eq1, eq2, eq3, eq4]), axis=None)
        return E_array

    def dynamics(t, s, tinitial, tfinal):
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
        derivatives = np.array(
            [x_dot, y_dot, theta_dot, lambx_dot, lamby_dot, lambtheta_dot])

        return (tfinal-tinitial)*derivatives/2

    global u_array, t_array

    u_array = []
    t_array = []

    xf = final_states[0]
    yf = final_states[1]
    thetaf = final_states[2]
    x0 = initial_states[0]
    y0 = initial_states[1]
    theta0 = initial_states[2]

    t0 = 0

    start = time.time()

    tau = np.linspace(-1, +1, K+1)

    ptot0guess = np.ones([2*nx, K-1])
    reshaped = np.reshape(ptot0guess, 2*nx*(K-1))

    # Decision vector: the initial states at each partition
    changing = np.concatenate(([guesses[0:nx+1], reshaped]))

    obj_sol = root(objective, changing, method="hybr", tol=1e-4)
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)

    obj_sol = obj_sol.x

    tf = obj_sol[nx]
    ptot0 = np.reshape(obj_sol[nx+1:], [2*nx, K-1])
    points = 1000
    tau_array = np.empty([K, points])
    control_val = np.empty([K, points])
    states_val = np.empty([K*points, 2*nx])

    for k in range(K):
        if k == 0:
            p0 = [x0, y0, theta0, obj_sol[0], obj_sol[1], obj_sol[2]]
        else:
            p0 = ptot0[:, k-1]
        tau_eval = np.linspace(tau[k], tau[k+1], points)

        soly = np.zeros((points+1, 2*nx))
        soly[0, :] = p0
        dt = (tau[k+1] - tau[k])/points
        for ii, t in enumerate(tau_eval):
            soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], t0, tf)*dt

        # Collect values of each partition to plot
        control_val[k] = u_array
        tau_array[k] = tau_eval
        states_val[points*k:points + points*k,
                   :] = soly[0:-1, :]
        u_array = []

    # Scale back time from (-1, 1) to (t0, tf)
    time_s = t0 + (tf-t0)*(np.reshape(tau_array, K*points) + 1)/2

    end = time.time()

    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", time_s[-1])

    plot_shooting(time=time_s, states_val=states_val, states_str=states_str,
                  nx=nx, control_val=np.reshape(control_val, K*points), control_str=control_str, nu=nu, control_time=time_s, is_costates=True)


def plot_shooting(time, states_val, states_str, nx, control_val, control_str, nu, control_time, is_costates=False):
    """ Plots all states, all costates (if is_costate == True), the beta control and the orbit transfer in polar coord.
        nx - number of states
        nu - number of controls
    """
    fig1, axs1 = plt.subplots(nx)
    fig1.suptitle("State evolution of {} ".format(states_str[0:nx]))
    fig2, axs2 = plt.subplots(nu)
    fig2.suptitle("State evolution of {} ".format(control_str))
    for jj in range(nx):
        axs1[jj].plot(time, states_val[:, jj])
        axs1[jj].set_ylabel(states_str[jj])
    if is_costates:
        fig3, axs3 = plt.subplots(nx)
        fig3.suptitle("Co-state evolution of {} ".format(states_str[nx:]))
        for jj in range(nx):
            axs3[jj].plot(time, states_val[:, nx+jj])
            axs3[jj].set_ylabel(states_str[nx+jj])
    axs2.plot(control_time, control_val)
    axs2.set_ylabel(control_str[0])
    plt.xlabel("time [s]")
    plt.show()


def main():

    nx = 3  # number of states
    nu = 1  # number of controls

    K = 10

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

    lamb0_guess = [1, 1, 1]
    tf_guess = [1]
    u_guess = [1]

    guesses = lamb0_guess + tf_guess + u_guess  # list of element guesses

    multiple_shooting(initial_states, final_states, guesses,
                      nx, K, states_str, nu, control_str, False)


if __name__ == "__main__":
    main()

    print("Done")
