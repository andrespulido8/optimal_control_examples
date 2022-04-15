""" Script that solves the optimal control problem for the midterm using indirect multiple shooting
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

        global beta_array, t_array
        tf = obj0[nx]
        ptot0 = np.reshape(obj0[nx+1:], [2*nx, K-1])
        E_array = np.empty([K-1, nx*2])

        for k in range(K):
            if k == 0:
                p0 = [r0, vr0, theta0, vtheta0, m0, obj0[0],
                      obj0[1], obj0[2], obj0[3], obj0[4]]
            else:
                p0 = ptot0[:, k-1]

            sol = solve_ivp(dynamics, [tau[k], tau[k+1]], p0,
                            args=(t0, tf), method='Radau')

            r = sol.y[0]
            vr = sol.y[1]
            theta = sol.y[2]
            vtheta = sol.y[3]
            m = sol.y[4]
            lambr = sol.y[5]
            lambvr = sol.y[6]
            lambtheta = sol.y[7]
            lambvtheta = sol.y[8]
            lambm = sol.y[9]

            if k < K-1:
                ptots_end = ptot0[:, k]
                eq1k = r[-1] - ptots_end[0]
                eq2k = vr[-1] - ptots_end[1]
                eq3k = theta[-1] - ptots_end[2]
                eq4k = vtheta[-1] - ptots_end[3]
                eq5k = m[-1] - ptots_end[4]
                eq6k = lambr[-1] - ptots_end[5]
                eq7k = lambvr[-1] - ptots_end[6]
                eq8k = lambtheta[-1] - ptots_end[7]
                eq9k = lambvtheta[-1] - ptots_end[8]
                eq10k = lambm[-1] - ptots_end[9]
                E_array[k, :] = [eq1k, eq2k, eq3k, eq4k,
                                 eq5k, eq6k, eq7k, eq8k, eq9k, eq10k]
                #print([eq1k, eq2k, eq3k, eq4k, eq5k, eq6k, eq7k, eq8k, eq9k, eq10k])

        x_tf = np.array([r[-1], vr[-1], theta[-1], theta[-1], m[-1]])
        lamb_tf = np.array(
            [lambr[-1], lambvr[-1], lambtheta[-1], lambvtheta[-1], lambm[-1]])
        comb = np.concatenate((x_tf, lamb_tf), axis=0)
        H = np.matmul(lamb_tf, dynamics(tf, comb, 0, tf)[5:])

        beta_array = []
        t_array = []

        eq1 = r[-1] - rf
        eq2 = vr[-1] - vrf
        eq3 = vtheta[-1] - vthetaf
        eq4 = H
        eq5 = lambtheta[-1]
        eq6 = lambm[-1] - 1
        print("Final eqs: ", [eq1, eq2, eq3, eq4, eq5, eq6])

        E_array = np.reshape(E_array, 2*nx*(K-1))
        E_array = np.concatenate(
            (E_array, [eq1, eq2, eq3, eq4, eq5, eq6]), axis=None)
        return E_array

    def dynamics(t, s, tinitial, tfinal):
        """Dynamics of the spacecraft"""
        global beta_array, t_array

        r_t = s[0]
        vr_t = s[1]
        theta_t = s[2]
        vtheta_t = s[3]
        m_t = s[4]
        lambr_t = s[5]
        lambvr_t = s[6]
        lambtheta_t = s[7]
        lambvtheta_t = s[8]
        lambm_t = s[9]

        beta = np.arctan2(lambvr_t, lambvtheta_t)

        beta_array.append(beta)
        t_array.append(t)

        r_dot = vr_t
        vr_dot = -mu/r_t**2 + vtheta_t**2/r_t + T*np.sin(beta)/m_t
        theta_dot = vtheta_t/r_t
        vtheta_dot = -vr_t*vtheta_t/r_t + T*np.cos(beta)/m_t
        m_dot = -T/ve
        lambr_dot = -2*lambvr_t*mu/r_t**3 + vtheta_t * \
            (lambvr_t*vtheta_t + vr_t*lambvtheta_t)/r_t**2
        lambvr_dot = lambvtheta_t*vtheta_t/r_t - lambr_t
        lambtheta_dot = 0
        lambvtheta_dot = -2*vtheta_t*lambvr_t/r_t - \
            lambtheta_t/r_t + lambvtheta_t*vr_t/r_t
        lambm_dot = T*(lambvr_t*np.sin(beta) +
                       lambvtheta_t*np.cos(beta))/m_t**2
        derivatives = np.array([r_dot, vr_dot, theta_dot, vtheta_dot, m_dot,
                                lambr_dot, lambvr_dot, lambtheta_dot, lambvtheta_dot, lambm_dot])

        return (tfinal-tinitial)*derivatives/2

    global beta_array, t_array

    beta_array = []
    t_array = []

    rf = final_states[0]
    vrf = final_states[1]
    vthetaf = final_states[2]
    r0 = initial_states[0]
    vr0 = initial_states[1]
    theta0 = initial_states[2]
    vtheta0 = initial_states[3]
    m0 = initial_states[4]

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
            p0 = [r0, vr0, theta0, vtheta0, m0, obj_sol[0],
                  obj_sol[1], obj_sol[2], obj_sol[3], obj_sol[4]]
        else:
            p0 = ptot0[:, k-1]
        tau_eval = np.linspace(tau[k], tau[k+1], points)

        soly = np.zeros((points+1, 2*nx))
        soly[0, :] = p0
        dt = (tau[k+1] - tau[k])/points
        for ii, t in enumerate(tau_eval):
            soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], t0, tf)*dt

        # Collect values of each partition to plot
        control_val[k] = beta_array[0:points]
        tau_array[k] = tau_eval
        states_val[points*k:points + points*k,
                   :] = soly[0:-1, :]

    # Scale back time from (-1, 1) to (t0, tf)
    time_s = t0 + (tf-t0)*(np.reshape(tau_array, K*points)+1)/2

    end = time.time()

    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", time_s[-1])
    print("m(tf): ", states_val[-1, -1])

    plot_shooting(time=time_s, states_val=states_val, states_str=states_str,
                  nx=nx, control_val=np.reshape(control_val, K*points), control_str=control_str, nu=nu, control_time=time_s, is_costates='True')


def plot_shooting(time, states_val, states_str, nx, control_val, control_str, nu, control_time, is_costates='False'):
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
    if is_costates == True:
        fig3, axs3 = plt.subplots(nx)
        fig3.suptitle("Co-state evolution of {} ".format(states_str[nx:]))
        for jj in range(nx):
            axs3[jj].plot(time, states_val[:, nx+jj])
            axs3[jj].set_ylabel(states_str[nx+jj])
    axs2.plot(control_time, control_val)
    axs2.set_ylabel(control_str[0])
    plt.xlabel("time [s]")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.axes(projection='polar')
    plt.polar(states_val[:, 2], states_val[:, 0])
    plt.title(f'Trajectory Curve')
    plt.show()


def main():
    global mu, T, ve
    mu = 1
    T = 0.1405
    ve = 1.8758344

    nx = 5  # number of states
    nu = 1  # number of controls

    s_str = ['$r$', '$v_r$', '$theta$', '$v_theta$', 'm']
    cs_str = ['$\lambda_r$', '$\lambda_vr$',
              '$\lambda_theta$', '$\lambda v_theta$', '$\lambda_m$']
    states_str = s_str + cs_str
    control_str = ['$beta$']

    K = 20

    # Initial conditions
    r0 = 1
    rf = 1.5
    vr0 = 0
    vrf = 0
    theta0 = 0
    vtheta0 = np.sqrt(mu/r0)
    vthetaf = np.sqrt(mu/rf)

    m0 = 1

    initial_states = [r0, vr0, theta0, vtheta0, m0]
    final_states = [rf, vrf, vthetaf]

    lamb0_guess = [1, 1, 1, 1, 1]
    tf_guess = [2]
    beta_guess = [0.1]

    guesses = lamb0_guess + tf_guess + beta_guess  # list of element guesses

    multiple_shooting(initial_states, final_states, guesses,
                      nx, K, states_str, nu, control_str, False)


if __name__ == "__main__":
    main()

    print("Done")
