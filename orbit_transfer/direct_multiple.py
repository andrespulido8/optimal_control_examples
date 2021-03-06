""" Script that solves the optimal control problem for the Orbit-Transfer problem using direct multiple shooting
    Author: Andres Pulido
    Date: April 2022
"""

import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def direct_multiple_shooting_method(initial_states, final_states, tf_guess, coeff_guess, nx, N, K, bound, states_str, nu, control_str):
    """Solves a optimal control problem with the multiple shooting method
    """

    def nonlinear_equality(obj0):
        """ Nonlinear equality constraint for the optimization function. The equality is the difference in the final states and the initial
            states in the next partition and the difference between the final states at the final partition and the boundary condition
        """
        global beta_array, t_array, m_tf
        tf = obj0[0]
        points = 100
        # initial states at the beginning of each partition
        ptot0 = np.reshape(obj0[(N+1)*K+1:], [nx, K-1])
        # coefficient polynomials for every partition
        C_0 = np.reshape(obj0[1:(N+1)*K+1], (N+1, K))
        # matrix for the difference of each partition
        E_array = np.empty([K-1, nx])

        for k in range(K):
            if k == 0:
                p0 = initial_states
            else:
                p0 = ptot0[:, k-1]

            sol = solve_ivp(dynamics, [tau[k], tau[k+1]], p0,
                            args=(t0, tf, C_0[:, k]), method='Radau')

            r = sol.y[0]
            vr = sol.y[1]
            theta = sol.y[2]
            vtheta = sol.y[3]
            m = sol.y[4]

            if k < K-1:
                ptots_end = ptot0[:, k]
                eq1k = r[-1] - ptots_end[0]
                eq2k = vr[-1] - ptots_end[1]
                eq3k = theta[-1] - ptots_end[2]
                eq4k = vtheta[-1] - ptots_end[3]
                eq5k = m[-1] - ptots_end[4]
                E_array[k, :] = [eq1k, eq2k, eq3k, eq4k, eq5k]

        m_tf = m[-1]

        beta_array = []
        t_array = []

        eq1 = r[-1] - rf
        eq2 = vr[-1] - vrf
        eq3 = vtheta[-1] - vthetaf
        eqs = [eq1, eq2, eq3]
        print("Final eqs: ", eqs)

        E_array = np.reshape(E_array, nx*(K-1))
        E_array = np.concatenate(
            (E_array, eqs), axis=None)
        return E_array

    def dynamics(t, s, tinitial, tfinal, coeff):
        global beta_array, t_array

        r_t = s[0]
        vr_t = s[1]
        theta_t = s[2]
        vtheta_t = s[3]
        m_t = s[4]

        beta = np.polynomial.polynomial.polyval(
            t, coeff, )  # polynomial of degree N

        beta_array.append(beta)
        t_array.append(t)

        r_dot = vr_t
        vr_dot = -mu/r_t**2 + vtheta_t**2/r_t + T*np.sin(beta_array[-1])/m_t
        theta_dot = vtheta_t/r_t
        vtheta_dot = -vr_t*vtheta_t/r_t + T*np.cos(beta_array[-1])/m_t
        m_dot = -T/ve
        derivatives = np.array([r_dot, vr_dot, theta_dot, vtheta_dot, m_dot])

        return (tfinal-tinitial)*derivatives/2

    def objective(obj, t0):
        """ Cost function for the optimization function, in this problem is the
            maximization of the final mass (scaled to be in interval -1 to 1)
        """
        return -(t0-obj[0])*m_tf/2

    global beta_array, t_array

    beta_array = []
    t_array = []

    # Boundary condition at the final time
    rf = final_states[0]
    vrf = final_states[1]
    vthetaf = final_states[2]

    start = time.time()

    C_guess = coeff_guess*np.ones((N+1, K))

    tau = np.linspace(-1, +1, K+1)
    t0 = 0

    ptot0guess = np.ones([nx, K-1])
    reshaped = np.reshape(ptot0guess, nx*(K-1))

    # Decision vector (final time, initial states at the partitions and coefficients)
    obj_vec = np.hstack(([tf_guess], np.reshape(C_guess, (N+1)*K), reshaped))

    eq_cons = {'type': 'eq', 'fun': nonlinear_equality}

    bounds = np.append([bound[0]], [bound[1], ]*(N+1)
                       * K, axis=0)  # bounds for the coefficients
    bounds = np.append(bounds, [bound[2], ]*nx *
                       (K-1), axis=0)  # bounds for iniital states at each partition

    obj_sol = minimize(objective, obj_vec, method='SLSQP',
                       constraints=eq_cons, options={
                           'ftol': 1e-4, 'disp': True},
                       bounds=tuple(bounds), args=(t0))
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)
    obj_sol = obj_sol.x

    tf = obj_sol[0]

    C = np.reshape(obj_sol[1:(N+1)*K+1], (N+1, K))
    ptot0 = np.reshape(obj_sol[(N+1)*K+1:], [nx, K-1])
    points = 1000
    tau_array = np.empty([K, points])
    control_val = np.empty([K, points])
    states_val = np.empty([K*points, nx])

    for k in range(K):
        if k == 0:
            p0 = initial_states
        else:
            p0 = ptot0[:, k-1]
        tau_eval = np.linspace(tau[k], tau[k+1], points)

        soly = np.zeros((points+1, nx))  # matrix of solution
        soly[0, :] = p0
        dt = (tau[k+1] - tau[k])/points

        for ii, t in enumerate(tau_eval):
            soly[ii+1, :] = soly[ii] + \
                dynamics(t, soly[ii], t0, tf, C[:, k])*dt
        # Collect values of each partition to plot
        control_val[k] = beta_array[0:points]
        tau_array[k] = tau_eval
        states_val[points*k:points + points*k,
                   :] = soly[0:-1, :]
        beta_array = []

    # Scale back time from (-1, 1) to (t0, tf)
    time_s = t0 + (tf-t0)*(np.reshape(tau_array, K*points)+1)/2

    end = time.time()

    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", time_s[-1])
    print("m(tf): ", states_val[-1, -1])

    plot_shooting(time=time_s, states_val=states_val, states_str=states_str,
                  nx=nx, control_val=np.reshape(control_val, K*points), control_str=control_str, nu=nu, control_time=time_s, is_costates=False)


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
    nu = 1

    N = 6  # number of degrees for polynomial

    K = 7

    # Initial conditions
    r0 = 1
    rf = 1.5
    vr0 = 0
    vrf = 0
    theta0 = 0
    vtheta0 = np.sqrt(mu/r0)
    vthetaf = np.sqrt(mu/rf)

    m0 = 1

    # Strings to plot
    states_str = ['$r$', '$v_r$', '$theta$', '$v theta$', 'm']
    control_str = ['$beta$']

    initial_states = [r0, vr0, theta0, vtheta0, m0]
    final_states = [rf, vrf, vthetaf]

    tf_guess = 4
    tf_ub = 5
    tf_lb = 1

    coeff_guess = 2
    coeff_ub = 25
    coeff_lw = -25

    states_ub = 30
    states_lw = 0

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub],
             [states_lw, states_ub]]  # states bounds at the end

    direct_multiple_shooting_method(
        initial_states, final_states, tf_guess, coeff_guess, nx, N, K, bound, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
