""" Script that solves the optimal control problem for the project problem 1 using direct multiple shooting
    Author: Andres Pulido
    Date: April 2022
"""

import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def direct_multiple_shooting_method(initial_states, final_states, states_guess, tf_guess, coeff_guess, nx, N, K, bound, states_str, nu, control_str):
    """Solves a optimal control problem with the multiple shooting method
    """

    def nonlinear_inequality1(obj0):
        return 1 - u

    def nonlinear_inequality2(obj0):
        return u + 1

    def nonlinear_equality(obj0):
        """ Nonlinear equality constraint for the optimization function. The equality is the difference in the final states and the initial
            states in the next partition and the difference between the final states at the final partition and the boundary condition
        """
        global u_array, t_array, tf
        tf = obj0[0]
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

            if k < K-1:
                ptots_end = ptot0[:, k]
                E_array[k, :] = sol.y[:, -1] - ptots_end

        u_array = []
        t_array = []

        eqs = sol.y[:, -1] - final_states
        print("Final eqs: ", eqs)

        E_array = np.reshape(E_array, nx*(K-1))
        E_array = np.concatenate(
            (E_array, eqs), axis=None)
        return E_array

    def dynamics(t, s, tinitial, tfinal, coeff):

        global u_array, t_array, u

        x_t = s[0]
        y_t = s[1]
        theta_t = s[2]

        u = np.clip(np.tanh(coeff[0]) + coeff[1], -1, 1)
        #u = np.polynomial.polynomial.polyval(t, coeff)
        # u = np.clip(np.polynomial.polynomial.polyval(
        #    t, coeff), -1, 1)
        # 1 if np.polynomial.polynomial.polyval(
        #    t, coeff) > 0 else -1  # polynomial of degree N

        u_array.append(u)
        t_array.append(t)

        x_dot = np.cos(theta_t)
        y_dot = np.sin(theta_t)
        theta_dot = u
        derivatives = np.array([x_dot, y_dot, theta_dot])

        return (tfinal-tinitial)*derivatives/2

    def objective(obj, t0):
        """ Cost function for the optimization function, in this problem is the
            maximization of the final mass (scaled to be in interval -1 to 1)
        """
        #
        return (t0-obj[0])*tf/2

    global u_array, t_array

    u_array = []
    t_array = []

    start = time.time()

    C_guess = coeff_guess*np.ones((N+1, K))

    tau = np.linspace(-1, +1, K+1)
    t0 = 0

    ptot0guess = states_guess*np.ones([nx, K-1])
    reshaped = np.reshape(ptot0guess, nx*(K-1))

    # Decision vector (final time, initial states at the partitions and coefficients)
    obj_vec = np.hstack(([tf_guess], np.reshape(C_guess, (N+1)*K), reshaped))

    con1 = {'type': 'eq', 'fun': nonlinear_equality}
    con2 = {'type': 'ineq', 'fun': nonlinear_inequality1}
    con3 = {'type': 'ineq', 'fun': nonlinear_inequality2}
    eq_cons = [con1, con2, con3]

    bounds = np.append([bound[0]], [bound[1], ]*(N+1)
                       * K, axis=0)  # bounds for the coefficients
    bounds = np.append(bounds, [bound[2], ]*nx *
                       (K-1), axis=0)  # bounds for iniital states at each partition

    obj_sol = minimize(objective, obj_vec, method='SLSQP',
                       constraints=eq_cons, options={
                           'ftol': 1e-8, 'disp': True},
                       bounds=tuple(bounds), args=(t0))
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)
    obj_sol = obj_sol.x

    tf_sol = obj_sol[0]

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
                dynamics(t, soly[ii], t0, tf_sol, C[:, k])*dt
        # Collect values of each partition to plot
        control_val[k] = u_array[0:points]
        tau_array[k] = tau_eval
        states_val[points*k:points + points*k,
                   :] = soly[0:-1, :]
        u_array = []

    # Scale back time from (-1, 1) to (t0, tf)
    time_s = t0 + (tf_sol-t0)*(np.reshape(tau_array, K*points)+1)/2

    end = time.time()

    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", time_s[-1])

    plot_shooting(time=time_s, states_val=states_val, states_str=states_str,
                  nx=nx, control_val=np.reshape(control_val, K*points), control_str=control_str, nu=nu, control_time=time_s, is_costates=False)


def plot_shooting(time, states_val, states_str, nx, control_val, control_str, nu, control_time, is_costates=False):
    """ Plots all states, all costates (if is_costate == True) and the 'u' control.
        nx - number of states
        nu - number of controls
    """
    fig1, axs1 = plt.subplots(nx)
    fig1.suptitle("State evolution of {} ".format(states_str[0:nx]))
    fig2, axs2 = plt.subplots(nu)
    fig2.suptitle("Control evolution of {} ".format(control_str))
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

    plt.plot(states_val[:, 0], states_val[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    nx = 3  # number of states
    nu = 1

    N = 1  # number of degrees for polynomial

    K = 8

    # Initial conditions
    x0 = 0
    y0 = 0
    theta0 = -np.pi
    xf = 0
    yf = 0
    thetaf = np.pi

    states_str = ['$x$', '$y$', '$theta$']
    control_str = ['$u$']
    initial_states = [x0, y0, theta0]
    final_states = [xf, yf, thetaf]

    tf_guess = 5
    tf_ub = 8
    tf_lb = 3

    coeff_guess = 0.5
    coeff_ub = 1
    coeff_lw = -1

    states_guess = 0.1
    states_ub = 3.5
    states_lw = -3.5

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub],
             [states_lw, states_ub]]  # states bounds at the end

    direct_multiple_shooting_method(
        initial_states, final_states, states_guess, tf_guess, coeff_guess, nx, N, K, bound, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
