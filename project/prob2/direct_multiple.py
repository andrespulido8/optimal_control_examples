""" Script that solves the optimal control problem for the project problem 2 using direct multiple shooting
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

    def nonlinear_equality(obj0):
        """ Nonlinear equality constraint for the optimization function. The equality is the difference in the final states and the initial
            states in the next partition and the difference between the final states at the final partition and the boundary condition
        """
        global u1_array, u2_array, u3_array, t_array, tf
        tf = obj0[0]
        # coefficient polynomials for every partition
        C_0 = np.reshape(obj0[1:(N+1)*K*nu+1], (N+1, K, nu))
        # initial states at the beginning of each partition
        ptot0 = np.reshape(obj0[(N+1)*K*nu+1:], [nx, K-1])
        # matrix for the difference of each partition
        E_array = np.empty([K-1, nx])

        for k in range(K):
            if k == 0:
                p0 = initial_states
            else:
                p0 = ptot0[:, k-1]

            sol = solve_ivp(dynamics, [tau[k], tau[k+1]], p0,
                            args=(t0, tf, C_0[:, k, :]), method='Radau')

            if k < K-1:
                ptots_end = ptot0[:, k]
                E_array[k, :] = sol.y[:, -1] - ptots_end

        u1_array = []
        u2_array = []
        u3_array = []
        t_array = []

        eqs = sol.y[:, -1] - final_states
        print("Final eqs: ", eqs)

        E_array = np.reshape(E_array, nx*(K-1))
        E_array = np.concatenate(
            (E_array, eqs), axis=None)
        return E_array

    def dynamics(t, s, tinitial, tfinal, coeff):

        global u1_array, u2_array, u3_array, t_array

        x1_t = s[0]
        x2_t = s[1]
        x3_t = s[2]
        x4_t = s[3]
        x5_t = s[4]
        x6_t = s[5]

        # u1 = np.clip(np.polynomial.polynomial.polyval(
        #    t, coeff[:, 0]), -1, 1)
        u1 = np.tanh(coeff[0, 0]) * coeff[1, 0]
        # u2 = np.clip(np.polynomial.polynomial.polyval(
        #    t, coeff[:, 1]), -1, 1)
        u2 = np.tanh(coeff[0, 1]) * coeff[1, 1]
        # u3 = np.clip(np.polynomial.polynomial.polyval(
        #    t, coeff[:, 2]), -1, 1)
        u3 = np.tanh(coeff[0, 2]) * coeff[1, 2]
        # 1 if np.polynomial.polynomial.polyval(
        #    t, coeff) > 0 else -1  # polynomial of degree N

        u1_array.append(u1)
        u2_array.append(u2)
        u3_array.append(u3)
        t_array.append(t)

        I_phi = ((L-x1_t)**3 + x1_t**3)/3
        I_theta = I_phi*np.sin(x5_t)**2

        x1_dot = x2_t
        x2_dot = u1/L
        x3_dot = x4_t
        x4_dot = u2/I_theta
        x5_dot = x6_t
        x6_dot = u3/I_phi

        derivatives = np.array(
            [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot])

        return (tfinal-tinitial)*derivatives/2

    def objective(obj, t0):
        """ Cost function for the optimization function, in this problem is the
            maximization of the final mass (scaled to be in interval -1 to 1)
        """
        #
        return (t0-obj[0])*tf/2

    global u1_array, u2_array, u3_array, t_array

    u1_array = []
    u2_array = []
    u3_array = []
    t_array = []

    start = time.time()

    C_guess = coeff_guess*np.ones((N+1, K, nu))

    tau = np.linspace(-1, +1, K+1)
    t0 = 0

    ptot0guess = states_guess*np.ones([nx, K-1])
    reshaped = np.reshape(ptot0guess, nx*(K-1))

    # Decision vector (final time, initial states at the partitions and coefficients)
    obj_vec = np.hstack(
        ([tf_guess], np.reshape(C_guess, (N+1)*K*nu), reshaped))

    eq_cons = {'type': 'eq', 'fun': nonlinear_equality}

    bounds = np.append([bound[0]], [bound[1], ]*(N+1)
                       * K * nu, axis=0)  # bounds for the coefficients
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

    tf_sol = obj_sol[0]

    C = np.reshape(obj_sol[1:(N+1)*K*nu+1], (N+1, K, nu))
    ptot0 = np.reshape(obj_sol[(N+1)*K*nu+1:], [nx, K-1])
    points = 1000
    tau_array = np.empty([K, points])
    control_val = np.empty([K, points, nu])
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
        control_val[k, :, 0] = u1_array[0:points]
        control_val[k, :, 1] = u2_array[0:points]
        control_val[k, :, 2] = u3_array[0:points]
        tau_array[k] = tau_eval
        states_val[points*k:points + points*k,
                   :] = soly[0:-1, :]
        u1_array = []
        u2_array = []
        u3_array = []

    # Scale back time from (-1, 1) to (t0, tf)
    time_s = t0 + (tf_sol-t0)*(np.reshape(tau_array, K*points)+1)/2

    end = time.time()

    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", time_s[-1])

    plot_shooting(time=time_s, states_val=states_val, states_str=states_str,
                  nx=nx, control_val=np.reshape(control_val, (K*points, nu)), control_str=control_str, nu=nu, control_time=time_s, is_costates=False)


def plot_shooting(time, states_val, states_str, nx, control_val, control_str, nu, control_time, is_costates=False):
    """ Plots all states, all costates (if is_costate == True), the beta control and the orbit transfer in polar coord.
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
    for ii in range(nu):
        axs2[ii].plot(control_time, control_val[:, ii])
        axs2[ii].set_ylabel(control_str[ii])
    plt.xlabel("time [s]")
    plt.show()


def main():
    global L
    nx = 6  # number of states
    nu = 3

    L = 5  # length of link

    N = 1  # number of degrees for polynomial

    K = 15

    # Initial conditions
    x1_0 = 4.5
    x2_0 = 0
    x3_0 = 0
    x4_0 = 0
    x5_0 = np.pi/4
    x6_0 = 0
    # Final conditions
    x1_f = 4.5
    x2_f = 0
    x3_f = 2*np.pi/3
    x4_f = 0
    x5_f = np.pi/4
    x6_f = 0

    states_str = ['$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_5$', '$x_6$']
    control_str = ['$u_1$', '$u_2$', '$u_3$']
    initial_states = [x1_0, x2_0, x3_0, x4_0, x5_0, x6_0]
    final_states = [x1_f, x2_f, x3_f, x4_f, x5_f, x6_f]

    tf_guess = 5
    tf_ub = 10
    tf_lb = 0.2

    coeff_guess = 0.5
    coeff_ub = 1
    coeff_lw = -1

    states_guess = 0.2
    states_ub = 3.5
    states_lw = -3.5

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub],
             [states_lw, states_ub]]  # states bounds at the end

    direct_multiple_shooting_method(
        initial_states, final_states, states_guess, tf_guess, coeff_guess, nx, N, K, bound, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
