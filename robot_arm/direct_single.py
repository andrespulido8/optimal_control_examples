""" Script that solves the optimal control problem for the project problem 2 using direct single shooting
    Author: Andres Pulido
    Date: April 2022
"""

import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def direct_single_shooting_method(initial_states, final_states, tf_guess, coeff_guess, nx, N, bound, states_str, nu, control_str):

    def nonlinear_equality(obj0):
        """ Nonlinear equality constraint for the optimization function. The equality is the difference in the final states
            and the boundary condition
        """
        global u1_array, u2_array, u3_array, t_array, tf

        tf = obj0[0]

        sol = solve_ivp(dynamics, [t0, tf], initial_states,
                        method='Radau', args=[obj0[1:]])

        u1_array = []
        u2_array = []
        u3_array = []
        t_array = []

        eqs = sol.y[:, -1] - final_states

        print('objective eqs: ', eqs)
        return eqs

    def dynamics(t, s, coeff):

        global u1_array, u2_array, u3_array, t_array

        x1_t = s[0]
        x2_t = s[1]
        x3_t = s[2]
        x4_t = s[3]
        x5_t = s[4]
        x6_t = s[5]

        # u1 = np.clip(np.polynomial.polynomial.polyval(
        #    t, coeff[:, 0]), -1, 1)
        u1 = coeff[6]*np.cos(coeff[0]) * coeff[1]
        # u2 = np.clip(np.polynomial.polynomial.polyval(
        #    t, coeff[:, 1]), -1, 1)
        u2 = coeff[7]*np.cos(coeff[2]) * coeff[3]
        # u3 = np.clip(np.polynomial.polynomial.polyval(
        #    t, coeff[:, 2]), -1, 1)
        u3 = coeff[8]*np.cos(coeff[4]) * coeff[5]
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

        return derivatives

    def objective(obj):
        """ Cost function for the optimization function, in this problem is the
            minimization of the final time 
        """
        tf = obj[0]
        return tf

    global u1_array, u2_array, u3_array, t_array

    u1_array = []
    u2_array = []
    u3_array = []
    t_array = []

    t0 = 0

    start = time.time()

    obj_vec = np.hstack(([tf_guess], coeff_guess*np.ones(((N+1)*nu))),)

    eq_cons = {'type': 'eq', 'fun': nonlinear_equality}

    bounds = np.append(bound, [bound[1], ]*((N+1)*nu-1), axis=0)

    obj_sol = minimize(objective, obj_vec, method='SLSQP',
                       constraints=eq_cons, options={
                           'ftol': 1e-4, 'disp': True, 'maxiter': 200},
                       bounds=tuple(bounds))
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)
    obj_sol = obj_sol.x

    tf = obj_sol[0]
    u1_array = []
    u2_array = []
    u3_array = []

    # euler forward solver
    points = 1000
    t_eval = np.linspace(0, tf, points)
    control_val = np.empty([points, nu])
    soly = np.zeros((points+1, nx))
    soly[0, :] = initial_states
    dt = tf/points
    for ii, t in enumerate(t_eval):
        soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], obj_sol[1:])*dt

    control_val[:, 0] = u1_array
    control_val[:, 1] = u2_array
    control_val[:, 2] = u3_array
    states_val = soly[0:-1, :]

    end = time.time()

    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", t_eval[-1])

    plot_shooting(time=t_eval, states_val=states_val, states_str=states_str,
                  nx=nx, control_val=control_val, control_str=control_str, nu=nu, control_time=t_eval, is_costates=False)


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

    N = 2

    L = 5
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
    tf_ub = 8
    tf_lb = 0.2

    coeff_guess = 5
    coeff_ub = 10
    coeff_lw = -10

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub]]  # states bounds at the end

    direct_single_shooting_method(
        initial_states, final_states, tf_guess, coeff_guess, nx, N, bound, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
