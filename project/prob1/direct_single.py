""" Script that solves the optimal control problem for the project using direct single shooting
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
        global u_array, t_array

        tf = obj0[0]

        sol = solve_ivp(dynamics, [0, tf], initial_states,
                        method='Radau', args=[obj0[1:]])
        x = sol.y[0]
        y = sol.y[1]
        theta = sol.y[2]

        u_array = []
        t_array = []

        eq1 = x[-1] - xf
        eq2 = y[-1] - yf
        eq3 = theta[-1] - thetaf
        eqs = [eq1, eq2, eq3]

        print('objective eqs: ', eqs)
        return eqs

    def dynamics(t, s, coeff):
        """Dynamics of the mobile robot"""

        global u_array, t_array

        x_t = s[0]
        y_t = s[1]
        theta_t = s[2]

        u = np.clip(np.polynomial.polynomial.polyval(
            t, coeff), -1, 1)  # polynomial of degree N

        u_array.append(u)
        t_array.append(t)

        x_dot = np.cos(theta_t)
        y_dot = np.sin(theta_t)
        theta_dot = u
        derivatives = np.array([x_dot, y_dot, theta_dot])

        return derivatives

    def objective(obj):
        """ Cost function for the optimization function, in this problem is the
            minimization of the final time 
        """
        tf = obj[0]
        return tf

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

    obj_vec = np.vstack((tf_guess, coeff_guess*np.ones((N+1, 1))),)

    eq_cons = {'type': 'eq', 'fun': nonlinear_equality}

    bounds = np.append(bound, [bound[1], ]*(N), axis=0)
    obj_sol = minimize(objective, obj_vec, method='SLSQP',
                       constraints=eq_cons, options={
                           'ftol': 1e-4, 'disp': True, 'maxiter': 200},
                       bounds=tuple(bounds))
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)
    obj_sol = obj_sol.x

    tf = obj_sol[0]

    # euler forward solver
    points = 1000
    t_eval = np.linspace(0, tf, points)
    soly = np.zeros((points+1, nx))
    soly[0, :] = initial_states
    dt = tf/points
    for ii, t in enumerate(t_eval):
        soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], obj_sol[1:])*dt

    control_val = u_array
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
    nu = 1

    N = 10  # number of degrees for polynomial

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

    tf_guess = [1]
    tf_ub = 10
    tf_lb = 0.2

    coeff_guess = 1
    coeff_ub = 20
    coeff_lw = -5

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub]]

    direct_single_shooting_method(
        initial_states, final_states, tf_guess, coeff_guess, nx, N, bound, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
