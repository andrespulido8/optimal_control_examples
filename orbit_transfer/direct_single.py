""" Script that solves the optimal control problem for the Orbit-Transfer problem using direct single shooting
    Author: Andres Pulido
    Date: April 2022
"""

import time
from scipy.optimize import minimize
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def direct_single_shooting_method(initial_states, final_states, tf_guess, coeff_guess, nx, N, bound, states_str, nu, control_str):

    def nonlinear_equality(obj0):
        """ Nonlinear equality constraint for the optimization function. The equality is the difference in the final states
            and the boundary condition
        """
        global beta_array, t_array, m_tf
        tf = obj0[0]
        t_eval = np.linspace(0, tf, 1000)

        # euler forward solver
        soly = np.zeros((1000+1, nx))
        soly[0, :] = initial_states
        dt = tf/1000
        for ii, t in enumerate(t_eval):
            soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], obj0[1:])*dt

        r = soly[:, 0]
        vr = soly[:, 1]
        vtheta = soly[:, 3]
        m = soly[:, 4]

        m_tf = m[-1]

        beta_array = []
        t_array = []

        eq1 = r[-1] - rf
        eq2 = vr[-1] - vrf
        eq3 = vtheta[-1] - vthetaf
        eqs = [eq1, eq2, eq3]

        print('objective eqs: ', eqs)
        return eqs

    def dynamics(t, s, coeff):
        global beta_array, t_array

        r_t = s[0]
        vr_t = s[1]
        theta_t = s[2]
        vtheta_t = s[3]
        m_t = s[4]

        beta = np.polynomial.polynomial.polyval(t, coeff, )

        beta_array.append(beta)
        t_array.append(t)

        r_dot = vr_t
        vr_dot = -mu/r_t**2 + vtheta_t**2/r_t + T*np.sin(beta_array[-1])/m_t
        theta_dot = vtheta_t/r_t
        vtheta_dot = -vr_t*vtheta_t/r_t + T*np.cos(beta_array[-1])/m_t
        m_dot = -T/ve

        return np.array([r_dot, vr_dot, theta_dot, vtheta_dot, m_dot])

    def objective(obj):
        """ Cost function for the optimization function, in this problem is the
            maximization of the final mass 
        """
        return -m_tf

    global beta_array, t_array

    beta_array = []
    t_array = []

    rf = final_states[0]
    vrf = final_states[1]
    vthetaf = final_states[2]

    start = time.time()

    obj_vec = np.vstack(([tf_guess], coeff_guess*np.ones((N+1, 1))),)

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

    control_val = np.degrees(beta_array)
    states_val = soly[0:-1, :]

    end = time.time()

    print('Elapsed time: ', end - start,
          'seconds, or: ', (end-start)/60, 'minutes')
    print("tf: ", t_eval[-1])
    print("m(tf): ", states_val[-1, -1])

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

    N = 19  # number of degrees for polynomial

    # Initial conditions
    r0 = 1
    rf = 1.5
    vr0 = 0
    vrf = 0
    theta0 = 0
    vtheta0 = np.sqrt(mu/r0)
    vthetaf = np.sqrt(mu/rf)

    m0 = 1

    states_str = ['$r$', '$v_r$', '$theta$', '$v_theta$', 'm']
    control_str = ['$beta$']
    initial_states = [r0, vr0, theta0, vtheta0, m0]
    final_states = [rf, vrf, vthetaf]

    tf_guess = 4
    tf_ub = 5
    tf_lb = 1

    coeff_guess = 2
    coeff_ub = 25
    coeff_lw = -25

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub]]

    direct_single_shooting_method(
        initial_states, final_states, tf_guess, coeff_guess, nx, N, bound, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
