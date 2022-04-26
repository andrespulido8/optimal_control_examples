""" Script that solves the optimal control problem for the Robot Arm problem using indirect single shooting
    Author: Andres Pulido
    Date: April 2022
"""

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

        global u1_array, u2_array, u3_array, t_array
        tf = obj0[nx]
        t_eval = np.linspace(0, tf, 100)
        in0 = np.concatenate((initial_states, obj0[:-1]))

        # solve_ivp solver
        sol = solve_ivp(dynamics, [0, tf], in0, t_eval=t_eval, method='Radau',)

        q_end = sol.y[:, -1]
        # Dot product of lambda and the derivatives of the states
        H_tf = np.dot(q_end[nx:], dynamics(tf, q_end)[:nx])

        u1_array = []
        u2_array = []
        u3_array = []
        t_array = []

        eqs = np.empty(nx+1)
        eqs[:nx] = q_end[:nx] - final_states
        eqs[-1] = H_tf + 1

        # include value at end of list
        print('objective eqs: ', eqs)
        return eqs

    def dynamics(t, s):
        """Dynamics of the robot arm"""

        global u1_array, u2_array, u3_array, t_array

        x1_t = s[0]
        x2_t = s[1]
        x3_t = s[2]
        x4_t = s[3]
        x5_t = s[4]
        x6_t = s[5]
        lambx1_t = s[6]
        lambx2_t = s[7]
        lambx3_t = s[8]
        lambx4_t = s[9]
        lambx5_t = s[10]
        lambx6_t = s[11]

        u1 = 1 if lambx2_t < 0 else -1
        u2 = 1 if lambx4_t < 0 else -1
        u3 = 1 if lambx6_t < 0 else -1

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
        lambx1_dot = -9*(lambx4_t*u2 + lambx6_t*u3/(np.sin(x5_t)**2)) * \
            ((L-2*x2_t)/(L*(3*x1_t**2 - 3*x1_t*L + L**2)**2))
        lambx2_dot = -lambx1_t
        lambx3_dot = 0
        lambx4_dot = -lambx3_t
        lambx5_dot = 2*lambx4_t*u2/(np.tan(x5_t)*I_phi*np.sin(x5_t)**2)
        lambx6_dot = -lambx5_t

        derivatives = np.array(
            [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, lambx1_dot, lambx2_dot, lambx3_dot, lambx4_dot, lambx5_dot, lambx6_dot])

        return derivatives

    global u1_array, u2_array, u3_array, t_array

    u1_array = []
    u2_array = []
    u3_array = []
    t_array = []

    start = time.time()

    obj_sol = root(objective, guesses[0:-1], method="hybr", tol=1e-5,)
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)

    obj_sol = obj_sol.x

    _ = objective(obj_sol)

    sol_initial_states = np.concatenate((initial_states, obj_sol[:-1]))

    tf = obj_sol[nx]
    t_eval = np.linspace(0, tf, 1000)

    control_val = np.empty([1000, nu])

    # euler forward solver
    soly = np.zeros((1000+1, 2*nx))
    soly[0, :] = sol_initial_states
    dt = tf/1000
    for ii, t in enumerate(t_eval):
        soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii])*dt

    control_val[:, 0] = u1_array
    control_val[:, 1] = u2_array
    control_val[:, 2] = u3_array
    control_time = np.array(t_array)

    # states_val = np.transpose(sol.y)
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

    states_str = ['$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_5$', '$x_6$',
                  '$lam x_1$', '$lam x_2$', '$lam x_3$', '$lam x_4$', '$lam x_5$', '$lam x_6$']
    control_str = ['$u_1$', '$u_2$', '$u_3$']
    initial_states = [x1_0, x2_0, x3_0, x4_0, x5_0, x6_0]
    final_states = [x1_f, x2_f, x3_f, x4_f, x5_f, x6_f]

    lamb0_guess = [-8, -20, 0.1, -2, 5, 5]
    tf_guess = [10]
    u_guess = [0.5]

    guesses = lamb0_guess + tf_guess + u_guess  # list of element guesses

    indirect_single_shooting(
        initial_states, final_states, guesses, nx, states_str, nu, control_str)


if __name__ == "__main__":
    main()
    print("Done")
