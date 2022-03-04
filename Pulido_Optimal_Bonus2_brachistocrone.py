from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def multiple_shooting(initial_states, final_states, guesses, K):

    def objective(obj0):
        global theta_array
        tf = obj0[3]
        ptot0 = np.reshape(obj0[4:], [2*nx, K-1])
        points = 100
        tau_array = np.empty([K, points])
        states = np.empty([K, points])
        E_array = np.empty([K-1, nx*2])

        for k in range(K):
            if k == 0:
                p0 = [x0, y0, v0, obj0[0], obj0[1], obj0[2]]
            else:
                p0 = ptot0[:, k-1]

            tau_eval = np.linspace(tau[k], tau[k+1], points)

            sol = solve_ivp(dynamics, [tau_eval[0], tau_eval[-1]],
                            p0, t_eval=tau_eval, args=(t0, tf), dense_output=True)

            x = sol.y[0]
            y = sol.y[1]
            v = sol.y[2]
            lambx = sol.y[3]
            lamby = sol.y[4]
            lambv = sol.y[5]

            if k < K-1:
                ptots_end = ptot0[:, k]
                eq1 = x[-1] - ptots_end[0]
                eq2 = y[-1] - ptots_end[1]
                eq3 = v[-1] - ptots_end[2]
                eq4 = lambx[-1] - ptots_end[3]
                eq5 = lamby[-1] - ptots_end[4]
                eq6 = lambv[-1] - ptots_end[5]
                E_array[k, :] = [eq1, eq2, eq3, eq4, eq5, eq6]
                #print([eq1, eq2, eq3, eq4])

        H = lambx[-1]*v[-1]*np.sin(theta_array[-1]) + lamby[-1] * \
            v[-1]*np.cos(theta_array[-1]) + lambv[-1] * \
            g*np.cos(theta_array[-1])

        theta_array = []  # restart vector

        eq1f = x[-1] - xf
        eq2f = y[-1] - yf
        eq3f = lambv[-1]
        eq4f = H + 1
        E_array = np.reshape(E_array, 2*nx*(K-1))
        E_array = np.concatenate((E_array, [eq1, eq2, eq3, eq4]), axis=None)

        return E_array

    def dynamics(t, s, tinitial, tfinal):
        global theta, theta_array

        v_t = s[2]
        lambx_t = s[3]
        lamby_t = s[4]
        lambv_t = s[5]

        def solve_control(th):
            ht = lambx_t*v_t*np.cos(th) - (lamby_t*v_t + lambv_t*g)*np.sin(th)
            # print(ht)
            return ht

        theta, = fsolve(solve_control, theta_guess)
        theta_array.append(theta)

        x_dot = v_t*np.sin(theta)
        y_dot = v_t*np.cos(theta)
        v_dot = g*np.cos(theta)
        lambx_dot = 0
        lamby_dot = 0
        lambv_dot = -s[3]*np.cos(theta) - s[4]*np.cos(theta)
        return (tfinal-tinitial)*np.array([x_dot, y_dot, v_dot, lambx_dot, lamby_dot, lambv_dot])/2

    global x0, y0, v0, xf, yf, theta_guess, theta_array

    theta_guess = guesses[4]
    theta_array = []

    xf = final_states[0]
    yf = final_states[1]
    x0 = initial_states[0]
    y0 = initial_states[1]
    v0 = initial_states[2]

    t0 = 0

    nx = len(initial_states)

    tau = np.linspace(-1, +1, K+1)

    ptot0guess = np.zeros([2*nx, K-1])
    ptot0guess[1, 0] = 5
    ptot0guess[1, 1] = 8
    reshaped = np.reshape(ptot0guess, 2*nx*(K-1))
    changing = np.concatenate([guesses[0:4], reshaped])

    obj_sol, obj_dict, ier, mesg = fsolve(objective, changing, xtol=10e-8,
                                          full_output=True)

    print("Solution found? ", ier)
    print("msg: ", mesg)
    print("n func calls: ", obj_dict["nfev"])

    tf = obj_sol[3]
    ptot0 = np.reshape(obj_sol[4:], [2*nx, K-1])
    points = 100
    tau_array = np.empty([K, points])
    y_array = np.empty([K, points])
    x_array = np.empty([K, points])
    for k in range(K):
        if k == 0:
            p0 = [x0, y0, v0, obj_sol[0], obj_sol[1], obj_sol[2]]
        else:
            p0 = ptot0[:, k-1]
        tau_eval = np.linspace(tau[k], tau[k+1], points)
        sol = solve_ivp(dynamics, [tau_eval[0], tau_eval[-1]],
                        p0, t_eval=tau_eval, args=(t0, tf), dense_output=True)
        tau_array[k] = sol.t
        x_array[k] = sol.y[0]
        y_array[k] = sol.y[1]

    plt.figure(figsize=(10, 8))
    plt.plot(np.reshape(x_array, K*points), -1*np.reshape(y_array, K*points))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Brachistocrone Curve')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(np.reshape(tau_array, K*points),
             np.reshape(x_array, K*points), label='x')
    plt.plot(np.reshape(tau_array, K*points),
             np.reshape(y_array, K*points), label='y')
    #plt.plot(sol.t, sol.y[2], label='v')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.title(f'States')
    plt.legend()
    plt.show()


def main():
    global g
    g = 10

    x0 = 0
    xf = 2
    y0 = 0
    yf = 2
    v0 = 0
    #vf = free

    K = 3

    initial_states = [x0, y0, v0]
    final_states = [xf, yf]

    lamb0_guess = [1.2, 0.8, 1]
    tf_guess = [0.9]
    theta_guess = [0.5]

    guesses = lamb0_guess + tf_guess + theta_guess  # list of element guesses

    multiple_shooting(initial_states, final_states, guesses, K)


if __name__ == "__main__":
    main()
    print("Done")
