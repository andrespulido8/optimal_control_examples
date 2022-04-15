from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def brachistocrone_shooting_method(initial_states, final_states, guesses):

    def objective(obj0):
        global theta_array
        tf = obj0[3]
        t_eval = np.linspace(0, tf, 1000)

        sol = solve_ivp(F1, [0, tf],
                        [x0, y0, v0, obj0[0], obj0[1], obj0[2]], t_eval=t_eval)
        x = sol.y[0]
        y = sol.y[1]
        v = sol.y[2]
        lambx = sol.y[3]
        lamby = sol.y[4]
        lambv = sol.y[5]

        H = lambx[-1]*v[-1]*np.sin(theta_array[-1]) + lamby[-1] * \
            v[-1]*np.cos(theta_array[-1]) + lambv[-1]*g*np.cos(theta_array[-1])
        theta_array = []

        eq1 = x[-1] - xf
        eq2 = y[-1] - yf
        eq3 = lambv[-1]
        eq4 = H + 1
        print([eq1, eq2, eq3, eq4])
        return [eq1, eq2, eq3, eq4]

    def F1(t, s):
        global theta, theta_array

        v_t = s[2]
        lambx_t = s[3]
        lamby_t = s[4]
        lambv_t = s[5]

        def solve_control(th):
            ht = lambx_t*v_t*np.cos(th) - (lamby_t*v_t + lambv_t*g)*np.sin(th)
            # print(ht)
            return ht

        theta, = fsolve(
            solve_control, theta_guess)

        theta_array.append(theta)

        x_dot = v_t*np.sin(theta)
        y_dot = v_t*np.cos(theta)
        v_dot = g*np.cos(theta)
        lambx_dot = 0
        lamby_dot = 0
        lambv_dot = -s[3]*np.cos(theta) - s[4]*np.cos(theta)
        return [x_dot, y_dot, v_dot, lambx_dot, lamby_dot, lambv_dot]

    global x0, y0, v0, xf, yf, theta_guess, theta_array

    theta_guess = guesses[4]
    theta_array = []

    xf = final_states[0]
    yf = final_states[1]
    x0 = initial_states[0]
    y0 = initial_states[1]
    v0 = initial_states[2]

    obj_sol = fsolve(objective, guesses[0:4])

    # x0, y0, v0, lambx0, lamby0, lambv0
    sol_initial_states = [x0, y0, v0, obj_sol[0], obj_sol[1], obj_sol[2]]

    tf = obj_sol[3]
    t_eval = np.linspace(0, tf, 100)

    sol = solve_ivp(F1, [0, tf], sol_initial_states, t_eval=t_eval)

    print("theta array shape: ", len(theta_array))  # not used right now
    plt.figure(figsize=(10, 8))
    plt.plot(sol.y[0], -1*sol.y[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Brachistocrone Curve')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(sol.t, sol.y[0], label='x')
    plt.plot(sol.t, sol.y[1], label='y')
    plt.plot(sol.t, sol.y[2], label='v')
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

    initial_states = [x0, y0, v0]
    final_states = [xf, yf]

    lamb0_guess = [1, 1, 1]
    tf_guess = [1]
    theta_guess = [0.5]

    guesses = lamb0_guess + tf_guess + theta_guess  # list of element guesses

    brachistocrone_shooting_method(initial_states, final_states, guesses)


if __name__ == "__main__":
    main()
    print("Done")
