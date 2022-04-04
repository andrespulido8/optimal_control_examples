from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def direct_single_shooting_method(initial_states, final_states, tf_guess, coeff_guess, nx, N, bound):

    def nonlinear_inequality(obj0):
        global beta_array, t_array, m_tf
        tf = obj0[0]
        t_eval = np.linspace(0, tf, 1000)

        # solve_ivp solver
        # sol = solve_ivp(dynamics, [0, tf], initial_states, t_eval=t_eval, method='Radau', args=(obj_vec[1:],))
        # r = sol.y[0]
        # vr = sol.y[1]
        # theta = sol.y[2]
        # vtheta = sol.y[3]
        # m = sol.y[4]

        # euler forward solver
        soly = np.zeros((1000+1, nx))
        soly[0, :] = initial_states
        dt = tf/1000
        for ii, t in enumerate(t_eval):
            soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], obj0[1:])*dt

        # soly = odeint(dynamics, initial_states, t_eval, tfirst=True, printmessg=1)
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
        tf = obj[0]
        t_eval = np.linspace(0, tf, 1000)

        # euler forward solver
        soly = np.zeros((1000+1, nx))
        soly[0, :] = initial_states
        dt = tf/1000
        for ii, t in enumerate(t_eval):
            soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], obj[1:])*dt

        m_tf = soly[:, 4][-1]

        # return -obj_vec[0]
        #print("Coefficients: ", obj[1:])
        #print('tf: ', obj[0])
        return -m_tf

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

    obj_vec = np.vstack((tf_guess, coeff_guess*np.ones((N+1, 1))),)

    ineq_cons = {'type': 'ineq',
                 'fun': []}

    eq_cons = {'type': 'eq', 'fun': nonlinear_inequality}

    bounds = np.append(bound, [bound[1], ]*(N), axis=0)
    obj_sol = minimize(objective, obj_vec, method='SLSQP',
                       constraints=eq_cons, options={
                           'ftol': 1e-4, 'disp': True},
                       bounds=tuple(bounds))
    # r0, vr0, theta0, vtheta, m
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)
    obj_sol = obj_sol.x

    tf = obj_sol[0]
    t_eval = np.linspace(0, tf, 100)

    sol = solve_ivp(dynamics, [0, tf], initial_states,
                    t_eval=t_eval, args=(obj_sol[1:],))

    print("beta array shape: ", len(beta_array))
    print("max beta: ", max(beta_array))
    control = np.array([t_array, np.degrees(beta_array)])
    sortedCon = control[:, control[0].argsort()]

    plt.figure(figsize=(10, 8))
    plt.plot(sortedCon[0], sortedCon[1])
    plt.xlabel('t')
    plt.ylabel('Beta')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(sol.t, sol.y[0], label='r')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.title(f'States')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.plot(sol.t, np.degrees(sol.y[2]), label='theta')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.axes(projection='polar')
    plt.polar(sol.y[2], sol.y[0])
    plt.title(f'Trajectory Curve')
    plt.show()


def main():
    global mu, T, ve
    mu = 1
    T = 0.1405
    ve = 1.8758344

    nx = 5  # number of states

    N = 10  # number of degrees for polynomial

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

    tf_guess = [4]
    tf_ub = 5
    tf_lb = 1

    coeff_guess = 2
    coeff_ub = 25
    coeff_lw = -25

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub]]

    direct_single_shooting_method(
        initial_states, final_states, tf_guess, coeff_guess, nx, N, bound)


if __name__ == "__main__":
    main()
    print("Done")
