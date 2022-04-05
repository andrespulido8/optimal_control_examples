from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def direct_multiple_shooting_method(initial_states, final_states, tf_guess, coeff_guess, nx, N, K, bound):

    def nonlinear_inequality(obj0):
        global beta_array, t_array, m_tf
        tf = obj0[0]

        ptot0 = np.reshape(obj0[(N+1)*K+1:], [nx, K-1])
        C_0 = np.reshape(obj0[1:(N+1)*K+1], (N+1, K))
        E_array = np.empty([K-1, nx])

        for k in range(K):
            if k == 0:
                p0 = initial_states
            else:
                p0 = ptot0[:, k-1]

            #tau_eval = np.linspace(tau[k], tau[k+1], points)

            # euler forward solver
            #soly = np.zeros((1000+1, nx))
            #soly[0, :] = initial_states
            #dt = tf/1000
            # for ii, t in enumerate(t_eval):
            #    soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], obj0[1:])*dt
            #r = soly[:, 0]
            #vr = soly[:, 1]
            #vtheta = soly[:, 3]
            #m = soly[:, 4]

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

        beta = np.polynomial.polynomial.polyval(t, coeff, )

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
        #t_eval = np.linspace(0, tf, 1000)

        # euler forward solver
        #soly = np.zeros((1000+1, nx))
        #soly[0, :] = initial_states
        #dt = tf/1000
        # for ii, t in enumerate(t_eval):
        #    soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii], obj[1:])*dt

        #m_tf = soly[:, 4][-1]

        # return -obj_vec[0]
        #print("Coefficients: ", obj[1:])
        #print('tf: ', obj[0])
        return -(t0-obj[0])*m_tf/2

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

    C_guess = coeff_guess*np.ones((N+1, K))

    tau = np.linspace(-1, +1, K+1)
    t0 = 0

    ptot0guess = np.ones([nx, K-1])
    reshaped = np.reshape(ptot0guess, nx*(K-1))
    cres = np.reshape(C_guess, (N+1)*K)
    obj_vec = np.hstack((tf_guess, cres, reshaped))

    ineq_cons = {'type': 'ineq',
                 'fun': []}

    eq_cons = {'type': 'eq', 'fun': nonlinear_inequality}

    bounds = np.append([bound[0]], [bound[1], ]*(N+1)
                       * K, axis=0)  # bounds for c
    bounds = np.append(bounds, [bound[2], ]*nx *
                       (K-1), axis=0)  # bounds for states
    obj_sol = minimize(objective, obj_vec, method='SLSQP',
                       constraints=eq_cons, options={
                           'ftol': 1e-4, 'disp': True},
                       bounds=tuple(bounds), args=(t0))
    # r0, vr0, theta0, vtheta, m
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)
    obj_sol = obj_sol.x

    tf = obj_sol[0]

    C = np.reshape(obj_sol[1:(N+1)*K+1], (N+1, K))
    ptot0 = np.reshape(obj_sol[(N+1)*K+1:], [nx, K-1])
    points = 1000
    tau_array = np.empty([K, points])
    r_array = np.empty([K, points])
    theta_array = np.empty([K, points])

    for k in range(K):
        if k == 0:
            p0 = initial_states
        else:
            p0 = ptot0[:, k-1]
        tau_eval = np.linspace(tau[k], tau[k+1], points)
        sol = solve_ivp(dynamics, [tau_eval[0], tau_eval[-1]],
                        p0, t_eval=tau_eval, args=(t0, tf, C[:, k]), dense_output=True)
        tau_array[k] = sol.t
        r_array[k] = sol.y[0]
        theta_array[k] = sol.y[2]

    print("beta array shape: ", len(beta_array))
    print("max beta: ", max(beta_array))
    #control = np.array([t_array, np.degrees(beta_array)])
    #sortedCon = control[:, control[0].argsort()]

    #plt.figure(figsize=(10, 8))
    #plt.plot(sortedCon[0], sortedCon[1])
    # plt.xlabel('t')
    # plt.ylabel('Beta')
    # plt.legend()
    # plt.show()

    #plt.figure(figsize=(10, 8))
    #plt.plot(sol.t, sol.y[0], label='r')
    # plt.xlabel('t')
    # plt.ylabel('Value')
    # plt.title(f'States')
    # plt.legend()
    # plt.show()
    #plt.figure(figsize=(10, 8))
    #plt.plot(sol.t, np.degrees(sol.y[2]), label='theta')
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(10, 8))
    plt.axes(projection='polar')
    plt.polar(np.reshape(theta_array, K*points), np.reshape(r_array, K*points))
    plt.title(f'Trajectory Curve')
    plt.show()


def main():
    global mu, T, ve
    mu = 1
    T = 0.1405
    ve = 1.8758344

    nx = 5  # number of states

    N = 3  # number of degrees for polynomial

    K = 10

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

    bound = [[tf_lb, tf_ub], [coeff_lw, coeff_ub],
             [0, 30]]  # states bounds at the end

    direct_multiple_shooting_method(
        initial_states, final_states, tf_guess, coeff_guess, nx, N, K, bound)


if __name__ == "__main__":
    main()
    print("Done")
