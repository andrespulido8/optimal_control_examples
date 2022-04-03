from email import message
from re import S
from scipy.optimize import root
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def indirect_single_shooting_method(initial_states, final_states, guesses, nx):

    def objective(obj0):
        global beta_array, t_array
        tf = obj0[nx]
        t_eval = np.linspace(0, tf, 1000)
        x0 = [r0, vr0, theta0, vtheta0, m0, obj0[0],
              obj0[1], obj0[2], obj0[3], obj0[4]]

        # solve_ivp solver
        sol = solve_ivp(dynamics, [0, tf], x0, t_eval=t_eval, method='Radau')
        r = sol.y[0]
        vr = sol.y[1]
        theta = sol.y[2]
        vtheta = sol.y[3]
        m = sol.y[4]
        lambr = sol.y[5]
        lambvr = sol.y[6]
        lambtheta = sol.y[7]
        lambvtheta = sol.y[8]
        lambm = sol.y[9]

        # euler forward solver
        #soly = np.zeros((1000+1, 2*nx))
        #soly[0, :] = x0
        #dt = tf/1000
        # for ii, t in enumerate(t_eval):
        #    soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii])*dt

        ##soly = odeint(dynamics, x0, t_eval, tfirst=True, printmessg=1)
        #r = soly[:, 0]
        #vr = soly[:, 1]
        #theta = soly[:, 2]
        #vtheta = soly[:, 3]
        #m = soly[:, 4]
        #lambr = soly[:, 5]
        #lambvr = soly[:, 6]
        #lambtheta = soly[:, 7]
        #lambvtheta = soly[:, 8]
        #lambm = soly[:, 9]

        # H = lambr[-1]*vr[-1] + lambvr[-1]*(-mu/r[-1] + vtheta[-1]**2/r[-1] + T*np.sin(beta_array[-1])/m[-1]) + lambtheta[-1] * \
        #    vtheta[-1]/r[-1] + lambvtheta[-1] * \
        #    (-vr[-1]*vtheta[-1]/r[-1] + T *
        #     np.cos(beta_array[-1])/m[-1]) - lambm[-1]*T/ve

        x = np.array([r[-1], vr[-1], theta[-1], theta[-1], m[-1]])
        lamb_tf = np.array(
            [lambr[-1], lambvr[-1], lambtheta[-1], lambvtheta[-1], lambm[-1]])
        comb = np.concatenate((x, lamb_tf), axis=0)
        H = np.matmul(lamb_tf, dynamics(tf, comb)[5:])

        beta_array = []
        t_array = []

        eq1 = r[-1] - rf
        eq2 = vr[-1] - vrf
        eq3 = vtheta[-1] - vthetaf
        eq4 = H
        eq5 = lambtheta[-1]
        eq6 = lambm[-1] - 1
        eqs = [eq1, eq2, eq3, eq4, eq5, eq6]
        print('objective eqs: ', eqs)
        return eqs

    def dynamics(t, s):
        global beta_array, t_array

        r_t = s[0]
        vr_t = s[1]
        theta_t = s[2]
        vtheta_t = s[3]
        m_t = s[4]
        lambr_t = s[5]
        lambvr_t = s[6]
        lambtheta_t = s[7]
        lambvtheta_t = s[8]
        lambm_t = s[9]

        def solve_control(bt):
            Hbeta = lambvr_t*T*np.cos(bt)/m_t - lambvtheta_t*T*np.sin(bt)/m_t
            # print(bt)
            return Hbeta

        #full_output = False
        # beta = root(
        #    solve_control, beta_guess, tol=1e-6)
        # if full_output:
        #    print("Theta solution found? ",
        #          "yes!" if beta.success == 1 else "No :(")
        #    print("msg: ", beta.message)
        # beta = beta.x[0]  # np.clip(beta.x, -np.pi/2, np.pi/2)
        beta = np.arctan2(lambvr_t, lambvtheta_t)

        beta_array.append(beta)
        t_array.append(t)

        r_dot = vr_t
        vr_dot = -mu/r_t**2 + vtheta_t**2/r_t + T*np.sin(beta)/m_t
        theta_dot = vtheta_t/r_t
        vtheta_dot = -vr_t*vtheta_t/r_t + T*np.cos(beta)/m_t
        m_dot = -T/ve
        lambr_dot = -2*lambvr_t*mu/r_t**3 + vtheta_t * \
            (lambvr_t*vtheta_t + vr_t*lambvtheta_t)/r_t**2
        lambvr_dot = lambvtheta_t*vtheta_t/r_t - lambr_t
        lambtheta_dot = 0
        lambvtheta_dot = -2*vtheta_t*lambvr_t/r_t - \
            lambtheta_t/r_t + lambvtheta_t*vr_t/r_t
        lambm_dot = T*(lambvr_t*np.sin(beta) +
                       lambvtheta_t*np.cos(beta))/m_t**2

        return np.array([r_dot, vr_dot, theta_dot, vtheta_dot, m_dot, lambr_dot, lambvr_dot, lambtheta_dot, lambvtheta_dot, lambm_dot])

    global beta_array, t_array

    beta_guess = guesses[-1]
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

    # r0, vr0, theta0, vtheta, m, lambr0, lambvr0, lambtheta0, lambvtheta_0, lambm_0
    obj_sol = root(objective, guesses[0:-1], method="hybr", tol=1e-4)
    print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    print("msg: ", obj_sol.message)
    print("n func calls: ", obj_sol.nfev)
    #obj_sol = root(objective, obj_sol.x, method="hybr", tol=1e-8)
    #print("Solution found? ", "yes!" if obj_sol.success == 1 else "No :(")
    #print("msg: ", obj_sol.message)
    #print("n func calls: ", obj_sol.nfev)

    obj_sol = obj_sol.x

    _ = objective(obj_sol)

    sol_initial_states = [r0, vr0, theta0, vtheta0,
                          m0, obj_sol[0], obj_sol[1], obj_sol[2], obj_sol[3], obj_sol[4]]

    tf = obj_sol[nx]
    t_eval = np.linspace(0, tf, 1000)

    sol = solve_ivp(dynamics, [0, tf], sol_initial_states, t_eval=t_eval)

    # euler forward solver
    #soly = np.zeros((1000+1, 2*nx))
    #soly[0, :] = sol_initial_states
    #dt = tf/1000
    # for ii, t in enumerate(t_eval):
    #    soly[ii+1, :] = soly[ii] + dynamics(t, soly[ii])*dt

    #soly = odeint(dynamics, x0, t_eval, tfirst=True, printmessg=1)
    #r = soly[:, 0]
    #vr = soly[:, 1]
    #theta = soly[:, 2]
    #vtheta = soly[:, 3]
    #m = soly[:, 4]

    print("beta array shape: ", len(beta_array))
    print("max beta: ", max(beta_array))
    aa = np.array(t_array)
    bb = np.degrees(beta_array)[:]
    control = np.array([aa, bb])
    sortedCon = control[:, control[0].argsort()]

    plt.figure(figsize=(10, 8))
    plt.plot(sortedCon[0], sortedCon[1])
    plt.xlabel('t')
    plt.ylabel('Beta')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(t_eval, sol.y[0], label='r')
    #plt.plot(sol.t, sol.y[1], label='vr')
    #plt.plot(sol.t, sol.y[2], label='theta')
    #plt.plot(sol.t, sol.y[3], label='vtheta')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.title(f'States')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.plot(t_eval, np.degrees(sol.y[2]), label='theta')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.axes(projection='polar')
    #plt.plot(sol.y[0]*np.cos(sol.y[2]), sol.y[0]*np.sin(sol.y[2]))
    plt.polar(sol.y[2], sol.y[0])
    plt.title(f'Trajectory Curve')
    plt.show()


def main():
    global mu, T, ve
    mu = 1
    T = 0.1405
    ve = 1.8758344

    nx = 5  # number of states

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

    lamb0_guess = [1, 1, 1, 1, 1]
    tf_guess = [2]
    beta_guess = [0.1]

    guesses = lamb0_guess + tf_guess + beta_guess  # list of element guesses

    indirect_single_shooting_method(initial_states, final_states, guesses, nx)


if __name__ == "__main__":
    main()
    print("Done")
