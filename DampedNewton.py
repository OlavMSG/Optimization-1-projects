import numpy as np
import scipy.sparse as sparse             # Sparse matrices
from scipy.sparse.linalg import norm, lsqr
from scipy.linalg import eig
import matplotlib.pyplot as plt
newparams = {'figure.figsize': (6.0, 6.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)

# function to get configuration space
def config_space(l):
    r_max = np.sum(l)
    l_max = np.max(l)
    r_min = 2 * l_max - r_max
    return r_min, r_max


def get_config_space_to_plot(r_max, r_min, p, N=1000):
    x_conf = np.linspace(-r_max, r_max, N)
    y_conf_up = np.sqrt(r_max ** 2 - x_conf ** 2)
    y_conf_low = np.zeros(N)
    min_dist = 0
    if r_min > 0:
        arg_low = np.argwhere(np.abs(x_conf) <= r_min).flatten()
        y_conf_low[arg_low] = np.sqrt(r_min ** 2 - x_conf[arg_low] ** 2)
        length_p = np.linalg.norm(p)
        min_dist = r_min - length_p
        print("Not possible to reach p.")
    return x_conf, y_conf_up, y_conf_low, min_dist


def get_xy_to_plot(l, thetas):
    n = len(l)
    # start in the Origin
    x = [0]
    y = [0]
    for i in range(n):
        Stheta = np.sum(thetas[:i + 1])
        x.append(x[i] + l[i] * np.cos(Stheta))
        y.append(y[i] + l[i] * np.sin(Stheta))
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def Plot(l, thetas, p, case, save=False, num="0"):
    x, y = get_xy_to_plot(l, thetas)
    r_min, r_max = config_space(l)
    x_conf, y_conf_up, y_conf_low, min_dist = get_config_space_to_plot(r_max, r_min, p)

    dist = np.linalg.norm(np.array([x[-1], y[-1]]) - p)
    print("Minimal distance from Robot arm end to p:", min_dist)
    print("Robot arm end to p:", dist)
    epsilon = 2e-15
    if min_dist - epsilon <= dist <= min_dist + epsilon:
        print("Minimum is reached.")

    fig, ax = plt.subplots(1, 1, num="Robot " + num, figsize=(8, 8))
    ax.plot(x, y, "b-o", label="Robot")
    ax.plot(x[-1], y[-1], "r.", label="Robot arm end")
    ax.plot(p[0], p[1], "rx", label="p : (" + "{:.0f}".format(p[0]) + "," + "{:.0f}".format(p[1]) + ")")
    ax.fill_between(x_conf, y_conf_up, y_conf_low, label="Configuration space", color='g')
    ax.fill_between(x_conf, - y_conf_up, - y_conf_low, color='g')

    ax.set_xlim(-1.02 * r_max, 1.02 * r_max)
    ax.set_ylim(-1.02 * r_max, 1.02 * r_max)

    ax.set_title("Robot position, Case: " + "{:.0f}".format(case)
                 + "\n" + "Robot arm end to p: " + "{:.4f}".format(dist))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.grid(True)
    # legend
    # asking matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc=9, bbox_to_anchor=(0.50, -0.12), ncol=2)
    # Set to "save" to True to save the plot
    if save:
        plt.savefig("Robot" + num + ".pdf", bbox_inches='tight')


def get_conv_to_plot(l ,thetas_list, p):
    dist = []
    for thetas in thetas_list:
        x, y = get_xy_to_plot(l, thetas)
        arm_end = np.array([x[-1], y[-1]])
        dist.append(np.linalg.norm(arm_end - p))
    dist = np.asarray(dist)
    return dist

def Plot_convergence(l, thetas_list, p, case, save=False, num="0"):
    dist = get_conv_to_plot(l, thetas_list, p)
    fig, ax = plt.subplots(1, 1, num="Conv " + num, figsize=(8, 5))
    ax.semilogy(dist, "b-o", label="$\|F(\\vartheta) - p \|$")

    ax.set_title("Convergence rate, Case: " + "{:.0f}".format(case))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("$\|F(\\vartheta) - p \|$")
    ax.grid(True)
    # legend
    # asking matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc=9, bbox_to_anchor=(0.9, 1.135), ncol=2)
    # Set to "save" to True to save the plot
    if save:
        plt.savefig("Conv" + num + ".pdf", bbox_inches='tight')


def r1(l, thetas, p1):
    n = len(l)
    return np.sum([l[i] * np.cos(np.sum(thetas[:i + 1])) for i in range(n)]) - p1


def del_r1(l, thetas, k):
    # k, number of theta to differentiate on
    n = len(l)
    return - np.sum([l[i] * np.sin(np.sum(thetas[:i + 1])) for i in range(k, n)])


def del2_r1(l, thetas, k1, k2):
    n = len(l)
    s = max(k1, k2)
    return - np.sum([l[i] * np.cos(np.sum(thetas[:i + 1])) for i in range(s, n)])



def r2(l, thetas, p2):
    n = len(l)
    return np.sum([l[i] * np.sin(np.sum(thetas[:i + 1])) for i in range(n)]) - p2


def del_r2(l, thetas, k):
    # k, number of theta to differentiate on
    n = len(l)
    return np.sum([l[i] * np.cos(np.sum(thetas[:i + 1])) for i in range(k, n)])


def del2_r2(l, thetas, k1, k2):
    n = len(l)
    s = max(k1, k2)
    return - np.sum([l[i] * np.sin(np.sum(thetas[:i + 1])) for i in range(s, n)])


def Hessian_ri(l, thetas, del2_ri):
    n = len(l)
    ri = sparse.dok_matrix((n, n))
    for k1 in range(n):
        for k2 in range(n):
            ri[k1, k2] = del2_ri(l, thetas, k1, k2)
    return ri


def r(l, thetas, p):
    r = sparse.dok_matrix((2, 1))
    r[0] = r1(l, thetas, p[0])
    r[1] = r2(l, thetas, p[1])
    return r


def Jacobi(l, thetas):
    n = len(l)
    J = sparse.dok_matrix((2, n))
    for k in range(n):
        J[0, k] = del_r1(l, thetas, k)
        J[1, k] = del_r2(l, thetas, k)
    return J


def thetas_0_2pi(thetas):
    for i in range(len(thetas)):
        while thetas[i] >= 2 * np.pi or thetas[i] < 0:
            if thetas[i] >= 2 * np.pi:
                thetas[i] -= 2 * np.pi
            if thetas[i] < 0:
                thetas[i] += 2 * np.pi
    return thetas


def wolfeLineSearch(l, thetas_k, p, fk, gk, pk, c1, c2, rho, ak=1.0, nmaxls=100):
    pkgk = pk @ gk
    # Increase the step length until the Armijo rule is (almost) not satisfied
    while 0.5 * norm(r(l, thetas_k + rho * ak * pk, p)) < fk + c1 * rho * ak * pkgk[0]:
        ak *= rho

    # Use bisection to find the optimal step length
    aU = ak  # upper step length limit
    aL = 0  # lower step length limit
    for i in range(nmaxls):

        # Find the midpoint of aU and aL
        ak = 0.5 * (aU + aL)

        if 0.5 * norm(r(l, thetas_k + ak * pk, p)) > fk + c1 * ak * pkgk:
            # Armijo condition is not satisfied, decrease the upper limit
            aU = ak
            continue

        gk_ak_pk = Jacobi(l, thetas_k + ak * pk).T @ r(l, thetas_k + ak * pk, p)
        if pk @  gk_ak_pk > -c2 * pkgk:
            # Upper Wolfe condition is not satisfied, decrease the upper limit
            aU = ak
            continue

        if pk @ gk_ak_pk < c2 * pkgk:
            # Lower Wolfe condition is not satisfied, increase the lower limit
            aL = ak
            continue

        # Otherwise, all conditions are satisfied, stop the search
        break
    return ak


def searchDirection(gk, Hk, epsilon=1e-8):
    pk = - lsqr(Hk.tocsr(), gk.toarray().ravel())[0]  # compute the search direction
    if - pk @ gk <= epsilon * norm(gk) * np.linalg.norm(pk):
        gk = gk.toarray()  # needed here to not get fail later
        return - gk.ravel()  # ensure that the directional derivative is negative in this direction
    return pk



def DampedNewton(l, p, thetas0, tol=1e-10, nmax=1000, nmaxls=100,
                c1=1e-4, c2=0.9, rho=2, epsilon=1e-8):

    thetas_k_list = [thetas0]  # Start condition
    ak = 1  # current step length
    count = 0
    n = len(l)
    for i in range(nmax):
        thetas_k = thetas_0_2pi(thetas_k_list[-1])  # thetas between 0 and 2*pi
        fk = 0.5 * norm(r(l, thetas_k, p))  # current function value
        Jk = Jacobi(l, thetas_k)  # current Jacobi
        rk = r(l, thetas_k, p)
        gk = Jk.T @ rk  # current gradient
        Hk = Jk.T @ Jk + rk[0, 0] * Hessian_ri(l, thetas_k, del2_r1) \
             + rk[1, 0] * Hessian_ri(l, thetas_k, del2_r2)  # current Hessian

        # Compute the search direction
        pk = searchDirection(gk, Hk, epsilon)

        # Perform line search
        ak = wolfeLineSearch(l, thetas_k, p, fk, gk, pk, c1, c2, rho, ak=1.0, nmaxls=nmaxls)

        # Perform the step, add the step to the list, and compute the f and the gradient of f for the next step
        thetas_k_list.append(thetas_k + ak * pk)

        # count number of iterations
        count += 1

        if norm(gk) < tol:
            eigval= eig(Hk.toarray())[0]
            min_eigval = np.min(eigval)
            max_eigval = np.max(eigval)
            if min_eigval < 0 < max_eigval:  # converged to a saddle point
                print("Converged to å saddle point, trying with new initial values.")
                # try new thetas0
                thetas0 = thetas_0_2pi(thetas0 + np.random.rand(len(thetas0)))  # Get new initial guess away form old
                print("New thetas0: ", thetas0 / np.pi, " * pi")
                thetas_k_list2, count2 = DampedNewton(l, p, thetas0, nmax=nmax-count)
                for i in range(len(thetas_k_list2)):
                    thetas_k_list.append(thetas_k_list2[i])
                count += count2
            break

    thetas_k_list = np.asarray(thetas_k_list)
    return thetas_k_list, count


def print_thetas(thetas):
    print("Thetas:", thetas)
    print("Thetas:", thetas / np.pi, "* pi")



def run_test_cases(save=False):
    l_dict = {0: [3, 2, 2], 1: [1, 4, 1], 2: [3, 2, 1, 1], 3: [3, 2, 1, 1]}
    p_dict = {0: [3, 2], 1: [1, 1], 2: [3, 2], 3: [0, 0]}
    thetas0_dict = {0: np.array([3/2, 1/2, 1/2])*np.pi, 1: np.array([4/5, 4/5, 4/5]) * np.pi,
                    2: np.array([0.80089218, 0.64076372, 0.61571798, 0.50032875]) * np.pi,
                    3: np.array([3/2, 1/2, 1/2, 1/2]) * np.pi}

    # Number after "2:" are choosen by running the code once and taking the last thetas0 for New thetas0.
    # 0.80089218 0.64076372 0.61571798 0.50032875
    # 1.80234842, 1.2842212,  0.65702182, 1.95688468



    for i in range(4):
        l = l_dict[i]
        p = p_dict[i]
        thetas0 = thetas0_dict[i]
        print("Case:", i + 1)
        print("l:", l)
        print("p:", p)
        thetas_list, count = DampedNewton(l, p, thetas0)
        print("Converged in", count, "iterations")
        Plot(l, thetas_list[-1], p, i + 1, num=str(i+1), save=save)
        print_thetas(thetas_list[-1])
        Plot_convergence(l, thetas_list, p, i + 1, num=str(i+1), save=save)
        print("---------------------------------------")




# A special case discussed in the theory
# There is a saddle point which the robot arm gets stuck in
def run_special_case(save=False):
    l = [3, 2, 2]
    p = [1, 0.5]  # Somewhere along the first l
    thetas0 = np.array([2.07111284, 0, np.pi])  # [x, 0, pi], for x in |R 2.07111284 + n*2pi = 2728
    print("Case: 5")
    print("l:", l)
    print("p:", p)
    thetas_list, count = DampedNewton(l, p, thetas0)
    print("Converged in", count, "iterations")
    Plot(l, thetas_list[-1], p, 5, num=str(5), save=save)
    print_thetas(thetas_list[-1])
    Plot_convergence(l, thetas_list, p, 5, num=str(5), save=save)
    print("---------------------------------------")


"run code"
# save = True to save the plots
save = False
run_test_cases(save=save)
run_special_case(save=save)
plt.show()
