"""M3C PROJECT.

Tudor Trita Trita
CID: 01199397
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from m1 import bmodel as bm  # assumes p2.f90 has been compiled with:
# f2py3 -c p2_dev1.f90 -m m1

# pylama:ignore=E501


def simulate_jacobi(n, input_num=(20000, 1e-8), input_mod=(1, 1, 1, 2, 1.5), display=False):
    """Solve contamination model equations with jacobi iteration.

    Input:
    input_num: 2-element tuple containing kmax (max number of iterations
    and tol (convergence test parameter)

    input_mod: 5-element tuple containing g,k_bc,s0,r0,t0 --
        g: bacteria death rate
        k_bc: r=1 boundary condition parameter
        s0,r0,t0: source function parameters
    display: if True, a contour plot showing the final concetration field is generated
    Output:C,deltac: Final concentration field and |max change in C| each iteration
    """
    # Set model parameters------

    kmax, tol = input_num
    g, k_bc, s0, r0, t0 = input_mod

    # -------------------------------------------

    # Set Numerical parameters
    Del = np.pi/(n+1)
    r = np.linspace(1, 1+np.pi, n+2)
    t = np.linspace(0, np.pi, n+2)  # theta
    tg, rg = np.meshgrid(t, r)  # r-theta grid

    # Factors used in update equation
    rinv2 = 1.0/(rg*rg)
    fac = 1.0/(2 + 2*rinv2+Del*Del*g)
    facp = (1+0.5*Del/rg)*fac
    facm = (1-0.5*Del/rg)*fac
    fac2 = fac*rinv2

    # Set initial condition/boundary conditions
    C = (np.sin(k_bc*tg)**2)*(np.pi+1.-rg)/np.pi

    # Set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0*np.exp(-20.*((rg-r0)**2+(tg-t0)**2))*(Del**2)*fac

    deltac = []
    Cnew = C.copy()

    # Jacobi iteration
    for k in range(kmax):
        # Compute Cnew
        Cnew[1:-1, 1:-1] = (Sdel2[1:-1, 1:-1] + C[2:, 1:-1]*facp[1:-1, 1:-1]
                            + C[:-2, 1:-1]*facm[1:-1, 1:-1] + (C[1:-1, :-2]
                            + C[1:-1, 2:])*fac2[1:-1, 1:-1])  # Jacobi update

        # Compute delta_p
        deltac += [np.max(np.abs(C-Cnew))]
        C[1:-1, 1:-1] = Cnew[1:-1, 1:-1]
        if k % 1000 == 0:
            print("k,dcmax:", k, deltac[k])

        # Check for convergence
        if deltac[k] < tol:
            print("Converged, k=%d, dc_max=%28.16f " % (k, deltac[k]))
            break

    deltac = deltac[:k+1]

    if display:
        plt.figure()
        plt.contour(t, r, C, 50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Final concentration field')
        plt.show()

    return C, deltac, k


def simulate(n, input_num=(100000, 1e-8), input_mod=(1, 1, 1, 2, 1.5), display=False):
    """Solve contamination model equations.

    Using OSI method, input/output same as in simulate_jacobi above.
    """
    # Set model parameters------

    kmax, tol = input_num
    g, k_bc, s0, r0, t0 = input_mod

    # Set numerical parameters
    Del = np.pi/(n+1)
    r = np.linspace(1, 1+np.pi, n+2)
    t = np.linspace(0, np.pi, n+2)  # theta
    tg, rg = np.meshgrid(t, r)  # r-theta grid

    # Factors used in update equation
    rinv2 = 1.0/(rg*rg)
    fac = 1.0/(2 + 2*rinv2+Del*Del*g)
    facp = (1+0.5*Del/rg)*fac
    facm = (1-0.5*Del/rg)*fac
    fac2 = fac*rinv2

    # Set initial condition/boundary conditions
    C = (np.sin(k_bc*tg)**2)*(np.pi+1.-rg)/np.pi

    # Set source function, Sdel2 = S*del^2*fac
    Sdel2 = s0*np.exp(-20.*((rg-r0)**2+(tg-t0)**2))*(Del**2)*fac

    deltac = []
    Cnew = C.copy()

    # OSI iteration;
    for k in range(kmax):
        for i in range(n-1):
            for j in range(n-1):
                Cnew[i+1, j+1] = (1/2)*(3*(Sdel2[i+1, j+1] + C[i+2, j+1]*facp[i+1, j+1]
                                        + Cnew[i, j+1]*facm[i+1, j+1] + (Cnew[i+1, j]
                                        + C[i+1, j+2])*fac2[i+1, j+1]) - C[i+1, j+1])
        # Compute delta_p
        deltac += [np.max(np.abs(C-Cnew))]
        C[1:-1, 1:-1] = Cnew[1:-1, 1:-1]
        if k % 1000 == 0:
            print("k,dcmax:", k, deltac[k])

        # Check for convergence
        if deltac[k] < tol:
            print("Converged, k=%d, dc_max=%28.16f " % (k, deltac[k]))
            break

    deltac = deltac[:k+1]
    if display:
        plt.figure()
        plt.contour(t, r, C, 50)
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title('Final concentration field')
        plt.show()
    return C, deltac, k


def performance(display):
    """Analysis of performance.

    We compare the 2 algorithms for computing the result in both Python and
    Fortran, giving us the 4 methods: Python using Jacobi iterations (PJAC),
    Python using OSI iterations (POSI), Fortran using Jacobi iterations (FJAC)
    and Fortran using OSI iterations (FOSI). In my analysis of the performance,
    I vary n and concentrate in comparing performance of the methods in terms of
    wall time and converge to k.

    The two algorithms converge at different k’s, and this is shown in
    figure 2.3.4, where we can see that the OSI method converges at a much faster
    rate than the Jacobi method for different n. This is especially true for larger n,
    as we can see that at n=100, the Jacobi method converges at k approx. 12000 whereas
    the OSI method converges at k approx. 2000, giving a large difference in k of 10000.
    I also noted that both PJAC and FJAC converge at the same k, and similarly for POSI
    and FOSI, provided that all the other parameters are the same.

    In terms of time, in Figure 2.3.1 we can see that the Python Jacobi is
    significantly slower than both the Fortran Jacobi and the Fortran OSI. This is most
    pronounced when n is small, which shows that the compiler is particularly efficient
    for small problem sizes.

    The difference in wall time between the PJAC and FJAC is due to the compiled nature of Fortran,
    as even though much of the code is vectorized in PJAC, there is still a loop for every k,
    which is optimized by the compiler.

    When comparing FJAC vs FOSI in Figure 2.3.3, we can see that FOSI is faster than FJAC,
    except for very small problem sizes. We can see that the ratio of times FJAC/FOSI peaks
    at around n= 40, and then it slowly stabilizes at around a ratio of 1.5 for n large (bigger than 200).
    Trials of n bigger than 300 were impractical, as for n larger than 300, k would become too large
    in the Jacobi iterations.

    This difference between FJAC and FOSI is due to the OSI method being more expensive for
    each iteration of k, but the Jacobi method converges for a much larger k (seen in Figure 2.3.4),
    thus the OSI method makes up for lost time each iteration by converging at a smaller k.
    What is interesting is that for very small problem sizes (less than 20), this difference in
    convergence is not that significant, and the expensiveness of each iteration makes the
    OSI method slower than the Jacobi method.

    This is even more exaggerated when comparing POSI with PJAC which can be seen in Figure 2.3.1.
    JOSI , for n = 100 is in fact approx. 50 times slower than PJAC. Whereas, in Fortran,
    the time it takes to run each OSI loop is minimized due to the compiler, in Python this
    does not happen. Therefore, each iteration is very slow, and the fact that
    it converges at a lower k is no longer enough.

    The fact that the OSI is more expensive is due to being unable to be
    vectorized, as iterations depend on each other.
    """
    # Figure 2.3.1 illustrating poor performance of Python OSI
    narray = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    length = len(narray)
    tot_time = np.zeros((length, 4))  # Array of times executed
    speedup = np.zeros((length, 3))   # Array of speedups

    for i in range(length):
            n = int(narray[i])
            print("N = ", n)
            t1 = time.time()
            simulate_jacobi(n)
            t2 = time.time()
            tot_time[i, 0] = (t2-t1)

            t3 = time.time()
            simulate(n)
            t4 = time.time()
            tot_time[i, 1] = (t4-t3)

            t5 = time.time()
            bm.simulate_jacobi(n)
            t6 = time.time()
            tot_time[i, 2] = (t6-t5)

            t7 = time.time()
            bm.simulate(n)
            t8 = time.time()
            tot_time[i, 3] = (t8-t7)

    print(tot_time[:, 2], tot_time[:, 3])
    speedup[:, 0] = tot_time[:, 1]/tot_time[:, 0]  # Python OSI / Python Jacobi
    speedup[:, 1] = tot_time[:, 0]/tot_time[:, 2]  # Python Jacobi / Fortran Jac
    speedup[:, 2] = tot_time[:, 0]/tot_time[:, 3]  # Python Jacobi / Fortran OSI
    plt.figure(figsize=(10, 5))
    plt.plot(narray, speedup[:, 0], 'k--', label='POSI/PJAC')
    plt.plot(narray, speedup[:, 1], 'b--', label='PJAC/FJAC')
    plt.plot(narray, speedup[:, 2], 'r--', label='PJAC/FOSI')
    plt.legend(loc='upper left')
    plt.xlabel('n')
    plt.ylabel('Time ratios')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.3.1, Function:performance \n Execution Time Ratios of Python OSI vs other methods")
    plt.grid(True)
    if display:
        plt.show()

    # Figure 2.3.2 illustrating times for Fortran methods and Python Jacobi Method
    narray = np.linspace(5, 150, 30)
    length = len(narray)
    tot_time = np.zeros((length, 3))  # Array of times executed

    for i in range(length):
            n = int(narray[i])
            print("N = ", n)
            t1 = time.time()
            simulate_jacobi(n)
            t2 = time.time()
            tot_time[i, 0] = (t2-t1)

            t3 = time.time()
            bm.simulate_jacobi(n)
            t4 = time.time()
            tot_time[i, 1] = (t4-t3)

            t5 = time.time()
            bm.simulate(n)
            t6 = time.time()
            tot_time[i, 2] = (t6-t5)

    plt.figure(figsize=(10, 5))
    plt.plot(narray, tot_time[:, 0], 'k',
             narray, tot_time[:, 1], 'b',
             narray, tot_time[:, 2], 'r',)
    plt.xlabel('n')
    plt.ylabel('Time elapsed (s)')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.3.2, Function:performance \n Execution Times of Different Methods")
    plt.grid(True)
    plt.legend(('Python Jacobi', 'Fortran Jacobi', 'Fortran OSI'), loc='upper left')
    if display:
        plt.show()

    # # Figure 2.3.3 illustrating times for Fortran methods
    narray = np.array([2, 5, 10, 20, 30, 40, 50, 60, 70, 100, 150, 200, 250, 300])
    length = len(narray)
    tot_time = np.zeros((length, 2))  # Array of times executed
    speedup = np.zeros(length)

    for i in range(length):
            n = int(narray[i])
            print("N = ", n)
            t1 = time.time()
            bm.simulate_jacobi(n)
            t2 = time.time()
            tot_time[i, 0] = (t2-t1)

            t3 = time.time()
            bm.simulate(n)
            t4 = time.time()
            tot_time[i, 1] = (t4-t3)
    speedup = tot_time[:, 0]/tot_time[:, 1]

    plt.figure(figsize=(10, 5))
    plt.plot(narray, speedup, 'b--')
    plt.xlabel('n')
    plt.ylabel('Time Ratio FJAC/FOSI')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.3.3, Function:performance \n Wall Time Ratios of Fortran Methods")
    plt.grid(True)
    if display:
        plt.show()

    # Figure 2.3.4 performance vs K for Jacobi vs OSI method
    narray = np.linspace(10, 100, 10)
    length = len(narray)
    karray = np.zeros((length, 2))  # Array of times executed

    for i in range(length):
            n = int(narray[i])
            print("N = ", n)
            karray[i, 0] = simulate_jacobi(n)[2]
            karray[i, 1] = simulate(n)[2]

    plt.figure(figsize=(10, 5))
    plt.plot(narray, karray[:, 0], 'r',
             narray, karray[:, 1], 'b')
    plt.xlabel('n')
    plt.ylabel('k')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.3.4, Function:performance \n Performance of Jacobi vs OSI Methods measured by K")
    plt.grid(True)
    plt.legend(('Jacobi Method', 'OSI Method'), loc='upper left')
    if display:
        plt.show()
    return None


def analyze(display):
    """Influence of antibacterial agent on concentration.

    This part is concerned with the effects of theta* on the levels of bacteria in the model. In my answer,
    I only consider varying the parameter k to illustrate the effects of theta* properly. In my answer,
    I will assume that total bacteria = sum of all concentrations in the model using N = 50. I also assume
    that the best place to apply antibac is one which minimizes the total amount of bacteria. t = theta.

    Firstly, to be able to understand the effects of the antibac, I went ahead to compare having antibac vs
    not having antibac effects in the concentration of bacteria. set theta* = theta for Figure 2.4.6, which
    gave me the same result as in Q2, then used theta* = pi/2 in Figure 2.4.7. This shows a significant change,
    with the concentration of bacteria decreased in the latter case, and in fact, the sum of C in 2.4.6 is approx. 340
    where as in 2.4.7 is approx. 147, thus confirming that the antibac works at decreasing the concentration.

    In figures 2.4.1-2.4.4, the plot shows how the total concentration is related to theta*.

    An interesting behaviour to be noted is that, the larger k gets, the lower the total concentration of
    bacteria is, which is due to the fact that the larger k is, the bacteria is more ‘spread out’ as the peaks
    of sin^2(kt) are more numerous. One would expect the total concentration to be lower when this is the case.

    Another interesting effect is that, as theta goes further from theta*, the concentration levels go to 0.

    For all of the figures 2.4.1-2.4.4, we can see that the optimal place to place the antibac is at either 0 or pi.

    But even more interesting behaviours happen. From Figures 2.4.2-2.4.4, we can see that there exist local
    minima in the graph, e.g. if theta*= pi/2 for k=2, then the concentration is at a local minimum. The same
    occurs as k increases, those these minima are less pronounced, and beyond k =4, some of these are lost.
    These local minima correspond to the roots of sin^2(kt). It is also interesting to note that if theta* is such
    that sin^2(ktheta*) = 1, the total concentration is greatest.

    Thus my conclusion is that the best values for theta* are 0 and pi, otherwise, theta* should be such that
    sin^(ktheta*)=0, as these give the local minima in concentration.
    """
    N = 50

    # Figure 2.4.1
    k = [2, 1 + np.pi/2, 1, 10000]  # inputs: s0, r0, k, kmax
    C_summed = np.zeros(N+2)
    thetas = np.linspace(0, np.pi, N+2)
    thetastar = np.zeros((N+2, N+2))
    count = 0
    for i in thetas:
        thetastar[:, :] = i
        C = bm.simulate_q4(k, thetastar, N)
        C_summed[count] = float(np.sum(C))
        count += 1

    plt.figure()
    plt.plot(thetas, C_summed, 'k--')
    plt.xlabel('theta*')
    plt.ylabel('Total concentration')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4.1, Function:analyze \n Sum of concentration levels for bacterial model when k = 1")
    plt.grid(True)

    if display:
        plt.show()

    # Figure 2.4.2
    k = [2, 1 + np.pi/2, 2, 10000]  # inputs: s0, r0, k, kmax
    C_summed = np.zeros(N+2)
    thetas = np.linspace(0, np.pi, N+2)
    thetastar = np.zeros((N+2, N+2))
    count = 0
    for i in thetas:
        thetastar[:, :] = i
        C = bm.simulate_q4(k, thetastar, N)
        C_summed[count] = float(np.sum(C))
        count += 1

    plt.figure()
    plt.plot(thetas, C_summed, 'k--')
    plt.xlabel('theta*')
    plt.ylabel('Total concentration')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4.2, Function:analyze \n Sum of concentration levels for bacterial model when k = 2")
    plt.grid(True)
    if display:
        plt.show()

    # Figure 2.4.3
    k = [2, 1 + np.pi/2, 3, 10000]  # inputs: s0, r0, k, kmax
    C_summed = np.zeros(N+2)
    thetas = np.linspace(0, np.pi, N+2)
    thetastar = np.zeros((N+2, N+2))
    count = 0
    for i in thetas:
        thetastar[:, :] = i
        C = bm.simulate_q4(k, thetastar, N)
        C_summed[count] = float(np.sum(C))
        count += 1

    plt.figure()
    plt.plot(thetas, C_summed, 'k--')
    plt.xlabel('theta*')
    plt.ylabel('Max concentration')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4.3, Function:analyze \n Sum concentration for bacterial model when k = 3")
    plt.grid(True)
    if display:
        plt.show()

    # Figure 2.4.4
    k = [2, 1 + np.pi/2, 4, 10000]  # inputs: s0, r0, k, kmax
    C_summed = np.zeros(N+2)
    thetas = np.linspace(0, np.pi, N+2)
    thetastar = np.zeros((N+2, N+2))
    count = 0
    for i in thetas:
        thetastar[:, :] = i
        C = bm.simulate_q4(k, thetastar, N)
        C_summed[count] = float(np.sum(C))
        count += 1

    plt.figure()
    plt.plot(thetas, C_summed, 'k--')
    plt.xlabel('theta*')
    plt.ylabel('Max concentration')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4.4, Function:analyze \n Sum concentration for bacterial model when k = 4")
    plt.grid(True)
    if display:
        plt.show()

    # Figure 2.4.5
    k = [2, 1 + np.pi/2, 5, 10000]  # inputs: s0, r0, k, kmax
    C_summed = np.zeros(N+2)
    thetas = np.linspace(0, np.pi, N+2)
    thetastar = np.zeros((N+2, N+2))
    count = 0
    for i in thetas:
        thetastar[:, :] = i
        C = bm.simulate_q4(k, thetastar, N)
        C_summed[count] = float(np.sum(C))
        count += 1

    plt.figure()
    plt.plot(thetas, C_summed, 'k--')
    plt.xlabel('theta*')
    plt.ylabel('Total concentration')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4.2, Function:analyze \n Sum of concentration levels for bacterial model when k = 5")
    plt.grid(True)
    if display:
        plt.show()

    # Figure 2.4.6
    k = [2, 1 + np.pi/2, 1, 10000]  # inputs: s0, r0, k, kmax
    r = np.linspace(1, 1+np.pi, N+2)
    t = np.linspace(0, np.pi, N+2)  # theta
    tg, rg = np.meshgrid(t, r)  # r-theta grid
    thetastar = np.zeros((N+2, N+2))
    thetastar = tg
    C = bm.simulate_q4(k, thetastar, N)
    print(np.sum(C))
    plt.figure(figsize=(10, 7))
    plt.contour(t, r, C, 50)
    plt.xlabel('theta')
    plt.ylabel('r')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4.6, Function:analyze \n Final concentration field, no antibacterial agent, k = 1.")
    plt.grid(True)
    if display:
        plt.show()

    # Figure 2.4.7
    thetastar[:, :] = np.pi/2
    C = bm.simulate_q4(k, thetastar, N)
    print(np.sum(C))
    plt.figure(figsize=(10, 7))
    plt.contour(t, r, C, 50)
    plt.xlabel('theta')
    plt.ylabel('r')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.4.7, Function:analyze \n Final concentration field, with antibacterial agent, thetastar = pi/2, k = 1.")
    plt.grid(True)
    if display:
        plt.show()
    return None


if __name__ == '__main__':
    perf, analz = False, False
    display = True
    if perf:
        performance(display)
    elif analz:
        analyze(display)
    print("Everything executed")
