"""M3C 2018 Homework 3.

Tudor Trita Trita
CID: 01199397
M3C 2018 CW3
Date: 29/11/2018

Contains five functions:
    plot_S: plots S matrix -- use if you like
    simulate2: Simulate tribal competition over m trials. Return: all s matrices at final time and fc at nt+1 times averaged across the m trials.
    simulate3: Simulate tribal competition over m trials. Return: Average of all s matrices at all times in the form of Sarray.
    performance: Anlyzes and assesses performance of python, fortran, and fortran+openmp simulation codes
    analyze: Analyzes influence of model parameter, g
    visualize: Generates animation illustrating "interesting" tribal dynamics
"""

import subprocess
import time
import numpy as np
from m1 import tribes as tr
# Assumes that hw3.f90 has been compiled with:
# f2py3 --f90flags='-fopenmp' -c hw3.f90 -m m1 -lgomp

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

# pylama:ignore=E501,W0611,C901


def simulate2(N, Nt, b, e, g, m):
    """Simulate m trials of C vs. M competition.

    Output: S: Status of each gridpoint at end of simulation, 0=M, 1=C
            fc_ave: fraction of villages which are C at all Nt+1 times
                    averaged over the m trials
    """
    # Set initial condition
    S = np.ones((N, N, m), dtype=int)  # Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    S[j, j, :] = 0
    N2inv = 1./(N*N)

    fc_ave = np.zeros(Nt+1)  # Fraction of points which are C
    fc_ave[0] = S.sum()

    # Initialize matrices
    NB = np.zeros((N, N, m), dtype=int)  # Number of neighbors for each point
    NC = np.zeros((N, N, m), dtype=int)  # Number of neighbors who are Cs
    S2 = np.zeros((N+2, N+2, m), dtype=int)  # S + border of zeros
    F = np.zeros((N, N, m))  # Fitness matrix
    F2 = np.zeros((N+2, N+2, m))  # Fitness matrix + border of zeros
    A = np.ones((N, N, m))  # Fitness parameters, each of N^2 elmnts is 1 or b
    P = np.zeros((N, N, m))  # Probability matrix
    Pden = np.zeros((N, N, m))
    # ---------------------

    # Calculate number of neighbors for each point
    NB[:, :, :] = 8
    NB[0, 1:-1, :], NB[-1, 1:-1, :], NB[1:-1, 0, :], NB[1:-1, -1, :] = 5, 5, 5, 5
    NB[0, 0, :], NB[-1, -1, :], NB[0, -1, :], NB[-1, 0, :] = 3, 3, 3, 3
    NBinv = 1.0/NB
    # -------------

    # ----Time marching-----
    for t in range(Nt):
        R = np.random.rand(N, N, m)  # Random numbers used to update S

        # Set up coefficients for fitness calculation
        A = np.ones((N, N, m))
        ind0 = np.where(S == 0)
        A[ind0] = b

        # Add boundary of zeros to S
        S2[1:-1, 1:-1, :] = S

        # Count number of C neighbors for each point
        NC = (S2[:-2, :-2, :] + S2[:-2, 1:-1, :] + S2[:-2, 2:, :] + S2[1:-1, :-2, :]
              + S2[1:-1, 2:, :] + S2[2:, :-2, :] + S2[2:, 1:-1, :] + S2[2:, 2:, :])

        # Calculate fitness matrix, F----
        F = NC*A
        F[ind0] = F[ind0] + (NB[ind0]-NC[ind0])*e
        F = F*NBinv
        # -----------

        # Calculate probability matrix, P-----
        F2[1:-1, 1:-1, :] = F
        F2S2 = F2*S2
        # Total fitness of cooperators in community
        P = (F2S2[:-2, :-2, :] + F2S2[:-2, 1:-1, :] + F2S2[:-2, 2:, :]
             + F2S2[1:-1, :-2, :] + F2S2[1:-1, 1:-1, :] + F2S2[1:-1, 2:, :]
             + F2S2[2:, :-2, :] + F2S2[2:, 1:-1, :] + F2S2[2:, 2:, :])

        # Total fitness of all members of community
        Pden = (F2[:-2, :-2, :] + F2[:-2, 1:-1, :] + F2[:-2, 2:, :]
                + F2[1:-1, :-2, :] + F2[1:-1, 1:-1, :] + F2[1:-1, 2:, :]
                + F2[2:, :-2, :] + F2[2:, 1:-1, :] + F2[2:, 2:, :])

        P = (P/Pden)*g + 0.5*(1.0-g)  # probability matrix
        # ---------

        # Set new affiliations based on probability matrix and random numbers stored in R
        S[:, :, :] = 0
        S[R <= P] = 1

        fc_ave[t+1] = S.sum()
        # ----Finish time marching-----

    fc_ave = fc_ave*N2inv/m

    return S, fc_ave
# ------------------


def simulate3(N, Nt, b, e, g, m):
    """Simulate m trials of C vs. M competition.

    Output: Sarray: Status of each gridpoint in S at every single timestep
                    averaged over m trials.
    """
    # Set initial condition
    S = np.ones((N, N, m), dtype=int)  # Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    S[j, j, :] = 0

    # Initialize matrices
    NB = np.zeros((N, N, m), dtype=int)  # Number of neighbors for each point
    NC = np.zeros((N, N, m), dtype=int)  # Number of neighbors who are Cs
    S2 = np.zeros((N+2, N+2, m), dtype=int)  # S + border of zeros
    F = np.zeros((N, N, m))  # Fitness matrix
    F2 = np.zeros((N+2, N+2, m))  # Fitness matrix + border of zeros
    A = np.ones((N, N, m))  # Fitness parameters, each of N^2 elements is 1 or b
    P = np.zeros((N, N, m))  # Probability matrix
    Pden = np.zeros((N, N, m))
    Sarray = np.zeros((N, N, Nt+1))
    Sarray[:, :, 0] = S[:, :, 0]
    # ---------------------

    # Calculate number of neighbors for each point
    NB[:, :, :] = 8
    NB[0, 1:-1, :], NB[-1, 1:-1, :], NB[1:-1, 0, :], NB[1:-1, -1, :] = 5, 5, 5, 5
    NB[0, 0, :], NB[-1, -1, :], NB[0, -1, :], NB[-1, 0, :] = 3, 3, 3, 3
    NBinv = 1.0/NB
    # -------------

    # ----Time marching-----
    for t in range(Nt):
        R = np.random.rand(N, N, m)  # Random numbers used to update S every time step

        # Set up coefficients for fitness calculation
        A = np.ones((N, N, m))
        ind0 = np.where(S == 0)
        A[ind0] = b

        # Add boundary of zeros to S
        S2[1:-1, 1:-1, :] = S

        # Count number of C neighbors for each point
        NC = (S2[:-2, :-2, :] + S2[:-2, 1:-1, :] + S2[:-2, 2:, :] + S2[1:-1, :-2, :]
              + S2[1:-1, 2:, :] + S2[2:, :-2, :] + S2[2:, 1:-1, :] + S2[2:, 2:, :])

        # Calculate fitness matrix, F----
        F = NC*A
        F[ind0] = F[ind0] + (NB[ind0]-NC[ind0])*e
        F = F*NBinv
        # -----------

        # Calculate probability matrix, P-----
        F2[1:-1, 1:-1, :] = F
        F2S2 = F2*S2

        # Total fitness of cooperators in community
        P = (F2S2[:-2, :-2, :] + F2S2[:-2, 1:-1, :] + F2S2[:-2, 2:, :]
             + F2S2[1:-1, :-2, :] + F2S2[1:-1, 1:-1, :] + F2S2[1:-1, 2:, :]
             + F2S2[2:, :-2, :] + F2S2[2:, 1:-1, :] + F2S2[2:, 2:, :])

        # Total fitness of all members of community
        Pden = (F2[:-2, :-2, :] + F2[:-2, 1:-1, :] + F2[:-2, 2:, :]
                + F2[1:-1, :-2, :] + F2[1:-1, 1:-1, :] + F2[1:-1, 2:, :]
                + F2[2:, :-2, :] + F2[2:, 1:-1, :] + F2[2:, 2:, :])

        P = (P/Pden)*g + 0.5*(1.0-g)  # probability matrix
        # ---------

        # Set new affiliations based on probability matrix and random numbers stored in R
        S[:, :, :] = 0
        S[R <= P] = 1

        # ----Finish time marching-----
        Sarray[:, :, t+1] = np.round(np.average(S, axis=2))

    return Sarray
# ------------------


def performance(N, Nt, Ntarray, b, e, g, M, marray, times, Narray, display):
    """Assess performance of simulate2, simulate2_f90, and simulate2_omp.

    (The following was run on a library computer using software hub vm)

    Figure 1.1 using 2 threads testing on M:
    Python function is the slowest. It is approx. 3.5 times slower than Fortran and
    7 times slower than Fortran+OMP routines.
    Fortran+OMP routine is fastest. It is approx. 2 times faster than the Fortran
    without OMP.

    Figure 1.2 using 2 threads testing on Nt:
    Python function is the slowest. It is approx. 7 times slower than Fortran and
    12 times slower than Fortran+OMP routines.
    Fortran+OMP routine is fastest. It is approx. 2 times faster than the Fortran
    without OMP.

    Figure 1.3 using 2 threads testing on N:
    Python function is the slowest. It is approx. 5 times slower than Fortran and
    7 times slower than Fortran+OMP routines.
    Fortran+OMP routine is fastest. It is approx. 1.5 times faster than the Fortran
    without OMP. When ran on the library computer, performance of the OMP version was
    actually slower for larger N than the non-OMP version.

    Figure 1.4:
    This figure displays the speedup difference of the different methods averaged out.
    The figure averages the times in the results obtained in figures 1.1 to 1.3.

    Conclusion: The difference between simualte2_f90 and simulate2_omp is most pronounced
    when tested against larger M values. Overall, we can see that the Python function is
    much slower than both fortran routines.
    """
    tr.tr_b = b
    tr.tr_e = e
    tr.tr_g = g

    # Comparing performance of times against M
    length = len(marray)

    tot_time = np.zeros((length, 3))  # Array of times executed

    for i in range(length):
        for j in range(times):
            t1 = time.time()
            simulate2(N, Nt, b, e, g, marray[i])
            t2 = time.time()
            tot_time[i, 0] += (t2-t1)/length

            t3 = time.time()
            tr.simulate2_f90(N, Nt, marray[i])
            t4 = time.time()
            tot_time[i, 1] += (t4-t3)/length

            t5 = time.time()
            tr.simulate2_omp(N, Nt, marray[i])
            t6 = time.time()
            tot_time[i, 2] += (t6-t5)/length

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(marray, tot_time[:, 0], 'k',
             marray, tot_time[:, 1], 'b',
             marray, tot_time[:, 2], 'r')
    plt.xlabel("m")
    plt.ylabel("Time elapsed, seconds")
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 1.1, Function:performance \n Performance against different values of m (N = 21, Nt = 50).")
    plt.grid(True)
    plt.legend(('Python Only', 'Fortran', 'Fortran + OMP'), loc='upper left')
    fig1.savefig("hw311.png")
    if display:
        plt.show()
    plt.cla()

    speedup1 = sum(tot_time[:, 0])/sum(tot_time[:, 1])  # Speedup of Python/Fortran
    speedup2 = sum(tot_time[:, 1])/sum(tot_time[:, 2])  # Speedup of Fortran/Fortran+OMP
    speedup3 = sum(tot_time[:, 0])/sum(tot_time[:, 2])  # Speedup of Python/Fortran+OMP

    # Comparing performance of times against NT
    length = len(Ntarray)

    tot_time = np.zeros((length, 3))  # Array of times executed

    for i in range(length):
        for j in range(times):
            t1 = time.time()
            simulate2(N, Ntarray[i], b, e, g, M)
            t2 = time.time()
            tot_time[i, 0] += (t2-t1)/length

            t3 = time.time()
            tr.simulate2_f90(N, Ntarray[i], M)
            t4 = time.time()
            tot_time[i, 1] += (t4-t3)/length

            t5 = time.time()
            tr.simulate2_omp(N, Ntarray[i], M)
            t6 = time.time()
            tot_time[i, 2] += (t6-t5)/length

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(Ntarray, tot_time[:, 0], 'k',
             Ntarray, tot_time[:, 1], 'b',
             Ntarray, tot_time[:, 2], 'r')
    plt.xlabel("Nt")
    plt.ylabel("Time elapsed, seconds")
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 1.2, Function:performance \n Performance against different values of Nt (N = 21, M = 50).")
    plt.grid(True)
    plt.legend(('Python Only', 'Fortran', 'Fortran + OMP'), loc='upper left')
    fig2.savefig("hw312.png")
    if display:
        plt.show()
    plt.cla()

    speedup4 = sum(tot_time[:, 0])/sum(tot_time[:, 1])  # Speedup of Python/Fortran
    speedup5 = sum(tot_time[:, 1])/sum(tot_time[:, 2])  # Speedup of Fortran/Fortran+OMP
    speedup6 = sum(tot_time[:, 0])/sum(tot_time[:, 2])  # Speedup of Python/Fortran+OMP

    # Comparing performance of times against N
    length = len(Narray)

    tot_time = np.zeros((length, 3))  # Array of times executed

    for i in range(length):
        for j in range(times):
            t1 = time.time()
            simulate2(Narray[i], Nt, b, e, g, M)
            t2 = time.time()
            tot_time[i, 0] += (t2-t1)/length

            t3 = time.time()
            tr.simulate2_f90(Narray[i], Nt, M)
            t4 = time.time()
            tot_time[i, 1] += (t4-t3)/length

            t5 = time.time()
            tr.simulate2_omp(Narray[i], Nt, M)
            t6 = time.time()
            tot_time[i, 2] += (t6-t5)/length

    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(Narray, tot_time[:, 0], 'k',
             Narray, tot_time[:, 1], 'b',
             Narray, tot_time[:, 2], 'r')
    plt.xlabel("N")
    plt.ylabel("Time elapsed, seconds")
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 1.3, Function:performance \n Performance against different values of N (Nt, M = 50).")
    plt.grid(True)
    plt.legend(('Python Only', 'Fortran', 'Fortran + OMP'), loc='upper left')
    fig3.savefig("hw313.png")
    if display:
        plt.show()
    plt.cla()

    speedup7 = sum(tot_time[:, 0])/sum(tot_time[:, 1])  # Speedup of Python/Fortran
    speedup8 = sum(tot_time[:, 1])/sum(tot_time[:, 2])  # Speedup of Fortran/Fortran+OMP
    speedup9 = sum(tot_time[:, 0])/sum(tot_time[:, 2])  # Speedup of Python/Fortran+OMP

    # Figure for speedups
    fig4 = plt.figure(figsize=(10, 5))
    plt.plot(["P/F", "F/FOMP", "P/FOMP"], [speedup1, speedup2, speedup3], 'ro',
             ["P/F", "F/FOMP", "P/FOMP"], [speedup4, speedup5, speedup6], 'bo',
             ["P/F", "F/FOMP", "P/FOMP"], [speedup7, speedup8, speedup9], 'yo')
    plt.xlabel("P - Python, F - Fortran, FOMP - Fortran + OMP")
    plt.ylabel("Speedup")
    plt.ylim((0, 18))
    plt.legend(('Speedup on M', 'Speedup on Nt', 'Speedup on N'), loc='upper left')
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 1.4, Function:performance \n Speedup of different Methods.")
    fig4.savefig("hw314.png")
    if display:
        plt.show()
    plt.cla()
    return None
# ------------------


def analyze(N, Nt, Ntarray, e, m, display):
    """Analyze influence of model parameter, g.

    Figure 2.1: The 3D plot illustrates the effect of g and b for fixed Nt = 30.
    The main trend displayed in the plot is that if the lower g is, the lower
    fc_average will be for all m values at Nt = 30.

    Figure 2.2: Here we can see that for fixed M = 1.5, fc_ave and g are inversely
    correlated, i.e. the lower g is, the long term behaviour of fc_ave results in a
    higher value. It is also to be noted that the lower g is, fc_ave 'stabilizes'
    faster.

    Figure 3.3: Here g = 0.9, M = 30.
    This graph shows that for fixed, b is inversely correlated with fc_ave, i.e.
    the higher b is, the lower fc_ave will be. This is in agreement with results
    found in coursework 1.

	In fact, if g = 1, then if M villages take over, then there's no chance for C villages
	to spontaneously 'appear'. This is not the case, if g != 1. e.g when g = 0.8, there is
	a chance of C villages appearing spontaneously in S, which changes the dynamics
	of the village competition.
    """
    tr.tr_e = e

    b = np.arange(1, 1.5, 0.005)
    g = np.arange(0.8001, 1.000001, 0.002)

    # Contains final fv_ave value for each permutation of b and g:
    lenb = len(b)
    leng = len(g)
    fv_ave_fin = np.zeros((lenb, leng))

    # Loop to load fv_ave_fin
    for i in range(lenb):
        for j in range(leng):
            tr.tr_b = b[i]
            tr.tr_g = g[j]
            fv_ave_fin[i, j] = tr.simulate2_omp(N, Nt, m)[1][Nt]

    b, g = np.meshgrid(b, g[::-1])  # Setting orientation of plot

    #  3D PLOT
    fig5 = plt.figure(figsize=(11, 6))
    ax = fig5.gca(projection='3d')
    plot = ax.plot_surface(b, g, fv_ave_fin, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_zlim(0, 1)
    ax.set_xlim(1.5, 1)
    ax.set_ylim(1, 0.8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax.set_xlabel('b')
    ax.set_ylabel('g (gamma)')
    ax.set_zlabel('Final averaged error fv_ave')
    ax.set_title('Name: Tudor Trita Trita, CID:01199397 \n Figure 2.1, Function:analyze \n Plot of fv_average against values of b and g (Nt = 30).')
    fig5.colorbar(plot, shrink=0.5, aspect=5)
    plt.savefig("hw321.png")
    if display:
        plt.show()
    plt.cla()

    # Figure 5:
    b = 1.5
    m = 30
    tr.tr_b = b
    garray = 0.99, 0.95, 0.9, 0.85, 0.8
    x = len(Ntarray)
    y = len(garray)
    fc_ave_fin = np.zeros((x, y))
    for i1 in range(y):
        tr.tr_g = garray[i1]
        for i2 in range(x):
            fc_ave_fin[i2, i1] = tr.simulate2_omp(N, Ntarray[i2], m)[1][Ntarray[i2]]

    fig6 = plt.figure(figsize=(11, 6))
    plt.plot(Ntarray, fc_ave_fin[:, 0], 'r',
             Ntarray, fc_ave_fin[:, 1], 'g',
             Ntarray, fc_ave_fin[:, 2], 'k',
             Ntarray, fc_ave_fin[:, 3], 'b',
             Ntarray, fc_ave_fin[:, 4], 'y')
    plt.xlabel("Nt")
    plt.ylabel("fc_ave")
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.2, Function:analyze \n Behaviour of fc_ave against Nt for different values of g. (b = 1.5, M = 30)")
    plt.grid(True)
    plt.legend(('g = 0.99', 'g = 0.95', 'g = 0.9', 'g = 0.85', 'g = 0.8'), loc='upper right')
    fig6.savefig("hw322.png")
    if display:
        plt.show()
    plt.cla()

    # Figure 6:
    g = 0.9
    m = 30
    tr.tr_g = g
    barray = 1.4, 1.1, 1.05, 1.01
    x = len(Ntarray)
    y = len(barray)
    fc_ave_fin = np.zeros((x, y))
    for i1 in range(y):
        tr.tr_b = barray[i1]
        for i2 in range(x):
            fc_ave_fin[i2, i1] = tr.simulate2_omp(N, Ntarray[i2], m)[1][Ntarray[i2]]

    fig7 = plt.figure(figsize=(11, 6))
    plt.plot(Ntarray, fc_ave_fin[:, 0], 'r',
             Ntarray, fc_ave_fin[:, 1], 'k',
             Ntarray, fc_ave_fin[:, 2], 'b',
             Ntarray, fc_ave_fin[:, 3], 'y')
    plt.xlabel("Nt")
    plt.ylabel("fc_ave")
    plt.title("Name: Tudor Trita Trita, CID:01199397 \n Figure 2.3, Function:analyze \n Behaviour of fc_ave against Nt for different values of b.")
    plt.grid(True)
    plt.legend(('b = 1.4', 'b = 1.1', 'b = 1.05', 'b = 1.01'), loc='upper right')
    fig7.savefig("hw323.png")
    if display:
        plt.show()
    plt.cla()
    return None
# ------------------


def visualize(Nt, N, m, b, e, g, display):
    """Animation illustrating evolution of C and M villages.

    Note: simulate3 returns an array with averaged S's at each timestep
    2 Movies submitted
    """
    Sarray = simulate3(N, Nt, b, e, g, m)

    fig = plt.figure(figsize=(10, 7))

    def animate(i):
        ind_s0 = np.where(Sarray[:, :, i] == 0)  # M locations
        ind_s1 = np.where(Sarray[:, :, i] == 1)  # C locations
        animation = plt.plot(ind_s0[1], ind_s0[0], 'rs',
                             ind_s1[1], ind_s1[0], 'bs')
        plt.title("Name: Tudor Trita Trita, CID:01199397 \n Function:visualize, Nt = %d \n (b = 1.01, e = 0.01, g = 0.96)" % (i))
        return animation

    anim = animation.FuncAnimation(fig, animate, frames=Nt, repeat=False,  blit=True)
    anim.save("hw3movie.mp4")
    if display:
        plt.show()
    plt.cla()
    return None
# ------------------


if __name__ == '__main__':
    time1 = time.time()
    perf, analz, vis = 0, 0, 1
    numthreads = 2
    display = False  # Toggle True if you want graphs to appear

    if perf == 1:
        tr.numthreads = numthreads
        N, b, e, g = 21, 1.1, 0.01, 0.95
        Nt, m, times = 30, 30, 10
        marray = np.arange(50, 501, 50)[1:]
        Ntarray = np.arange(50, 501, 50)[1:]
        Narray = np.arange(11, 52, 10)
        performance(N, Nt, Ntarray, b, e, g, m, marray, times, Narray, display)

    if analz == 1:
        tr.numthreads = numthreads
        Ntarray = np.arange(1, 100, 1)
        N, Nt, e, m = 51, 30, 0.01, 20
        analyze(N, Nt, Ntarray, e, m, display)

    if vis == 1:
        movie = 1  # Movie = 1 for N=21, Movie = 2 for N=51
        if movie == 1:
            Nt, N, m, b, e, g = 150, 21, 10, 1.01, 0.01, 0.96
            visualize(Nt, N, m, b, e, g, display)
        else:
            Nt, N, m, b, e, g = 150, 51, 10, 1.01, 0.01, 0.96
            visualize(Nt, N, m, b, e, g, display)
    time2 = time.time()
    tim = time2-time1
    print("Everything has been executed in %d seconds! Have a nice day!" % (tim))
# ------------------ END OF PROGRAM -----------------
