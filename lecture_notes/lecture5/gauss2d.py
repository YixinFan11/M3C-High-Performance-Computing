"""Apply different (unconstrained) optimzation methods to find minimum of "gauss2d" function specified below
    ncg1: Newton-CG with approximate Hessian
    ncg2: Newton-CG with exact Hessian
    bfgs1: BFGS with approximate Jacobian
    bfgs2: BFGS with exact Jacobian
"""

#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


#objective function-------
def gauss2d(xf,x0,y0,a,b):
    """Compute 2d gaussian function, exp(-a*(x-x0)^2-b*(y-y0)^2)
    x = xf[0], y = xf[1]
    """
    x = xf[0]
    y = xf[1]

    return -np.exp(-a*(x-x0)**2 - b*(y-y0)**2)
#----------------------------------------------

#gradient of objective function
def gauss2d_grad(xf,x0,y0,a,b):
    """Compute gradietn of 2d gaussian function
    defined in gauss2d. Returns two-element tuple
    containing (df/dx,df/dy)
    """

    #compute function
    f = gauss2d(xf,x0,y0,a,b)

    x = xf[0]
    y = xf[1]

    return np.array([-2.0*a*(x-x0)*f,-2.0*b*(y-y0)*f])
#------------------------------------------------------

#Hessian for objective function
def gauss2d_hess(xf,x0,y0,a,b):
    """Compute Hessian for function defined in guass2d.
    Return Hessian matrix, [[fxx fxy],[fxy fyy]]"""

    f = gauss2d(xf,x0,y0,a,b)
    fx,fy = gauss2d_grad(xf,x0,y0,a,b)

    #unpack input
    x = xf[0]
    y = xf[1]

    #compute 2nd derivatives
    fxx = -2.0*a*(f + (x-x0)*fx)
    fxy = -2.0*b*(y-y0)*fx
    fyy = -2.0*b*(f + (y-y0)*fy)

    #construct Hessian
    H = np.zeros((2,2))
    H[0,0] = fxx
    H[0,1] = fxy
    H[1,0] = fxy
    H[1,1] = fyy
    return H
#------------

#display objective
def display_gauss2d(args):

    x = np.linspace(-5,5,101)
    y = np.linspace(-5,5,101)

    xg,yg=np.meshgrid(x,y)
    xf = (xg,yg)

    x0,y0,a,b=args

    f = gauss2d(xf,x0,y0,a,b)

#    plt.figure()
#    plt.contour(xg,yg,f)
    plt.figure()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xg, yg, f,rstride=1,cstride=5)
    plt.show()
#-------------
if __name__ == "__main__":
    parameters = (0.5,0.25,1.0,1.0) #x0,y0,a,b
#    display_gauss2d(parameters) #uncomment to create surface plot

    (ncg1,ncg2,bfgs1,bfgs2)=(True,True,True,True) #specify which methods to run
    xguess = (0.0,0.0)

    if ncg1:
        result_newt = minimize(gauss2d,xguess,args=parameters,method='Newton-CG',jac=gauss2d_grad)
        print("Gauss2d, Newton-CG, approximate Hessian")
        print("location of optimum:",result_newt.x)
        print("info: ", result_newt)
        print("---------------------------------------")

    if ncg2:
        result_newt2 = minimize(gauss2d,xguess,args=parameters,method='Newton-CG',jac=gauss2d_grad,hess=gauss2d_hess)
        print("Gauss2d, Newton-CG, exact Hessian")
        print("location of optimum:",result_newt2.x)
        print("info: ", result_newt2)
        print("---------------------------------------")

    if bfgs1:
        result_bfgs1 = minimize(gauss2d,xguess,args=parameters,method='BFGS',jac=False)
        print("Gauss2d, BFGS, approximate gradient")
        print("location of optimum:",result_bfgs1.x)
        print("info: ", result_bfgs1)
        print("---------------------------------------")

    if bfgs2:
        result_bfgs2 = minimize(gauss2d,xguess,args=parameters,method='BFGS',jac=gauss2d_grad)
        print("Gauss2d, BFGS, exact gradient")
        print("location of optimum:",result_bfgs2.x)
        print("info: ", result_bfgs2)
        print("---------------------------------------")
