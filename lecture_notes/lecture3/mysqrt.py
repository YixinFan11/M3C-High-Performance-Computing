""" Lecture 3 example, will be developed further in lecture 4
Module for computing sqrt with Newton's method
"""

def sqrt2(a,x0=1,debug=False):
    """ Function for computing sqrt(a) with Newton's method
    x0: initial guess
    debug: iteration-level info is displayed when True
    """

    #Check input
    assert type(a) is int or type(a) is float, "error, input must be numeric"
    assert a>=0, "error, input must be non-negative"

    tol = 1e-12 # tolerance for convergence check
    maxit = 100000 # maximum number of iterations

    #Newton's method
    for i in range(maxit):
        x = x0/2 + a/(2*x0)
        delta_x = abs(x-x0)
        if debug:
            print("after iteration %d, x = %18.16f,dx=%18.16f" %(i+1,x,delta_x))
        if delta_x<tol:
            if debug:
                print("converged")
            break
        x0  = x
    return x

def test_sqrt2():
    """test sqrt2 function via comparison with
    math.sqrt for values stored in avalues
    """
    from math import sqrt
    avalues = [0.25, 3.45,23.5,13342534.1]
    tol = 1.e-14
    for a in avalues:
        s = sqrt(a)
        s2 = sqrt2(a)
        delta_s = abs(s-s2)
        assert delta_s<tol, "error, test failed with a=%f" %(a)
    print("all tests passed")
