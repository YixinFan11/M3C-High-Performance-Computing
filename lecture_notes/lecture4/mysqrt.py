"""Lecture 3/4 example, computing sqrt with Newton's method
"""

def sqrt2(a,x0=1,debug=False):
    """function to compute sqrt(a) with Newton's method
    optional inputs:
        x0: initial guess
        debug: iteration details are output to screen if True
    """

    #Check input
    assert type(a) is int or type(a) is float, "error, input must be numeric"
    assert a>=0, "a must be non-negative"

    tol = 1.0e-12 #convergence criteria
    imax = 10000 #maximum number of iterations

    #Newton's method
    for i in range(imax):
        x1 = x0/2 + a/(2*x0)
        if  debug:
            print("iteration %d, x = %16.14f" %(i+1,x1))
        del_x = abs(x1-x0)
        if del_x < tol:
            if debug:
                print("converged!")
            break
        x0 = x1

    return x1

def test_sqrt2():
    from math import sqrt
    avalues = [0.1,4,25,143.323]
    for a in avalues:
        s = sqrt(a)
        s2 = sqrt2(a)
        del_s = abs(s-s2)
        assert del_s < 1e-14, "error, test failed for a = %f" %(a)
    print("all tests ok")



if __name__ == '__main__':
    test_sqrt2()



"""Note: There are three ways to 'run' the code. One is to import the module in the
ipython terminal and call the individual functions. You can also use "run mysqrt"
in the ipython terminal or "python mysqrt.py" in the general unix terminal. In these
latter two cases, __name__ == "__main__" and the if statement above is true and the
test function is called. Flags can be used with the run command to obtain timming and
profiling information: run -t, run -p
"timeit" can be used in the ipython terminal to time any available function
See also lecture 4 slides.
"""
