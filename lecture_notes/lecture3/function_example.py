def example(x,y,z):
    '''Example of a python function,
    returns twice the first input variable
    and the product of the 2nd and 3rd input
    variables'''
    
    x2 = 2*x 

    return x2,y*z
    
    
def example2(x,y,z):
    '''Another example of a python function which
    returns twice the first input variable
    and the product of the 2nd and 3rd input
    variables, but now we assume that x is a list and
    only double its 1st element.'''
    
    x[0] = x[0]+1

    return x,y*z
    
    
def example2b(x,y,z):
    '''safer implementation of example2'''
    
    x2[0] = x[0]+1

    return x2,y*z
    
    
def example3(x,y,z=1):
    '''Example of a python function,
    returns twice the first input variable
    and the product of the 2nd and 3rd input
    variables, and z has a default value of 1'''

    return 2*x,y*z
    