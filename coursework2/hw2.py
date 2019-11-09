"""M3C 2018 Homework 2
Name: Tudor Mihai Trita Trita
CID: 01199397
TID: 167
Course: Mathematics G103 Year 3
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy import optimize
from m5 import nmodel as nm #assumes that hw2_dev.f90 has been
#compiled with: f2py -c hw2_dev.f90 -m m5
#from sklearn.neural_network import MLPClassifier
# May also use scipy, scikit-learn, and time modules as needed

def read_data(tsize=60000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 70000 matrix, X
    and the corresponding labels are stored in a 70000 element array, y.
    The final 70000-tsize images and labels are stored in X_test
    and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int)%2 #rescale the image, convert the labels to 0s and 1s (For even and odd integers)
    Data = None

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]
    print("processed dataset")
    return X,y,X_test,y_test
#----------------------------

def snm_test(X,y,X_test,y_test,omethod,input=(None)):
    """Train single neuron model with input images and labels (i.e. use data in X and y), then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """
    n = X.shape[0]
    fvec = np.random.randn(n+1) #initial fitting parameters

    #Add code to train SNM and evaluate testing test_error
    d = X.shape[1]
    nm.nm_x = X
    nm.nm_y = y

    #Train snm using appropriate model
    if omethod==1:
        res = minimize(nm.snmodel,fvec,args=(d,),method='L-BFGS-B',jac=True)
        fvec_f = res.x
    elif omethod==2:
        fvec_f = nm.sgd(fvec,n,0,d,0.1)
    else:
        print("error, omethod must be 1 or 2")
        return None

    #Compute testing error
    z = np.dot(fvec_f[:-1],X_test) + fvec_f[-1]
    a_int = np.round(1.0/(1.0+np.exp(-z)))
    eps = np.abs(a_int - y_test)
    test_error = eps.sum()/y_test.size

    output = (None) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------


def nnm_test(X,y,X_test,y_test,m,omethod,input=(None)):
    """Train neural network model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=60000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of neurons in inner layer
    omethod=1: use l-bfgs-b optimizer
    omethod=2: use stochastic gradient descent
    input: tuple, set if and as needed
    """
    n = X.shape[0]
    fvec = np.random.randn(m*(n+2)+1) #initial fitting parameters

    #Add code to train NNM and evaluate testing error, test_error
    d = X.shape[1]
    nm.nm_x = X.copy()
    nm.nm_y = y.copy()

    #Train snm using appropriate model
    if omethod==1:
        res = minimize(nm.nnmodel,fvec,args=(n,m,d),method='L-BFGS-B',jac=True)
        fvec_f = res.x
    elif omethod==2:
        fvec_f = nm.sgd(fvec,n,m,d,0.1)
    else:
        print("error, omethod must be 1 or 2")
        return None

    nm.nm_x = X_test.copy()
    nm.nm_y = y_test.copy()

    test_error = nm.run_nnmodel(fvec_f,n,m,d)

    output = (None) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------


def nm_analyze(mvalues,dvalues):
    """ Analyze performance of single neuron and neural network models
    on even/odd image classification problem
    Add input variables and modify return statement as needed.
    Should be called from
    name==main section below
    """

    #Read in data, and initialize arrays
    Xfull,yfull,X_test,y_test = read_data()
    dt = np.zeros((len(dvalues),len(mvalues),2))
    e_test = np.zeros((len(dvalues),len(mvalues),2))
    n = Xfull.shape[0]

    nm.nm_x_test = X_test
    nm.nm_y_test = y_test
    d_test = y_test.size

    #loop through d and m values storing wall time and testing error
    for i,d in enumerate(dvalues):
        nm.nm_x = Xfull[:,:d]
        nm.nm_y = yfull[:d]
        for j,m in enumerate(mvalues):
            for k,omethod in enumerate([1,2]):
                print("i,j,k=",i,j,k)
                args_net = (n,m,d)
                if m==0: #SNM
                    fvec = np.random.randn(n+1)
                    args = (d,)
                    if omethod==1: #BFGS
                        t1=time()
                        res = minimize(nm.snmodel,fvec,args,method='L-BFGS-B',jac=True)
                        fvec_f = res.x
                        t2=time()

                    else: #SGD
                        t1=time()
                        fvec_f = nm.sgd(fvec,n,m,d,0.1)
                        t2 = time()
                    #Compute testing error
                    z = np.dot(fvec_f[:-1],X_test) + fvec_f[-1]
                    a_int = np.round(1.0/(1.0+np.exp(-z)))
                    eps = np.abs(a_int - y_test)
                    e_test[i,j,k] = eps.sum()/y_test.size
                else: #NNM
                    fvec = np.random.randn(m*(n+2)+1)
                    args = (n,m,d)
                    if omethod==1: #BFGS
                        t1=time()
                        res = minimize(nm.nnmodel,fvec,args,method='L-BFGS-B',jac=True)
                        fvec_f = res.x
                        t2=time()

                    else: #SGD
                        t1=time()
                        fvec_f = nm.sgd(fvec,n,m,d,0.1)
                        t2 = time()

                    e_test[i,j,k] = nm.run_nnmodel(fvec_f,n,m,d_test)
                dt[i,j,k] = t2-t1


    plt.figure()
    plt.semilogx(dt,e_test[:,:,0],'x--')
    plt.xlabel('time (s)')
    plt.ylabel('testing error')
    plt.title('Testing error vs. wall time for l-bfgs-b classification calculations \n d=%s' %str(dvalues))
    plt.legend(('SNM','m=1','2','4'))

    return (e_test,dt)
#--------------------------------------------



if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code

    #output = nm_analyze()

    #Getting data
    X,y,X_test,y_test = read_data()

    figure = [True,False]

    #ADD ANALYSE:
    nm_analyze(X,y,X_test,y_test,figure)
