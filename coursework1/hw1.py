"""M3C 2018 Homework 1
   Name: Tudor Mihai Trita Trita
   Course: Mathematics Year 3
   CID: 01199397
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time

def simulate1(N,Nt,b,e):
	"""Simulate C vs. M competition on N x N grid over
	Nt generations. b and e are model parameters
	to be used in fitness calculations
	Output: S: Status of each gridpoint at tend of somulation, 0=M, 1=C
	fc: fraction of villages which are C at all Nt+1 times
	Do not modify input or return statement without instructor's permission.

	Strategy to achieve the correct output:
	1. Calculate fitness of C's and M's separately
	2. Calculate probability of villages remaining the same the next timestep and at the same time:
	3. Execute changes to villages on S
	"""

	#Set initial condition
	S = np.ones((N,N),dtype=int)
	j = int((N-1)/2)
	S[j-1:j+2,j-1:j+2]=0

	#Initialise matrix of ones
	Eye = np.ones((N,N),dtype=int)

	#Fraction of points which are C
	fc = np.zeros(Nt+1)
	fc[0] = S.sum()/(N*N)

	#Initialise fitness matrices

	CFit = np.zeros((N,N),dtype=float) #fitness of C's
	MFit = np.zeros((N,N),dtype=float) #fitness of M's
	TFit = np.zeros((N,N),dtype=float) #total fitness


	#----------------- Main loop begins

	for tstep in range(Nt):

		#Calculate 'complement' of S, where the values of 1's and 0's are swapped
		#This is used for the calculation of fitness of M villages
		Scomp = Eye-S


		#-------------- Fitness calculations begin

		#---------------Calculating fitness of villages in the corners
		#Achieved by slicing S and Scomp accordingly and summing through entries of slices

		nghbs = 3 #each corner village has three neighbours

		#"Bottom left"
		if S[0,0]==1:
			CFit[0,0]=((S[0:2,0:2]).sum()-1)/nghbs # The '-1' to account for target village point. slices are 2x2
		else:
			MFit[0,0]=(S[0:2,0:2].sum()*b+(Scomp[0:2,0:2].sum()-1)*e)/nghbs #C villages get b points, M villages get e points

		#"Bottom right"
		if S[0,N-1]==1:
			CFit[0,N-1]=(S[0:2,N-2:N].sum()-1)/nghbs
		else:
			MFit[0,N-1]=((S[0:2,N-2:N].sum()*b+(Scomp[0:2,N-2:N].sum()-1)*e))/nghbs

		#"Top left"
		if S[N-1,0]==1:
			CFit[N-1,0]=((S[N-2:N,0:2]).sum()-1)/nghbs
		else:
			MFit[N-1,0]=(S[N-2:N,0:2].sum()*b+(Scomp[N-2:N,0:2].sum()-1)*e)/nghbs

		#"Top right"
		if S[N-1,N-1]==1:
			CFit[N-1,N-1]=(S[N-2:N,N-2:N].sum()-1)/nghbs
		else:
			MFit[N-1,N-1]=(S[N-2:N,N-2:N].sum()*b+(Scomp[N-2:N,N-2:N].sum()-1)*e)/nghbs


		#---------------- Moving on to border villages fitness calculations

		nghbs = 5 #each border village has five neighbours

		for i in range(1,N-1):
			#First to do "bottom border"
			if S[0,i]==1:
				CFit[0,i]=(S[0:2,i-1:i+2].sum()-1)/nghbs #slices are either 3x2 or 2x3
			else:
				MFit[0,i]=(S[0:2,i-1:i+2].sum()*b+(Scomp[0:2,i-1:i+2].sum()-1)*e)/nghbs

			#"top border"
			if S[-1,i]==1:
				CFit[-1,i]=(S[-2:,i-1:i+2].sum()-1)/nghbs
			else:
				MFit[-1,i]=(S[-2:,i-1:i+2].sum()*b+(Scomp[-2:,i-1:i+2].sum()-1)*e)/nghbs
            if S[i,0] == 1:
                CFit[i,0]=(S[i-1:i+2,0:2].sum()-1)/nghbs
            else:
				MFit[i,0]=(S[i-1:i+2,0:2].sum()*b+(Scomp[i-1:i+2,0:2].sum()-1)*e)/nghbs
			#"right border"
            if S[i,-1]==1:
				CFit[i,-1]=(S[i-1:i+2,-2:].sum()-1)/nghbs
			else:
				MFit[i,-1]=(S[i-1:i+2,-2:].sum()*b+(Scomp[i-1:i+2,-2:].sum()-1)*e)/nghbs


		#------------Moving on to rest of villages fitness calculations

		nghbs = 8 #all other villages have 8 nghbs,

		for i in range(1, N-1):
			for k in range(1, N-1):
				if S[k,i]==1:
					CFit[k,i]=(S[k-1:k+2,i-1:i+2].sum()-1)/nghbs #slices are 3x3
				else:
					MFit[k,i]=(S[k-1:k+2,i-1:i+2].sum()*b+(Scomp[k-1:k+2,i-1:i+2].sum()-1)*e)/nghbs

		#------------Fitness calculations finished for all villages

		#######################################################################################

		#--Begin probability calculations combined with changes to matrix S

		#Preliminary calculations
		TFit = MFit+CFit #Matrix storing total fitness of community around villages

		#------------------Probability/S corner villages
		#Similar method as calculating fitness, though now slicing through CFit and MFit
		#decision function is used, see below...

		#"Bottom left"
		if S[0,0]==1:
			S[0,0]=decision(CFit[0:2,0:2].sum()/TFit[0:2,0:2].sum()) #This is decision(P(C village in next turn))
		else:
			S[0,0]=decision(1-(MFit[0:2,0:2].sum()/TFit[0:2,0:2].sum())) #This is decision(1-P(M village in next turn))

		#"Bottom right"
		if S[0,N-1]==1:
			S[0,N-1]=decision(CFit[0:2,N-2:N].sum()/TFit[0:2,N-2:N].sum())
		else:
			S[0,N-1]=decision(1-(MFit[0:2,N-2:N].sum()/TFit[0:2,N-2:N].sum()))
		#"Top left"
		if S[N-1,0]==1:
			S[N-1,0]=decision(CFit[N-2:N,0:2].sum()/TFit[N-2:N,0:2].sum())
		else:
			S[N-1,0]=decision(1-(MFit[N-2:N,0:2].sum()/TFit[N-2:N,0:2].sum()))
		#"Top right"
		if S[N-1,N-1]==1:
			S[N-1,N-1]=decision(CFit[N-2:N,N-2:N].sum()/TFit[N-2:N,N-2:N].sum())
		else:
			S[N-1,N-1]=decision(1-(MFit[N-2:N,N-2:N].sum()/TFit[N-2:N,N-2:N].sum()))


		#------------------Probability/S border villages

		for i in range(1,N-1):
			#First to do "bottom border"
			if S[0,i]==1:
				S[0,i]=decision(CFit[0:2,i-1:i+2].sum()/TFit[0:2,i-1:i+2].sum())
			else:
				S[0,i]=decision(1-(MFit[0:2,i-1:i+2].sum()/TFit[0:2,i-1:i+2].sum()))
			#"top border"
			if S[-1,i]==1:
				S[-1,i]=decision(CFit[-2:,i-1:i+2].sum()/TFit[-2:,i-1:i+2].sum())
			else:
				S[-1,i]=decision(1-(MFit[-2:,i-1:i+2].sum()/TFit[-2:,i-1:i+2].sum()))
			#"left border"
			if S[i,0]==1:
				S[i,0]=decision(CFit[i-1:i+2,0:2].sum()/TFit[i-1:i+2,0:2].sum())
			else:
				S[i,0]=decision(1-(MFit[i-1:i+2,0:2].sum()/TFit[i-1:i+2,0:2].sum()))
			#"right border"
			if S[i,-1]==1:
				S[i,-1]=decision(CFit[i-1:i+2,-2:].sum()/TFit[i-1:i+2,-2:].sum())
			else:
				S[i,-1]=decision(1-(MFit[i-1:i+2,-2:].sum()/TFit[i-1:i+2,-2:].sum()))

		#--------------------Probability/S rest of villages

		for i in range(1, N-1):
			for k in range(1, N-1):
				if S[k,i]==1:
					S[k,i]=decision(CFit[k-1:k+2,i-1:i+2].sum()/TFit[k-1:k+2,i-1:i+2].sum())
				else:
					S[k,i]=decision(1-(MFit[k-1:k+2,i-1:i+2].sum()/TFit[k-1:k+2,i-1:i+2].sum()))

		#---------------------Changes in S completed

		#Resetting fitness matrices
		CFit[:,:],MFit[:,:],TFit[:,:]=0,0,0

		#Updating entry in fc
		fc[tstep+1] = S.sum()/(N*N)

	return S,fc

def analyze(f1,f2,f3):
	""" Analysis of qualitative trends found:
	I will first focus on the case where N=21 and e=0.01:

	Case 1, b>1.2: the evolution of the average of fc quickly goes to zero as time passes, and as
	b increases, the rate of fc going to zero increases. This can be seen in figure1.

	Case 2 1.05<b<1.2: the evolution of the average of fc stabilises between 0 and 1, and the rate
	at which it stabilises is faster the larger b. This can be seen in figure1.

	Case 3 b<1.01: C villages tend to wipe out M villages, as the average of fc over time
	is very close to 1.

	Regarding the dynamics of the process in the three cases, illustrated in figure3:
	Case 1, b>1.1: Most orbits of fc head to zero by Nt=125. The orbits will head to zero faster
	if b is increased.
	Case 2, 1.05<b<1.1: Orbits of fc are unpredictable, and many will either end up at 1 in the beginning
	or will head to zero eventually. In this case, fc does not reach a relatively stable value at long times.
	Case 3, b<1.01: Most orbits of fc go to 1 quickly. There is some deviation by a few stray orbits.

	It can be said that fc reaches a relatively stable value at long times for values of b larger than 1.1.
	For values of b below 1.1, fc has unstable behaviour at long times

	Now we turn to case where N=21 and b=1.01:
	Figure2 illustrates the fact that in this case, increasing the value of e is negatively correlated
	with the ratio of fc at long times.

	Similar trends are found when altering the value of N.
	"""

	#Function to plot figure 1
	if f1==True:
		N,Nt,bs,e,iters=21,250,[1.01,1.05,1.075,1.1,1.15,1.2,1.25,1.3,1.35],0.01,50
		figure = figure1(N,Nt,bs,e,iters)


	#Function to plot figure 2
	if f2==True:
		N,Nt,b,es,iters=21,250,1.01,[0.01,0.1,0.2,0.3,0.4],45
		figure = figure2(N,Nt,b,es,iters)

	#To go to figure3
	if f3==True:
		N,Nt,b,e=21,250,[1.2,1.075,1.06,1.01],0.01
		figure = figure3(N,Nt,b,e)

	message = 'All calculations have finished'
	return message

def figure1(N,Nt,bs,e,iters):
	"""Function to plot behaviour of averages of fc against time for
	differnet values of 1.01<=b<=1.35 given fixed e=0.01
	Input: N,Nt,bs,e,iters
	Uses the function simulate1 iters number of times then averages to create mean of fc through time
	Plots the various graphs and saves plots as hw11.jpg
	"""

	#Auxiliary variables
	fcmeanmat = []

	for i in bs:
		fcmean = np.zeros(Nt+1)
		for j in range(iters):
			fc = simulate1(N,Nt,i,e)[1] #Getting fc values
			fcmean += fc
		fcmean /= iters #Calculating average of fc
		fcmeanmat.append(fcmean)

	Ntarray = range(Nt+1)

	plt.figure(figsize=(13,8))

	#plotting all difference averages of fc

	plt.plot(Ntarray,fcmeanmat[0],label='b=1.01')
	plt.hold(True)
	plt.plot(Ntarray,fcmeanmat[1],label='b=1.05')
	plt.plot(Ntarray,fcmeanmat[2],label='b=1.075')
	plt.plot(Ntarray,fcmeanmat[3],label='b=1.1')
	plt.plot(Ntarray,fcmeanmat[4],label='b=1.15')
	plt.plot(Ntarray,fcmeanmat[5],label='b=1.2')
	plt.plot(Ntarray,fcmeanmat[6],label='b=1.25')
	plt.plot(Ntarray,fcmeanmat[7],label='b=1.3')
	plt.plot(Ntarray,fcmeanmat[8],label='b=1.35')
	plt.hold(False)

	plt.xlabel('Nt, Amount of Time Elapsed')
	plt.ylabel('fc, Proportion of Villages which are C ')

	plt.title("Name: Tudor Trita Trita, CID: 01199397 \n Function: figure1 \n Behaviour of Average of fc Against Time for Different Values of b given e=0.01")

	plt.legend()
	plt.grid(True)

	plt.savefig('hw11.png')
	plt.show()
	return

def figure2(N,Nt,b,es,iters):
	"""Function to plot behaviour of averages of fc against time for
	differnet values of 0.01<=e<=0.4 given fixed b=1.01
	Input: N,Nt,bs,e,iters
	Uses the function simulate1 iters number of times then averages to create mean of fc through time
	Plots the various graphs and saves plots as hw12.jpg
	"""

	#Auxiliary variables
	fcmeanmat = []

	for i in es:
		fcmean = np.zeros(Nt+1)
		for j in range(iters):
			fc = simulate1(N,Nt,b,i)[1] #calculates fc through time
			fcmean += fc #calculates mean of fc
		fcmean /= iters
		fcmeanmat.append(fcmean)

	Ntarray = range(Nt+1)

	plt.figure(figsize=(13,8))

	#plots means of fc through time

	plt.plot(Ntarray,fcmeanmat[0],label='e=0.01')
	plt.hold(True)
	plt.plot(Ntarray,fcmeanmat[1],label='e=0.1')
	plt.plot(Ntarray,fcmeanmat[2],label='e=0.2')
	plt.plot(Ntarray,fcmeanmat[3],label='e=0.3')
	plt.plot(Ntarray,fcmeanmat[4],label='e=0.4')

	plt.hold(False)

	plt.xlabel('Nt, Amount of Time Elapsed')
	plt.ylabel('fc, Proportion of Villages which are C ')

	plt.title("Name: Tudor Trita Trita, CID: 01199397 \n Function: figure2 \n Behaviour of fc Against Time for Different Values of e given b=1.01")

	plt.legend()
	plt.grid(True)

	plt.savefig('hw12.png')
	plt.show()
	return

def figure3(N,Nt,b,e):
	"""Function to plot behaviour of random fc's against time for
	differnet values of 1.01<=b<=1.2 given fixed e=0.01
	Input: N,Nt,bs,e
	Uses the function simulate1 to create many versions of fc,
	to investigate dynamics for different values of b.
	Plots the various graphs and saves plots as hw13.jpg
	"""

	#Initialising fc matrices for different values of b
	fcmat0 = []
	fcmat1 = []
	fcmat2 = []
	fcmat3 = []

	Ntarray = range(Nt+1)
	iters=20 #Number of iterations


	for i in range(iters): #Fetching different random fc values
		fcmat0.append(simulate1(N,Nt,b[0],e)[1])
		fcmat1.append(simulate1(N,Nt,b[1],e)[1])
		fcmat2.append(simulate1(N,Nt,b[2],e)[1])
		fcmat3.append(simulate1(N,Nt,b[3],e)[1])

	#Plotting 4 subfigures
	plt.figure(figsize=(15,10))

	plt.hold(True)

	plt.subplot(2,2,1)
	for i in range(iters):
		plt.plot(Ntarray,fcmat0[i])
	plt.grid(True)
	plt.xlabel('Nt')
	plt.ylabel('fc')
	plt.title('b=1.2')


	plt.subplot(2,2,2)
	for i in range(iters):
		plt.plot(Ntarray,fcmat1[i])
	plt.grid(True)
	plt.xlabel('Nt')
	plt.ylabel('fc')
	plt.title('b=1.075')

	plt.subplot(2,2,3)
	for i in range(iters):
		plt.plot(Ntarray,fcmat2[i])
	plt.grid(True)
	plt.xlabel('Nt')
	plt.ylabel('fc')
	plt.title('b=1.06')


	plt.subplot(2,2,4)
	for i in range(iters):
		plt.plot(Ntarray,fcmat3[i])
	plt.grid(True)
	plt.xlabel('Nt')
	plt.ylabel('fc')
	plt.title('b=1.01')

	plt.hold(False)

	plt.suptitle("Name: Tudor Trita Trita, CID: 01199397 \n Function: figure3 \n Behaviour of many fc's against Nt over time for different \n values of b with e=0.01 demonstrating different dynamics")

	plt.savefig('hw13.png')
	plt.show()
	return

def decision(p):
    """Function to execute probabilities
    outputting 1 with probability p
    """
    if random.random() < p:
        return 1
    else:
        return 0

def plot_S(S):
    """Simple function to create plot from input S matrix
    """
    ind_s0 = np.where(S==0) #C locations
    ind_s1 = np.where(S==1) #M locations
    plt.plot(ind_s0[1],ind_s0[0],'rs',ind_s1[1],ind_s1[0],'bs')
    plt.show()
    plt.pause(0.05)
    return None

if __name__ == '__main__':

    #Choose which figure to generate below below
	f1,f2,f3=False,False,False

	output = analyze(f1,f2,f3)
	print(output)
