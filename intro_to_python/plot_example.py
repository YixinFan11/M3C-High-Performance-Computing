"""Illustrative example of plotting with matplotlib
To run: import plot_example

"""



import numpy as np
import matplotlib.pyplot as plt #Import matplotlib


#Create some arrays to be plotted
Nx = 100
Ny = 200
x = np.linspace(0.0,np.pi,Nx)
y = np.linspace(-np.pi,np.pi,Ny)

f = np.sin(x)
g = np.cos(y)



#Create plot
plt.figure() #make new figure

plt.plot(x,f,'b-',label='sin') #blue line
plt.plot(y,g,'r--',label='cos') #red dashed line

#add axis labels,legend, and figure title
plt.xlabel('time')
plt.ylabel('f(t),g(t)')
plt.legend(loc='best')
plt.title('Illustrative figure prepared by Prasun Ray')

#adjust x-axis limits, turn on grid, display and save figure
plt.xlim(0,np.pi)
plt.grid()
plt.show()

plt.savefig('plot_example.png')





