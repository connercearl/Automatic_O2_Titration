import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

Patm = 760 #mmHg atmospheric pressure
PH2O = 20 #mmHg partial pressure of Water in lungs
PalvCO2 = 40 #mmHg partial pressure of CO2
Pgrad = 10 #mmHG pressure gradient between alveoli and blood


def SpO2(FiO2,A,B,R):
    PO2 = FiO2*(Patm - PH2O)-(PalvCO2/R)-Pgrad
    return ((((PO2**3+A*PO2)**-1*B)+1)**-1)*100

A = [150,150,150]
B = [23400,15000,50000]
R = [.2,.8,.8]

FiO2 = np.linspace(.21,.5,50)

y1 = SpO2(FiO2,A[0],B[0],R[1])
y2 = SpO2(FiO2,A[1],B[1],R[1])
y3 = SpO2(FiO2,A[2],B[2],R[1])


plt.plot(FiO2,y1,'b-',label='1')
plt.plot(FiO2,y2,'k-',label='2')
plt.plot(FiO2,y3,'r-',label='3')
plt.ylabel('SPO2')
plt.xlabel('FiO2')
plt.legend(loc='best')

#%%
#model

#values (assuming atmospheric)
Ptot=760 #mm Hg
T = 36+273.15 #K
R=8.31446 #J/mol*K
V=2 #m^3
n=((Ptot*(101325/760))*V)/(R*T)
t_const=1 #min (time constants)
q=n/t_const
tau=5 #min for the dynamic model

m=.08575

#paramters
A=150
B=50000



def model(x,t,u,H):
    #phyical model
    FiO2=x[0]
    SpO2=x[1]

    PO2 = FiO2*(Patm - PH2O)-(PalvCO2/R)-Pgrad
    SpO2_ss=((((PO2**3+A*PO2)**-1*B)+1)**-1)

    dFiO2dt=((u-FiO2)*(q/n)) #-(m*H)/n)
    dSpO2dt=( -SpO2 + SpO2_ss)*(1/tau)
    return np.array([dFiO2dt,dSpO2dt])

t=np.linspace(0,500,500)

FiO2= np.ones(len(t)) *.21
SpO2= np.ones(len(t)) * .95


u=np.ones(len(t))
#u=np.linspace(0,.21,500)
u[0:200]=.20
u[200:300]=.05
u[300:]=.20


#H_rate=np.zeros(len(t))
#H_rate[0:500]=130


x01 = np.empty(2)
x01[0] = .21
x01[1] = .95

for i in range(len(t)-1):
    ts = [t[i],t[i+1]]
    y = odeint(model,x01,ts,args=(u[i+1],H_rate[i+1]))
    FiO2[i+1] = y[-1][0]
    SpO2[i+1] = y[-1][1]
    x01[0]=FiO2[i+1]
    x01[1] = SpO2[i+1]
plt.figure
plt.subplot(3,1,1)
plt.plot(t,FiO2,'r',label='FiO2')
plt.plot(t,u,'m',label='Oxygen Fraction')

plt.ylabel('fraction of O2')
plt.legend(loc='best')
plt.subplot(3,1,2)
plt.plot(t,SpO2,'b--',label='SpO2')
plt.legend(loc='best')
#plt.subplot(3,1,3)
#plt.plot(t,H_rate,'c:',label='Heart Rate')
#plt.legend(loc='best')
#plt.xlabel('time')
#plt.show()


data = np.vstack((t,u,SpO2)) # vertical stack
data = data.T             # transpose data
np.savetxt('data.txt',data,delimiter=',')


#%%

from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Import CSV data file
# Column 1 = time (t)
# Column 2 = input (u)
# Column 3 = output (yp)
data = np.loadtxt('data.txt',delimiter=',')
u0 = data[0,1]
yp0 = data[0,2]
t = data[:,0].T - data[0,0]
u = data[:,1].T
yp = data[:,2].T

# specify number of steps
ns = len(t)
delta_t = t[1]-t[0]
# create linear interpolation of the u data versus time
uf = interp1d(t,u)

# define first-order plus dead-time approximation
def fopdt(y,t,uf,Km,taum,thetam):
    # arguments
    #  y      = output
    #  t      = time
    #  uf     = input linear function (for time shift)
    #  Km     = model gain
    #  taum   = model time constant
    #  thetam = model time constant
    # time-shift u
    try:
        if (t-thetam) <= 0:
            um = uf(0.0)
        else:
            um = uf(t-thetam)
    except:
        #print('Error with time extrapolation: ' + str(t))
        um = u0
    # calculate derivative
    dydt = (-(y-yp0) + Km * (um-u0))/taum
    return dydt

# simulate FOPDT model with x=[Km,taum,thetam]
def sim_model(x):
    # input arguments
    Km = x[0]
    taum = x[1]
    thetam = x[2]
    # storage for model values
    ym = np.zeros(ns)  # model
    # initial condition
    ym[0] = yp0
    # loop through time steps
    for i in range(0,ns-1):
        ts = [t[i],t[i+1]]
        y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
        ym[i+1] = y1[-1]
    return ym

# define objective
def objective(x):
    # simulate model
    ym = sim_model(x)
    # calculate objective
    obj = 0.0
    for i in range(len(ym)):
        obj = obj + (ym[i]-yp[i])**2
    # return result
    return obj

# initial guesses
x0 = np.zeros(3)
x0[0] = 2.0 # Km
x0[1] = 3.0 # taum
x0[2] = 0.0 # thetam

# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize Km, taum, thetam
solution = minimize(objective,x0)

# Another way to solve: with bounds on variables
#bnds = ((0.4, 0.6), (1.0, 10.0), (0.0, 30.0))
#solution = minimize(objective,x0,bounds=bnds,method='SLSQP')
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

print('Kp: ' + str(x[0]))
print('taup: ' + str(x[1]))
print('thetap: ' + str(x[2]))

# calculate model with updated parameters
ym1 = sim_model(x0)
ym2 = sim_model(x)
# plot results
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,yp,'kx-',linewidth=2,label='Process Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
plt.ylabel('Output')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(t,u,'bx-',linewidth=2)
plt.plot(t,uf(t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.ylabel('Input Data')
plt.show()