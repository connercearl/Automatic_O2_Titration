"""
This code was originally developed by the following authors
Conner Earl, Emilee Hunter, Miranda Burnham, Payden Yates
December 14, 2017
"""






import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d

Patm = 760 #mmHg atmospheric pressure
PH2O = 20 #mmHg partial pressure of Water in lungs
PalvCO2 = 40 #mmHg partial pressure of CO2
Pgrad = 10 #mmHG pressure gradient between alveoli and blood

#values (assuming atmospheric)
Ptot=760 #mm Hg
T = 36+273.15 #K
R=40 # Respiratory Coeficient 
V=2 #m^3
n=((Ptot*(101325/760))*V)/(R*T) #total moles in the incubator
t_const=1 #sec (time constants)
q=n/t_const
tau=5 #sec for the dynamic model 

#paramters 
m=.002 # coeficient to describe how change in heart rate disturbs SpO2
A=150 # coeficients used for blood oxygen saturation
B=500000 # adjusted coeficient to represent low compliance in baby


def model(x,t,u,H):
   #phyical model
   FiO2=x[0] #Initial condition for Fraction of inspired oxygen
   SpO2=x[1] #Initial condition for Satruated oxygen

   PO2 = FiO2*(Patm - PH2O)-(PalvCO2/R)-Pgrad #describes arterial PO2 from Patm
   SpO2_ss=((((PO2**3+A*PO2)**-1*B)+1)**-1) #hemoglobin saturation curve
   dFiO2dt=((u-FiO2)*(q/n)) #Oxygen changing in incubator
   dSpO2dt=( -SpO2 + SpO2_ss + m*(H-H_ss))*(1/tau) #SPO2 changing in baby
   
   return np.array([dFiO2dt,dSpO2dt])

x01 = np.array([0.21, 0.95]) # initial conditions 
u_ss = 0.21 # mole fraction
H_ss = 130.0 # HR
tf = 30      # simulate for 30 sec
ns = (tf*4)+1  # sample time = 10 min

t = np.linspace(0,tf,ns)
FiO2 = np.ones(len(t)) * x01[0]  #creating vectors to store variables
SpO2 = np.ones(len(t)) * x01[1]
u = np.ones(len(t)) * u_ss
H = np.ones(len(t)) * H_ss

# Doublet test
u[0:] = 0.30  
u[25:50] = 0.05
u[50:75] = 0.1
u[75:] = 0.21

plt.figure(figsize=(10,7))
#odeint
for i in range(len(t)-1):
   ts = [t[i],t[i+1]]
   y = odeint(model,x01,ts,args=(u[i+1],H[i+1]))
   FiO2[i+1] = y[-1][0]
   SpO2[i+1] = y[-1][1]
   x01[0]=FiO2[i+1]
   x01[1] = SpO2[i+1]
#plotting
ticks = np.linspace(0,tf,5)
ax=plt.subplot(3,1,1)
ax.grid() # turn on grid
plt.plot(t[0:i+1],u[0:i+1],'b--',linewidth=3,label='yO2')
plt.plot(t[0:i+1],FiO2[0:i+1],'r-',linewidth=3,label='FiO2')
plt.ylabel('Fraction')
plt.legend(loc='best')
plt.xlim([0,tf])
plt.xticks(ticks)
ax=plt.subplot(3,1,2)
ax.grid() # turn on grid

plt.plot(t[0:i+1],SpO2[0:i+1],'k.-',linewidth=3,label='SpO2')
plt.ylabel('Fraction')
plt.legend(loc=2)
plt.xlim([0,tf])
plt.xticks(ticks)
ax=plt.subplot(3,1,3)
ax.grid() # turn on grid
plt.plot(t[0:i+1],H[0:i+1],'g:',linewidth=3,label='HR')
plt.ylabel('Heart Rate')
plt.xlabel('Time (min)')
plt.xlim([0,tf])
plt.legend(loc='best')
plt.xticks(ticks)
plt.show()

data=np.vstack((t,u,SpO2))
data=data.T
np.savetxt('data.txt',data, delimiter=',')
#%%
# Used to find FOPDT parameters
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
x0[0] = .1 # Km
x0[1] = 41 # taum
x0[2] = 3 # thetam

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


#%%
#PI CONTROLLER

#Heart rate disturbance variabel
mydata = np.loadtxt('simulated_HR.csv',delimiter=',') 
#PI tuning
Kp = x[0]
taup = x[1]
thetap = x[2]
tauc = max(10*taup,80*thetap)
#Agressive tuning to meet specifications
Kc = ((1.0/Kp)*((taup + 0.5*thetap)/(tauc + 0.5*thetap)))*135
tauI = (taup + 0.5*thetap)*4.5
tauD = ((taup*thetap)/(2*taup + thetap))*0 # taud=0 PI controller

print('PI CONTROLLER:')
print(' ')
print('Kc: ' + str(Kc))
print('tauI: ' + str(tauI))
print('tauD: ' + str(tauD))

FiO2 = np.ones(len(t)) * x01[0]
SpO2 = np.ones(len(t)) * x01[1] #Create vector to store variables
u = np.ones(len(t)) * u_ss #input variable (yO2)
H = np.ones(len(t)) * H_ss #Heart rate disturbances
op = np.zeros(len(t))  # controller output
pv = np.zeros(len(t))  # process variable
e = np.zeros(len(t))   # error
ie = np.zeros(len(t))  # integral of the error
dpv = np.zeros(len(t)) # derivative of the pv
P = np.zeros(len(t))   # proportional
I = np.zeros(len(t))   # integral
D = np.zeros(len(t))   # derivative
sp = np.zeros(len(t))  # set point

sp[0:] = 0.95 #initial setpoitnt
op_hi = 1
op_lo = 0.21

for i in range(len(mydata)):
    H[i] = mydata[i]
#Heart rate disturbance data 
pv[0] = .80 #Initial parameter infant in respiratory distress

# Create plot
plt.figure(figsize=(10,7))

# Note: Animating plot slows down script
animate = False #True
if animate:
    plt.ion()
    plt.show()

mydata = np.genfromtxt('simulated_HR.csv')

for i in range(len(t)-1):
    delta_t = t[i+1]-t[i]
    e[i] = sp[i] - pv[i]
    if i >= 1:  # calculate starting on second cycle
        dpv[i] = (pv[i]-pv[i-1])/delta_t
        ie[i] = ie[i-1] + e[i] * delta_t
    P[i] = Kc * e[i]
    I[i] = Kc/tauI * ie[i]
    D[i] = - Kc * tauD * dpv[i]
    op[i] = op[0] + P[i] + I[i] + D[i]
    if op[i] > op_hi:  # check upper limit
        op[i] = op_hi
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    if op[i] < op_lo:  # check lower limit
        op[i] = op_lo
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    ts = [t[i],t[i+1]]
    u[i+1] = op[i]
    y = odeint(model,x01,ts,args=(u[i+1],H[i+1]))
    FiO2[i+1] = y[-1][0]
    SpO2[i+1] = y[-1][1]
    x01[0] = FiO2[i+1]
    x01[1] = SpO2[i+1]
    pv[i+1] = SpO2[i+1]
    op[len(t)-1] = op[len(t)-2]
    ie[len(t)-1] = ie[len(t)-2]
    P[len(t)-1] = P[len(t)-2]
    I[len(t)-1] = I[len(t)-2]
    D[len(t)-1] = D[len(t)-2]

    if animate or i==(len(t)-2):
        # clear plot if animating
        if animate:
            plt.clf()
            
        ticks = np.linspace(0,tf,5)
        ax=plt.subplot(3,1,1)
        ax.grid() # turn on grid
        plt.plot(t[0:i+1],u[0:i+1],'b--',linewidth=3,label='yO2')
        plt.plot(t[0:i+1],FiO2[0:i+1],'r-',linewidth=3,label='FiO2')
        plt.ylabel('Fraction')
        plt.legend(loc='best')
        plt.xlim([0,tf])
        plt.xticks(ticks)
        
        ax=plt.subplot(3,1,2)
        ax.grid() # turn on grid
        plt.plot(t[0:i+1],SpO2[0:i+1],'k.-',linewidth=3,label='SpO2')
        plt.plot(t,sp,'r--',linewidth=2,label='Set Point')
        plt.plot([0,30],[.98,.98],'k--',linewidth=2,label='Limits')
        plt.plot([0,30],[.92,.92],'k--',linewidth=2)
        #plt.plot(t[0:i+1],S[0:i+1],'b-',linewidth=3,label='S')
        plt.ylabel('Fraction')
        plt.legend(loc='best')
        plt.xlim([0,tf])
        plt.xticks(ticks)
        plt.ylim([.75,1])
        
        ax=plt.subplot(3,1,3)
        ax.grid() # turn on grid
        plt.plot(t[0:i+1],H[0:i+1],'g:',linewidth=3,label='HR')
        plt.ylabel('Heart Rate')
        plt.xlabel('Time (min)')
        plt.xlim([0,tf])
        plt.legend(loc='best')
        plt.xticks(ticks)

        if animate:
            plt.draw()
            plt.pause(.001)
        else:
            plt.show()
            
