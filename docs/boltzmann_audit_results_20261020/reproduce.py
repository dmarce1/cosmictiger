import numpy as np
from scipy.linalg import expm
import math,json

g=1-1/math.sqrt(2)
print('Variable opacity yprime=-exp(x)y on [0,1], exact=',math.exp(1-math.e))
for h in [.1,.05,.025,.0125,.00625]:
    n=round(1/h); h=1/n
    vals=[]
    for shifted in [True,False]:
        y=1.
        for j in range(n):
            x=j*h
            t=x+(.25*h if shifted else 0)
            lam1=-math.exp(t+g*h)
            f1=lam1*y/(1-g*h*lam1)
            tmp=y+(1-2*g)*h*f1
            lam2=-math.exp(t+(1-g)*h)
            f2=lam2*tmp/(1-g*h*lam2)
            y+=.5*h*(f1+f2)
        vals.append(y)
    print(h, 'as-coded=',vals[0],'relative error=',vals[0]/math.exp(1-math.e)-1,'corrected times=',vals[1],'relative error=',vals[1]/math.exp(1-math.e)-1)

# Frozen-coefficient photon/baryon subsystem, no expansion/metric/pressure.
# State [delta_gamma, theta_gamma/k, theta_b/k]; k=1; R=4rho_gamma/(3rho_b)=1.
E=np.array([[0.,-4/3,0.],[.25,0.,0.],[0.,0.,0.]])
C0=np.array([[0.,0.,0.],[0.,-1.,1.],[0.,1.,-1.]])
I=np.eye(3)
period=2*math.pi*math.sqrt(6)
y0=np.array([1.,0.,0.])
print('Tight coupling, one acoustic period',period)
for n in [160,320,640,1280]:
    h=period/n
    Ep=I+E*h/2+(E@E)*(h/2)**2/2+(E@E@E)*(h/2)**3/6
    for sigma in [1e2,1e5,1e9]:
        C=sigma*C0
        B=np.linalg.solve(I-g*h*C,C)
        Cstep=I+.5*h*(B+B@(I+(1-2*g)*h*B))
        step=Ep@Cstep@Ep
        y=np.linalg.matrix_power(step,n)@y0
        ref=expm((E+C)*period)@y0
        amplitude=math.sqrt(y[0]**2+(y[1]+y[2])**2*2/3)
        print('n=',n,'h=',h,'sigma=',sigma,'delta=',y[0],'exact finite sigma delta=',ref[0],'diff=',y[0]-ref[0],'slip=',y[1]-y[2])
