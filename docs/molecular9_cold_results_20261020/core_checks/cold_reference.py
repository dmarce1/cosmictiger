"""Independent binary64 reference using CDS downward data, SI constants, RK4."""
import json,math,pathlib
root=pathlib.Path(__file__).resolve().parent.parent/'reference_data'
rows={tuple(map(int,s.split()[:4])):list(map(float,s.split()[4:])) for s in (root/'table2.dat').read_text().splitlines()}
keys=((2,0,0,0),(2,1,0,1),(3,0,1,0),(3,1,1,1))
weights=(1/16,3/16,3/16,9/16)
kb=8.617333262145e-5;ev=1.602176634e-12

def cool(T):
 value=0.
 for key,weight in zip(keys,weights):
  gap,*p,error=rows[key]
  down=sum(coefficient*T**j for j,coefficient in enumerate(p))*1e-14
  degeneracy=(2*key[0]+1)*(2*key[1]+1)/((2*key[2]+1)*(2*key[3]+1))
  value+=weight*down*degeneracy*math.exp(-gap/T)*kb*gap
 return value

def capacity(T):
 result=1.5
 for parity,weight in ((0,.25),(1,.75)):
  levels=[85.4*(j*(j+1)-2*parity) for j in range(parity,202,2)]
  w=[(2*j+1)*math.exp(-e/T) for j,e in zip(range(parity,202,2),levels)]
  z=sum(w);mean=sum(e*v for e,v in zip(levels,w))/z
  result+=weight*(sum(e*e*v for e,v in zip(levels,w))/z-mean*mean)/T**2
 q=5987/T;occ=math.exp(-q);result+=q*q*occ/(1-occ)**2
 return .5*kb*result

def rhs(T):
 t=T/1000
 lte=(9.5e-22/ev)*t**3.76/(1+.12*t**2.1)*math.exp(-(0.13/t)**3)+(3e-24/ev)*math.exp(-.51/t)+(6.7e-19/ev)*math.exp(-5.86/t)+(1.6e-18/ev)*math.exp(-11.7/t)
 low=1e4*.5*cool(T)
 return -.5*low/(1+low/lte)/capacity(T)

def solve(n):
 # Match binary32 interval used by runtime driver exactly.
 interval=99999997952.;h=interval/n;t=80.
 for step in range(n):
  a=rhs(t);b=rhs(t+.5*h*a);c=rhs(t+.5*h*b);d=rhs(t+h*c)
  t+=h*(a+2*b+2*c+d)/6
 return t
result={'table2_anchors_ev':{str(T):cool(T) for T in (10,20,30,50,80)},'rk4_temperature_1000_steps':solve(1000),'rk4_temperature_2000_steps':solve(2000)}
print(json.dumps(result,indent=2))
