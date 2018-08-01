

#  NLP written by GAMS Convert at 05/24/17 13:20:40
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#          1        1        0        0        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#          9        9        0        0        0        0        0        0
#  FX      3        3        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#          9        1        8        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(0,None))
m.x2 = Var(within=Reals,bounds=(0,None))
m.x3 = Var(within=Reals,bounds=(0,None))
m.x4 = Var(within=Reals,bounds=(0,None))
m.x5 = Var(within=Reals,bounds=(0,None))
m.x6 = Var(within=Reals,bounds=(-0.05,-0.05))
m.x7 = Var(within=Reals,bounds=(0.1,0.1))
m.x8 = Var(within=Reals,bounds=(0,0))
m.x9 = Var(within=Reals,bounds=(0,0))
m.x10 = Var(within=Reals,bounds=(0,0))
m.x11 = Var(within=Reals,bounds=(0,0))
m.x12 = Var(within=Reals,bounds=(0,0))
m.x13 = Var(within=Reals,bounds=(0,0))

m.obj = Objective(expr=(8.39*m.x3*m.x4 - 0.727*m.x2*m.x3 - 684.4*m.x4*m.x5 + 63.5*m.x4*m.x2 + 0.107*m.x2 + 0.126*m.x3 - 
                       9.99*m.x5 - 3.933*m.x1 - 45.83*m.x7 - 7.64*m.x8)**2 + (0.949*m.x1*m.x3 + 0.173*m.x1*m.x5 - 0.987*
                       m.x2 - 22.95*m.x4 - 28.37*m.x6)**2 + (-0.716*m.x1*m.x2 - 1.578*m.x1*m.x4 + 1.132*m.x4*m.x2 + 
                       0.002*m.x1 - 0.235*m.x3 + 5.67*m.x5 - 0.921*m.x7 - 6.51*m.x8)**2 + (m.x2 - m.x1*m.x5 - m.x4 - 
                       0.168*m.x6)**2 + (m.x1*m.x4 - m.x3 - 0.196*m.x5 - 0.0071*m.x7)**2, sense=minimize)

m.c1 = Constraint(expr = m.x1 + m.x2 <= 10)


from pyomo.opt import SolverStatus, TerminationCondition

solver = SolverFactory('baron')
results = solver.solve(model, options={"MaxIter": 1135, "CompIIS":1, "results": 1, "ResName": "finres", "summary": 1, "SumName": "test.lst", "times": 1, "TimName": "timtest.lst"}, tee = True, keepfiles=True) # Solving a model instance  
soln_status = (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal)
print (soln_status)
