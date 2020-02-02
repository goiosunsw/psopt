import matplotlib.pyplot as plt
import numpy as np

from psopt import FunctionOptimizer

# define initial parameters
b=1.1
c=-1

# define fitting variables
xm = np.linspace(0,1,100)
ym = b*xm+c+np.random.randn(len(xm))/10.

# define fitting function 
# provide default variables for parameters
fit_f = lambda x, a=0, b=0, c=0: a*x*x+b*x+c

opt = FunctionOptimizer()

opt.set_objective_function(fit_f,independent_var='x')
#opt.switch_fitting('a',False)
opt.set_bounds('c',[-1.5,1.5])

# call optimizer with dependent and independent variables
pars=opt.optimize(ym,xm)
print('Initial parameters')
print(opt.params)

print('Fitted parameters (unconstrained)')
print (pars)

fig, ax = plt.subplots(1)
ax.plot(xm,ym,'o',label='data')
ax.plot(xm, fit_f(xm,**pars),label='unconstrained fit')

# Do a constrained fit:
bounds = [-.5,.5]
opt.set_bounds('c',bounds)
pars=opt.optimize(ym,xm)
print('Fit parameters (bounds in c: {})'.format(bounds))
print(pars)
ax.plot(xm, fit_f(xm,**pars),label='bounds in c')

# Remove boundaries
print('Bounds reset')
opt.reset_bounds()
pars=opt.optimize(ym,xm)
print(pars)

# Freeze parameter c 
new_c = 1.0
opt.set_param('c',new_c)
print('Freeze parameter c={:f}'.format(new_c))
opt.fitting_off('c')
pars=opt.optimize(ym,xm)
print(pars)
ax.plot(xm, fit_f(xm,**pars),label='Frozen c')


ax.legend()

plt.show()