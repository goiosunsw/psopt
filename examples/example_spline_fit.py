import matplotlib.pyplot as plt
import numpy as np

from psopt import SplineOptimizer


xt = np.linspace(0,10,100)
yt = np.sin(xt) + np.random.randn(100)*.1
so = SplineOptimizer(x=xt,y=yt,n_nodes=7)
ax = so.plot()
so.fit_spline()
ax = so.plot(ax)

#ax.legend(['data','init','','','fit'])

so.set_method('basinhopping')
so.fit_spline()
so.plot(ax)

plt.show()