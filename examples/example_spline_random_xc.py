import matplotlib.pyplot as plt
import numpy as np

from psopt import SplineOptimizer
import traceback

freeze_edges = True

xt = np.linspace(0,10,50)
y = np.sin(xt)
yt = y + np.random.randn(len(xt))*.5
so = SplineOptimizer(x=xt,y=yt,n_nodes=7,degree=3,freeze_edges=freeze_edges)
fig,ax = plt.subplots(1)
ax.plot(xt,yt,'o')
ax.plot(xt,y)
so.fit_spline()
xc_st = []
xc_fit = []
yc_fit = []
xc_cost = []
xc_ret = []
xc_par = []
for ii in range(20):
    if freeze_edges:
        newxc = np.random.uniform(so.xc[0],so.xc[-1],len(so.xc)-2)
        so.xc[1:-1] = np.sort(newxc)
    else:
        dx = np.max(xt)-np.min(xt)
        xmin = np.min(xt) - dx/2
        xmax = np.max(xt) + dx/2
        newxc = np.random.uniform(xmin,xmax,len(so.xc))
        so.xc = newxc
        for parname in so.params:
            if parname[0]=='x':
                #so.set_bounds(parname,(xmin,xmax))
                so.set_bounds(parname,(-np.inf,np.inf))
    xcst = so.xc
    so.yc = None
    try:
        so.init_control_points()
    except  ValueError:
        continue
    xc_st.append(xcst)
    try:
        par=so.fit_spline()
        #ax = so.plot(ax)
        xc_cost.append(so._ret['fun'])
        xc_ret.append(so._ret)
    except ValueError:
        xc_cost.append(np.inf)
        xc_ret.append(traceback.format_exc())
        par=[]
    xc_fit.append(so.xc.copy())
    yc_fit.append(so.yc.copy())
    xc_par.append(par)

xc_cost = np.array(xc_cost)
xc_fit = np.array(xc_fit)
yc_fit = np.array(yc_fit)

xp = np.linspace(np.min(xt),np.max(xt),1000)
ninf = np.sum(np.isinf(xc_cost))
for ii in np.argsort(xc_cost)[[0,1,2,-1-ninf]]:
    xc = xc_fit[ii]
    yc = yc_fit[ii]
    so.set_control_points(xc=xc,yc=yc)
    lns=ax.plot(xc,yc,'.')
    ax.plot(xp,so.spl(xp),color=lns[0].get_color())

ax.set_ylim([-2,2])

plt.show()