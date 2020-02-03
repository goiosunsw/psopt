from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import LinearConstraint
from psopt import FunctionOptimizer
import numpy as np


def nuarmax(x,y,rad=1.0):
    maxx = []
    maxy = []
    for xx,yy in zip(x,y):
        idxr = np.abs(x-xx)<rad
        if not any(y[idxr]>yy):
            maxx.append(xx)
            maxy.append(yy)
    return np.array(maxx),np.array(maxy)
    

class SplineOptimizer(FunctionOptimizer):
    def __init__(self,degree=3,n_nodes=3,
                 x=None,y=None,
                 xc=None,yc=None,
                 freeze_edges=True,**kwargs):
        self.degree = degree
        self.n_nodes=n_nodes
        self.freeze_edges=freeze_edges
        FunctionOptimizer.__init__(self, **kwargs)

        self.min_dx = 0.001
        if x is not None and y is not None:
            self.set_data(x,y)
            
        if xc is None:
            self.xc,yg=self.guess_nodes()
        else:
            self.xc=xc
            yg = np.interp(xc,x,y)
        if yc is None:
            self.yc=yg
            
        self.set_control_points(self.xc,self.yc)
        
        self.set_objective_function(self.fpred,params=self.param_tuplist)
        
        #print(super())
        
    @property
    def param_tuplist(self):
        return [(k,v) for k,v in self.param_dict.items()]
        
    def guess_nodes(self):
        node_int = len(self.xm)//(self.n_nodes-1)
        node_idx = list(range(0,len(self.xm),node_int))[:-1]+[-1]
        node_x = self.xm[node_idx]
        node_y = self.ym[node_idx]
        return node_x, node_y
        
    def set_control_points(self, xc, yc):
        print(xc,yc)
        xmin=min(self.xm)
        xmax = max(self.xm)
        for xi,xx in enumerate(xc):
            self.set_param('x{}'.format(xi),xx,bounds=(xmin,xmax))
            #self.set_bounds('x{}'.format(xi),[xmin,xmax])
        if self.freeze_edges:
            self.fitting_off('x0')
            self.fitting_off('x{}'.format(xi))
        for yi,yy in enumerate(yc):
            self.set_param('y{}'.format(yi),yy)
            
        if self.freeze_edges:
            nconst = len(xc)-3
        else:
            nconst = len(xc)-1
        self.constraint_mx = np.zeros((nconst,nconst+1+len(yc)))
        self.constraint_lb = np.ones(nconst)*self.min_dx
        for ii in range(nconst):
            self.constraint_mx[ii,ii]=-1
            self.constraint_mx[ii,ii+1]=1
        self.constraints = LinearConstraint(self.constraint_mx,self.constraint_lb,np.inf)
    
        self._control_xy_to_spl(xc,yc)
        
    def _control_xy_to_spl(self, xc, yc):
        idx = np.argsort(xc)
        self.xc=np.array(xc)[idx]
        self.yc=np.array(yc)[idx]
        for ii,xx in enumerate(self.xc[:-1]):
            if xx==self.xc[ii+1]:
                try:
                    self.xc[ii+1]=(self.xc[ii+2]+xx)/2
                except IndexError:
                    self.xc[ii+1]+=.001
        self.spl = InterpolatedUnivariateSpline(self.xc, self.yc, k=self.degree) 
                
        
    def _param_list_to_spl(self, parlist):
        xi = 0
        yi = 0
        xc = []
        yc = []
        for pp in parlist:
            try:
                self.params['x{}'.format(xi)]
                self.params['x{}'.format(xi)]=pp
                xc.append(pp)
                xi+=1
                continue
            except KeyError:
                pass
            self.params['y{}'.format(yi)]=pp
            yc.append(pp)
            yi+=1
        self._control_xy_to_spl(xc,yc)
            
    def obj_fun_wrap(self, x, geom_func):
        return geom_func(x)
    
    def fpred(self,x,*parlist):
        self._param_list_to_spl(parlist)
        y = self.obj_fun_wrap(x, self.spl)
        return(y)
            
    def fit_spline(self):
        self.optimize(self.ym,self.xm)#,constraints=self.constraints)
        
    def set_data(self,x,y):
        self.xm = np.asarray(x)
        self.ym = np.asarray(y)
        self.min_dx = (max(self.xm)-min(self.xm))*1e-4
        
    @property
    def knots(self):
        return self.spl.get_knots()
        
    def plot(self,ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig,ax = plt.subplots(1)
            ax.plot(self.xm,self.ym,'o')
        lns=ax.plot(self.xm,self.spl(self.xm))
        color = lns[-1].get_color()
        ax.plot(self.xc,self.yc,'x',ms=12,color=color)
        for kk in self.knots:
            ax.axvline(kk,color=color,alpha=.5,ls='--')
        return ax

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.linspace(-3, 3, 50)
    y = np.exp(-x**2) + 0.1 * np.random.randn(50)

    so=SplineOptimizer(x=x,y=y,n_nodes=7,degree=2)#,method='basinhopping')
    ax=so.plot()
    so.fit_spline()
    ax=so.plot(ax=ax)
    plt.show()
    
