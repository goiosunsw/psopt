import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import LinearConstraint
from psopt import FunctionOptimizer


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
    def __init__(self,degree=3,n_nodes=5,
                 x=None,y=None,
                 xc=None,yc=None,
                 xclim=None,
                 yclim=None,
                 freeze_edges=True,**kwargs):
        self.degree = degree
        self.n_nodes=n_nodes
        self.freeze_edges=freeze_edges
        FunctionOptimizer.__init__(self, **kwargs)

        self.min_dx = 0.001
        self.xc = xc
        self.yc = yc
        if x is not None and y is not None:
            self.set_data(x,y)
        self._take_step = self._take_step_spline
        self.set_lims(xclim=xclim, yclim=yclim)
        self._register_free_controls()

    def set_lims(self,xclim=None,yclim=None):
        if xclim is None:
            try:
                self.xmin
            except AttributeError:
                self.xmin = np.min(self.xm)
                self.xmax = np.max(self.xm)
        else:
            self.xmin = np.min(xclim)
            self.xmax = np.max(xclim)
        if yclim is None:
            try:
                self.ymin
            except AttributeError:
                self.ymin = np.min(self.ym)
                self.ymax = np.max(self.ym)
        else:
            self.ymin = np.min(yclim)
            self.ymax = np.max(yclim)
            
    def init_control_points(self):
        if self.xc is None:
            self.xc,yg=self.guess_nodes()
        else:
            yg = np.interp(self.xc,self.xm,self.ym)
        if self.yc is None:
            self.yc=yg
            
        self.set_control_points(self.xc,self.yc)
        
        
        #print(super())

    def _register_free_controls(self):
        self.free_x_idx = []
        self.free_y_idx = []
        for ii, parname in enumerate(self.free_params):
            if parname[0]=='x':
                self.free_x_idx.append(ii)
            if parname[0]=='y':
                self.free_y_idx.append(ii)

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
            parname = 'x{}'.format(xi)
            try:
                bounds = self.bounds[parname]
            except KeyError:
                self.set_param(parname,xx,bounds=(xmin,xmax))
            #self.set_bounds('x{}'.format(xi),[xmin,xmax])
        if self.freeze_edges:
            self.fitting_off('x0')
            self.fitting_off('x{}'.format(xi))
        for yi,yy in enumerate(yc):
            self.set_param('y{}'.format(yi),yy)
        self._control_xy_to_spl(xc,yc)
        self.set_constraints()

    def set_constraints(self):
        n_free_x = sum([1 for x in self.free_params if x[0]=='x'])
        if n_free_x>0:
            nconst = n_free_x - 1
            self.constraint_mx = np.zeros((nconst,len(self.free_params)))
            self.constraint_lb = np.ones(nconst)*self.min_dx
            for ii in range(nconst):
                self.constraint_mx[ii,ii]=-1
                self.constraint_mx[ii,ii+1]=1
            self.constraints = LinearConstraint(self.constraint_mx,self.constraint_lb,np.inf,keep_feasible=True)
        else:
            self.constraints = None
    
        
    def _take_step_spline(self, args):
        min_dx = 0.001
        newx = np.random.uniform(self.xmin,self.xmax,len(self.free_x_idx))
        newx = np.sort(newx)
        dx = np.diff(newx)
        dx[dx<min_dx]=min_dx
        newx = np.cumsum(np.insert(dx,0,newx[0]))
        stepx = newx - args[self.free_x_idx]
        newy = np.random.uniform(self.ymin,self.ymax,len(self.free_y_idx))
        return np.concatenate((newx, newy))

        
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

    def obj_fun_wrap(self, x, func, *extra):
        return func(x)
            
    def fpred(self,x,*parlist):
        self._param_list_to_spl(parlist)
        y = self.obj_fun_wrap(x, self.spl)
        return(y)

    def _param_dict_to_spl(self,pardict):
        ii=0
        xc=[]
        yc=[]
        while True:
            try:
                xc.append(pardict['x{}'.format(ii)])
                yc.append(pardict['y{}'.format(ii)])
                ii+=1
            except KeyError:
                break
        self._control_xy_to_spl(xc,yc)
                
    def fit_spline(self):
        ret = self.optimize(self.ym,self.xm)#,constraints=self.constraints)
        self._param_dict_to_spl(ret)
        return ret

    def random_x_controls_inside_data(self):
        """
        returns a random set of controls
        and boundaries
        """
        xmin = np.min(self.xm)
        xmax = np.max(self.xm)
        newxc = np.random.uniform(xmin,xmax,len(self.xc)-2)
        newxc = np.sort(newxc)
        newxc = np.concatenate(([xmin], newxc, [xmax]))
        return newxc, [(xmin, xmax)] * len(newxc)

    def random_x_controls_around_data(self, range_mult=1):
        """
        returns a random set of controls
        and boundaries
        """

        dx = np.max(self.xm)-np.min(self.xm)
        xmin = np.min(self.xm) - dx*range_mult
        xmax = np.max(self.xm) + dx*range_mult
        newxc = np.random.uniform(xmin,xmax,len(self.xc))
        newxc = np.sort(newxc)
        return newxc, [(-np.inf, np.inf)] * len(newxc)

    def run_multi_fit(self, iterations=10):
        xc_st = []
        xc_fit = []
        yc_fit = []
        xc_cost = []
        xc_ret = []
        xc_par = []

        for ii in range(iterations):
            if self.freeze_edges:
                newxc, bounds = self.random_x_controls_inside_data()
            else:
                newxc, bounds = self.random_x_controls_around_data()
            self.xc = newxc
            bound_dict = {'x%d'%ii:v for ii,v in enumerate(bounds)}
            self.set_bound_dict(bound_dict)
            xcst = self.xc
            self.yc = None
            try:
                self.init_control_points()
            except  ValueError:
                continue
            xc_st.append(xcst)
            try:
                par=self.fit_spline()
                #ax = so.plot(ax)
                xc_cost.append(self._ret['fun'])
                xc_ret.append(self._ret)
            except ValueError:
                xc_cost.append(np.inf)
                xc_ret.append(traceback.format_exc())
                par=[]
            xc_fit.append(self.xc.copy())
            yc_fit.append(self.yc.copy())
            xc_par.append(par)

        xc_cost = np.array(xc_cost)
        xc_fit = np.array(xc_fit)
        yc_fit = np.array(yc_fit)

        return xc_cost, xc_fit, yc_fit
            
        
    def set_data(self,x,y):
        self.xm = np.asarray(x)
        self.ym = np.asarray(y)
        self.min_dx = (max(self.xm)-min(self.xm))*1e-4
        self.init_control_points()
        self.set_objective_function(self.fpred,params=self.param_tuplist)
      
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
    
