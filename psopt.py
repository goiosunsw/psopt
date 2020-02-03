from scipy.optimize import basinhopping, minimize
import numpy as np
import inspect

class Optimizer(object):
    def __init__(self, method=None, init_params=None):
        if method == 'basinhopping':
            self._optimize_caller = self._basinhopping
        else:
            self._optimize_caller = self._minimize
        self._params = []
        self._param_vals = []
        self._param_mask=[]
        self._bounds=None
        if init_params:
            self.set_params(init_params)
        self.constraints = ()
        
    
    @property
    def params(self):
        return {k:v for k,v in zip(self._params,self._param_vals)}
    
    @params.setter
    def params(self, par):
        self._params = []
        self._param_vals = []
        self._param_mask = []
        for kk, vv in par.items():
            self._params.append(kk)
            self._param_vals.append(vv)
            self._param_mask.append(True)
            
    def set_param(self,par,default_val=0., bounds=None,fit=None):
        try:
            idx=self._params.index(par)
            self._param_vals[idx]=default_val
        except ValueError:
            self._params.append(par)
            self._param_vals.append(default_val)
            self._param_mask.append(True)
            if self._bounds:
                self.bounds.append((-np.inf,np.inf))
        if fit is not None:
            self.switch_fitting(par,fit)
        if bounds:
            self.set_bounds(par,bounds)
        
         
            
    def apply_mask_dict(self,mask_dict):
        for k,v in mask_dict.items():
            self.switch_fitting(k,v)

    def switch_fitting(self,param,val):
        try:
            idx = self._params.index(param)
        except ValueError:
            logging.warn('apply mask: %s not in parameter list. Skipping'%param)
            return
        self._param_mask[idx]=bool(val)

    def fitting_on(self, param):
        self.switch_fitting(param,True)

    def fitting_off(self, param):
        self.switch_fitting(param,False)

    @property
    def param_mask(self):
        return self._param_mask
    
    @param_mask.setter
    def param_mask(self, param_mask):
        for kk, vv in param_mask.items:
            self.switch_fitting(kk,vv)
    
    @property
    def free_params(self):
        return [p for p, m in zip(self._params,self._param_mask) if m]

    @property
    def free_param_vals(self):
        return [v for v, m in zip(self._param_vals, self._param_mask) if m]
        
    @property
    def param_dict(self):
        return {p:v for p, v, m in zip(self._params,self._param_vals,self._param_mask) if m}
    
    @property
    def vals(self):
        return self._param_vals
        
    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, intervals):
        try:
            self._bounds = []
            for bb, param, default in zip(intervals, 
                                        self._params, 
                                        self._param_vals):
                bbmax = max(*bb)
                bbmin = min(*bb)
                fv = [param, default, bbmin, bbmax]
                assert (default>bbmin) and (default<bbmax), 'Default value of {} ({}) exceeds boundaries ({} - {})'.format(*fv)
                self._bounds.append([bbmin,bbmax])
        except TypeError:
            logging.warn('Boundaries not set')
            self._bounds = None
            
    @property
    def free_bounds(self):
        if self._bounds:
            return [b for b,m in zip(self._bounds, self._param_mask) if m]
        else:
            return None
        
    @property
    def fitted_param_dict(self):
        ii=0
        par_dict={}
        for param, val_orig, mask in zip(self._params, self._param_vals, self._param_mask):
            if mask: 
                par_dict[param]=self._ret.x[ii]
                ii+=1
            else:
                par_dict[param]=val_orig
        return par_dict
        
    @property 
    def fitted_params(self):
        ii=0
        pars = []
        for param, val_orig, mask in zip(self._params, self._param_vals, self._param_mask):
            if mask: 
                pars.append(self._ret.x[ii])
                ii+=1
            else:
                pars.append(val_orig)
        return pars
    
    def set_bounds(self, param, bounds):
        try:
            idx = self._params.index(param)
        except ValueError:
            logging.warn('apply bound: %s not in parameter list. Skipping'%param)
            return
        if not self._bounds:
            self._bounds = [(-np.inf,np.inf) for p in self._params]
        try:
            self._bounds[idx]=bounds
        except IndexError:
            for ii in range(len(self._bounds),idx-1):
                self._bounds.append((-np.inf,np.inf))
            self.bounds.append(bounds)

    def reset_bounds(self):
        self._bounds = None
        
    def set_cost_func(self,f):
        self._cost_func_all_params = f
        for par in inspect.signature(self._cost_func_all_params).parameters.values():
            if par.default == inspect.Parameter.empty:
                self.set_param(par.name)
            else:
                self.set_param(par.name,par.default)        
    
    def _cost_func(self, free_params):
        all_params = []
        ii = 0
        for param, val_orig, mask in zip(self._params, self._param_vals, self._param_mask):
            if mask: 
                all_params.append(free_params[ii])
                ii+=1
            else:
                all_params.append(val_orig)
        return self._cost_func_all_params(*all_params)
    
    def _basinhopping(self):
        return basinhopping(self._cost_func, self.free_param_vals,minimizer_kwargs={"constraints":self.constraints,"bounds":self.free_bounds})
                
    def _minimize(self):
        return minimize(self._cost_func, self.free_param_vals, bounds=self.free_bounds, constraints=self.constraints)
    
    def optimize(self):
        ret = self._optimize_caller()
        self._ret = ret
        return self.fitted_param_dict


class FunctionOptimizer(Optimizer):
    def __init__(self,*args,**kwargs):
        try:
            f = kwargs.pop('func')
        except KeyError:
            f = None
            
        try:
            params = kwargs.pop('params')
        except KeyError:
            params = None
                    

        super().__init__(*args,**kwargs)
        self._dist_func = self._default_dist
        self.weights = []
        self.xm = []
        self.ym = []
        if f:
            self.set_objective_function(f,params=params)
        
    def _default_dist(self, ym, yt, weights=None):
        """
        Distance measure between two vectors ym measured and yt trial
        """
        return sum((ym - yt)**2)
    
    @property
    def residuals(self):
        yt = self._objective_function(self.xm, *self.fitted_params)
        return yt-self.ym
    
    def _cost_func_all_params(self, *args):
        yt = self._objective_function(self.xm, *args)
        return self._dist_func(self.ym, yt, self.weights)
        
    def set_objective_function(self,f,independent_var=None,params=None):
        """
        set the objective function:
        the function to be fitted
        
        Must have the form:
        
        f(x, p1=X1, p2=X2,...)
        
        and return y with length similar to x
        
        p1, p2, etc. are the parameters
        """
        self._objective_function = f
        if params is None:
            params=[]
            for par in inspect.signature(f).parameters.values():
                if independent_var is None:
                    independent_var = par.name
                    continue
                if par.default == inspect.Parameter.empty:
                    params.append((par.name,None))
                else:
                    params.append((par.name,par.default))

        
        for par,val in params:
            if par == independent_var:
                continue
            if par is None:
                self.set_param(par)
            else:
                self.set_param(par,val)
    
    def set_dist_func(self,f):
        """
        set the function to calculate the distance between two vectors
        
        f(ym, yt, weights)
        
        should return a metric between ym and yt weighted by the weighting vector
        """
        self._dist_func = f
        
    def optimize(self, y, x):
        """
        Optimize parameters so that y-f(x) has minimal residues
        """
        self.xm = np.asarray(x)
        self.ym = np.asarray(y)
        ret = self._optimize_caller()
        self._ret = ret
        return self.fitted_param_dict
    
    def predict(self, x):
        return self._objective_function(x, *self.fitted_params)
    
    def plot(self):
        fig,ax = subplots(1)
        plot(self.xm,self.ym,'.',alpha=.2,label='obs.')
        plot(self.xm,self._objective_function(self.xm,*self._param_vals),label='init')
        plot(self.xm,self.predict(self.xm),label='fit')
        legend()
        return ax