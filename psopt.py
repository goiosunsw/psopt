from scipy.optimize import basinhopping, minimize
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
            
    def set_param(self,par,default_val=0., bounds=None,fit=True):
        try:
            idx=self._params.index(par)
            self._param_vals[idx]=default_val
        except ValueError:
            self._params.append(par)
            self._param_vals.append(default_val)
            self._param_mask.append(True)
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
        self._bounds[idx]=bounds
        
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
        return basinhopping(self._cost_func, self.free_param_vals)
                
    def _minimize(self):
        return minimize(self._cost_func, self.free_param_vals, bounds=self.free_bounds)
    

    def optimize(self):
        ret = self._optimize_caller()
        self._ret = ret
        return self.fitted_param_dict