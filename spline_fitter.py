from scipy.interpolate import PchipInterpolator
from psopt import FunctionOptimizer


class SplineFitter(object):
    def __init__(self, n=6, xtot=1.0, y0=1.0):
        self.n=n
        self.xd = np.ones(n-1)*xtot/(n-1)
        self.y = np.ones(n)*y0
        self.xmin = xtot/n/1000.
        self.ymin = y0/1000.
        
    def get_nodes(self,*args):
        xs = [0.]
        ys = [args[0]]
        for ii in range(self.n-1):
            xs.append(xs[-1]+args[2*ii+1])
            ys.append(args[2*ii+2])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        return xs, ys
        
    
    def __call__(self,x,*args):
        xs,ys = self.get_nodes(*args)
        f = interpolate.PchipInterpolator(xs,ys)
        return(f(x))
        
    def get_param_dict(self):
        params = {}
        params['y0']=self.y[0]
        for ii in range(self.n-1):
            params['l{}'.format(ii)] = self.xd[ii]
            params['y{}'.format(ii+1)] = self.y[ii+1]
        return params

    def get_param_bounds(self):
        params = {'y0':[self.ymin,np.inf]}
        for ii in range(self.n-1):
            params['l{}'.format(ii)] = [self.xmin,np.inf]
            params['y{}'.format(ii)] = [self.ymin,np.inf]
        return params
    
    def get_param_tuples(self):
        params = [('y0',self.y[0])]
        for ii in range(self.n-1):
            params.append(('l{}'.format(ii), self.xd[ii]))
            params.append(('y{}'.format(ii+1), self.y[ii+1]))
        return params

    def guess_nodes(self, n=7):
        


if __name__ == '__main__':
    
