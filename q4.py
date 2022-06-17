import numpy as np
from q2 import GradientMethod

class Newtons_Method(GradientMethod):
    def __init__(
        self, obj, grad, hess, x0, y0, tol=10 ** -5, 
        maxit=100, a=2, sigma=0.5, gamma=0.1, exact_tol=10 ** -6,
        gamma1 = 10**(-6), gamma2 = 0.1, ):
        super().__init__(obj, grad, x0, y0, tol, maxit, a, sigma, gamma, exact_tol)

        self.hess = hess
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.x_iter = [self.x0]
        self.y_iter = [self.y0]

        self.Newton_direction_number = 0
        self.alpha_is_1_number = 0

    def get_direction(self, x, y):
        s = np.linalg.solve(self.hess(x, y), -self.grad(x, y))
        s_norm = np.linalg.norm(s)
        grad_vec = np.array(self.grad(x,y))
        if -grad_vec.T*s >= self.gamma1*min(1,s_norm**self.gamma2)*s_norm**2:
            d = s
            self.Newton_direction_number += 1
        else:
            d = -grad_vec  
        return  d

    def newton_method(self):
        x = self.x0
        y = self.y0
        k = 0
        while np.linalg.norm(self.grad(x, y)) > self.tol:
            if(k>874670888):break
            count_a = 0
            d = self.get_direction(x, y)
            #find step size
            alpha = 1
            while self.obj(x-alpha*self.grad(x,y)[0], y-alpha*self.grad(x,y)[1]) - f(x,y) > -self.gamma*alpha*(self.grad(x,y)[0]**2+(self.grad(x,y)[1])**2):
                count_a += 1
                alpha = alpha*self.sigma
            if(alpha == 1): self.alpha_is_1_number += 1
            t=alpha*d   
            x +=t[0,0]
            y +=t[1,0]
            self.x_iter.append(x)
            self.y_iter.append(y)
            k+=1
        print("x = ", x)
        print("y = ", y)
        print("count = ", k)
        print("Newton_direction_number = ", self.Newton_direction_number)
        print("alpha_is_1_number", self.alpha_is_1_number)



def f(x1,x2):
    f=100*(x2-x1**2)**2 + (1-x1)**2
    return f

def Grad(x1,x2):
    G=np.mat([[-400*x1*(x2-x1**2)-2*(1-x1)],
              [200*(x2-x1**2)]])
    return G

def Hess(x1,x2):
    H=np.mat([[-400*x2+1200*x1**2+2,-400*x1],
              [-400*x1,200]])
    return H

def main():

    gm = GradientMethod(obj=f, grad = lambda x, y:[-400*x*(y-x**2)-2*(1-x),200*(y-x**2)], 
    x0 = -1, y0 = -0.5,  sigma= 0.5, gamma=10**(-4), tol=10**(-7))
    gm.grand_descend_BT()

    nm = Newtons_Method(obj = f, grad=Grad, hess=Hess, x0 = -1, y0 = -0.5,
    sigma= 0.5, gamma=10**(-4), gamma1=10**(-6), gamma2=0.1, tol=10**(-7))
    nm.newton_method()


if __name__ == "__main__":
    main()