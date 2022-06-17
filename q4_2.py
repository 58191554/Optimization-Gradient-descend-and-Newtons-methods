from q4 import Newtons_Method
import matplotlib.pyplot as plt
import numpy as np
from q2 import f

def Grad(a, b):
    return np.mat([[4*a**3+2*a**2+a-4*a*b], [-2*a**2+8/3*b]])  

def Hess(a, b):
    return np.mat([
        [12*a**2 + 4*a + 1 -4*b, -4*a],
        [-4*a,8/3 ]])

def main():
    nm1 = Newtons_Method(obj = f, grad=Grad, hess=Hess, x0 = -3, y0 = -3,
    sigma= 0.5, gamma=10**(-4), gamma1=10**(-6), gamma2=0.1, tol=10**(-7))
    nm1.newton_method()
    x1_trajectory = nm1.x_iter
    y1_trajectory = nm1.y_iter

    nm2 = Newtons_Method(obj = f, grad=Grad, hess=Hess, x0 = 3, y0 = -3,
    sigma= 0.5, gamma=10**(-4), gamma1=10**(-6), gamma2=0.1, tol=10**(-7))
    nm2.newton_method()
    x2_trajectory = nm2.x_iter
    y2_trajectory = nm2.y_iter

    nm3 = Newtons_Method(obj = f, grad=Grad, hess=Hess, x0 = -3, y0 = 3,
    sigma= 0.5, gamma=10**(-4), gamma1=10**(-6), gamma2=0.1, tol=10**(-7))
    nm3.newton_method()
    x3_trajectory = nm3.x_iter
    y3_trajectory = nm3.y_iter

    nm4 = Newtons_Method(obj = f, grad=Grad, hess=Hess, x0 = 3, y0 = 3,
    sigma= 0.5, gamma=10**(-4), gamma1=10**(-6), gamma2=0.1, tol=10**(-7))
    nm4.newton_method()
    x4_trajectory = nm4.x_iter
    y4_trajectory = nm4.y_iter

    plt.plot(x1_trajectory, y1_trajectory, "red")
    plt.plot(x2_trajectory, y2_trajectory, "green")
    plt.plot(x3_trajectory, y3_trajectory, "blue")
    plt.plot(x4_trajectory, y4_trajectory, "purple")


    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    plt.contour(X, Y, Z, 30)
    plt.show()

main()
