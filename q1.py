import numpy
import scipy
import scipy.optimize as opt

def func(x:float):
    f = -(2*numpy.log(x))/((x-1)**3) + 1/(x*(x-1)**2) + (4*x)/((x-1)**2*(x+1)**2)
    return f


def bisection(a:float, b:float, epsilon = 10**(-5)):
    # define function
    x_1 = a
    x_2 = b
    x_mid = (x_1+x_2)/2
    func(x_1)
    func(x_2)

    func(x_mid)
    # see if root at twoi side
    if func(x_1) == 0:
        print("case 1")
        root = x_1
    elif func(x_2) == 0:
        root = x_2
        print("case 2")
    # check root existance
    elif func(x_1) * func(x_2) > 0:
        print("case 3")
        root = False
    elif func(x_mid) == 0:
        print("case 4")
        root = x_mid
    else: 
        print("case 5")
        count = 0
        while(func(x_mid)**2 > epsilon**2):
            print("x_mid = ", x_mid)
            fmid = func(x_mid)
            count += 1
            if(func(x_1) *  func(x_mid) < 0):
                x_2 = x_mid
                x_mid = (x_1 + x_2)/2

            elif( func(x_2) *  func(x_mid) < 0):
                x_1 = x_mid
                x_mid = (x_1 + x_2)/2
                
    print("count = ", count)
    root = x_mid
    return root
        
print(bisection(1.5,4.5))
