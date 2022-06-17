from q2 import GradientMethod, f, grad
import numpy as np
import math

class Limited_Memory_Gradient_Descent(GradientMethod):
    def __init__(self, obj, grad, x0, y0, m = 25, epsilon = 10**(-6), tol=10 ** -5, maxit=100, a=2, sigma=0.5, gamma=0.1, exact_tol=10 ** -6):
        super().__init__(obj, grad, x0, y0, tol, maxit, a, sigma, gamma, exact_tol)
        self.m = m
        self.epsilon = epsilon
        self.x_iter = [self.x0]
        self.y_iter = [self.y0]
        self.k = 0


    def get_D(self, k):
        sum1 = 0
        sum2 = 0
        for i in range(max(k-self.m, 0), k):
            sum1 += self.grad(self.x_iter[i], self.y_iter[i])[0]**2 
            sum2 += self.grad(self.x_iter[i], self.y_iter[i])[1]**2
        v1 = math.sqrt(self.epsilon+sum1)
        v2 = math.sqrt(self.epsilon+sum2)
        return [v1, v2]

    def Memory_Grad_Descent(self):
        k = 0
        x = self.x0
        y = self.y0

        gamma = self.gamma
        sigma = self.sigma

        while grad(x,y)[0]**2 + grad(x,y)[1]**2 > self.tol**2:
            k += 1
            get_D = self.get_D(k)
            count_a = 0
            alpha = 1
            while f(x-alpha*grad(x,y)[0], y-alpha*grad(x,y)[1]) - f(x,y) > -gamma*alpha*(grad(x,y)[0]**2+(grad(x,y)[1])**2):
                count_a += 1
                alpha = alpha*sigma

            x_new = x - alpha*grad(x,y)[0]/get_D[0]; self.x_iter.append(x_new)
            y_new = y - alpha*grad(x,y)[1]/get_D[1]; self.y_iter.append(y_new)
            x = x_new
            y = y_new
            

        # print("x = ", x)
        # print("y = ", y)
        # print("f(x,y) = ", f(x, y))
        print("count = ", k)
        self.k = k



def main():
    lm = Limited_Memory_Gradient_Descent(f,grad, 3, 3, m = 25)
    lm.Memory_Grad_Descent()

if __name__ == "__main__":
    main()