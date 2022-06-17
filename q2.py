
class GradientMethod:

    def __init__(self, obj, grad, x0, y0, tol = 10**(-5), maxit = 100, 
    a = 2, sigma = 0.5, gamma = 0.1, exact_tol = 10**(-6)):
        self.obj = obj      #return the objective f(x) at an input vector x ∈ Rn
        self.grad = grad    #return the gradient ∇f(x) at an input vector x ∈ Rn
        self.x0 = x0        #the initial point.
        self.y0 = y0
        self.tol = tol      #a tolerance parameter. stop at ||∇f(xk)||<=tol
        self.maxit = maxit
        self.a = a
        self.sigma = sigma
        self.gamma = gamma
        self.exact_tol = exact_tol
        self.x_iter = []
        self.y_iter = []

    #backtracking line search
    def grand_descend_BT(self):
        count1 = 0      #count for finding iterationtime
        x = self.x0
        y = self.y0
        f = self.obj
        grad = self.grad
        gamma = self.gamma
        sigma = self.sigma

        x_iter = [self.x0]
        y_iter = [self.y0]

        #threshold check
        while grad(x,y)[0]**2 + grad(x,y)[1]**2 > self.tol**2:
            count2 = 0      #count for finding step size

            #find step size
            alpha = 1
            while f(x-alpha*grad(x,y)[0], y-alpha*grad(x,y)[1]) - f(x,y) > -gamma*alpha*(grad(x,y)[0]**2+(grad(x,y)[1])**2):
                count2 += 1
                alpha = alpha*sigma

            x_new = x - alpha*grad(x,y)[0]; x_iter.append(x_new)
            y_new = y - alpha*grad(x,y)[1]; y_iter.append(y_new)
            x = x_new
            y = y_new
            count1 += 1

        print("x = ", round(x,4))
        print("y = ", round(y,4))
        print("f(x,y) = ", round(f(x, y), 4))
        print("count = ", count1)

        self.x_iter = x_iter
        self.y_iter = y_iter

    # use golden method to do the exact line search
    def grand_descend_Exact(self):
        f = self.obj
        grad = self.grad
        x = self.x0
        y = self.y0
        count = 0      #count for finding step size
        x_iter = [self.x0]
        y_iter = [self.y0]


        while grad(x,y)[0]**2 + grad(x,y)[1]**2 > self.tol**2:
            alpha = 1
            count_a = 0
            xl = 0
            xr = self.a

            while xr-xl >= self.exact_tol and count_a < self.maxit:
                #get alpha
                count_a += 1
                xl2=0.382*xr+0.618*xl
                xr2=0.382*xl+0.618*xr
                if f(x-xl2*grad(x,y)[0], y-xl2*grad(x,y)[1]) < f(x-xr2*grad(x,y)[0], y-xr2*grad(x,y)[1]):
                    xr = xr2
                else:
                    xl = xl2

            alpha = (xl+xr)/2
            x_new = x - alpha*grad(x,y)[0]; x_iter.append(x_new)
            y_new = y - alpha*grad(x,y)[1]; y_iter.append(y_new)
            x = x_new
            y = y_new
            count += 1
        print("x = ", round(x,4))
        print("y = ", round(y,4))
        print("f(x,y) = ", round(f(x, y), 4))
        print("count = ", count)
        self.x_iter = x_iter
        self.y_iter = y_iter


def f(a,b):
    f=a**4+2/3*a**3+1/2*a**2-2*a**2*b+4/3*b**2
    return f

def grad(a, b):
    grad=[4*a**3+2*a**2+a-4*a*b , -2*a**2+8/3*b]
    return grad

def main():
    gm = GradientMethod(f, grad, 3, 3)
    print("back tracking line search")
    gm.grand_descend_BT()
    print("-"*50+"\n","exact line serch")
    gm.grand_descend_Exact()

if __name__ == "__main__":
    main()
