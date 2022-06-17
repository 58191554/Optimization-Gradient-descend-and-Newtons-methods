from q3 import Limited_Memory_Gradient_Descent as LMGD
from q2 import f, grad
import matplotlib.pyplot as plt
import numpy as np

def main():
    k_ls1 = []
    k_ls2 = []
    k_ls3 = []
    k_ls4 = []
    x_ls = [5,10,15,25,35, 45, 55, 65, 75, 85, 95, 100]
    for i in x_ls:
        gm1 = LMGD(f, grad, -3, -3, m=i)
        gm1.Memory_Grad_Descent()
        k_ls1.append(gm1.k)

        gm2 = LMGD(f, grad, -3, 3, m=i)
        gm2.Memory_Grad_Descent()
        k_ls2.append(gm2.k)

        gm3 = LMGD(f, grad, 3, -3, m=i)
        gm3.Memory_Grad_Descent()
        k_ls3.append(gm3.k)

        gm4 = LMGD(f, grad, 3, 3, m=i)
        gm4.Memory_Grad_Descent()
        k_ls4.append(gm4.k)

    plt.plot(x_ls, k_ls1,'r')
    plt.plot(x_ls, k_ls2,'b')
    plt.plot(x_ls, k_ls3,'pink')
    plt.plot(x_ls, k_ls4,'g')
    plt.xlabel("m")
    plt.ylabel("iteration time")
    plt.show()
main()
