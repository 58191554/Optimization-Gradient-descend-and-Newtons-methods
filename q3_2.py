from turtle import shape
from q3 import Limited_Memory_Gradient_Descent as LMGD
from q2 import f, grad
import matplotlib.pyplot as plt
import numpy as np

def main():
    gm1 = LMGD(f, grad, -3, -3)
    gm1.Memory_Grad_Descent()
    x1_trajectory = gm1.x_iter
    y1_trajectory = gm1.y_iter

    gm2 = LMGD(f, grad, 3, -3)
    gm2.Memory_Grad_Descent()
    x2_trajectory = gm2.x_iter
    y2_trajectory = gm2.y_iter

    gm3 = LMGD(f, grad, -3, 3)
    gm3.Memory_Grad_Descent()
    x3_trajectory = gm3.x_iter
    y3_trajectory = gm3.y_iter

    gm4 = LMGD(f, grad, 3, 3)
    gm4.Memory_Grad_Descent()
    x4_trajectory = gm4.x_iter
    y4_trajectory = gm4.y_iter
    plt.plot(x1_trajectory, y1_trajectory, "red")
    plt.plot(x2_trajectory, y2_trajectory, "green")
    plt.plot(x3_trajectory, y3_trajectory, "blue")
    plt.plot(x4_trajectory, y4_trajectory, "purple")
    plt.scatter(x1_trajectory[-1], y1_trajectory[-1],c="blue",edgecolors="hotpink",marker="^")
    plt.scatter(x2_trajectory[-1], y2_trajectory[-1],c="blue",edgecolors="hotpink",marker="^")
    plt.scatter(x3_trajectory[-1], y3_trajectory[-1],c="blue",edgecolors="hotpink",marker="^")
    plt.scatter(x4_trajectory[-1], y4_trajectory[-1],c="blue",edgecolors="hotpink",marker="^")

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    plt.contour(X, Y, Z, 30)
    plt.show()

main()