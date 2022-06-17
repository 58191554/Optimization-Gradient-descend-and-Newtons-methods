from q2 import GradientMethod, f, grad
import matplotlib.pyplot as plt
import numpy as np

def main():
    gm1 = GradientMethod(f, grad, -3, -3)
    gm1.grand_descend_BT()
    x1_trajectory = gm1.x_iter
    y1_trajectory = gm1.y_iter

    gm2 = GradientMethod(f, grad, 3, -3)
    gm2.grand_descend_BT()
    x2_trajectory = gm2.x_iter
    y2_trajectory = gm2.y_iter

    gm3 = GradientMethod(f, grad, -3, 3)
    gm3.grand_descend_BT()
    x3_trajectory = gm3.x_iter
    y3_trajectory = gm3.y_iter

    gm4 = GradientMethod(f, grad, 3, 3)
    gm4.grand_descend_BT()
    x4_trajectory = gm4.x_iter
    y4_trajectory = gm4.y_iter

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

    gm1 = GradientMethod(f, grad, -3, -3)
    gm1.grand_descend_Exact()
    x1_trajectory = gm1.x_iter
    y1_trajectory = gm1.y_iter

    gm2 = GradientMethod(f, grad, 3, -3)
    gm2.grand_descend_Exact()
    x2_trajectory = gm2.x_iter
    y2_trajectory = gm2.y_iter

    gm3 = GradientMethod(f, grad, -3, 3)
    gm3.grand_descend_Exact()
    x3_trajectory = gm3.x_iter
    y3_trajectory = gm3.y_iter

    gm4 = GradientMethod(f, grad, 3, 3)
    gm4.grand_descend_Exact()
    x4_trajectory = gm4.x_iter
    y4_trajectory = gm4.y_iter

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