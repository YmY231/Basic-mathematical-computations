import numpy as np
import matplotlib.pyplot as plt
import lagrange_polynomials as lp #previously written functions

def poly_interpolation(a,b,p,n,x,f,produce_fig):
    
    xhat = np.linspace(a, b, p+1)
    tol  = 1.0e-10
    L = lp.lagrange_poly(p, xhat, n, x, tol)[0]
    interpolant = np.array([sum([f(xhat[i])*L[i][j] for i in range(p+1)]) for j in range(n)])
    
    if produce_fig == True:
        fig = plt.figure()
        plt.plot(x, f(x), 'o', linestyle='-', label='$f(x)$')
        plt.plot(x, interpolant, 's', linestyle='-', label='interpolant')
        plt.xlabel('x')
        plt.legend()
        plt.show()
    else: fig = None

    return interpolant, fig


def poly_interpolation_2d(p,a,b,c,d,X,Y,n,m,f,produce_fig):

    tol  = 1.0e-10   
    xhat = np.linspace(a, b, p+1)
    yhat = np.linspace(c, d, p+1)
    Lp_x = lp.lagrange_poly(p, xhat, n, X[0], tol)[0]
    Lp_y = lp.lagrange_poly(p, yhat, m, Y[:,0], tol)[0]
    F = np.array([[f(xhat[i],yhat[j]) for i in range(p+1)] for j in range(p+1)])
    interpolant = np.array([[np.dot(np.dot(F,Lp_x[:,j]),Lp_y[:,i]) for j in range(n)] for i in range(m)])
    
    if produce_fig == True:
        fig = plt.figure()
        plt.contour(X[0], Y[:,0], interpolant)
    else: fig = None

    return interpolant,fig  


def approximate_derivative(x,p,h,k,f):

    tol = 1.0e-10
    x = [x]
    xhat = np.linspace(x[0]-k*h, x[0]+(p-k)*h, p+1)
    L = lp.deriv_lagrange_poly(p, xhat, 1, x, tol)[0]
    deriv_approx = np.float64(np.dot(f(xhat),L))

    return deriv_approx


#################################################################
## Test Code ##
#################################################################
print("\nAny outputs above this line are due to importing lagrange_polynomials.py.\n")

# Initialise
a = 0.5
b = 1.5
p = 3
n = 10
x = np.linspace(0.5,1.5,n)
f = lambda x: np.exp(x)+np.sin(np.pi*x)
#Run the function
interpolant, fig = poly_interpolation(a,b,p,n,x,f,False)

print("\n################################")
print('Q2 TEST OUTPUT:\n')
print("interpolant = \n")
print(interpolant)

################

f = lambda x,y : np.exp(x**2+y**2)
n = 4; m = 3
a = 0; b = 1; c = -1; d = 1 
x = np.linspace(a,b,n)
y = np.linspace(c,d,m)
X,Y = np.meshgrid(x,y)

interpolant,fig = poly_interpolation_2d(11,a,b,c,d,X,Y,n,m,f,False)

print("\n################################")
print('Q4 TEST OUTPUT:\n')
print("interpolant = \n")
print(interpolant)

################

print("\n################################")
print("Q6 TEST OUTPUT:\n")
#Initialise
p = 3
h = 0.1
x = 0.5
f = lambda x: np.cos(np.pi*x)+x

for k in range(4):
    #Run test 
    deriv_approx = approximate_derivative(x,p,h,k,f)
    print("k = " + str(k)+ ", deriv_approx = " + str(deriv_approx))
# %%
