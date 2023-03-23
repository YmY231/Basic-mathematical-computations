#################################################################
## Functions to compute some approximation errors
#################################################################

#################################################################
## Imports
## - No further imports should be necessary
## - If you wish to import a non-standard modules, ask Ed if that 
## - is acceptable
#################################################################
import numpy as np
import matplotlib.pyplot as plt
import approximations as approx #import previously written functions
#################################################################

#################################################################
## Functions to be completed by student
#################################################################

#%% Q3 code

def interpolation_errors(a,b,n,P,f):
    
    '''
    (a) The error is continuously decreasing when p increases. This is beacuse that, by the Burden & Faires theorem of error identity for polynomial interpolation, the error of this function will monotonic decreasing with increasing p.
    (b) The error is waving at first and will suddenly increasing to a large value. This is because that f is always near zero on [-5,5] but a maximum of 1 in x=0.
    '''
    
    x = np.linspace(a, b, 2000)
    interpolant = [approx.poly_interpolation(a, b, P[i], 2000, x, f, False)[0] for i in range(n)]
    error_matrix = np.array([max([abs(interpolant[i][j]-f(x[j])) for j in range(2000)])  for i in range(n)])

    fig = plt.figure()
    plt.plot(P, error_matrix, 'o', linestyle='-')
    plt.semilogy()
    plt.xlabel('$p_j$')
    plt.ylabel('error(in the form of log)')
    plt.legend()
    plt.show()
    
    return error_matrix, fig


#%% Q7 code 

def derivative_errors(x,P,m,H,n,f,fdiff):
    '''
    (a) The error is generally smaller when p is larger. For each p, the error is increasing when x increases.
    (b) The error will be zero when x<0. For x great or equal than 0, the errors are two parallel line for the last two p's.
    '''
    E = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            D_approx = approx.approximate_derivative(x, P[i], H[j], i+1, f)
            E[i][j] = abs(fdiff(x)-D_approx)
            
    fig = plt.figure()
    for i in range(m):
        plt.loglog(H,E[i,:],'o',linestyle = '-',label = 'p'+str(i))
        plt.xlabel("log(x)")
        plt.ylabel("log(E)")
    plt.legend()
    plt.show()
        
    return E, fig


#################################################################
## Test Code ##
## You are highly encouraged to write your own tests as well,
## but these should be written in a separate file
#################################################################
print("\nAny outputs above this line are due to importing approximations.py.\n")

################
#%% Q3 Test
################

# Initialise
n = 5
P = np.arange(1,n+1)
a = -1
b = 1
f = lambda x: 1/(x+2)

#Run the function
error_matrix, fig = interpolation_errors(a,b,n,P,f)

print("\n################################")
print('Q3 TEST OUTPUT:\n')

print("error_matrix = \n")
print(error_matrix)

################
#%% Q7 Test
################

#Initialise
P = np.array([2,4,6])
H = np.array([1/4,1/8,1/16])
x = 0
f = lambda x: 1/(x+2)
fdiff = lambda x: -1/((x+2)**2)

#Run the function
E, fig = derivative_errors(x,P,3,H,3,f,fdiff)

print("\n################################")
print("Q7 TEST OUTPUT:\n")

print("E = \n")
print(E)
plt.show()

