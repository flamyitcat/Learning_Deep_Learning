import numpy as np

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).transpose()
#print(y)
def nonlin(x):
    return 1/(1+np.exp(-x))

def nonlin_deriv(x):
    return x*(1-x)

np.random.seed(1)

syn0 = 2*np.random.random((3,4))-1
syn1 = 2*np.random.random((4,1))-1

for i in range(60000):

    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    #print(l2)

    l2_error = y - l2
    #print(l2_error)
    if(i%10000)==0:
        print("Error"+str(np.mean(np.abs(l2_error))))
    l2_delta = l2_error * nonlin_deriv(l2)
    #print(l2_delta)
    l1_error = l2_delta.dot(syn1.transpose())
    l1_delta = l1_error*nonlin_deriv(l1)

    syn1 += l1.transpose().dot(l2_delta)
    syn0 += l0.transpose().dot(l1_delta)

print(syn0)
print(syn1)