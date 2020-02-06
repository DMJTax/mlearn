import numpy 
import mlearn as ml
import matplotlib.pyplot as plt

# create some dataset
N = [50,50]
x0 = numpy.random.randn(N[0],2)
x1 = numpy.random.randn(N[1],2)
x1[:,0] = x1[:,0] + 3.  # move data from class 1
x = numpy.concatenate((x0,x1),axis=0)
y = numpy.concatenate((-numpy.ones([N[0],1]),numpy.ones([N[1],1])),axis=0)

# define a model
f = ml.model_linear(dim=2)
# define loss
L = ml.decomposableloss(ml.loss_hinge,ml.reg_l2,0.1)
# optimise:
f,l = L.train_gd(f,x,y,learnrate=0.001,T=100)

# how good is it?
pred = f(x)
print("The classification error is %f."%numpy.mean(ml.loss_01(pred,y)))

# show
plt.scatter(x[:,0],x[:,1],c=y.flatten())
f.plot()
plt.show()

