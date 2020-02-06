"""
Machine Learning toolbox
"""
import numpy
import matplotlib.pyplot as plt
import copy  # tsk tsk, this python

# === ML model =============================================

class mlmodel(object):
    """General ML prediction model

    Defines a model that contains:
    .name  the model name
    .dim   the input dimensionality for the data
    .w     the model parameters
    .pred  the output (plus gradient, if required)

    """

    def __init__(self):
        self.name = ''
        self.pred = None
        self.dim = 0
        self.w = None

    def __str__(self):
        return "%s model with %dD input." % (self.name,self.dim)

    def __call__(self,x,give_grad=False):
        if self.pred is None:
            return None # or raise an error??
        else:
            # keep on torturing the data until it becomes a numpy array of
            # the right size:
            # (did I ever told you that I hate python?)
            if not isinstance(x,numpy.ndarray):
                x = numpy.array(+x)
            if (len(x.shape)==1):
                x = numpy.array(x,ndmin=2)
            return self.pred(x,give_grad)

    def __len__(self):
        return len(self.w)

    def gradientcheck(self,x):
        "Gradient checking of the function. If the test fails, output is false"
        # claimed exact outcome and gradient:
        [pred,grad] = self.pred(x,give_grad=True)
        # the original weights:
        W = numpy.copy(self.w)
        # now approximate
        smallval = 1e-8
        approx = numpy.zeros(grad.shape)
        for i in range(0, len(W)):
            self.w = W + smallval
            f1 = self.pred(x)
            self.w = W - smallval
            f2 = self.pred(x)
            df = f1-f2
            approx[:,i] = df[:,0]/(2*smallval)
        err = abs(grad - approx)

        return (err.max()<1e-8)

    def plot(self,levels=[0.0],colors=None,gridsize = 30):
        ax = plt.gca()
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        dx = (xl[1]-xl[0])/(gridsize-1)
        dy = (yl[1]-yl[0])/(gridsize-1)
        x = numpy.arange(xl[0],xl[1]+0.01*dx,dx)
        y = numpy.arange(yl[0],yl[1]+0.01*dy,dy)
        z = numpy.zeros((gridsize,gridsize))
        for i in range(0,gridsize):
            for j in range(0,gridsize):
                # have I already told you that I hate python?
                featvec = numpy.array([x[i],y[j]],ndmin=2)
                z[j,i] = self(featvec)
        plt.contour(x,y,z,levels,colors=colors)

# === decomposable loss ====================================

class decomposableloss(object):
    """Loss function class

    Encapsulate an optimisation problem like:
           L = sum_i loss(fx_i,y_i)  +  lambda * Regularizer"
    in an object.

    Input arguments:
    loss    loss function
    reg     regularizer
    l       tradeoff parameter lambda
    """

    def __init__(self,loss,reg,l):
        if not callable(loss):
            raise ValueError('The loss function should be a function.')
        if not callable(reg):
            raise ValueError('The regularization function should be a function.')
        self.name = ''
        self.dataloss = loss
        self.regularizer = reg
        self.lambd = l

    def __str__(self):
        outstr = "Decomp.loss: "+self.dataloss.func_name
        if (self.lambd>0):
            outstr+=" + %f %s" % (self.lambd,self.regularizer.func_name)
        return outstr

    def __call__(self,f,x,y):
        #prediction
        (pred,dfdw) = f(x,give_grad=True)
        # losses:
        (l,dldf) = self.dataloss(pred,y)
        (r,drdw) = self.regularizer(f.w)
        # total loss:
        l = sum(l) + self.lambd*r
        # and derivative 
        # (this matrix multiplication is ridiculous! I hate python :-( )
        dldw = numpy.dot(dfdw.transpose(), dldf) + self.lambd*drdw

        return (l,dldw)

    def train_gd(self,f,x,y,learnrate=0.0001,T=1000):
        f = copy.deepcopy(f)
        loss = numpy.zeros(T)
        t = 0
        deltal = -numpy.inf
        while (t<T) and (deltal<1e-7):
            (loss[t],dldw) = self(f,x,y)
            f.w = f.w -learnrate*dldw

            if (numpy.remainder(t,100)==0):
                print('Epoch %d: loss=%f' % (t,loss[t]))
            if (t>0):
                deltal = loss[t]-loss[t-1]
            t += 1
           
        return (f,loss)

    def train_adam(self,f,x,y,learnrate=0.001,T=10000,beta1=0.9,beta2=0.999):
        f = copy.deepcopy(f)
        n = len(f)
        loss = numpy.zeros(T)
        eps = 1e-8
        t = 0
        m = numpy.zeros((n,1))
        v = numpy.zeros((n,1))
        deltal = -numpy.inf
        while (t<T) and (deltal<1e-7):
            (loss[t],dldw) = self(f,x,y)
            m = beta1*m + (1.-beta1)*dldw
            v = beta2*v + (1.-beta2)*dldw*dldw
            m = m/(1.-beta1**(t+1))
            v = v/(1.-beta2**(t+1))
            f.w = f.w -learnrate*m/(numpy.sqrt(v) + eps)

            if (numpy.remainder(t,100)==0):
                print('Epoch %d: loss=%f' % (t,loss[t]))
            if (t>0):
                deltal = loss[t]-loss[t-1]
            t += 1
           
        return (f,loss)
            
    def train_adam2(self,f,x,y,learnrate=0.001,T=10000,beta1=0.9,beta2=0.999):
        f = copy.deepcopy(f)
        loss = numpy.zeros(T)
        eps = 1e-8
        (loss[0],dldw) = self(f,x,y)
        m = dldw
        v = dldw*dldw
        t = 1
        deltal = -numpy.inf
        while (t<T) and (deltal<1e-7):
            (loss[t],dldw) = self(f,x,y)
            m = beta1*m + (1.-beta1)*dldw
            v = beta2*v + (1.-beta2)*dldw*dldw
            m = m/(1.-beta1**t)
            v = v/(1.-beta2**t)
            f.w = f.w -learnrate*m/(numpy.sqrt(v) + eps)

            if (numpy.remainder(t,100)==0):
                print('Epoch %d: loss=%f' % (t,loss[t]))
            if (t>0):
                deltal = loss[t]-loss[t-1]
            t += 1
           
        return (f,loss)
            
# === nondecomposable loss ====================================

class nondecomposableloss:
    "General non-decomposable loss like PRAUC or AUC"

    def __init__(self,loss,dataloss,alpha,param=None,lambd=0,reg=[]):
        self.name = ''
        self.loss = loss
        self.dataloss = dataloss
        self.alpha = alpha   # the lagrange multipliers
        self.param = param
        self.lambd = lambd
        self.reg = reg

    def __call__(self,f,x,y,v):
        # prediction
        pred,dfdw = f(x)
        # losses:
        l,dldf = self.dataloss(pred,y)
        if (self.lambd==0):
            r = 0
            drdw = numpy.zeros(f.w.shape)
        else:
            weight = f.w
            (r,drdw) = self.reg(weight)
        # total loss
        l = numpy.dot(v.transpose(),l) + self.lambd*r
        # and derivative 
        dldw = numpy.dot(dfdw.transpose(), v*dldf) + self.lambd*drdw
        return l, dldw

    def train_minmax_gd(self,f,x,y,learnrate=0.0001,T=1000):
        # initialise:
        f = copy.deepcopy(f)
        l = numpy.inf
        t = 1
        deltal = -numpy.inf

        # iterate over training epochs:
        while ((t<T) & (deltal<-1e-7)):
            # update the model parameters:
            newl,dldw,tmp = self.loss(self.param,self.dataloss,f,x,y,self.alpha)
            f.w = f.w - learnrate*dldw
            l = numpy.vstack((l,newl))  # tsk tsk, python

            # update the lagrange multipliers
            newl,newdl,dalphadw = self.loss(self.param,self.dataloss,f,x,y,self.alpha)
            self.alpha = self.alpha + learnrate*dalphadw
            self.alpha[self.alpha<0] = 0  # Hmmm, good enough?

            if (numpy.remainder(t,100)==0):
                print('Epoch %d: loss=%f' % (t,newl))
            if (t>10):
                deltal = l[t]-l[t-1]
            t += 1

        return f,l

    
# === loss definitions ======================================

def loss_01(fx,y):
    l = (numpy.sign(fx) != y)*1  # ugly way to convert logical->int
    dldf = numpy.zeros(fx.shape)
    return (l,dldf)

def loss_hinge(fx,y):
    score = fx*y
    I = (score<1)*1  # ugly way to convert logical->int
    l = (1-score)*I
    dldf = -I*y
    return (l,dldf)

def loss_logistic(fx,y):
    score = fx*y
    l = numpy.log(1.+numpy.exp(-score))/numpy.log(2.)
    dldf = -y/(1.+numpy.exp(score))/numpy.log(2.)
    return (l,dldf)

def loss_squared(fx,y):
    dff = fx-y
    l = dff*dff 
    dldf = 2*dff
    return (l,dldf)

def loss_l1(fx,y):
    dff = fx-y
    l = numpy.abs(dff)
    dldf = numpy.sign(dff)
    return (l,dldf)

# weirdo one:
def loss_ownloss(fx,y):
    p = 6
    dff = numpy.abs(fx - y)
    sg = numpy.sign(fx - y)
    l = dff**p
    dldf = (p*dff**(p-1))*sg
    return (l,dldf)

# --- asymmetric loss functions ---

def loss_exp_sigm(fx,y):
    A = 1.
    B = 1.
    n = len(fx)
    l = numpy.zeros((n,1))
    dldf = numpy.zeros((n,1))
    # sigmoid for positive class
    I = (y==1).nonzero()[0]   # shitty Python
    l[I] = 2/(1.+numpy.exp(fx[I]))
    dldf[I] = -l[I]*(1.-l[I]/2)
    # exp for the negative class
    I = (y==-1).nonzero()[0]   # shitty Python
    ef = numpy.exp(fx[I])
    l[I] = A*ef**B
    dldf[I] = A*B*(ef**(B-1))*ef

    return (l,dldf)

# --- complicated loss functions ---

def loss_auc(fx,y,bnd=None):
    if bnd is None:
        bnd = [0.0,1.0]
    if (bnd[1]<bnd[0]):
        raise ValueError('Upper limit should be larger than lower limit.')
    if (bnd[0]<0.):
        raise ValueError('Lower limit should be larger or equal than 0.')
    if (bnd[1]>1.):
        raise ValueError('Upper limit should be smaller or equal than 1.')
    fnr,fpr = roc(fx,y)

    out = numpy.trapz(1.-fpr,fnr)
    return out

def loss_prc(fx,y):
    prec,rec = prc(fx,y)

    out = numpy.trapz(prec,rec)
    return out

def loss_RatP(prec,surrogateL,f,x,y,alpha):
    Ip = numpy.where(y==+1)[0]
    In = numpy.where(y==-1)[0]
    nrY = len(Ip)
    v = numpy.zeros(y.shape)

    # predict
    pred,dfdw = f(x)
    lh,dlhdf = surrogateL(pred,y)
    v[Ip] = 1+alpha
    v[In] = alpha*(prec/(1-prec))
    l = numpy.dot(v.transpose(),lh) - alpha*nrY

    # weighted derivative wrt w
    dldw = numpy.dot(dfdw.transpose(), v*dlhdf)
    # derivative wrt alpha
    v[Ip] = 1
    v[In] = prec/(1-prec)
    dalphadw = numpy.dot(v.transpose(),lh) - nrY

    return l,dldw,dalphadw
    
def loss_PatR(rec,surrogateL,f,x,y,alpha):
    Ip = numpy.where(y==+1)[0]
    In = numpy.where(y==-1)[0]
    nrY = len(Ip)
    v = numpy.zeros(y.shape)

    # predict
    pred,dfdw = f(x)
    lh,dlhdf = surrogateL(pred,y)
    v[Ip] = alpha/nrY
    v[In] = 1
    l = numpy.dot(v.transpose(),lh) - alpha*(rec-1)

    # weighted derivative wrt w
    dldw = numpy.dot(dfdw.transpose(), v*dlhdf)
    # derivative wrt alpha
    v[Ip] = 1/nrY
    v[In] = 0
    dalphadw = numpy.dot(v.transpose(),lh) + rec - 1

    return l,dldw,dalphadw
    
def loss_aucpr(prec,surrogateL,f,x,y,alpha):
    K = len(alpha)
    delta = numpy.diff(prec)
    Ip = numpy.where(y==+1)[0]
    In = numpy.where(y==-1)[0]
    nrY = len(Ip)
    v = numpy.zeros(y.shape)

    # compute the loss over the different biases
    l = 0;
    dldw = numpy.zeros(f.w.shape);
    dalphadw = numpy.zeros((K,1));
    for t in range(0,K):
        # prediction
        pred,dfdw = f(x,t)
        lh,dlhdf = surrogateL(pred,y)
        # weighted loss
        v[Ip] = 1 + alpha[t]
        v[In] = alpha[t]*prec[t+1]*(1-prec[t+1])
        l = l + delta[t]*(numpy.dot(v.transpose(),lh) - alpha[t]*nrY)

        # derivative wrt w
        dldw = numpy.dot(dfdw.transpose(), v*dlhdf)
        # derivative wrt alpha
        v[Ip] = 1
        v[In] = prec[t+1]/(1-prec[t+1])
        dalphadw[t] = delta[t]*(numpy.dot(v.transpose(),lh) - nrY)

    return l,dldw,dalphadw

def roc(pred,y,targetlab=1,targetid=0):
    Ntarget = numpy.sum(y==targetlab)
    Noutl = numpy.sum(y!=targetlab)

    lab = (y==targetlab)*1.  # ugly to convert logical->float
    # and sort:
    ind = pred[:,targetid].argsort()
    labt = lab[ind]
    labo = labt-1.

    # how many ones left from threshold:
    FNr = numpy.cumsum(labt)/(Ntarget)
    # how many zeros right from threshold:
    FPr = 1. + numpy.cumsum(labo)/Noutl

    return FNr,FPr

def plotroc(FNr,FPr=None):
    plt.plot(FPr,1.-FNr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

def prc(pred,y,targetlab=1,targetid=0):
    Ntarget = numpy.sum(y==targetlab)
    lab = (y==targetlab)*1.  # ugly to convert logical->float

    # sort predictions:
    pred = -pred
    ind = pred[:,targetid].argsort()
    labt = lab[ind]
    labo = 1.-labt

    clabt = numpy.cumsum(labt)
    clabo = numpy.cumsum(labo)

    prec = clabt/(clabt+clabo)
    rec = clabt/Ntarget

    return prec,rec

 
# --- standard regularizers ----------------------------

def reg_l1(w):
    r = sum(abs(w))
    drdw = numpy.sign(w)
    return (r,drdw)

def reg_l2(w):
    r = w.transpose().dot(w)  # this is ridiculous!
    drdw = 2*w
    return (r,drdw)

# === plotting loss ======================================
def plotloss2D(loss,f,wi,wj,x,y,nrlevels=10,colors=None,gridsize = 30):
    dwi = (wi[2]-wi[1])/(gridsize-1)
    dwj = (wj[2]-wj[1])/(gridsize-1)
    wis = numpy.arange(wi[1],wi[2]+0.01*dwi,dwi)
    wjs = numpy.arange(wj[1],wj[2]+0.01*dwj,dwj)
    z = numpy.zeros((gridsize,gridsize))
    weights = f.w
    for i in range(0,gridsize):
        for j in range(0,gridsize):
            f.w[wi[0]] = wis[i]
            f.w[wj[0]] = wjs[j]
            fx,_ = f(x)
            z[j,i] = numpy.mean(loss(fx,y)) # average over all data
    levels = numpy.linspace(numpy.min(z),numpy.max(z),nrlevels)
    plt.contour(wis,wjs,z,levels,colors=colors)
    plt.xlabel('w_%d'%wi[0])
    plt.ylabel('w_%d'%wj[0])


#  = interface to prtools...
def m2p(f,*args):
    "ML to PRtools mapping"
    if isinstance(f,str):
        if (f=='untrained'):
            return 'M2P '
    if isinstance(f,mlearn.mlmodel):
        # store the model in a prmapping:
        newm = prmapping(m2p)
        newm.data = (f,args) # a bit ugly, but needed
        newm.name += f.name
        newm.shape[0] = f.dim
        newm.shape[1] = 1
        newm.mapping_type = 'trained'
        return newm
    else:
        # we are applying to new data, stored in args[0]
        if isinstance(f[0],mlearn.mlmodel):
            functionargs = f[1]
            out = f[0](args[0],*functionargs)  # bloody Python magic
        else:
            print("we did not get a ml model!!")

        return out[0]


# === simple models ======================================

class model_linear(mlmodel):
   "Linear model"

   def __init__(self,dim=2,sigm=0.0001):
       self.name = 'Linear'
       self.dim = dim
       self.w = sigm*numpy.random.randn(dim+1,1)
   
   def pred(self,x,give_grad=False):
       # Function output and derivative wrt. weights
       N = x.shape[0]
       xx = numpy.concatenate((x,numpy.ones((N,1))),axis=1)
       if give_grad:
           return (xx.dot(self.w), xx)
       else:
           return xx.dot(self.w)
        
class model_linear_nobias(mlmodel):
   "Linear model without bias"

   def __init__(self,dim=2,sigm=0.001):
       self.name = 'Non-biased linear'
       self.dim = dim
       self.w = sigm*numpy.random.randn(dim,1)
   
   def pred(self,x,give_grad=False):
       # Function output and derivative wrt. weights
       if give_grad:
           return (x.dot(self.w), x)
       else:
           return x.dot(self.w)

class model_linear_multib(mlmodel):
    "Linear model with multiple biases"

    def __init__(self,dim=2,k=5,sigm=0.001):
        self.name = 'Multiple-bias linear'
        self.dim = dim
        w1 = sigm*numpy.random.randn(dim,1)
        w2 = numpy.linspace(-2*sigm, 2*sigm, k)
        self.w = numpy.vstack((w1,w2[:,None]))  # WTF! I hate python!

    def pred(self,x,give_grad=False):
        sz = x.shape
        nr = len(self.w)-sz[1]
        f = x.dot(w[0:sz[1]]) + w[sz[1]+k[0]]
        if give_grad:
            dfdw = numpy.concatenate((x,numpy.zeros((sz[0],nr))),axis=1)
            dfdw[:,sz[1]+k[0]] = 1
            return f,dfdw
        else:
            return f

# some standard classifiers:

def ols(X,y,lambda1=0.):
    " Ordinary least-squares "
    n = X.shape[0]
    X = numpy.concatenate((X,numpy.ones((n,1))),axis=1)
    dim = X.shape[1]
    C = numpy.matmul(X.T,X) + lambda1*numpy.eye(dim)
    Cinv = numpy.linalg.inv(C)
    tmp = Cinv.dot(X.T)
    f = model_linear(dim=dim)
    f.w = tmp.dot(y)
    return f

def logistic(X,y,lambda1=0.):
    " Logistic classifier "
    N,dim = X.shape
    f = model_linear(dim=dim)
    L = decomposableloss(ml.loss_logistic,ml.reg_l2,lambda1)
    f,l = L.train_gd(f,X,y,learnrate=0.0001,T=100)
    return f



