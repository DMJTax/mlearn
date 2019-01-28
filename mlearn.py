import numpy
import matplotlib.pyplot as plt
import copy  # tsk tsk, this python

# === ML model =============================================

class mlmodel(object):
    'General prediction model given some input feature vector'

    def __init__(self,modelfunc,dim,*args):
        if not callable(modelfunc):
            raise ValueError('The modelfunction should be a function.')
        if not isinstance(dim,int):
            raise ValueError('The dimensionality should be an integer.')
        self.name = ''
        self.modelfunc = modelfunc
        self.dim = dim
        (self.w,self.name) = modelfunc('init',dim,args)

    def __str__(self):
        return "%s model with %dD input." % (self.name,self.dim)

    def w(self):
        print(self.w)

    def __call__(self,x,*args):
        # keep on torturing the data until it becomes a numpy array of
        # the right size:
        # (did I ever told you that I hate python?)
        if not isinstance(x,numpy.ndarray):
            x = numpy.array(+x)
        if (len(x.shape)==1):
            x = numpy.array(x,ndmin=2)
        f,df = self.modelfunc(self.w,x,args)
        return f,df

    def gradientcheck(self,x):
        "Gradient checking of the function. If the test fails, output is false"
        # claimed exact outcome and gradient:
        [pred,grad] = self.modelfunc(self.w,x)
        # the original weights:
        W = self.w
        # now approximate
        smallval = 1e-8
        approx = numpy.zeros(grad.shape)
        for i in range(0, len(W)):
            w1 = numpy.copy(W); w1[i]+=smallval  # WTF! explicit copy!!
            f1 = self.modelfunc(w1,x)[0]
            w2 = numpy.copy(W); w2[i]-=smallval
            f2 = self.modelfunc(w2,x)[0]
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
                z[j,i] = self(featvec)[0]
        plt.contour(x,y,z,levels,colors=colors)

# === decomposable loss ====================================

class decomposableloss:
    "Loss function like   sum_i loss(fx_i,y_i) + lambda * Regularizer"

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
        (pred,dfdw) = f(x)
        # losses:
        (l,dldf) = self.dataloss(pred,y)
        (r,drdw) = self.regularizer(f.w)
        # total loss:
        l = sum(l) + self.lambd*r
        # and derivative 
        # (this matrix multiplication is ridiculous! I hate python :-( )
        dldw = numpy.dot(dfdw.transpose(), dldf) + self.lambd*drdw

        return (l,dldw)

    def train_gd(self,f,x,y,learnrate=0.0001,T=100):
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

    
# === simple definitions ======================================

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

# --- asymmetric ones ---

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

# --- complicated ones ---

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

def reg_l1(w):
    r = sum(abs(w))
    drdw = numpy.sign(w)
    return (r,drdw)

def reg_l2(w):
    r = w.transpose().dot(w)  # this is ridiculous!
    drdw = 2*w
    return (r,drdw)

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

# === simple models ======================================

def model_linear(w,x,s=None):
   "Linear model"
   
   if isinstance(w,basestring):
       # Define the initialisation:
       if s is None:
           s=0.0001
       if (len(s)==0):
           s=0.0001
       w = s*numpy.random.randn(x+1,1)
       return (w,"Linear")     # parameters and name
   else:
       # Function output and derivative wrt. weights
       sz = x.shape
       x1 = numpy.concatenate((x,numpy.ones((sz[0],1))),axis=1)
       return (x1.dot(w), x1)  # model output and derivative
        
def model_linear_nob(w,x,s):
   "Linear model without bias term"
   if isinstance(w,basestring):
       # First define the initialisation:
       if (len(s)<1):
           s=0.0001
       w = s*numpy.random.randn(x,1)
       return (w,"Nonbias linear")
   else:
       # Next define the function output and derivative wrt. weights
       return (x.dot(w), x)

def model_linear_multib(w,x,k=5):
    "Linear model with multiple biases"
    if isinstance(w,basestring):
        if (len(k)<2):
            s = 0.0001
        else:
            s = k[1]
        w1 = s*numpy.random.randn(x,1)
        w2 = numpy.linspace(-2*s, 2*s, k[0])
        w = numpy.vstack((w1,w2[:,None]))  # WTF! I hate python!
        return (w,"Linear multibias")
    else:
        sz = x.shape
        nr = len(w)-sz[1]
        f = x.dot(w[0:sz[1]]) + w[sz[1]+k[0]]
        dfdw = numpy.concatenate((x,numpy.zeros((sz[0],nr))),axis=1)
        dfdw[:,sz[1]+k[0]] = 1
        return f,dfdw



