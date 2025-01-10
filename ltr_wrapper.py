## LTR example main file 
## data generation by random polynomial function
import sys
import time

import numpy as np
## ###################################################
## for demonstration
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmplot
import sklearn.model_selection 
from sklearn.metrics import mean_squared_error
## ###################################################

import ltr_solver_multiview_013 as ltr

## ################################################################
## ################################################################
class ltr_cls:


  ## --------------------------------
  def __init__(self, **parameters):


    self.llinks = None
    self.cmodel = ltr.ltr_solver_cls(norder = parameters['order'], \
                                     rank = parameters['rank'])

    self.__set_basic_parameters()
    self.__set_additional_parameters()
    
    return

  ## public methods  
  ## --------------------------------
  def set_params(self,**hyperparams):

    if 'order'  in hyperparams:
      norder = hyperparams['order']
    else:
      norder = self.cmodel.norder
      
    if 'rank' in hyperparams:
      nrank = hyperparams['rank']
    else:
      nrank = self.cmodel.nrank0
    
    self.cmodel.update_parameters(norder = norder, \
                                  nrank0 = nrank, \
                                  nrank = nrank, \
                                  nrankuv = nrank)

    return

  ## --------------------------------
  def fit(self,X,Y):

    lX = [X,X]
    self.llinks = [0,0]
    self.cmodel.fit(lX,Y, llinks = self.llinks, nepoch = self.cmodel.nrepeat)

    return(self)

  ## --------------------------------
  def predict(self,Xtest):

    lXtest = [Xtest,Xtest]
    Ypred = self.cmodel.predict(lXtest, llinks = self.llinks).ravel()
    
    return(Ypred)

  ## ###########################################################  
  ## private methods
  ## ----------------------------------------
  def __set_basic_parameters(self):

    cmodel = self.cmodel

    ## -------------------------------------
    ## Parameters to learn
    ## the most important parameter
    norder=2      ## maximum power, valid if no design llinks = None
    rank=20       ## number of ranks
    rankuv=10      ## internal rank for bottlenesck if rankuv<rank
    sigma=0.02    ## learning step size
    nsigma=1      ## step size correction interval
    gammanag=0.9     ## discount for the ADAM method
    gammanag2=0.9    ## discount for the ADAM method norm

    # mini-bacht size,
    mblock=500

    ## number of epochs
    nrepeat=20

    ## regularizatin constant for xlambda optimization parameter
    cregular=1

    ## activation function
    iactfunc = 4  ## =0 identity, =1 arcsinh, =2 2*sigmoid-1, =3 tanh, =4 relu
    iactfunc_ext = 0  ## =0 identity, =1 arcsinh, =2 2*sigmoid-1, =3 tanh, =4 relu

    ## cmodel.lossdegree = 0  ## =0 L_2^2, =1 L^2, =0.5 L_2^{0.5}, ...L_2^{z}
    lossdegree = 0  ## default L_2^2 =0
    regdegree = 1   ## regularization degree, Lasso

    norm_type  = 0  ## parameter normalization
                    ## =0 L2 =1 L_{infty} =2 arcsinh + L2  
                    ## =3 RELU, =4 tanh + L_2 

    perturb = 0.0   ## gradient perturbation

    report_freq = 100  ## frequency of the training reports

    ## set optimization parameters
    cmodel.update_parameters(mblock = mblock, \
                    sigma0 = sigma, \
                    nsigma = nsigma, \
                    gammanag = gammanag, \
                    gammanag2 = gammanag2, \
                    nrepeat = nrepeat, \
                    cregular = cregular, \
                    iactfunc = iactfunc, \
                    iactfunc_ext = iactfunc_ext, \
                    lossdegree = lossdegree, \
                    regdegree = regdegree, \
                    norm_type  = norm_type, \
                    perturb  =  perturb, \
                    report_freq  =  report_freq)
    
    return
    
  ## ----------------------------------------  
  def __set_additional_parameters(self):
    """
    Taks to set additional LTR parameters,
         in the basic case they might not be changed at all.
         In this version the default values are used. 
    Input: cmodel  reference of the solver object
    """

    cmodel = self.cmodel
    
    ## normalization
    ## output
    cmodel.iymean=1      ## =1 output vectors centralized
    cmodel.ymean=0        ## ymean
    cmodel.iyscale=1      ## =1 output vectors are scaled
    cmodel.yscale=0       ## the scaling value
    ## input
    cmodel.ixmean = 1   ## centralized inputs by mean
    cmodel.ixscale = 1  ## scale inputs by L_infty norm 
    cmodel.ixl2norm = 0 ## scale inputs by l_2 norm
    cmodel.ihomogeneous=1  ## =1 input views are homogenised =0 not 

    ## quantile regression
    cmodel.iquantile = 0   ## norm based regression, =1 quantile regression
    cmodel.quantile_alpha = 0.5  ## quantile, confidence, parameter
    cmodel.iquantile_hyperbola = 1  ## = hyperbola = 0 logistic approximation
    cmodel.quantile_smooth = 1    ## smoothing parameter of the pinball loss
                   ## in case of logistic 
                   ## if it is larger then it is closer to the pinball 
                   ## but less smooth 
                   ## in case of hyperbola
                   ## if it is smaller then it is closer to the pinball 
                   ## but less smooth
    
    cmodel.quantile_scale =1  ## scale the tangent direction 

    return

## #####################################################    
def report_params(cmodel):

  ## report the most important parameter values
  print('Order:',cmodel.norder)
  print('Rank:',cmodel.nrank)
  print('Rankuv:',cmodel.nrankuv)
  print('Step size:',cmodel.sigma0)
  print('Step freq:',cmodel.nsigma)
  print('Step scale:',cmodel.dscale)
  print('Epoch:',cmodel.nrepeat)
  print('Mini-batch size:',cmodel.mblock)
  print('Discount:',cmodel.gamma)
  print('Discount for NAG:',cmodel.gammanag)
  print('Discount for NAG norm:',cmodel.gammanag2)
  print('Bag size:',cmodel.mblock)
  print('Regularization:',cmodel.cregular)
  print('Gradient max ratio:',cmodel.sigmamax)
  print('Type of activation:',cmodel.iactfunc)
  print('Degree of loss:',cmodel.lossdegree)
  print('Degree of regularization:',cmodel.regdegree)
  print('Normalization type:',cmodel.norm_type)
  print('Gradient perturbation:', cmodel.perturb)
  print('Activation:', cmodel.iactfunc)
  print('Input centralization:', cmodel.ixmean)
  print('Input L_infty scaling:', cmodel.ixscale)
  print('Quantile regression:',cmodel.iquantile)
  print('Quantile alpha:',cmodel.quantile_alpha)    ## 0.5 for L_1 norm loss
  print('Quantile smoothing:',cmodel.quantile_smooth)

  return
    
## ################################################################
## test  
def main(iworkmode=None):
  """
  Task: to run the LTR solver vis the the ltr_wrapper
  """

  igraph=1   ## =0 no graph =1 graph is shown
  
  ## data preparation
  irandom = 0
  if irandom == 1:
    m = 1000  ## number of examples
    n = 10    ## dimension of the examples
    ny = 10   ## dimension of the response
    rng = np.random.default_rng()
    X=rng.standard_normal(size=(m,n))  
    W=rng.standard_normal(size=(n,ny))  ## linear mapping
    Y=np.dot(X,W)           ## reference variables  
  elif irandom == 0:
    zdata = np.load('davis.npz')
    Xtrain = zdata['Xtrain']
    Ytrain = zdata['Ytrain']
    Xtest = zdata['Xtest']
    Ytest = zdata['Ytest']

  ## U,S,V = np.linalg.svd(Xtrain)
      
  parameters = {'order':2, 'rank': 70}
  
  iorder_or_rank = 1  ## =0 order =1 rank
  if iorder_or_rank == 0:  
    hyperparams = {'order':[2]}
  elif iorder_or_rank == 1:
    hyperparams = {'rank':[10,20,30,40,50,60,70,80]}
    
  for key in hyperparams:
    nkeylen = len(hyperparams[key]) 
    xcorr = np.zeros((nkeylen,2))

    ihp = 0
    for hp in hyperparams[key]:
      model = ltr_cls(**parameters)
      print(key,hp)
      report_params(model.cmodel)
      model.set_params(**{str(key):hp})
      model.fit(Xtrain,Ytrain)
      Ypred = model.predict(Xtrain)
      ycorr = np.corrcoef(Ytrain.ravel(),Ypred.ravel())[0,1]
      xcorr[ihp] = (hp,ycorr)
      ihp += 1
      print('Correlation of the prediction and original output:',str('%7.4f'%ycorr))  

      if igraph == 1:

        fig=plt.figure(figsize=(6,6))
        fig.suptitle('LTR results')

        tgrid = (1,1)

        ax=plt.subplot2grid(tgrid,(0,0),colspan=1,rowspan=1)
        ax.scatter(Ytrain,Ypred, s=1)
        ax.set_xlabel('y')
        ax.set_ylabel('p')
        ax.set_title('y versus p, '+'pcorr: '+str('%7.4f'%ycorr))
        ax.grid('on')
        ## ax.legend()

        plt.tight_layout(pad=1)
        plt.show()
    
      
      
    ## summary  
    print('Summary:',xcorr)


  print('Bye')

  return(0)

## ###################################################
## ################################################################
if __name__ == "__main__":
  if len(sys.argv)==1:
    iworkmode=0
  elif len(sys.argv)>=2:
    iworkmode=eval(sys.argv[1])
  main(iworkmode)
