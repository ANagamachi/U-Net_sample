import keras
import sys
sys.path.append('ANN')
from unetxztanh import UNet
import numpy as np
#import GPy, GPyOpt
import random
import os
import h5py
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU,PReLU
#leaky_relu = LeakyReLU()
from keras.layers.normalization import BatchNormalization
#from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json,load_model
from keras import optimizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#import sklearn.cross_validation as crv
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
from keras.utils import plot_model
from keras.applications.vgg16 import VGG16
import keras.callbacks
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
#old_session = KTF.get_session()

#session = tf.Session('')
#KTF.set_session(session)
#KTF.set_learning_phase(1)

def RRSE(y_true, y_pred):
    #Root relative squared error
    return K.sqrt(K.mean(K.square(y_pred - y_true)) / K.mean(K.square(K.mean(y_true) - y_true)))

def R2(y_true, y_pred):
    #Coefficient of determination
    return K.sqrt(K.mean(K.square(y_pred - K.mean(y_true,keepdims=True))) / K.mean(K.square(y_true - K.mean(y_true,keepdims=True))))

def RMSPE(y_true, y_pred):
    #Root mean squared percentage error
    return K.sqrt(K.mean(K.square((y_pred - y_true)/(y_true + K.epsilon()))))

print '_____Input position of x-z plane!!!!!(4 or 16 or 48)_____'
ypl = input()
#position of x-z plane
#4 or 16 or 48

print '_____Input index number(cxx=0,cxy=1,...)!!!!!_____'
index = input()

CIJ = ['CXX','CXY','CXZ','CYY','CYZ','CZZ']
cij = ['cxx','cxy','cxz','cyy','cyz','czz']
dir_name = 'infertanh'
out_name = 'unet_y{0}stuicij'.format(ypl)
case_name = 'unet_y{0}stui{1}'.format(ypl,cij[index])

head = ('head','<i')
tail = ('tail','<i')

dtu = np.dtype([head,('U','<538229d'),tail])
dtv = np.dtype([head,('V','<538229d'),tail])
dtw = np.dtype([head,('W','<538229d'),tail])

dtcxx = np.dtype([head,(CIJ[index],'<466578d'),tail])

U=[]

CXX=[]

#dirpath=os.path.dirname(os.getcwd())

for i in xrange(384,484):
    with open('data/383+20000per40/CHAN%03dDU.DAT' % i,'r') as fd1:
        chunk1 = np.fromfile(fd1, dtype=dtu, count=1)
    with open('data/383+20000per40/CHAN%03dDV.DAT' % i,'r') as fd2:
        chunk2 = np.fromfile(fd2, dtype=dtv, count=1)
    with open('data/383+20000per40/CHAN%03dDW.DAT' % i,'r') as fd3:
        chunk3 = np.fromfile(fd3, dtype=dtw, count=1)
    with open('data/383+20000per40/CHAN%03dD{0}.DAT'.format(CIJ[index]) % i,'r') as fd11:
        chunk11 = np.fromfile(fd11, dtype=dtcxx, count=1)
    U1 = chunk1[0]['U'].reshape((73,101,73),order='F')
    V1 = chunk2[0]['V'].reshape((73,101,73),order='F')
    W1 = chunk3[0]['W'].reshape((73,101,73),order='F')
    CXX1 = chunk11[0][CIJ[index]].reshape((69,98,69),order='F')
    Ui = np.concatenate((U1[5:69,ypl+2,5:69][:,:,None],V1[5:69,ypl+2,5:69][:,:,None],W1[5:69,ypl+2,5:69][:,:,None]),axis=2)
#    Ui = np.transpose(Ui, (0,2,1))
    U.append(Ui)
    CXX.append(CXX1[3:67,ypl,3:67][:,:,None])
    print i

x = np.asarray(U)

y = np.asarray(CXX)
#y = np.concatenate([UFD,VFD,WFD],axis=1)

#data=pd.read_csv('CHANLEARNALL100_cxx.csv')
#data.index
#x=np.array(data.loc[:,14:79])
#x=np.array(data.loc[1:442368,['U','DPDX','UDUDX','VDUDY','WDUDZ','D2UDX2','D2UDY2','D2UDZ2','DCXXDX','DCXYDY','DCZDZ']].astype(np.float128))
#x=np.array(data.loc[:,['CXX','CXXDISC','R(J)']].astype(np.float128))
#xnp=x.as_matrix()
#y=np.array(data.loc[:,0:8])
#y=np.array(data.loc[1:442368,['U100t']].astype(np.float128))
#y=np.array(data.loc[:,['CXX100t']].astype(np.float128))
#ynp=y.as_matrix()

xmean = x.mean(axis=(0,1,2))
ymean = y.mean(axis=(0,1,2))
xstd  = np.std(x, axis=(0,1,2))
ystd  = np.std(y,axis=(0,1,2))
np.savetxt('./{0}/{2}/{1}/{1}xmean.csv'.format(dir_name,case_name,out_name),xmean,delimiter=',')
np.savetxt('./{0}/{2}/{1}/{1}ymean.csv'.format(dir_name,case_name,out_name),ymean,delimiter=',')
np.savetxt('./{0}/{2}/{1}/{1}xstd.csv'.format(dir_name,case_name,out_name),xstd,delimiter=',')
np.savetxt('./{0}/{2}/{1}/{1}ystd.csv'.format(dir_name,case_name,out_name),ystd,delimiter=',')

#x = preprocessing.scale(x)
#y = preprocessing.scale(y)

x = (x-xmean)/xstd
y = (y-ymean)/ystd

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
#inlen=len(x_train[0])
#outlen=len(y_train[0])

def unet_model():
    input_channel_count = 3
    output_channel_count = 1
    first_layer_filter_count = 64
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape',RRSE,R2,RMSPE])
    return model

#-----------------------------------------Bayesian Optimization---------------------------------------
#!!!!If you want to use Bayse Optimization,
#      1. reg_model() => reg_model(params)
#     2. the parameters you'd like to optimize => params[(take a number)].!!!!
#
#from skopt import gp_minimize
#from skopt.acquisition import gaussian_lcb
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import Matern
#noise_level=0.1
#
#gp = GaussianProcessRegressor(kernel=Matern(length_scale_bounds='fixed'), 
#                              alpha=noise_level**2, random_state=0)
#
#res = gp_minimize(reg_model,                  # the function to minimize
#                  [(inlen,500),(10,100),(10,20)],      # the bounds on each dimension of x
#                  base_estimator=gp,  # a GP estimator (optional)
#                  acq='LCB',          # the acquisition function (optional)
#                  n_calls=10,         # the number of evaluations of f including at x0
#                 n_random_starts=10,  # the number of random initialization points
#                  random_state=None)
#------------------------------------------------------------------------------------------------------------------

es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')
csv_logger = keras.callbacks.CSVLogger('./{0}/{2}/{1}/{1}log.csv'.format(dir_name,case_name,out_name))
#tb_cb = keras.callbacks.TensorBoard(log_dir='./tflog/', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
#tb_cb = keras.callbacks.TensorBoard(log_dir='./tflogcxx/', histogram_freq=1, write_graph=True, write_images=True)
#cbks = [es_cb, tb_cb]
#cbks = [tb_cb]
cbks = [es_cb,csv_logger]
#cbks = tf.TensorShape([es_cb, tb_cb])

estimator = KerasRegressor(build_fn=unet_model, epochs=50, batch_size=16)

#-------------------------------------------cross validation------------------------------------------
#seed=7
#np.random.seed(seed)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, x, y, cv=kfold)
#print('Results: %.2f (%.2f) MSE' % (results.mean(), results.std()))

# evaluate model with standardized dataset
#np.random.seed(seed)
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=reg_model, epochs=50, batch_size=10, verbose=1)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, x, y, cv=kfold)
#print('Standardized: %.2f (%.2f) MSE' % (results.mean(), results.std()))
#------------------------------------------------------------------------------------------------------------------

estimator.fit(x_train, y_train, validation_split=0.1, callbacks=cbks)

y_pred = estimator.predict(x_test)
estimator.model.save('./{0}/{2}/{1}/{1}_model.h5'.format(dir_name,case_name,out_name))
np.savetxt('./{0}/{2}/{1}/{1}test_compare.csv'.format(dir_name,case_name,out_name),np.concatenate([y_test.reshape(-1,1),y_pred.reshape(-1,1)],axis=1),delimiter=',')
print('error: ',mean_squared_error(y_test,y_pred))
print('score: ',np.sqrt(np.mean(np.square(y_pred - np.mean(y_test,keepdims=True))) / np.mean(np.square(y_test - np.mean(y_test,keepdims=True)))))
print('persentage error: ',np.sqrt(np.mean(np.square((y_pred - y_test)/np.clip(np.absolute(y_test) , 1e-08, None)))))
#y_test = tf.convert_to_tensor(y_test, np.float32)
#y_pred = tf.convert_to_tensor(y_pred, np.float32)
#print('score: ',R2(y_test,y_pred))
#print('percentage error: ',RMSPE(y_test,y_pred))
#KTF.set_session(old_session)

#reg_model().model.save_weights('all100_mlp_weights.h5')


