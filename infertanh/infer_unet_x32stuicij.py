import keras
import pandas as pd
import numpy as np
from keras.models import model_from_json,load_model
import time
import keras.backend as K
import matplotlib.pyplot as plt
start_time = time.time()
#================================================================================================================================
#================================================================================================================================
#================================================================================================================================
xpl = 32
case_name = 'unet_x{0}stui'.format(xpl)
xmean = np.loadtxt('./{0}cij/{0}cxx/{0}cxxxmean.csv'.format(case_name),delimiter=',')
xstd = np.loadtxt('./{0}cij/{0}cxx/{0}cxxxstd.csv'.format(case_name),delimiter=',')
CXXmean = np.loadtxt('./{0}cij/{0}cxx/{0}cxxymean.csv'.format(case_name),delimiter=',')
CXXstd = np.loadtxt('./{0}cij/{0}cxx/{0}cxxystd.csv'.format(case_name),delimiter=',')
CXYmean = np.loadtxt('./{0}cij/{0}cxy/{0}cxyymean.csv'.format(case_name),delimiter=',')
CXYstd = np.loadtxt('./{0}cij/{0}cxy/{0}cxyystd.csv'.format(case_name),delimiter=',')
CXZmean = np.loadtxt('./{0}cij/{0}cxz/{0}cxzymean.csv'.format(case_name),delimiter=',')
CXZstd = np.loadtxt('./{0}cij/{0}cxz/{0}cxzystd.csv'.format(case_name),delimiter=',')
CYYmean = np.loadtxt('./{0}cij/{0}cyy/{0}cyyymean.csv'.format(case_name),delimiter=',')
CYYstd = np.loadtxt('./{0}cij/{0}cyy/{0}cyyystd.csv'.format(case_name),delimiter=',')
CYZmean = np.loadtxt('./{0}cij/{0}cyz/{0}cyzymean.csv'.format(case_name),delimiter=',')
CYZstd = np.loadtxt('./{0}cij/{0}cyz/{0}cyzystd.csv'.format(case_name),delimiter=',')
CZZmean = np.loadtxt('./{0}cij/{0}czz/{0}czzymean.csv'.format(case_name),delimiter=',')
CZZstd = np.loadtxt('./{0}cij/{0}czz/{0}czzystd.csv'.format(case_name),delimiter=',')
def RRSE(y_true, y_pred):
    #Root relative squared error
    return K.sqrt(K.mean(K.square(y_pred - y_true)) / K.mean(K.square(K.mean(y_true) - y_true)))

def R2(y_true, y_pred):
    #Coefficient of determination
    return K.sqrt(K.mean(K.square(y_pred - K.mean(y_true,keepdims=True))) / K.mean(K.square(y_true - K.mean(y_true,keepdims=True))))

def RMSPE(y_true, y_pred):
    #Root mean squared percentage error
    return K.sqrt(K.mean(K.square((y_pred - y_true)/(y_true + K.epsilon()))))

head = ('head','<i')
tail = ('tail','<i')

dtu = np.dtype([head,('U','<538229d'),tail])
dtv = np.dtype([head,('V','<538229d'),tail])
dtw = np.dtype([head,('W','<538229d'),tail])

dtcxx = np.dtype([head,('CXX','<466578d'),tail])
dtcxy = np.dtype([head,('CXY','<466578d'),tail])
dtcxz = np.dtype([head,('CXZ','<466578d'),tail])
dtcyy = np.dtype([head,('CYY','<466578d'),tail])
dtcyz = np.dtype([head,('CYZ','<466578d'),tail])
dtczz = np.dtype([head,('CZZ','<466578d'),tail])

print '_____Input file number(534 or 884)!!!!!_____'
fn=str(input())
#534 or 884
print '_____Input distance from T1(1 or 10000)!!!!!_____'
dstT=str(input())
#1 or 10000
file_name=fn+'_'+dstT

with open('./'+file_name+'/CHAN'+fn+'DU.DAT') as fd:
#with open('CHAN384DU.DAT') as fd:
    chunk1 = np.fromfile(fd, dtype=dtu, count=1)
with open('./'+file_name+'/CHAN'+fn+'DV.DAT') as fd:
#with open('CHAN384DV.DAT') as fd:
    chunk2 = np.fromfile(fd, dtype=dtv, count=1)
with open('./'+file_name+'/CHAN'+fn+'DW.DAT') as fd:
#with open('CHAN384DW.DAT') as fd:
    chunk3 = np.fromfile(fd, dtype=dtw, count=1)
U1 = chunk1[0]['U'].reshape((73,101,73),order='F')
V1 = chunk2[0]['V'].reshape((73,101,73),order='F')
W1 = chunk3[0]['W'].reshape((73,101,73),order='F')
#U2 = (U1[5:69,3:99,5:69].copy()-xmean)/std
#V2 = (V1[5:69,3:99,5:69].copy()-xmean)/std
#W2 = (W1[5:69,3:99,5:69].copy()-xmean)/std
#Ui = np.concatenate((U2[xpl-1,:,:][:,:,None],V2[xpl-1,:,:][:,:,None],W2[xpl-1,:,:][:,:,None]),axis=2)
Ui = np.concatenate((U1[xpl+4,3:99,5:69][:,:,None].copy(),V1[xpl+4,3:99,5:69][:,:,None].copy(),W1[xpl+4,3:99,5:69][:,:,None].copy()),axis=2)

Ui = (Ui-xmean)/xstd

with open('./'+file_name+'/CHAN'+fn+'DCXX.DAT') as fd:
#with open('CHAN384DCXX.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcxx, count=1)
CXX1 = chunk[0]['CXX'].reshape((69,98,69),order='F')
#CXX2 = CXX1[3:67,1:97,3:67].copy()
CXXdns = CXX1[xpl+2,1:97,3:67][:,:,None].copy().reshape(96,64)
CXXdins = CXXdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCXY.DAT') as fd:
#with open('CHAN384DCXY.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcxy, count=1)
CXY1 = chunk[0]['CXY'].reshape((69,98,69),order='F')
CXYdns = CXY1[xpl+2,1:97,3:67][:,:,None].copy().reshape(96,64)
CXYdins = CXYdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCXZ.DAT') as fd:
#with open('CHAN384DCXZ.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcxz, count=1)
CXZ1 = chunk[0]['CXZ'].reshape((69,98,69),order='F')
CXZdns = CXZ1[xpl+2,1:97,3:67][:,:,None].copy().reshape(96,64)
CXZdins = CXZdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCYY.DAT') as fd:
#with open('CHAN384DCYY.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcyy, count=1)
CYY1 = chunk[0]['CYY'].reshape((69,98,69),order='F')
CYYdns = CYY1[xpl+2,1:97,3:67][:,:,None].copy().reshape(96,64)
CYYdins = CYYdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCYZ.DAT') as fd:
#with open('CHAN384DCYZ.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcyz, count=1)
CYZ1 = chunk[0]['CYZ'].reshape((69,98,69),order='F')
CYZdns = CYZ1[xpl+2,1:97,3:67][:,:,None].copy().reshape(96,64)
CYZdins = CYZdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCZZ.DAT') as fd:
#with open('CHAN384DCZZ.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtczz, count=1)
CZZ1 = chunk[0]['CZZ'].reshape((69,98,69),order='F')
CZZdns = CZZ1[xpl+2,1:97,3:67][:,:,None].copy().reshape(96,64)
CZZdins = CZZdns.reshape(-1,1)

CXX_model = load_model('./{0}cij/{0}cxx/{0}cxx_model.h5'.format(case_name),custom_objects={'RRSE':RRSE,'R2':R2,'RMSPE':RMSPE})
CXY_model = load_model('./{0}cij/{0}cxy/{0}cxy_model.h5'.format(case_name),custom_objects={'RRSE':RRSE,'R2':R2,'RMSPE':RMSPE})
CXZ_model = load_model('./{0}cij/{0}cxz/{0}cxz_model.h5'.format(case_name),custom_objects={'RRSE':RRSE,'R2':R2,'RMSPE':RMSPE})
CYY_model = load_model('./{0}cij/{0}cyy/{0}cyy_model.h5'.format(case_name),custom_objects={'RRSE':RRSE,'R2':R2,'RMSPE':RMSPE})
CYZ_model = load_model('./{0}cij/{0}cyz/{0}cyz_model.h5'.format(case_name),custom_objects={'RRSE':RRSE,'R2':R2,'RMSPE':RMSPE})
CZZ_model = load_model('./{0}cij/{0}czz/{0}czz_model.h5'.format(case_name),custom_objects={'RRSE':RRSE,'R2':R2,'RMSPE':RMSPE})

Ui=Ui[None,:]

CXX_infer=CXX_model.predict(Ui)
CXY_infer=CXY_model.predict(Ui)
CXZ_infer=CXZ_model.predict(Ui)
CYY_infer=CYY_model.predict(Ui)
CYZ_infer=CYZ_model.predict(Ui)
CZZ_infer=CZZ_model.predict(Ui)

CXX_infer=CXX_infer*CXXstd+CXXmean
CXY_infer=CXY_infer*CXYstd+CXYmean
CXZ_infer=CXZ_infer*CXZstd+CXZmean
CYY_infer=CYY_infer*CYYstd+CYYmean
CYZ_infer=CYZ_infer*CYZstd+CYZmean
CZZ_infer=CZZ_infer*CZZstd+CZZmean

CXXann=CXX_infer.reshape((96,64))
CXXanndins=CXXann.reshape(-1,1)
CXYann=CXY_infer.reshape((96,64))
CXYanndins=CXYann.reshape(-1,1)
CXZann=CXZ_infer.reshape((96,64))
CXZanndins=CXZann.reshape(-1,1)
CYYann=CYY_infer.reshape((96,64))
CYYanndins=CYYann.reshape(-1,1)
CYZann=CYZ_infer.reshape((96,64))
CYZanndins=CYZann.reshape(-1,1)
CZZann=CZZ_infer.reshape((96,64))
CZZanndins=CZZann.reshape(-1,1)

dz=3.2/64
yy = np.array([[np.tanh((y/96.0-0.5)*np.log(49.0))/0.96+1 for _ in range(64)] for y in range(1,97)]).reshape(-1,1)
zz = np.array([[z*dz for z in range(1,65)] for _ in range(96)]).reshape(-1,1)

dins=np.concatenate([yy,zz,CXXdins,CXYdins,CXZdins,CYYdins,CYZdins,CZZdins,CXXanndins,CXYanndins,CXZanndins,CYYanndins,CYZanndins,CZZanndins],axis=1)

np.savetxt('./{0}cij/{1}/{0}cijdinsx_infer.dat'.format(case_name,file_name),dins,delimiter='  ')

Y, Z = np.meshgrid(np.array([np.tanh((y/96.0-0.5)*np.log(49.0))/0.96+1 for y in range(1,97)]),np.arange(dz,3.2+dz,dz))

#plt.pcolormesh(X, Z, CXXann, cmap='RdBu_r')
view = [[CXXdns.T,CXXann.T],[CXYdns.T,CXYann.T],[CXZdns.T,CXZann.T],[CYYdns.T,CYYann.T],[CYZdns.T,CYZann.T],[CZZdns.T,CZZann.T]]
view_name = [['CXXdns','CXXann'],['CXYdns','CXYann'],['CXZdns','CXZann'],['CYYdns','CYYann'],['CYZdns','CYZann'],['CZZdns','CZZann']]
contour=[np.arange(0,303,3),np.arange(-15,15.15,0.15),np.arange(-30,30.6,0.6),np.arange(0,15.15,0.15),np.arange(-8,8.16,0.16),np.arange(0,25.25,0.25)]
for ij in range(6):
    for i in range(2):
        plt.contourf(Y, Z, view[ij][i], contour[ij], extend='both', cmap='jet')
        plt.axes().set_aspect('equal')
        plt.colorbar (orientation='vertical')
        plt.xlabel('Y', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.savefig('./{0}cij/{3}/png/unet{1}x{2}.png'.format(case_name,view_name[ij][i],xpl,file_name), bbox_inches='tight')
        plt.close()

cxx_cc = np.array([])
cxy_cc = np.array([])
cxz_cc = np.array([])
cyy_cc = np.array([])
cyz_cc = np.array([])
czz_cc = np.array([])

cxx_nn_m = np.array([])
cxy_nn_m = np.array([])
cxz_nn_m = np.array([])
cyy_nn_m = np.array([])
cyz_nn_m = np.array([])
czz_nn_m = np.array([])

cxx_dns_m = np.array([])
cxy_dns_m = np.array([])
cxz_dns_m = np.array([])
cyy_dns_m = np.array([])
cyz_dns_m = np.array([])
czz_dns_m = np.array([])

#Uall = []

CXX_infer_all = np.empty((0,96,64))
CXY_infer_all = np.empty((0,96,64))
CXZ_infer_all = np.empty((0,96,64))
CYY_infer_all = np.empty((0,96,64))
CYZ_infer_all = np.empty((0,96,64))
CZZ_infer_all = np.empty((0,96,64))

for x in range(1,65):
    Ux = np.concatenate([U1[x+4,3:99,5:69][:,:,None].copy(),V1[x+4,3:99,5:69][:,:,None].copy(),W1[x+4,3:99,5:69][:,:,None].copy()],axis=2)
    Ux = Ux[None,:]
    CXX_inferx = CXX_model.predict(Ux)
    CXY_inferx = CXY_model.predict(Ux)
    CXZ_inferx = CXZ_model.predict(Ux)
    CYY_inferx = CYY_model.predict(Ux)
    CYZ_inferx = CYZ_model.predict(Ux)
    CZZ_inferx = CZZ_model.predict(Ux)
#    print np.shape(CXX_inferx[:,:,:,0])
    CXX_infer_all = np.concatenate([CXX_infer_all,CXX_inferx.reshape(-1,96,64)],axis=0)
    CXY_infer_all = np.concatenate([CXY_infer_all,CXY_inferx.reshape(-1,96,64)],axis=0)
    CXZ_infer_all = np.concatenate([CXZ_infer_all,CXZ_inferx.reshape(-1,96,64)],axis=0)
    CYY_infer_all = np.concatenate([CYY_infer_all,CYY_inferx.reshape(-1,96,64)],axis=0)
    CYZ_infer_all = np.concatenate([CYZ_infer_all,CYZ_inferx.reshape(-1,96,64)],axis=0)
    CZZ_infer_all = np.concatenate([CZZ_infer_all,CZZ_inferx.reshape(-1,96,64)],axis=0)
#    Uall.append(Ux)
#Uin = np.asarray(Uall)

CXXall = CXX1[3:67,1:97,3:67].copy()
CXYall = CXY1[3:67,1:97,3:67].copy()
CXZall = CXZ1[3:67,1:97,3:67].copy()
CYYall = CYY1[3:67,1:97,3:67].copy()
CYZall = CYZ1[3:67,1:97,3:67].copy()
CZZall = CZZ1[3:67,1:97,3:67].copy()

#CXX_infer_all = (CXX_model.predict(Uin)*CXXstd+CXXmean).reshape(64,96,64)
#CXY_infer_all = (CXY_model.predict(Uin)*CXYstd+CXYmean).reshape(64,96,64)
#CXZ_infer_all = (CXZ_model.predict(Uin)*CXZstd+CXZmean).reshape(64,96,64)
#CYY_infer_all = (CYY_model.predict(Uin)*CYYstd+CYYmean).reshape(64,96,64)
#CYZ_infer_all = (CYZ_model.predict(Uin)*CYZstd+CYZmean).reshape(64,96,64)
#CZZ_infer_all = (CZZ_model.predict(Uin)*CZZstd+CZZmean).reshape(64,96,64)

for y in range(96):
    CXX_xz = CXXall[:,y,:].reshape(-1).copy()
    CXY_xz = CXYall[:,y,:].reshape(-1).copy()
    CXZ_xz = CXZall[:,y,:].reshape(-1).copy()
    CYY_xz = CYYall[:,y,:].reshape(-1).copy()
    CYZ_xz = CYZall[:,y,:].reshape(-1).copy()
    CZZ_xz = CZZall[:,y,:].reshape(-1).copy()
    CXX_xz_infer = CXX_infer_all[:,y,:].reshape(-1).copy()
    CXY_xz_infer = CXY_infer_all[:,y,:].reshape(-1).copy()
    CXZ_xz_infer = CXZ_infer_all[:,y,:].reshape(-1).copy()
    CYY_xz_infer = CYY_infer_all[:,y,:].reshape(-1).copy()
    CYZ_xz_infer = CYZ_infer_all[:,y,:].reshape(-1).copy()
    CZZ_xz_infer = CZZ_infer_all[:,y,:].reshape(-1).copy()
    cxx1_nn_m = np.mean(CXX_xz_infer).reshape(-1).copy()
    cxx1_dns_m = np.mean(CXX_xz).reshape(-1).copy()
    cxy1_nn_m = np.mean(CXY_xz_infer).reshape(-1).copy()
    cxy1_dns_m = np.mean(CXY_xz).reshape(-1).copy()
    cxz1_nn_m = np.mean(CXZ_xz_infer).reshape(-1).copy()
    cxz1_dns_m = np.mean(CXZ_xz).reshape(-1).copy()
    cyy1_nn_m = np.mean(CYY_xz_infer).reshape(-1).copy()
    cyy1_dns_m = np.mean(CYY_xz).reshape(-1).copy()
    cyz1_nn_m = np.mean(CYZ_xz_infer).reshape(-1).copy()
    cyz1_dns_m = np.mean(CYZ_xz).reshape(-1).copy()
    czz1_nn_m = np.mean(CZZ_xz_infer).reshape(-1).copy()
    czz1_dns_m = np.mean(CZZ_xz).reshape(-1).copy()
    cxx1_cc = np.corrcoef(CXX_xz_infer,CXX_xz)[0,1].reshape(-1).copy()
    cxy1_cc = np.corrcoef(CXY_xz_infer,CXY_xz)[0,1].reshape(-1).copy()
    cxz1_cc = np.corrcoef(CXZ_xz_infer,CXZ_xz)[0,1].reshape(-1).copy()
    cyy1_cc = np.corrcoef(CYY_xz_infer,CYY_xz)[0,1].reshape(-1).copy()
    cyz1_cc = np.corrcoef(CYZ_xz_infer,CYZ_xz)[0,1].reshape(-1).copy()
    czz1_cc = np.corrcoef(CZZ_xz_infer,CZZ_xz)[0,1].reshape(-1).copy()
    cxx_nn_m = np.concatenate([cxx_nn_m,cxx1_nn_m],axis=0)
    cxy_nn_m = np.concatenate([cxy_nn_m,cxy1_nn_m],axis=0)
    cxz_nn_m = np.concatenate([cxz_nn_m,cxz1_nn_m],axis=0)
    cyy_nn_m = np.concatenate([cyy_nn_m,cyy1_nn_m],axis=0)
    cyz_nn_m = np.concatenate([cyz_nn_m,cyz1_nn_m],axis=0)
    czz_nn_m = np.concatenate([czz_nn_m,czz1_nn_m],axis=0)
    cxx_dns_m = np.concatenate([cxx_dns_m,cxx1_dns_m],axis=0)
    cxy_dns_m = np.concatenate([cxy_dns_m,cxy1_dns_m],axis=0)
    cxz_dns_m = np.concatenate([cxz_dns_m,cxz1_dns_m],axis=0)
    cyy_dns_m = np.concatenate([cyy_dns_m,cyy1_dns_m],axis=0)
    cyz_dns_m = np.concatenate([cyz_dns_m,cyz1_dns_m],axis=0)
    czz_dns_m = np.concatenate([czz_dns_m,czz1_dns_m],axis=0)
    cxx_cc = np.concatenate([cxx_cc,cxx1_cc],axis=0)
    cxy_cc = np.concatenate([cxy_cc,cxy1_cc],axis=0)
    cxz_cc = np.concatenate([cxz_cc,cxz1_cc],axis=0)
    cyy_cc = np.concatenate([cyy_cc,cyy1_cc],axis=0)
    cyz_cc = np.concatenate([cyz_cc,cyz1_cc],axis=0)
    czz_cc = np.concatenate([czz_cc,czz1_cc],axis=0)

R2 = np.array([np.tanh((y/96.0-0.5)*np.log(49.0))/0.96+1 for y in range(1,97)]).reshape(-1,1)

mean=np.c_[R2,cxx_nn_m,cxx_dns_m,cxy_nn_m,cxy_dns_m,cxz_nn_m,cxz_dns_m,cyy_nn_m,cyy_dns_m,cyz_nn_m,cyz_dns_m,czz_nn_m,czz_dns_m]
cc=np.c_[R2,cxx_cc,cxy_cc,cxz_cc,cyy_cc,cyz_cc,czz_cc]

np.savetxt('{0}cij/{1}/{0}cijmean_infer.dat'.format(case_name,file_name),mean,delimiter='  ')
np.savetxt('{0}cij/{1}/{0}cijcc_infer.dat'.format(case_name,file_name),cc,delimiter='  ')

np.savetxt('{0}cij/{1}/{0}cxxinfer_compare.csv'.format(case_name,file_name),np.concatenate([CXXall.reshape(-1,1),CXX_infer_all.reshape(-1,1)],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cxyinfer_compare.csv'.format(case_name,file_name),np.concatenate([CXYall.reshape(-1,1),CXY_infer_all.reshape(-1,1)],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cxzinfer_compare.csv'.format(case_name,file_name),np.concatenate([CXZall.reshape(-1,1),CXZ_infer_all.reshape(-1,1)],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cyyinfer_compare.csv'.format(case_name,file_name),np.concatenate([CYYall.reshape(-1,1),CYY_infer_all.reshape(-1,1)],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cyzinfer_compare.csv'.format(case_name,file_name),np.concatenate([CYZall.reshape(-1,1),CYZ_infer_all.reshape(-1,1)],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}czzinfer_compare.csv'.format(case_name,file_name),np.concatenate([CZZall.reshape(-1,1),CZZ_infer_all.reshape(-1,1)],axis=1),delimiter=',')


end_time = time.time()
elapsed_time=end_time-start_time
print('elapsed_time:{0}s'.format(elapsed_time))

