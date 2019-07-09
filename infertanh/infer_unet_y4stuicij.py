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
print '_____Input position of x-z plane(4 or 16 or 48)!!!!!_____'
ypl = input()
#position of x-z plane
#4 or 16 or 48
case_name = 'unet_y{0}stui'.format(ypl)
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
Ui = np.concatenate((U1[5:69,ypl+2,5:69][:,:,None],V1[5:69,ypl+2,5:69][:,:,None],W1[5:69,ypl+2,5:69][:,:,None]),axis=2)

Ui = (Ui-xmean)/xstd

with open('./'+file_name+'/CHAN'+fn+'DCXX.DAT') as fd:
#with open('CHAN384DCXX.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcxx, count=1)
CXX1 = chunk[0]['CXX'].reshape((69,98,69),order='F')
CXXdns = CXX1[3:67,ypl,3:67][:,:,None].reshape(64,64)
#CXXdins = CXX1[3:67,4,3:67][:,:,None].reshape(-1,1)
CXXdins = CXXdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCXY.DAT') as fd:
#with open('CHAN384DCXY.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcxy, count=1)
CXY1 = chunk[0]['CXY'].reshape((69,98,69),order='F')
CXYdns = CXY1[3:67,ypl,3:67][:,:,None].reshape(64,64)
CXYdins = CXYdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCXZ.DAT') as fd:
#with open('CHAN384DCXZ.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcxz, count=1)
CXZ1 = chunk[0]['CXZ'].reshape((69,98,69),order='F')
CXZdns = CXZ1[3:67,ypl,3:67][:,:,None].reshape(64,64)
CXZdins = CXZdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCYY.DAT') as fd:
#with open('CHAN384DCYY.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcyy, count=1)
CYY1 = chunk[0]['CYY'].reshape((69,98,69),order='F')
CYYdns = CYY1[3:67,ypl,3:67][:,:,None].reshape(64,64)
CYYdins = CYYdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCYZ.DAT') as fd:
#with open('CHAN384DCYZ.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtcyz, count=1)
CYZ1 = chunk[0]['CYZ'].reshape((69,98,69),order='F')
CYZdns = CYZ1[3:67,ypl,3:67][:,:,None].reshape(64,64)
CYZdins = CYZdns.reshape(-1,1)

with open('./'+file_name+'/CHAN'+fn+'DCZZ.DAT') as fd:
#with open('CHAN384DCZZ.DAT') as fd:
    chunk = np.fromfile(fd, dtype=dtczz, count=1)
CZZ1 = chunk[0]['CZZ'].reshape((69,98,69),order='F')
CZZdns = CZZ1[3:67,ypl,3:67][:,:,None].reshape(64,64)
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

CXXann=CXX_infer.reshape((64,64))
CXXanndins=CXXann.reshape(-1,1)
CXYann=CXY_infer.reshape((64,64))
CXYanndins=CXYann.reshape(-1,1)
CXZann=CXZ_infer.reshape((64,64))
CXZanndins=CXZann.reshape(-1,1)
CYYann=CYY_infer.reshape((64,64))
CYYanndins=CYYann.reshape(-1,1)
CYZann=CYZ_infer.reshape((64,64))
CYZanndins=CYZann.reshape(-1,1)
CZZann=CZZ_infer.reshape((64,64))
CZZanndins=CZZann.reshape(-1,1)

CXX_cc=np.corrcoef(CXXdns.reshape(-1),CXXann.reshape(-1))[0,1]
CXY_cc=np.corrcoef(CXYdns.reshape(-1),CXYann.reshape(-1))[0,1]
CXZ_cc=np.corrcoef(CXZdns.reshape(-1),CXZann.reshape(-1))[0,1]
CYY_cc=np.corrcoef(CYYdns.reshape(-1),CYYann.reshape(-1))[0,1]
CYZ_cc=np.corrcoef(CYZdns.reshape(-1),CYZann.reshape(-1))[0,1]
CZZ_cc=np.corrcoef(CZZdns.reshape(-1),CZZann.reshape(-1))[0,1]
cc=np.array([CXX_cc,CXY_cc,CXZ_cc,CYY_cc,CYZ_cc,CZZ_cc])
np.savetxt('./{0}cij/{1}/{0}cijcc_infer.dat'.format(case_name,file_name), np.concatenate([np.repeat([np.tanh((ypl/96.0-0.5)*np.log(49.0))/0.96+1],6).reshape(-1,1),cc.reshape(-1,1)],axis=1), delimiter='  ')

CXXann_m=np.mean(CXXann.reshape(-1))
CXYann_m=np.mean(CXYann.reshape(-1))
CXZann_m=np.mean(CXZann.reshape(-1))
CYYann_m=np.mean(CYYann.reshape(-1))
CYZann_m=np.mean(CYZann.reshape(-1))
CZZann_m=np.mean(CZZann.reshape(-1))
m=np.array([CXXann_m,CXYann_m,CXZann_m,CYYann_m,CYZann_m,CZZann_m])
np.savetxt('./{0}cij/{1}/{0}cijmean_infer.dat'.format(case_name,file_name), np.concatenate([np.repeat([np.tanh((ypl/96.0-0.5)*np.log(49.0))/0.96+1],6).reshape(-1,1),m.reshape(-1,1)],axis=1), delimiter='  ')

dx=6.4/64
dz=3.2/64
xx = np.array([[x*dx for _ in range(64)] for x in range(1,65)]).reshape(-1,1)
#yy = np.array([[np.tanh((y/96.0-0.5)*np.log(49.0))/1.92+0.5 for y in range(1,97)] for _ in range(64)]).reshape(-1,1)
zz = np.array([[z*dz for z in range(1,65)] for _ in range(64)]).reshape(-1,1)

dins=np.concatenate([xx,zz,CXXdins,CXYdins,CXZdins,CYYdins,CYZdins,CZZdins,CXXanndins,CXYanndins,CXZanndins,CYYanndins,CYZanndins,CZZanndins],axis=1)

np.savetxt('./{0}cij/{1}/{0}cijdinsy_infer.dat'.format(case_name,file_name),dins,delimiter='  ')

X, Z = np.meshgrid(np.arange(dx,6.4+dx,dx),np.arange(dz,3.2+dz,dz))

#plt.pcolormesh(X, Z, CXXann, cmap='RdBu_r')
view = [[CXXdns.T,CXXann.T],[CXYdns.T,CXYann.T],[CXZdns.T,CXZann.T],[CYYdns.T,CYYann.T],[CYZdns.T,CYZann.T],[CZZdns.T,CZZann.T]]
view_name = [['CXXdns','CXXann'],['CXYdns','CXYann'],['CXZdns','CXZann'],['CYYdns','CYYann'],['CYZdns','CYZann'],['CZZdns','CZZann']]
contour=[[np.arange(0,303,3),np.arange(0,15.15,0.15),np.arange(-60,61.2,1.2),np.arange(0,2.02,0.02),np.arange(-8,8.16,0.16),np.arange(0,101,1)],[np.arange(0,151.5,1.5),np.arange(-20,20.4,0.4),np.arange(-25,25.5,0.5),np.arange(0,20.2,0.2),np.arange(-10,10.2,0.2),np.arange(0,40.4,0.4)],[np.arange(0,3.03,0.03),np.arange(-1,1.02,0.02),np.arange(-1,1.02,0.02),np.arange(0,3.03,0.03),np.arange(-1,1.02,0.02),np.arange(-1,1.02,0.02)]]

if ypl<=4:
    y_index=0
elif ypl<=26:
    y_index=1
else:
    y_index=2

for ij in range(6):
    for i in range(2):
        plt.contourf(X, Z, view[ij][i], contour[y_index][ij], extend='both', cmap='jet')
        plt.axes().set_aspect('equal')
        plt.colorbar (orientation='vertical')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Z', fontsize=12)
        plt.savefig('./{0}cij/{3}/png/unet{1}y{2}.png'.format(case_name,view_name[ij][i],ypl,file_name), bbox_inches='tight')
        plt.close()

np.savetxt('{0}cij/{1}/{0}cxxinfer_compare.csv'.format(case_name,file_name),np.concatenate([CXXdins,CXXanndins],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cxyinfer_compare.csv'.format(case_name,file_name),np.concatenate([CXYdins,CXYanndins],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cxzinfer_compare.csv'.format(case_name,file_name),np.concatenate([CXZdins,CXZanndins],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cyyinfer_compare.csv'.format(case_name,file_name),np.concatenate([CYYdins,CYYanndins],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}cyzinfer_compare.csv'.format(case_name,file_name),np.concatenate([CYZdins,CYZanndins],axis=1),delimiter=',')
np.savetxt('{0}cij/{1}/{0}czzinfer_compare.csv'.format(case_name,file_name),np.concatenate([CZZdins,CZZanndins],axis=1),delimiter=',')


end_time = time.time()
elapsed_time=end_time-start_time
print('elapsed_time:{0}s'.format(elapsed_time))

