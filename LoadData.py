import numpy as np
import scipy.io as scio  


def Con2Numpy(var_name):
    path = './/data//'
    dataFile = path + var_name 
    data = scio.loadmat(dataFile)  
    x = data[var_name]
    x1 = x.astype(float)
    return x1

def load_HSI_data( data_name ):
    Data = dict()
    img_gyh = data_name+''
    img_gt = data_name+'_gt'
    Data['useful_sp_lab'] = np.array(Con2Numpy('useful_sp_lab'), dtype='int')
    Data[img_gt] = np.array(Con2Numpy(img_gt), dtype='int')
    Data[img_gyh] = Con2Numpy(img_gyh)
    Data['trpos'] = np.array(Con2Numpy('trpos'), dtype='int')
    return Data
