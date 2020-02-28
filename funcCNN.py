import numpy as np
import scipy.io as scio  
import tensorflow as tf

def Con2Numpy(var_name):
    dataFile = var_name 
    data = scio.loadmat(dataFile)  
    x=data[var_name]
    x1=x.astype(float)
    return x1

def Con2Numpy_path(var_name, path):
    dataFile = var_name 
    data = scio.loadmat(path+dataFile)  
    x=data[var_name]
    x1=x.astype(float)
    return x1


def load_HSI_data_list(path1):
    
    inst = np.array(Con2Numpy_path('inst', path1), dtype = 'float32') 
    temask= np.array(Con2Numpy_path('temask', path1), dtype=bool)
    trmask=np.array(Con2Numpy_path('trmask', path1),  dtype=bool)
    y_train=Con2Numpy_path('y_train', path1)
    y_test=Con2Numpy_path('y_test', path1)
    return  inst,  temask, trmask, y_train, y_test

def load_HSI_data():
    features_pretrain = np.array(Con2Numpy('features_pretrain'), dtype = 'float32')
    support_pretrain = np.array(Con2Numpy('support_pretrain'), dtype = 'float32')
    
    trmask_pretrain = Con2Numpy('trmask_pretrain')
    trmask_pretrain = np.array(trmask_pretrain, dtype=bool)
    
    y_train_pretrain = Con2Numpy('y_train_pretrain')
    return  features_pretrain, support_pretrain, trmask_pretrain, y_train_pretrain

def AscSort(x1):
    x = x1.copy()
    B = np.sort(x)
    IX = np.ones(np.size(x))
    for i in range(np.size(x)):
        idx = np.argmin(x)
        x[idx] = np.max(x)+1
        IX[i] = idx
    return B, IX
        
    
    
def GetKnn1(inx, data, k):
    [datarow, datacol] = np.shape(data)
    diffMat = np.tile(inx, (datarow, 1)) - data
    distanceMat = np.sqrt(np.sum(diffMat*diffMat, 1))
    [B, IX] = AscSort(distanceMat)
    if B[0] == 0:
        IX = IX[1:k+1].copy()
    else:
        IX = IX[0:k].copy()
    return IX

def lle(X, K, ln):
    [D, N] = np.shape(X)
    index = np.zeros([K, N])
    for i in range(N):
        index[:, i] = GetKnn1(X[:, i].T, X[:, 0:ln].T, K)
    neighborhood = index[0:K, :]
    if K>D:
        tol = 0.001
    else:
        tol = 0
    W = np.zeros([K, N])
    W1 = np.zeros([N, N])
    for ii in range(N):
        z = X[:, np.array(neighborhood[:, ii], dtype=int)]-np.tile(X[:, ii], (K, 1)).T
        C = np.dot(z.T, z)
        C = C + np.dot(np.dot(np.eye(K), tol), np.trace(C))
        W[:, ii] = np.dot(np.linalg.inv(C), np.ones([K,1])).reshape((K))
        W[:, ii] = W[:, ii]/np.sum(W[:, ii])
    for i in range(N):
        for j in range(K):
            W1[np.array(index[j, i], dtype=int), i] = W[j, i]
    return W1

def GetWlle(S1):
    S = S1.copy()
    S = S.T
    Wlle = S+S.T-np.dot(S.T, S)
    Wlle = Wlle-np.diag(np.diag(Wlle))
    return Wlle
def GetLabeledData(img2d, trte_idx):
    intrte=np.zeros([np.shape(trte_idx)[0],np.shape(img2d)[1]])
    all_num=np.shape(trte_idx)[0]
    for i in range(all_num):
        print(i)
        intrte[i,:] = img2d[trte_idx[i,0], :]
    intrte=np.reshape(intrte, (all_num ,40))
    return intrte
def GetMats(trte_idx,y_test,y_train,trnum):
    nums=np.shape(trte_idx)[0]
    trmask=np.zeros([nums])
    temask=np.zeros([nums])
    ytr=np.zeros([nums,np.shape(y_test)[1]])
    yte=np.zeros([nums,np.shape(y_test)[1]])
    for i in range(trnum):
        trmask[i]=1
        ytr[i]=y_train[trte_idx[i],:]
    for i in range(nums-trnum):
        temask[i+trnum]=1
        yte[i+trnum]=y_test[trte_idx[i+trnum],:]
    return np.array(trmask,dtype=bool),np.array(temask,dtype=bool),ytr,yte

def Gettrtemask(dim0,trnum,trte):
    if trte==0:
        trte_mask1=np.ones([trnum])
        trte_mask2=np.zeros([dim0-trnum])
        return np.concatenate([trte_mask1,trte_mask2],axis=0)
    elif trte==1:
        trte_mask1=np.zeros([trnum])
        trte_mask2=np.ones([dim0-trnum])
        return np.concatenate([trte_mask1,trte_mask2],axis=0)

def mapminmax01(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(X) 

def CalSupport_tf(A, lam):
    lam1 = lam
    A_ = A+lam1*np.eye(np.shape(A)[0])
    D_ = tf.reduce_sum(A_,reduction_indices=0)
    D_05 = tf.diag(D_**(-0.5))
    support = tf.matmul(tf.matmul(D_05, A_), D_05)
    return support
def CalSupport(A, lam):
    lam1 = lam
    A_ = A+lam1*np.eye(np.shape(A)[0])
    D_ = np.sum(A_, 1)
    D_05 = np.diag(D_**(-0.5))
    support = np.matmul(np.matmul(D_05, A_), D_05)
    return support
def arr2sparse(arr):
    arr_tensor = tf.constant(np.array(arr),dtype='float32')
    arr_idx = tf.where(tf.not_equal(arr_tensor, 0))
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), arr_tensor.get_shape())
    return arr_sparse
def AssignLabels(useful_sp_lab, trlabels, telabels, trmask, temask): 
    [rows, cols] = np.shape(useful_sp_lab)
    output_labels = np.zeros([rows, cols])
    sp_num = np.max(useful_sp_lab)
    for sp_idx in range(1, sp_num+1):
        pos1 = np.argwhere(useful_sp_lab==sp_idx)
        if trmask[sp_idx-1, 0] == True:
            pred_label = trlabels[sp_idx-1]
        else:
            pred_label = telabels[sp_idx-1]
        output_labels[pos1[:,0], pos1[:,1]] = pred_label+1
    return output_labels
def PixelWiseAccuracy(gt, pred_labels, trpos):
    num_labels = np.max(gt)
    gt[trpos[:,0]-1, trpos[:,1]-1] = 0
    err_num = 0
    for label_idx in range(1, num_labels+1):
        pos1 = np.argwhere(gt == label_idx)
        mat1 = gt[pos1[:,0], pos1[:,1]]
        mat2 = pred_labels[pos1[:,0], pos1[:,1]]
        mat3 = mat1-mat2
        err_num += np.shape(np.argwhere(mat3!=0))[0]
    return 1-err_num/np.shape(np.argwhere(gt>0))[0]
def GetExcelData(gt, pred_labels, trpos):
    gt[trpos[:,0]-1, trpos[:,1]-1] = 0
    num_classes = np.max(gt)
    per_acc = []
    overall_err_num = 0
    for lab_idx in range(1, num_classes+1):
        pos1 = np.argwhere(gt==lab_idx)
        preds = pred_labels[pos1[:,0], pos1[:,1]]
        gts = gt[pos1[:,0], pos1[:,1]]
        mat3 = gts-preds
        per_err_num = np.shape(np.argwhere(mat3!=0))[0]
        per_acc.append(1-per_err_num/np.shape(pos1)[0])
        overall_err_num += np.shape(np.argwhere(mat3!=0))[0]
    per_acc = np.array(per_acc, dtype='float32')
    OA = 1-overall_err_num/np.shape(np.argwhere(gt>0))[0]
    AA = np.mean(per_acc)
    # kappa
    n = np.shape(np.argwhere(gt!=0))[0] 
    ab1 = 0
    pos0 = np.argwhere(gt==0)
    pred_labels[pos0[:,0], pos0[:,1]] = 0
    for lab_idx in range(1, num_classes+1):
        a1 = np.shape(np.argwhere(gt==lab_idx))[0]
        b1 = np.shape(np.argwhere(pred_labels==lab_idx))[0]
        ab1 += a1*b1
    Pe = ab1/(n*n)
    kappa_coef = (OA-Pe)/(1-Pe)
    outputs = np.zeros([num_classes+3])
    outputs[0:num_classes] = per_acc
    outputs[num_classes] = OA
    outputs[num_classes+1] = AA
    outputs[num_classes+2] = kappa_coef
    return outputs
        
        
    
    