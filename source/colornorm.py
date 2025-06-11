import os
import cv2
import sys
import numpy as np

from sklearn.decomposition import DictionaryLearning, sparse_encode

nstains: int = 2
lam: float = 0.02

param: dict = {
    'K': 2,
    'lambda1': 0.02,
    'numThreads': 4,
    'mode': 2,
    'iter': 200,
    'posAlpha': True,
    'posD': True,
    'batchsize': 400,
    'clean': True,
}

def blockPrint() -> None:
    sys.stdout = open(os.devnull, 'w')

def enablePrint() -> None:
    sys.stdout = sys.__stdout__

def stainsep(I: np.ndarray, nstains: int, lam: float):
    global param
    if I.ndim != 3:
        print('[stainsep] Input must be 3-D')
        return [], [], []
    rows, cols = I.shape[:2]
    V, V1 = BLtrans(I)
    param['batchsize'] = round(0.2 * V1.shape[0])
    Wi = get_staincolor_sparsenmf(V1)
    Hi = estH(V, Wi, rows, cols)
    return Wi, Hi

def get_staincolor_sparsenmf(v: np.ndarray) -> np.ndarray:
    blockPrint()
    model = DictionaryLearning(
        n_components=param['K'],
        alpha=param['lambda1'],
        max_iter=param['iter'],
        fit_algorithm='lars',
        transform_algorithm='lasso_lars',
        positive_code=True,
        positive_dict=True,
        random_state=0
    )
    D = model.fit(v).components_
    enablePrint()
    a_arg = np.argsort(D[:, 1])
    return D[a_arg]

def BLtrans(I: np.ndarray):
    Ivecd = np.reshape(I, [I.shape[0] * I.shape[1], I.shape[2]])
    V = np.float64(np.log(255) - np.log(Ivecd + 1))
    img_lab = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
    luminlayer = np.reshape(np.array(img_lab[:, :, 0], np.float64), [I.shape[0] * I.shape[1]])
    Inew = Ivecd[(luminlayer / 255) < 0.9]
    VforW = np.log(255) - np.log(Inew + 1)
    return V, np.float64(VforW)

def estH(v: np.ndarray, Ws: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    Hs_vec = sparse_encode(
        X=v,
        dictionary=Ws,
        algorithm='lasso_lars',
        alpha=param['lambda1'],
        positive=True
    )
    Hs = np.reshape(Hs_vec, [nrows, ncols, param['K']])
    return Hs

def SCN(source: np.ndarray, Hta: np.ndarray, Wta: np.ndarray, Hso: np.ndarray) -> np.ndarray:
    Hso = np.reshape(Hso, [Hso.shape[0] * Hso.shape[1], Hso.shape[2]])
    Hso_Rmax = np.percentile(Hso, 99, axis=0)
    Hta = np.reshape(Hta, [Hta.shape[0] * Hta.shape[1], Hta.shape[2]])
    Hta_Rmax = np.percentile(Hta, 99, axis=0)
    normfac = Hta_Rmax / Hso_Rmax
    Hsonorm = Hso * normfac
    Ihat = np.dot(Wta, np.transpose(Hsonorm))
    sourcenorm = np.uint8(255 * np.exp(-np.reshape(np.transpose(Ihat), source.shape)))
    return sourcenorm

def CN(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    _, His = stainsep(source, nstains, lam)
    Wi, Hi = stainsep(target, nstains, lam)
    out = SCN(source, Hi, Wi, His)
    return out