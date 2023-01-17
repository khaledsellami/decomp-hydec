import numpy as np


def generate_callsim_matrix(X):
    # incoming calls vector
    calls_inc = X.sum(axis=0)
    # change to inf so that we get 0 if we divide by this vector
    calls_inc[calls_inc == 0] = np.inf
    # generate a matrix where calls_inc_div[i,j]=0 if incoming calls to i and j are 0, =1 if i or j has 0 incoming calls
    # and =2 if both of them have at least 1 incoming call
    calls_inc_div = (calls_inc != np.inf).astype(int).reshape((1, -1)) + (calls_inc.transpose() != np.inf).astype(
        int).reshape((-1, 1))
    # ignore the warning given by the next line. The division by 0 is expected and the resulting nan values are changed to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.nan_to_num((((X / calls_inc) + (X / calls_inc).transpose()) / calls_inc_div))
    return sim


def generate_cousage_matrix(X):
    X = (X.T != 0)
    sim = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            inter = (X[i, :] * X[j, :]).sum()
            union = (X[i, :] + X[j, :]).sum()
            if union == 0:
                sim[i, j] = 0
            else:
                sim[i, j] = inter / union
    return sim
