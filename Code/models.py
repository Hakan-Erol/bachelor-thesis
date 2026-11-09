"""
The code used in order to develop the models
"""
import prince
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import linalg
import numpy as np
from matplotlib.patches import Ellipse
from stepmix.stepmix import StepMix

def MCA(X_train,X_test):
    """
    input:
    X_train : Train set 
    X_test: Test set

    output:

    mca : MCA model
    X_train_cor: Train set (row coordinates)
    X_test_cor: test set (row coordinates)
    """

    # MCA model
    mca = prince.MCA(
    n_components=7,
    engine='sklearn',
    random_state=42,
    one_hot=False, # one hot encoding is done
    check_input=True,
    copy=True)

    # Fitting model on trainset
    mca = mca.fit(X_train)

    # Taking row coordinates
    X_train_cor = mca.row_coordinates(X_train)
    X_test_cor = mca.row_coordinates(X_test)

    # Eigenvalue Summary
    print(mca.eigenvalues_summary)

    

    return mca,X_train_cor,X_test_cor

def GMM(X_train,X_test):
    """
    input:
    X_train : Train set 
    X_test: Test set

    output:

    models : Gaussian mixture model
    
    """
    # Creating the GMM 
    # gm = GaussianMixture(n_components=5,random_state=42).fit(X_train)

    # Testing different models using the BIC (choose lowest)
    components = range(1,7) # 7 different components

    models = [] # List of fitted models
    for n in components:
        gm = GaussianMixture(n_components=n,random_state=42,n_init=20).fit(X_train)
        print(f'components:{n}\n\tBIC:{gm.bic(X_test):.2f}\n\tAIC:{gm.aic(X_test):.2f}')
        models.append(gm)

    return models

def LCA(X_train,X_test):
    """
    input:
    X_train : Train set 
    X_test: Test set

    output:

    models : LCA models
    
    """

    components = range(1,7)

    models = [] # List of fitted models
    for n in components:
        model = StepMix(n_components=n, measurement="categorical",random_state=42,n_init=20).fit(X_train)
        print(f'components:{n}\n\tBIC:{model.bic(X_test):.2f}\n\tAIC:{model.aic(X_test):.2f}')
        models.append(model)
    return models
    