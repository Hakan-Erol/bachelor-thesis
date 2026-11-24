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
    (X_train,X_test): both sets without row coordinates
    """
    answers_43 = ["A","B","C","D","E","F","G","H","I","J"]
    cols_mca = [f"43.{i}" for i in answers_43] + ['44','45']
    """
    We will fit the MCA on question 43 (Learning strategies) and on
    practice and lecture attendance
    """

    X_train_mca =  X_train[cols_mca]
    X_test_mca =  X_test[cols_mca]
    # For this package
    # MCA model
    mca = prince.MCA(
    n_components=12,
    engine='sklearn',
    random_state=42,
    correction="greenacre",
    one_hot=True, 
    check_input=True,
    copy=True)

    # Fitting model on trainset
    mca = mca.fit(X_train_mca)

    # Taking row coordinates
    X_train_cor = mca.row_coordinates(X_train_mca)
    X_test_cor = mca.row_coordinates(X_test_mca)

    # Eigenvalue Summary
    print(mca.eigenvalues_summary)

    

    return mca,X_train_cor,X_test_cor,(X_train,X_test),cols_mca

def GMM(X_train,X_test):
    """
    input:
    X_train : Train set 
    X_test: Test set

    output:

    models : Gaussian mixture models
    
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
    