"""
The code used in order to plot all the data
"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
import numpy as np
from matplotlib.patches import Ellipse

def corr_cluster(df):
    """
    Plot correlation and clustermap
    """
    # Sub plots
    # fig, ax =plt.subplots(1,2)

    # Correlation
    corr = df.corr()
    sns.heatmap(corr,annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation matrix")
    plt.show()

    # Cluster map 
    sns.clustermap(df)
    plt.title("Cluster map")
    plt.show()
    return None

def plot_gmm(chosen_component,X_train,X_test,models):
        """
        
        """
        chosen_component = 3
        chosen_model = models[chosen_component - 1]
    
        colors = ['blue','green','red',"purple","grey"]
        fig,ax = plt.subplots()
        ax.scatter(X_train.iloc[:,0],X_train.iloc[:,1],label='train')
        ax.scatter(X_test.iloc[:,0],X_test.iloc[:,1],label='test')
    
        for n in range(chosen_component):
            mean = chosen_model.means_[n,:2]
            covariance = chosen_model.covariances_[n][:2,:2]
            
            v,w = linalg.eigh(covariance)
            
            angle = np.arctan2(w[0][1],w[0][0])
            angle = 180.0 * angle / np.pi
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            # covariance = 
            ellipse = Ellipse(mean,v[0],v[1],angle = 180.0 + angle,color=colors[n])
            ellipse.set_clip_box(fig.bbox)
            ellipse.set_alpha(0.5)
            ax.add_patch(ellipse)
            
            
        fig.legend()
        fig.show()
        return None