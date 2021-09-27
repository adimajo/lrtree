import glmtree
from glmtree.data_test import generate_data2
from joblib import Parallel, delayed
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

bon_critere1=[]
bon_critere2=[]

split1=[]
split2=[]
BIC=[]
bonne_forme=[]
bon_arbre=[]
theta_arbre=[]

n_data= [100, 300, 500, 700, 1000, 2000, 3000, 5000, 8000, 10000]

for n in n_data :
    print(n)
    BIC_i=[]
    forme_arbre=0
    arbre=0
    theta_i=[]
    split1_i = [[], [], []]
    split2_i = [[], [], []]
    X, y, theta, BIC_oracle = generate_data2(n, 3, theta=None)

    models = []

    # for k in range(20):
    #     model = glmtree.Glmtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=4, max_iter=100)
    #     model.fit(X, y, nb_init=n)

    def fit_func(X, y, n_para):
        model = glmtree.Glmtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=4, max_iter=100)
        model.fit(X, y, n_para)
        return model

    models = Parallel(n_jobs=2)(delayed(fit_func)(X, y, 5) for k in range(10))

    for k in range(10):
        model=models[k]
        BIC_i.append(model.best_criterion[0])
        # text_representation = tree.export_text(model.best_link)
        # print(text_representation)

        # tree.plot_tree(model.best_link)
        #plt.show()
        treee = model.best_link.tree_
        feature = treee.feature
        threshold = treee.threshold

        split1_i[feature[0]].append(threshold[0])
        #0, 1 et 4 sont les nodes de split
        split2_i[feature[1]].append(threshold[1])
        split2_i[feature[4]].append(threshold[4])

        if len(feature)==7:
            if (feature==[0, 1, -2, -2, 1, -2, -2]).all() or (feature==[1, 0, -2, -2, 0, -2, -2]).all() :
                forme_arbre+=1
                if threshold[0]<0.1 and threshold[0]>-0.1 and threshold[1]<0.1 and threshold[1]>-0.1 and threshold[4]<0.1 and threshold[4]>-0.1 :
                    arbre+=1
                    theta_model = []
                    log_reg=model.best_logreg
                    for i in range(len(log_reg)):
                        param = log_reg[i].params
                        theta_model.append([param["Intercept"], param["par_0"], param["par_1"], param["par_2"]])
                    theta_i.append(theta_model)



    BIC.append(BIC_i)
    theta_arbre.append(theta_i)
    bonne_forme.append(forme_arbre/40)
    bon_arbre.append(arbre/40)

    bon_critere1.append(1 - len(split1_i[2])/40)
    bon_critere2.append(1 - len(split2_i[2])/80)
    split1.append(split1_i[0]+split1_i[1])
    split2.append(split2_i[0]+split2_i[1])

print("Pourcentage bon critères :", bonne_forme)
print("Pourcentage bon modele :", bon_arbre)
print("Theta quand bon modèle :", theta_arbre)
print("BIC :", BIC)
print("Critere 1 :", bon_critere1)
print("Critere 2 :", bon_critere2)
print("Split 1 :", split1)
print("Split 2:", split2)