from glmtree.data_test import generate_data
from glmtree.fit import fit_parallelized
from copy import deepcopy
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score



# bon_critere1 = []
# bon_critere2 = []
#
# split1 = []
# split2 = []
# BIC = []
# bonne_forme = []
# bon_arbre = []
# theta_arbre = []
#
# # Affichage de l'arbre obtenu, None, texte ou image
# affichage = None
#
# n_data = [100, 300, 500, 700, 1000, 2000, 3000, 5000, 8000, 10000]
# n_iter = [80, 90, 100, 110, 120, 130, 140, 150]
# n_para = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#
# X, y, theta, BIC_oracle = generate_data(10000, 3)
# print(BIC_oracle)
#
# for n in n_para:
#     print("n_para", n)
#     BIC_i = []
#     forme_arbre = 0
#     arbre = 0
#     theta_i = []
#     split1_i = [[], [], []]
#     split2_i = [[], [], []]
#
#     for k in range(40):
#         X_train=deepcopy(X)
#         y_train=deepcopy(y)
#         model = fit_parallelized(X_train, y_train, criterion="bic", algo='SEM', nb_init=n, tree_depth=2, class_num=4,
#                                  max_iter=100, validation=False)
#         BIC_i.append(model.best_criterion)
#         # if affichage == 'texte':
#         #     text_representation = tree.export_text(model.best_link)
#         #     print(text_representation)
#         # elif affichage == 'image':
#         #     tree.plot_tree(model.best_link)
#         #     plt.show()
#         # treee = model.best_link.tree_
#         # feature = treee.feature
#         # threshold = treee.threshold
#         #
#         # split1_i[feature[0]].append(threshold[0])
#         # # 0, 1 et 4 sont les nodes de split
#         # split2_i[feature[1]].append(threshold[1])
#         # split2_i[feature[4]].append(threshold[4])
#         #
#         # if len(feature) == 7:
#         #     if (feature == [0, 1, -2, -2, 1, -2, -2]).all() or (feature == [1, 0, -2, -2, 0, -2, -2]).all():
#         #         forme_arbre += 1
#         #         if threshold[0] < 0.1 and threshold[0] > -0.1 and threshold[1] < 0.1 and threshold[1] > -0.1 and \
#         #                 threshold[4] < 0.1 and threshold[4] > -0.1:
#         #             arbre += 1
#         #             theta_model = []
#         #             log_reg = model.best_logreg
#         #             for i in range(len(log_reg)):
#         #                 param = log_reg[i].params
#         #                 theta_model.append([param["Intercept"], param["par_0"], param["par_1"], param["par_2"]])
#         #             theta_i.append(theta_model)
#     print("BIC",n, np.mean(BIC_i))
#     BIC.append(BIC_i)
#     # theta_arbre.append(theta_i)
#     # bonne_forme.append(forme_arbre / 40)
#     # bon_arbre.append(arbre / 40)
#     #
#     # bon_critere1.append(1 - len(split1_i[2]) / 40)
#     # bon_critere2.append(1 - len(split2_i[2]) / 80)
#     # split1.append(split1_i[0] + split1_i[1])
#     # split2.append(split2_i[0] + split2_i[1])
#
# # print("Bon_critere=", bonne_forme)
# # print("Bon_arbre=", bon_arbre)
# # print("Theta=", theta_arbre)
# print("BIC=", BIC)
# # print("Bon_split1=", bon_critere1)
# # print("Bon_split2=", bon_critere2)
# # print("Split1=", split1)
# # print("Split2=", split2)


X, y, theta, BIC_oracle = generate_data(10000, 3)
X_test, y_test, _, _ = generate_data(10000, 3)

model = fit_parallelized(X, y, algo='SEM', criterion="aic", nb_init=10, tree_depth=2, class_num=4, max_iter=100)
text_representation = tree.export_text(model.best_link)
print(text_representation)
# tree.plot_tree(model.best_link)
# plt.show()
# plt.close()
models = model.best_logreg
print([mod.coef_ for mod in models])
print("Precision", model.precision(X_test, y_test))
y_proba = model.predict_proba(X_test)
print("SEM test : ", roc_auc_score(y_test, y_proba))
