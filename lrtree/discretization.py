import itertools
from bisect import bisect_right
from copy import deepcopy
from math import log

import numpy as np
import pandas as pd
from loguru import logger
from pandas.core.common import flatten
from scipy.stats import chi2_contingency
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

FORMAT = "{:.2e}"


def bin_data_cate_train(data: pd.DataFrame, var_cible: str, categorical=None):
    # if categorical is None:
    #     categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
    #                    "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET",
    #                    "CRED_Null_Group_Dest_fin_Conso",
    #                    "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers",
    #                    "CRED_Null_Group_bien_fin_Conso",
    #                    "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers",
    #                    "CRED_Null_Group_interv_Conso",
    #                    "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
    #                    "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option",
    #                    "TOP_SEUIL_New_def", "segment", "incident"]
    X = data.copy()
    to_change = []
    if categorical:
        for column in categorical:
            if column in X.columns:
                to_change.append(column)
                X.loc[:, column] = X[column].astype(str)
    for column in X.columns:
        if column not in categorical and column != var_cible:
            X.loc[:, column] = X[column].astype(np.float64)
    if var_cible in X.columns:
        X.drop([var_cible], axis=1, inplace=True)
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = enc.fit_transform(X[to_change])
    X_cat = pd.DataFrame(X_cat)
    X_num = X.drop(to_change, axis=1)
    if not X_num.columns.empty:
        for column in X_num.columns:
            col = X_num[column]
            if col.dtypes not in ("int32", "int64", "float32", "float64"):
                X_num.loc[:, column] = X_num[column].astype(np.int32)

    # Need to reset the index for the concat to work well
    X_cat = X_cat.reset_index(drop=True)
    X_num = X_num.reset_index(drop=True)

    X_train = pd.concat([X_num, X_cat], axis=1, ignore_index=True)
    return X_train, enc


def chi2_test(liste):
    try:
        return chi2_contingency(liste)[1]
    except ValueError:
        # Happens when one modality of the variable has only 0 or only 1
        return 1


def grouping(X: pd.DataFrame, var: str, var_predite: str, seuil: float = 0.2) -> (pd.DataFrame, dict):
    """
    Chi2 independence algorithm to group modalities

    :param pandas.Dataframe X:
        Data with some variables being categorical
    :param str var:
        Column for which we apply the algorithm
    :param str var:
        Column we aim to predict
    :param str var_predite:
        Column we aim to predict
    :param float seuil:
        Value for the p-value over which we merge modalities
    """
    X_grouped = X.copy()

    # Necessary to be able to compare (and unique) categories
    X_grouped[var] = X_grouped[var].astype(str)
    initial_categories = np.unique(X_grouped[var])

    p_value = 1
    # We want a minimum of two categories, otherwise the variable becomes useless
    while p_value > seuil and len(np.unique(X_grouped[var])) > 2:
        # Counts the number of 0/1 by modality
        freq_table = X_grouped.groupby([var, var_predite]).size().reset_index()
        # All the combinations of values
        liste_paires_modalities = list(itertools.combinations(np.unique(X_grouped[var]), 2))
        # Chi2 between the pairs
        liste_chi2 = [chi2_test([freq_table.iloc[np.in1d(freq_table[var], pair[0]), 2],
                                 freq_table.iloc[np.in1d(freq_table[var], pair[1]), 2]]) for pair in
                      liste_paires_modalities]
        p_value = max(liste_chi2)
        if p_value > seuil and len(np.unique(X_grouped[var])) > 1:
            # Updates the modality to the new concatenated value
            X_grouped[var].iloc[
                np.in1d(X_grouped[var], liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))])] = \
                liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))][0] + ' - ' + \
                liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))][1]
            logger.debug(f"Feature {var} - levels merged: "
                         f"{str(liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))])}")
        else:
            break
    new_categories = np.unique(X_grouped[var])

    # Dictionary of the correspondence old category : merged category
    dico_regroupement = {}
    for regrouped_cate in new_categories:
        for cat in regrouped_cate.split(' - '):
            dico_regroupement[cat] = regrouped_cate

    logger.debug(f"Feature {var} went from {len(initial_categories)} to {len(new_categories)} levels.")

    return X_grouped, dico_regroupement


def entropy(variable):
    """
    Computes the entropy of the variable
    """
    def stable_log(input):
        copy = input.copy()
        copy[copy <= 1e-10] = 1
        return np.log(copy)

    # Proportion/probability of each unique value in variable
    prob = np.unique(variable, return_counts=True)[1] / len(variable)
    ent = -sum(prob * stable_log(prob))
    return ent


def stopping_criterion(cut_idx, target, ent, depth):
    """
    Decided whether we should cut target at cut_idx, knowing we imagine the new entropy to be
    """
    n = len(target)
    target_entropy = entropy(target)
    gain = target_entropy - ent

    k = len(np.unique(target))
    k1 = len(np.unique(target[: cut_idx]))
    k2 = len(np.unique(target[cut_idx:]))

    delta = (log(3 ** k - 2) - (k * target_entropy -
                                k1 * entropy(target[: cut_idx]) -
                                k2 * entropy(target[cut_idx:])))
    cond = log(n - 1) / n + delta / n
    # We want at least one separation
    if gain >= cond or depth == 1:
        return gain
    else:
        return None


def find_cut_index(x, y):
    """
    Finds the best place to split x
    """
    n = len(y)
    init_entropy = 999999
    current_entropy = init_entropy
    index = None
    # We can't test every single value, it would take too long
    step = max(1, n // 100)
    for i in range(0, n - 1, step):
        if x[i] != x[i + 1]:
            cut = (x[i] + x[i + 1]) / 2.
            # Return the index where to insert item cut in list x, assuming x is sorted
            cutx = bisect_right(x, cut)
            weight_cutx = cutx / n
            left_entropy = weight_cutx * entropy(y[: cutx])
            right_entropy = (1 - weight_cutx) * entropy(y[cutx:])
            # New entropy with the separation
            temp = left_entropy + right_entropy
            if temp < current_entropy:
                current_entropy = temp
                index = i + 1
    if index is not None:
        return [index, current_entropy]
    else:
        return None


def get_index(xo, yo, low, upp, depth):
    x = xo[low:upp]
    y = yo[low:upp]
    # Finds the best place to split, if we had to split
    cut = find_cut_index(x, y)
    if cut is None:
        return None
    cut_index = int(cut[0])
    current_entropy = cut[1]
    # Checks whether it is worth it to split
    ret = stopping_criterion(cut_index, np.array(y),
                             current_entropy, depth)
    if ret is not None:
        return [cut_index, depth + 1]
    else:
        return None


def part(xo, yo, low, upp, cut_points, depth):
    """
    Recursive function with returns the cuts_points
    """
    x = xo[low: upp]
    if len(x) < 2:
        return cut_points
    cc = get_index(xo, yo, low, upp, depth=depth)
    if cc is None:
        return cut_points
    ci = int(cc[0])
    depth = int(cc[1])
    cut_points = np.append(cut_points, low + ci)
    cut_points = cut_points.astype(int)
    cut_points.sort()
    # We choose to have a maximum of 8 categories in total
    if len(cut_points) > 2:
        return cut_points
    return (list(part(xo, yo, low, low + ci, cut_points, depth=depth)) +
            list(part(xo, yo, low + ci + 1, upp, cut_points, depth=depth)))


def cut_points(x, y):
    """
    Computes the cut values on x to minimize the entropy (on y)
    """
    res = part(xo=x, yo=y, low=0, upp=len(x) - 1, cut_points=np.array([]), depth=1)
    cut_value = []
    if res is not None:
        cut_index = res
        for indices in cut_index:
            # Gets the cut value from the cut index
            cut_value.append((x[indices - 1] + x[indices]) / 2.0)
    result = np.unique(cut_value)
    return result


def apply_discretization(X, var, cut_values):
    cut_values = list(np.sort(cut_values))
    cut_values.append(np.inf)
    x = X[var].to_numpy()
    x_discrete = np.array([1 for _ in range(len(x))])
    logger.debug(f"Feature {var} discretized in {len(cut_values)} bin(s).")
    for index in range(1, len(cut_values)):
        row_filter = (cut_values[index - 1] < x) & (x <= cut_values[index])
        x_discrete[np.where(row_filter)[0]] = index + 1
    X[var] = x_discrete
    return X


def discretize_feature(X: pd.DataFrame, var: str, var_predite: str):
    """
    Discretizes the continuous variable X[var], using the target value var_predite
    MDPL (Minimum Description Length Principle)

    :param pandas.DataFrame X:
    :param str var:
    :param str var_predite:
    :return:
    :rtype: (pandas.DataFrame, list)
    """
    x = X[var].sort_values()
    y = X[var_predite][x.index].to_numpy()
    binning = cut_points(x.to_numpy(), y)
    X = apply_discretization(X, var, binning)
    return X, binning


def _categorie_data_bin_train(data: pd.DataFrame, var_cible, categorical=None, discretize=True):
    """
    Binarise the categorical variables (after merging categories)
    Returns the data, the encoder and the column labels

    :param pandas.DataFrame data: data with some variables being categorical
    :param str var_cible: name of target
    :param list categorical: list of categorical features' names
    :param bool discretize: whether to discretize continuous features
    """
    X = data.copy()
    to_change = []
    merged_cat = {}
    if categorical:
        for column in categorical:
            if column in X.columns:
                to_change.append(column)
                X.loc[:, column] = X[column].astype(str)
                X, dico = grouping(X, column, var_cible)
                merged_cat[column] = dico

    discret_cat = {}
    if discretize:
        for column in X.columns:
            if column not in categorical and column != var_cible:
                to_change.append(column)
                X.loc[:, column] = X[column].astype(np.float64)
                X, binning = discretize_feature(X, column, var_cible)
                discret_cat[column] = binning

    if var_cible in X.columns:
        X.drop([var_cible], axis=1, inplace=True)

    X_cat = None
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    if to_change:
        X_cat = enc.fit_transform(X[to_change])
        labels_cat = enc.get_feature_names_out()
        X_cat = pd.DataFrame(X_cat)
        X_cat = X_cat.reset_index(drop=True)

    # Used when we had decided to keep the continuous numerical variables
    X_num = X.drop(to_change, axis=1)
    X_num_transformed = None
    col_num = X_num.columns
    labels_num = []
    scaler = StandardScaler()
    if not col_num.empty:
        for column in col_num:
            labels_num.append(column)
            col = X_num[column]
            if col.dtypes not in ("int32", "int64", "float32", "float64"):
                X_num.loc[:, column] = X_num[column].astype(np.int32)
        X_num_transformed = scaler.fit_transform(X_num)

    if X_cat is not None and X_num_transformed is not None:
        X_train = pd.concat([pd.DataFrame(X_num_transformed), X_cat], axis=1, ignore_index=True)
        labels = [*labels_num, *labels_cat]
    elif X_cat is not None:
        X_train = X_cat
        labels = labels_cat
    else:
        X_train = X_num
        labels = labels_num
    return X_train, labels, enc, merged_cat, discret_cat, scaler, len(col_num)


def _categorie_data_bin_test(data_val: pd.DataFrame, enc: OneHotEncoder, scaler: StandardScaler, merged_cat: dict,
                             discret_cat: dict, categorical=None, discretize=False) -> pd.DataFrame:
    """
    Data treatment of the test data, using the method (merged categories, discretisation, OneHotEncoder) learned on
    the train data

    :param pandas.Dataframe data_val:
        Data with some variables being categorical
    :param sklearn.preprocessing.OneHotEncoder enc:
        OneHotEncoder fitted on the categorical training data
    :param sklearn.preprocessing.StandardScaler scaler:
        StandardScaler fitted on the numerical training data
    :param dict merged_cat: merged categories
    :param dict discret_cat: discretized categories
    :param list categorical: list of categorical features' names
    :param bool discretize: whether numerical features were discretized
    :return: treated data
    :rtype:pandas.DataFrame
    """
    if not isinstance(categorical, list):
        categorical = [None]
    X_val = data_val.copy()
    to_change = []
    for column in set(list(merged_cat.keys()) + categorical):
        if column in X_val.columns:
            to_change.append(column)
            X_val.loc[:, column] = X_val[column].astype(str)
    # Merging the categories
    X_val.replace(merged_cat, inplace=True)
    if discretize:
        for column in X_val.columns:
            if column not in set(list(merged_cat.keys()) + categorical):
                to_change.append(column)
                X_val.loc[:, column] = X_val[column].astype(np.float64)
                X_val = apply_discretization(X_val, column, discret_cat[column])

    if to_change:
        assert set(enc.feature_names_in_) == set(to_change)  # nosec
        X_val_cat = enc.transform(X_val[enc.feature_names_in_])
        X_val_cat = pd.DataFrame(X_val_cat)

    X_val_num = X_val.drop(to_change, axis=1)
    for column in X_val_num.columns:
        col = X_val_num[column]
        if col.dtypes not in ("int32", "int64", "float32", "float64"):
            X_val_num.loc[:, column] = X_val_num[column].astype(np.int32)
    # Need to reset the index for the concat to work well (when not same index)
    X_val_cat = X_val_cat.reset_index(drop=True)
    if not X_val_num.empty:
        X_val_num_transformed = scaler.transform(X_val_num)
        X_test = pd.concat([pd.DataFrame(X_val_num_transformed), X_val_cat], axis=1, ignore_index=True)
        return X_test
    else:
        return X_val_cat


def create_reduction_matrix(reductions, clustered, labels, chi0, pd):
    # Matrix of the changes of chi2 for every pair grouped
    for count1, value1 in enumerate(labels):
        # 0, modality0 ...
        for count2, value2 in enumerate(labels[count1 + 1:], start=count1 + 1):
            # count1+1, modality7 ...
            if count1 != count2:
                # New matrix once we have merged the two
                contingency_matrix = metrics.cluster.contingency_matrix(
                    clustered.replace({value1: value2}, inplace=False), pd)
                # stat chi2, p-value, degree of freedom, expected frequencies
                g, p, dof, expctd = chi2_contingency(contingency_matrix, lambda_="log-likelihood")
                reductions[count1, count2] = FORMAT.format((1 - g / chi0))
    return reductions


def update_reduction_matrix(df, var, var_predite, matrix, clustered, ind, chi0):
    # Update reduction matrix after finding the two categories that will impact the least our chi2
    pd = df[var_predite]
    labels = df[var].unique()
    # Replace the second category by the first in the categorical variable to get the new chi2
    clustered.replace({labels[ind[1]]: labels[ind[0]]}, inplace=True)
    # Replace all the values of chi2 related to ind[0] (new index for the merged category)
    for count, value in enumerate(labels):
        contingency_matrix = metrics.cluster.contingency_matrix(
            clustered.replace({value: labels[ind[0]]}, inplace=False), pd)
        g, p, dof, expctd = chi2_contingency(contingency_matrix, lambda_="log-likelihood")
        # Fills in (in the good half) the new values
        if count > ind[0]:
            matrix[ind[0], count] = FORMAT.format((1 - g / chi0))
        if count < ind[0]:
            matrix[count, ind[0]] = FORMAT.format((1 - g / chi0))
    return matrix


def green_clust(X: pd.DataFrame, var: str, var_predite: str, num_clusters: int) -> (pd.DataFrame, dict):
    """
    GreenClust algorithm to group modalities

    :param pandas.Dataframe X:
        Data with some variables being categorical
    :param str var:
        Column for which we apply the algorithm
    :param str var_predite:
        Column we aim to predict
    :param int num_clusters:
        Number of modalities we want
     """
    df = X.copy()
    # Necessary to be able to compare (and unique) categories
    df[var] = df[var].astype(str)
    # Get the modalities
    labels = df[var].unique()
    clustered = df[var].copy()
    # Square matrix (each line/column = a category), each cell represents the impact on chi2 if we merge the row &
    # column categories
    reductions = np.empty((len(labels), len(labels)))
    reductions[:] = np.NaN
    # List of index of categories merged to another category, then ignored in the reduction matrix
    forgotten_rows = []
    pd = df[var_predite]
    clusters = [*range(0, len(labels), 1)]
    # Clusters is a list of lists that will contain the new groups
    clusters = [[i] for i in clusters]
    contingency_matrix = metrics.cluster.contingency_matrix(df[var], pd)
    # Get the inital chi2 for the impact of grouping modalities
    chi0, p, dof, expctd = chi2_contingency(contingency_matrix, lambda_="log-likelihood")
    reductions = create_reduction_matrix(reductions, clustered, labels, chi0, pd)
    # keep merging until we reach the number of desired groups
    while len(clusters) > num_clusters:
        # Get the minimum chi2 indices in the reduction matrix without taking into account the forgotten rows
        # Smallest reduction = biggest chi2 test statistic = most dependant variables
        min_chi = np.nanmin(np.delete(np.delete(reductions, forgotten_rows, 0), forgotten_rows, 1))
        ind = tuple(np.argwhere(reductions == min_chi)[0])
        # For the indices of the min chi2, add the second index to forgetten rows (ind[1] now be ind[0]
        forgotten_rows.append(ind[1])
        # update the values of chi2 in the reduction matrix after merging the indices in the clustered column
        reductions = update_reduction_matrix(df, var, var_predite, reductions, clustered, ind, chi0)
        # And finally merging them in the clusters variable
        cluster1 = list(flatten([sublist for sublist in clusters if ind[0] in sublist]))
        cluster2 = list(flatten([sublist for sublist in clusters if ind[1] in sublist]))
        clusters.remove(cluster1)
        clusters.remove(cluster2)
        clusters += [cluster1 + cluster2]

    dico_regroupement = {}
    for group in clusters:
        cluster = []
        for ind in group:
            cluster += [labels[ind]]
        name_cluster = ' - '.join(cluster)
        for ind in group:
            dico_regroupement[labels[ind]] = name_cluster

    df.replace({var: dico_regroupement}, inplace=True)
    return df, dico_regroupement


def categorie_data_labels(data: pd.DataFrame, data_val: pd.DataFrame, categorical=None) -> (pd.DataFrame, pd.DataFrame):
    # if categorical is None:
    #     categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
    #                    "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET",
    #                    "CRED_Null_Group_Dest_fin_Conso",
    #                    "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers",
    #                    "CRED_Null_Group_bien_fin_Conso",
    #                    "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers",
    #                    "CRED_Null_Group_interv_Conso",
    #                    "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
    #                    "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option",
    #                    "TOP_SEUIL_New_def", "segment", "incident"]
    X = data.copy()
    X_val = data_val.copy()
    to_change = []
    X_cat = []
    X_val_cat = []
    for column in categorical:
        if column in X.columns:
            to_change.append(column)
            X[column] = X[column].astype(str)
            X_val[column] = X_val[column].astype(str)
            enc = LabelEncoder()
            X_val_cat.append(enc.transform(X_val[column]))
            X_cat.append(enc.fit_transform(X[column]))

    X_cat = pd.DataFrame(X_cat)
    X_val_cat = pd.DataFrame(X_val_cat)

    X_num = X.drop(to_change, axis=1)
    if "Defaut_12_Mois_contagion" in X_num.columns:
        X_num.drop(["Defaut_12_Mois_contagion"], axis=1, inplace=True)
    X_val_num = X_val.drop(to_change, axis=1)
    if "Defaut_12_Mois_contagion" in X_val_num.columns:
        X_val_num.drop(["Defaut_12_Mois_contagion"], axis=1, inplace=True)

    for column in X_num.columns:
        col = X_num[column]
        if col.dtypes not in ("int32", "int64", "float32", "float64"):
            X_num[column] = col.astype(np.int32)
            X_val_num[column] = col.astype(np.int32)

    return pd.concat([X_num, X_cat], axis=1), pd.concat([X_val_num, X_val_cat], axis=1)


def categorie_data_bin_train_test(data: pd.DataFrame,
                                  data_val: pd.DataFrame, categorical=None) -> (pd.DataFrame, pd.DataFrame, list):
    # if categorical is None:
    #     categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
    #                    "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET",
    #                    "CRED_Null_Group_Dest_fin_Conso",
    #                    "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers",
    #                    "CRED_Null_Group_bien_fin_Conso",
    #                    "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers",
    #                    "CRED_Null_Group_interv_Conso",
    #                    "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
    #                    "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option",
    #                    "TOP_SEUIL_New_def", "segment", "incident"]
    X = data.copy()
    X_val = data_val.copy()
    to_change = []
    for column in categorical:
        if column in X.columns:
            to_change.append(column)
            X[column] = X[column].astype(str)
            X_val[column] = X_val[column].astype(str)
    enc = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    X_cat = enc.fit_transform(deepcopy(X[to_change]))
    X_val_cat = enc.transform(deepcopy(X_val[to_change]))
    labels_cat = enc.get_feature_names_out()

    X_cat = pd.DataFrame(X_cat)
    X_val_cat = pd.DataFrame(X_val_cat)

    X_num = X.drop(to_change, axis=1)
    if "Defaut_12_Mois_contagion" in X_num.columns:
        X_num.drop(["Defaut_12_Mois_contagion"], axis=1, inplace=True)
    col_num = X_num.columns
    labels_num = []
    for column in col_num:
        labels_num.append(column)
    X_val_num = X_val.drop(to_change, axis=1)
    if "Defaut_12_Mois_contagion" in X_val_num.columns:
        X_val_num.drop(["Defaut_12_Mois_contagion"], axis=1, inplace=True)

    for column in X_num.columns:
        col = X_num[column]
        if col.dtypes not in ["int32", "int64", "float32", "float64"]:
            X_num[column] = X_num[column].astype(np.int32)
            X_val_num[column] = X_val_num[column].astype(np.int32)

    # Need to reset the index for the concat to work well (when not same index)
    X_cat = X_cat.reset_index(drop=True)
    X_val_cat = X_val_cat.reset_index(drop=True)
    X_num = X_num.reset_index(drop=True)
    X_val_num = X_val_num.reset_index(drop=True)

    X_train = pd.concat([X_num, X_cat], axis=1, ignore_index=True)
    X_test = pd.concat([X_val_num, X_val_cat], axis=1, ignore_index=True)
    labels = [*labels_num, *labels_cat]
    return X_train, X_test, labels


def bin_data_cate_test(data_val, enc, categorical):
    X_val = data_val.copy()
    # categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
    #                "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET", "CRED_Null_Group_Dest_fin_Conso",
    #                "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers", "CRED_Null_Group_bien_fin_Conso",
    #                "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers", "CRED_Null_Group_interv_Conso",
    #                "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
    #                "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option",
    #                "TOP_SEUIL_New_def", "segment", "incident"]
    to_change = []
    for column in categorical:
        if column in X_val.columns:
            to_change.append(column)
            X_val.loc[:, column] = X_val[column].astype(str)

    for column in X_val.columns:
        if column not in categorical:
            X_val.loc[:, column] = X_val[column].astype(np.float64)

    X_val_cat = enc.transform(X_val[to_change])
    X_val_cat = pd.DataFrame(X_val_cat)

    X_val_num = X_val.drop(to_change, axis=1)
    for column in X_val_num.columns:
        col = X_val_num[column]
        if col.dtypes not in ("int32", "int64", "float32", "float64"):
            X_val_num.loc[:, column] = X_val_num[column].astype(np.int32)

    # Need to reset the index for the concat to work well (when not same index)
    X_val_cat = X_val_cat.reset_index(drop=True)
    X_val_num = X_val_num.reset_index(drop=True)
    X_test = pd.concat([X_val_num, X_val_cat], axis=1, ignore_index=True)
    return X_test


def extreme_values(data: pd.DataFrame, missing: bool = False) -> pd.DataFrame:
    """
    Deals with extreme values (ex : NaN, or not filled)
    Creates (or not) a column signaling which values were missing

    :param pandas.DataFrame data: data to clean
    :param bool missing: whether or not to create a column signaling missing values
    :rtype: pandas.DataFrame
    """
    # Valeurs extremes quand valeurs manquantes
    extremes = [99999, 99999.99, 999999.0, 9.999999999999E10, 99999999999.99, 999, 'NR']
    for column in data.columns:
        data[column].replace(to_replace=extremes, value=np.NaN, inplace=True)

    if missing:
        # Creation de colonnes de variables présentes/absentes
        for column in data.columns:
            values = data[column]
            if None in values or pd.isna(values).any():
                new_column_name = "Exists_" + str(column)
                new_column = []
                for i in range(len(data)):
                    if values.iloc[i] is None or pd.isna(values.iloc[i]):
                        new_column.append(0)
                    else:
                        new_column.append(1)
                data.insert(0, new_column_name, new_column)

    for column in data.columns:
        data.loc[:, column] = data[column].fillna(value=0)

    dav_null = ["DAV_Null_INDIC_TIERS_CONTENTIEUX", "DAV_Null_INDIC_TIERS_NEIERTZ", "DAV_Null_INDIC_PERS_FICP",
                "DAV_Null_PERS_SAISIE_ATTRIB",
                "DAV_Null_INDIC_PERS_TUTELLE", "DAV_Null_INDIC_PERS_CURATELLE", "DAV_Null_INDIC_PERS_NEIERTZ",
                "DAV_Null_INDIC_PERS_REDRST_JUDIC", "DAV_Null_INDIC_PERS_CONSTIT_STE",
                "DAV_Null_INDIC_PERS_LIQUID_JUDIC"]
    for column in dav_null:
        if column in data.columns:
            data[column].replace(['0', '1'], [0, 1], inplace=True)

    return data


def traitement_train_val(X: pd.DataFrame, X_val: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, list):
    """
    Traite les données et les données de test en gérant les valeurs extremes, les variables catégoriques et en
    normalisant
    Retourne les données traitées et les labels des colonnes

    :param pandas.DataFrame X:
        Data with some variables being categorical
    :param pandas.DataFrame X_val:
        Validation data with some variables being categorical
    :return: les données traitées et les labels des colonnes
    :rtype: tuple
    """
    X = extreme_values(X, missing=False)
    X_val = extreme_values(X_val, missing=False)

    X_train, X_test, labels = categorie_data_bin_train_test(X, X_val)

    # Vérifie qu'on a bien les mêmes colonnes dans les données train / test
    for column in X_train.columns:
        if column not in X_test.columns:
            X_train = X_train.drop(column, axis=1)
    for column in X_test.columns:
        if column not in X_train.columns:
            X_test = X_test.drop(column, axis=1)

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, labels


def traitement_val(data: pd.DataFrame, enc: OneHotEncoder, scaler: StandardScaler, merged_cat: dict,
                   discret_cat: dict, categorical: list, discretize: bool) -> pd.DataFrame:
    """
    Traite les données de test en gérant les valeurs extremes, les variables catégoriques et en normalisant
    Retourne les données traitées

    :param pandas.Dataframe data:
        Data with some variables being categorical
    :param sklearn.preprocessing.OneHotEncoder enc:
        OneHotEncoder fitted on the training data
    :param sklearn.preprocessing.StandardScaler scaler:
        StandardScaler fitted on the training data
    :param dict merged_cat:
        Merged categories for each column
    :param dict discret_cat:
        Binning for the discretisation for each column
    :param list categorical:
        Names for categorical features
    :param bool discretize:
        Whether numerical features were discretized
    :return: preprocessed validation data
    :rtype: pandas.DataFrame
    """
    X_val = data.copy()
    if "Defaut_12_Mois_contagion" in X_val.columns:
        X_val.drop(["Defaut_12_Mois_contagion"], axis=1, inplace=True)
    X_val = extreme_values(X_val, missing=False)
    X_val = _categorie_data_bin_test(data_val=X_val, enc=enc, scaler=scaler,
                                     merged_cat=merged_cat, discret_cat=discret_cat,
                                     categorical=categorical, discretize=discretize)
    return X_val


def traitement_train(data: pd.DataFrame, target: str, categorical=None, discretize=False) -> tuple:
    """
    Traite les données en gérant les valeurs extremes, les variables catégoriques et en normalisant

    :param pandas.Dataframe data: data with some variables being categorical
    :param str target: variable cible
    :param list categorical: list of categorical features' names
    :param bool discretize: whether to discretize continuous features
    :return: processed data, labels of columns, encoder for categorical data and scaler for numerical data
    :rtype: tuple
    """
    X = data.copy()
    if target in X.columns and X[target].dtypes == "object":
        X[target].replace(["N", "O"], [int(0), int(1)], inplace=True)
    X = extreme_values(X, missing=False)
    X, labels, enc, merged_cat, discrete_cat, scaler, len_col_num = _categorie_data_bin_train(
        X,
        var_cible=target,
        categorical=categorical,
        discretize=discretize)
    return X, labels, enc, scaler, merged_cat, discrete_cat, len_col_num


class Processing:
    def __init__(self, target: str, discretize: bool = False, merge_threshold: float = 0.2):
        self.target = target
        self.labels, self.enc, self.scaler, self.merged_cat, self.discrete_cat = [None] * 5
        self.discretize = discretize
        self.num_cols = []
        self.cat_cols = []
        self.merge_threshold = merge_threshold
        self.X_train = None

    def fit(self, X: pd.DataFrame, categorical: list):
        self.cat_cols = categorical
        # Check if all categorical are inside
        if self.cat_cols:
            for col in self.cat_cols:
                if col not in X.columns:
                    msg = f"Column {col} specified in argument 'categorical' not present."
                    logger.error(msg)
                    raise ValueError(msg)
            # Calculate remaining columns
            self.num_cols = [col for col in X.columns.to_list() if col not in self.cat_cols + [self.target]]
        else:
            self.num_cols = X.columns.to_list()
            if self.target in self.num_cols:
                self.num_cols.remove(self.target)

        # Application sur train
        X, self.labels, self.enc, self.scaler, self.merged_cat, self.discrete_cat, len_col_num = traitement_train(
            data=X, target=self.target, categorical=self.cat_cols, discretize=self.discretize
        )
        # assert len_col_num == len(self.num_cols)  # nosec
        self.X_train = X

    def fit_transform(self, X: pd.DataFrame, categorical: list):
        self.fit(X, categorical)
        return self.X_train

    def transform(self, X: pd.DataFrame):
        X_test = X.copy()
        if self.target in X_test.columns.to_list():
            del X_test[self.target]
        return traitement_val(data=X_test, enc=self.enc, scaler=self.scaler,
                              merged_cat=self.merged_cat, discret_cat=self.discrete_cat,
                              categorical=self.cat_cols, discretize=self.discretize)
