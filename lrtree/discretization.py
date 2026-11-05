import itertools
from bisect import bisect_right
from math import log

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def as_categories(column: pd.Series) -> pd.Series:
    """
    The levels of a categorical feature as strings. Missing values are a level of their own ("nan"): with pandas'
    string dtype ``astype(str)`` would keep them missing and they could not be compared with the other levels.
    """
    return column.astype(object).where(column.notna(), "nan").astype(str)


def chi2_test(liste):
    try:
        return chi2_contingency(liste)[1]
    except ValueError:
        # Happens when one modality of the variable has only 0 or only 1
        return 1


def grouping(X: pd.DataFrame, var: str, var_predite: str, group: bool = True,
             seuil: float = 0.2) -> (pd.DataFrame, dict):
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
    :param bool group:
        Whether to merge modalities (if False, the modalities are left as they are)
    :param float seuil:
        Value for the p-value over which we merge modalities
    """
    X_grouped = X.copy()

    # Necessary to be able to compare (and unique) categories
    X_grouped[var] = as_categories(X_grouped[var])
    initial_categories = np.unique(X_grouped[var])

    p_value = 1
    # We want a minimum of two categories, otherwise the variable becomes useless
    while group and p_value > seuil and len(np.unique(X_grouped[var])) > 2:
        # Counts the number of 0/1 by modality
        freq_table = X_grouped.groupby([var, var_predite]).size().reset_index(name=0)
        # All the combinations of values
        liste_paires_modalities = list(itertools.combinations(np.unique(X_grouped[var]), 2))
        # Chi2 between the pairs
        liste_chi2 = [chi2_test([freq_table.loc[np.isin(freq_table[var], pair[0]), 0],
                                 freq_table.loc[np.isin(freq_table[var], pair[1]), 0]]) for pair in
                      liste_paires_modalities]
        p_value = max(liste_chi2)
        if p_value > seuil and len(np.unique(X_grouped[var])) > 1:
            # Updates the modality to the new concatenated value
            merged_pair = liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))]
            X_grouped.loc[np.isin(X_grouped[var], merged_pair), var] = merged_pair[0] + ' - ' + merged_pair[1]
            logger.debug(f"Feature {var} - levels merged: {merged_pair}")
        else:
            break
    new_categories = np.unique(X_grouped[var])

    # Dictionary of the correspondence old category : merged category
    dico_regroupement = {}
    for regrouped_cate in new_categories:
        for cat in regrouped_cate.split(' - '):
            dico_regroupement[cat] = regrouped_cate

    if group:
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
    x_discrete = np.ones(len(x), dtype=int)
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


def categorie_data_bin_train(data: pd.DataFrame, var_cible, categorical=None, discretize=True, group=True):
    """
    Binarise the categorical variables (after merging categories) and standardise the remaining numerical ones.
    Returns the data, the column labels, the encoder, the merged categories, the cuts of the discretised
    features, the scaler and the number of numerical columns left as is

    :param pandas.DataFrame data: data with some variables being categorical
    :param str var_cible: name of target
    :param list categorical: list of categorical features' names
    :param bool discretize: whether to discretize continuous features
    :param bool group: whether to merge the levels of categorical features
    """
    categorical = [] if categorical is None else list(categorical)
    X = data.copy()
    to_change = []
    merged_cat = {}
    for column in categorical:
        if column in X.columns:
            to_change.append(column)
            X[column] = as_categories(X[column])
            X, dico = grouping(X, column, var_cible, group)
            merged_cat[column] = dico

    discret_cat = {}
    if discretize:
        for column in X.columns:
            if column not in categorical and column != var_cible:
                to_change.append(column)
                X[column] = X[column].astype(np.float64)
                X, binning = discretize_feature(X, column, var_cible)
                discret_cat[column] = binning

    if var_cible in X.columns:
        X = X.drop(columns=[var_cible])

    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = pd.DataFrame(index=range(len(X)))
    labels_cat = []
    if to_change:
        X_cat = pd.DataFrame(enc.fit_transform(X[to_change]))
        labels_cat = list(enc.get_feature_names_out())

    # Used when we had decided to keep the continuous numerical variables
    X_num = X.drop(columns=to_change)
    labels_num = list(X_num.columns)
    for column in labels_num:
        if not pd.api.types.is_numeric_dtype(X_num[column]):
            X_num[column] = X_num[column].astype(np.int32)
    scaler = StandardScaler()
    X_num_transformed = pd.DataFrame(scaler.fit_transform(X_num)) if labels_num else pd.DataFrame(
        index=range(len(X)))

    X_train = pd.concat([X_num_transformed, X_cat.reset_index(drop=True)], axis=1, ignore_index=True)
    return X_train, [*labels_num, *labels_cat], enc, merged_cat, discret_cat, scaler, len(labels_num)


def categorie_data_bin_test(data_val: pd.DataFrame, enc: OneHotEncoder, scaler: StandardScaler, merged_cat: dict,
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
    categorical = [] if categorical is None else list(categorical)
    X_val = data_val.copy()
    # Same column order as in training, otherwise the encoder (which checks feature names) complains
    to_change = [column for column in categorical if column in X_val.columns]
    for column in to_change:
        X_val[column] = as_categories(X_val[column])
    # Merging the categories
    X_val = X_val.replace(merged_cat)
    if discretize:
        for column in X_val.columns:
            if column not in categorical:
                to_change.append(column)
                X_val[column] = X_val[column].astype(np.float64)
                X_val = apply_discretization(X_val, column, discret_cat[column])

    if to_change:
        X_val_cat = pd.DataFrame(enc.transform(X_val[to_change]))
    else:
        X_val_cat = pd.DataFrame(index=range(len(X_val)))

    X_val_num = X_val.drop(columns=to_change)
    for column in X_val_num.columns:
        if not pd.api.types.is_numeric_dtype(X_val_num[column]):
            X_val_num[column] = X_val_num[column].astype(np.int32)
    if X_val_num.shape[1]:
        X_val_num = pd.DataFrame(scaler.transform(X_val_num))
    else:
        X_val_num = pd.DataFrame(index=range(len(X_val)))

    # Need to reset the index for the concat to work well (when not same index)
    return pd.concat([X_val_num, X_val_cat.reset_index(drop=True)], axis=1, ignore_index=True)


def create_reduction_matrix(reductions, clustered, labels, chi0, target):
    # Matrix of the changes of chi2 for every pair grouped
    for count1, value1 in enumerate(labels):
        # 0, modality0 ...
        for count2, value2 in enumerate(labels[count1 + 1:], start=count1 + 1):
            # count1+1, modality7 ...
            if count1 != count2:
                # New matrix once we have merged the two
                contingency_matrix = metrics.cluster.contingency_matrix(
                    clustered.replace({value1: value2}), target)
                # stat chi2, p-value, degree of freedom, expected frequencies
                g, p, dof, expctd = chi2_contingency(contingency_matrix, lambda_="log-likelihood")
                reductions[count1, count2] = "{:.2e}".format((1 - g / chi0))
    return reductions


def update_reduction_matrix(df, var, var_predite, matrix, clustered, ind, chi0):
    # Update reduction matrix after finding the two categories that will impact the least our chi2
    target = df[var_predite]
    labels = df[var].unique()
    # Replace the second category by the first in the categorical variable to get the new chi2
    clustered = clustered.replace({labels[ind[1]]: labels[ind[0]]})
    # Replace all the values of chi2 related to ind[0] (new index for the merged category)
    for count, value in enumerate(labels):
        contingency_matrix = metrics.cluster.contingency_matrix(
            clustered.replace({value: labels[ind[0]]}), target)
        g, p, dof, expctd = chi2_contingency(contingency_matrix, lambda_="log-likelihood")
        # Fills in (in the good half) the new values
        if count > ind[0]:
            matrix[ind[0], count] = "{:.2e}".format((1 - g / chi0))
        if count < ind[0]:
            matrix[count, ind[0]] = "{:.2e}".format((1 - g / chi0))
    return matrix, clustered


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
    df[var] = as_categories(df[var])
    # Get the modalities
    labels = df[var].unique()
    clustered = df[var].copy()
    # Square matrix (each line/column = a category), each cell represents the impact on chi2 if we merge the row &
    # column categories
    reductions = np.empty((len(labels), len(labels)))
    reductions[:] = np.nan
    # List of index of categories merged to another category, then ignored in the reduction matrix
    forgotten_rows = []
    target = df[var_predite]
    clusters = [*range(0, len(labels), 1)]
    # Clusters is a list of lists that will contain the new groups
    clusters = [[i] for i in clusters]
    contingency_matrix = metrics.cluster.contingency_matrix(df[var], target)
    # Get the inital chi2 for the impact of grouping modalities
    chi0, p, dof, expctd = chi2_contingency(contingency_matrix, lambda_="log-likelihood")
    reductions = create_reduction_matrix(reductions, clustered, labels, chi0, target)
    # keep merging until we reach the number of desired groups
    while len(clusters) > num_clusters:
        # Get the minimum chi2 indices in the reduction matrix without taking into account the forgotten rows
        # Smallest reduction = biggest chi2 test statistic = most dependant variables
        min_chi = np.nanmin(np.delete(np.delete(reductions, forgotten_rows, 0), forgotten_rows, 1))
        ind = tuple(np.argwhere(reductions == min_chi)[0])
        # For the indices of the min chi2, add the second index to forgetten rows (ind[1] now be ind[0]
        forgotten_rows.append(ind[1])
        # update the values of chi2 in the reduction matrix after merging the indices in the clustered column
        reductions, clustered = update_reduction_matrix(df, var, var_predite, reductions, clustered, ind, chi0)
        # And finally merging them in the clusters variable
        cluster1 = next(sublist for sublist in clusters if ind[0] in sublist)
        cluster2 = next(sublist for sublist in clusters if ind[1] in sublist)
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

    df = df.replace({var: dico_regroupement})
    return df, dico_regroupement


_SENTINEL_VALUES = [99999, 99999.99, 999999.0, 9.999999999999E10, 99999999999.99, 999, 'NR']
# Columns / target of the in-house credit datasets the scripts in ``scripts/`` were written for
_LEGACY_DROPPED = ["Defaut_12_Mois_contagion"]
_LEGACY_BINARY_FLAGS = [
    "DAV_Null_INDIC_TIERS_CONTENTIEUX", "DAV_Null_INDIC_TIERS_NEIERTZ", "DAV_Null_INDIC_PERS_FICP",
    "DAV_Null_PERS_SAISIE_ATTRIB", "DAV_Null_INDIC_PERS_TUTELLE", "DAV_Null_INDIC_PERS_CURATELLE",
    "DAV_Null_INDIC_PERS_NEIERTZ", "DAV_Null_INDIC_PERS_REDRST_JUDIC", "DAV_Null_INDIC_PERS_CONSTIT_STE",
    "DAV_Null_INDIC_PERS_LIQUID_JUDIC"]


def extreme_values(data: pd.DataFrame, missing: bool = False, legacy_schema: bool = False) -> pd.DataFrame:
    """
    Deals with extreme values (ex : NaN, or not filled): sentinel values (99999, 999, 'NR'...) become missing and
    missing values become 0 (or a "missing" level for non numerical columns).
    Creates (or not) a column signaling which values were missing

    :param pandas.DataFrame data: data to clean
    :param bool missing: whether or not to create a column signaling missing values
    :param bool legacy_schema: also recode the binary flags of the in-house credit datasets
    :rtype: pandas.DataFrame
    """
    data = data.replace(to_replace=_SENTINEL_VALUES, value=np.nan)

    if missing:
        # Creation de colonnes de variables présentes/absentes
        for column in list(data.columns):
            values = data[column]
            if values.isna().any():
                data.insert(0, "Exists_" + str(column), (~values.isna()).astype(int).to_numpy())

    # Missing numerical values become 0, missing levels of categorical features a level of their own
    numerical = [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]
    data = data.fillna({column: 0 if column in numerical else "missing" for column in data.columns})

    if legacy_schema:
        flags = [column for column in _LEGACY_BINARY_FLAGS if column in data.columns]
        data = data.replace({column: {'0': 0, '1': 1} for column in flags})

    return data


def traitement_val(data: pd.DataFrame, enc: OneHotEncoder, scaler: StandardScaler, merged_cat: dict,
                   discret_cat: dict, categorical: list, discretize: bool, clean_extremes: bool = False,
                   legacy_schema: bool = False) -> pd.DataFrame:
    """
    Applies to new data the treatment (extreme values, categorical variables, normalisation) learned by
    :func:`traitement_train`

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
    :param list categorical: list of categorical features' names
    :param bool discretize:
        Whether numerical features were discretized
    :param bool clean_extremes: whether to replace sentinel values (see :func:`extreme_values`)
    :param bool legacy_schema: drop / recode the columns specific to the in-house credit datasets
    :return: the treated data
    :rtype: pandas.DataFrame
    """
    X_val = data.copy()
    if legacy_schema:
        X_val = X_val.drop(columns=[column for column in _LEGACY_DROPPED if column in X_val.columns])
    if clean_extremes:
        X_val = extreme_values(X_val, missing=False, legacy_schema=legacy_schema)
    return categorie_data_bin_test(X_val, enc, scaler, merged_cat, discret_cat, categorical, discretize)


def traitement_train(data: pd.DataFrame, target: str, categorical=None, discretize=False, group=False,
                     clean_extremes: bool = False, legacy_schema: bool = False) -> tuple:
    """
    Treats the training data (extreme values, categorical variables, normalisation)

    :param pandas.Dataframe data: data with some variables being categorical
    :param str target: variable cible
    :param list categorical: list of categorical features' names
    :param bool discretize: whether to discretize continuous features
    :param bool group: whether to merge the levels of categorical features
    :param bool clean_extremes: whether to replace sentinel values (see :func:`extreme_values`)
    :param bool legacy_schema: recode the target / flags specific to the in-house credit datasets
    :return: processed data, labels of columns, encoder for categorical data, scaler for numerical data, merged
        categories, discretisation cuts and number of numerical columns
    :rtype: tuple
    """
    X = data.copy()
    if legacy_schema and target in X.columns and X[target].dtype == "object":
        X[target] = X[target].replace({"N": 0, "O": 1})
    if clean_extremes:
        X = extreme_values(X, missing=False, legacy_schema=legacy_schema)
    X, labels, enc, merged_cat, discrete_cat, scaler, len_col_num = categorie_data_bin_train(
        X,
        var_cible=target,
        categorical=categorical,
        discretize=discretize,
        group=group)
    return X, labels, enc, scaler, merged_cat, discrete_cat, len_col_num


class Processing:
    """
    Treatment of the features of a segment: merging of the levels of categorical features, discretisation of the
    numerical ones (optional), one-hot encoding and standardisation.

    :param str target: name of the target column
    :param bool discretize: whether to discretise numerical features (MDLP) before one-hot encoding them
    :param bool group: whether to merge the levels of categorical features (chi2 independence test)
    :param float merge_threshold: p-value over which the levels of a categorical feature are merged
    :param bool clean_extremes: whether to replace sentinel values (99999, 999...) by 0
    """
    def __init__(self, target: str, discretize: bool = False, group: bool = False, merge_threshold: float = 0.2,
                 clean_extremes: bool = False):
        self.target = target
        self.group = group
        self.labels, self.enc, self.scaler, self.merged_cat, self.discrete_cat = [None] * 5
        self.discretize = discretize
        self.num_cols = []
        self.cat_cols = []
        self.merge_threshold = merge_threshold
        self.clean_extremes = clean_extremes
        self.X_train = None

    def fit(self, X: pd.DataFrame, categorical: list = None):
        self.cat_cols = [] if categorical is None else list(categorical)
        # Check if all categorical are inside
        for col in self.cat_cols:
            if col not in X.columns:
                msg = f"Column {col} specified in argument 'categorical' not present."
                logger.error(msg)
                raise ValueError(msg)
        # Calculate remaining columns
        self.num_cols = [col for col in X.columns.to_list() if col not in self.cat_cols + [self.target]]
        # Application sur train
        X, self.labels, self.enc, self.scaler, self.merged_cat, self.discrete_cat, len_col_num = traitement_train(
            data=X, target=self.target, categorical=self.cat_cols, discretize=self.discretize,
            group=self.group, clean_extremes=self.clean_extremes
        )
        if len_col_num != len(self.num_cols) - (len(self.discrete_cat) if self.discretize else 0):
            raise ValueError("Inconsistent number of numerical columns after treatment.")
        self.X_train = X
        return self

    def fit_transform(self, X: pd.DataFrame, categorical: list = None):
        return self.fit(X, categorical).X_train

    def transform(self, X: pd.DataFrame):
        X_test = X.copy()
        if self.target in X_test.columns.to_list():
            X_test = X_test.drop(columns=[self.target])
        return traitement_val(data=X_test, enc=self.enc, scaler=self.scaler,
                              merged_cat=self.merged_cat, discret_cat=self.discrete_cat,
                              categorical=self.cat_cols, discretize=self.discretize,
                              clean_extremes=self.clean_extremes)
