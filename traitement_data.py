import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import chi2_contingency
from pandas.core.common import flatten
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import itertools


def clean_data(data):
    """Removes the columns that are not useful for the prediction, pr which were already predictions

    :param pandas.Dataframe data:
        Data to clean
    """
    data["Defaut_12_Mois_contagion"] = data["Defaut_12_Mois_contagion"].replace(["N", "O"], [int(0), int(1)])

    # A ne pas utiliser
    Pas_utiles = ['ACTIVITE_ECO_ENTREP_APE', 'ACTIVITE_ECO_ETABLIST_NAF', 'CATEGORIE_JURIDIQUE', 'CLASSE_NOTA_TIERS',
                  'CODE_AGENCE', 'CODE_COMMUNE_ADRESSE', 'CODE_RESIDENT_FISCALITE',
                  'CRED_Null_BIEN_FINANCE', 'CRED_Null_BIEN_FINANCE_CONSO', 'CRED_Null_BIEN_FINANCE_HAB',
                  'CRED_Null_CATEGORIE_PRET', 'CRED_Null_CATEGORIE_PRET_CONSO',
                  'CRED_Null_CATEGORIE_PRET_HAB', 'CRED_Null_CATEGORIE_PRET_REVLG', 'CRED_Null_DEST_FINANCE',
                  'CRED_Null_DEST_FINANCE_CONSO', 'CRED_Null_DEST_FINANCE_HAB',
                  'CRED_Null_DOMAINE_INTERV', 'CRED_Null_DOMAINE_INTERV_CONSO', 'CRED_Null_DOMAINE_INTERV_HAB',
                  'CRED_Null_NB_EC_DEF_HS_CONTAG', 'CSP_INITIALE', 'DATE_ARRETE',
                  'DATE_CLOS_GROUPE_RISQUE', 'DATE_CREATION_ENTREPRISE', 'DATE_CREATION_ETABLIST',
                  'DATE_CREATION_TIERS', 'DATE_DEBUT_OPT', 'DATE_INSTAL_DIRIGEANT', 'DATE_MODIF_GROUPE_RISQUE',
                  'DATE_MODIF_SEGMENT_GR', 'DATE_MODIF_SEGMENT_TIERS', 'DATE_NAISSANCE_DIRIGEANT',
                  'DATE_NAISSANCE_TIERS', 'DATE_SORTIE_DEFAUT_TIERS', 'DATE_SURVENUE_DEFAUT_New',
                  'DATE_SURVENUE_DEFAUT_TIERS',
                  'Date_fin_option', 'Defaut_12_Mois_No_contagion', 'Defaut_12_mois_New', 'FIN_TYPO_DFT_SCEN2',
                  'GRADE_ACTUEL', 'GRADE_PD_PA_SIM_New', 'IDENTIFIANT_COMMERCIAL', 'ID_CR',
                  'ID_GROUPE_RISQUE', 'ID_TIERS', 'INDIC_DA_NON_PERM_TIERS', 'INDIC_DA_PERMANENT_TIERS', 'INDIC_GPT',
                  'INDIC_PERS_DECES', 'INDIC_PERS_GLOBAL_12M', 'INDIC_PERS_GLOBAL_3M', 'INDIC_PERS_GLOBAL_6M',
                  'INDIC_PERS_GLOBAL_M',
                  'INDIC_PERS_INTERDIT_BANC', 'INDIC_TIERS_DOUTEUX', 'INDIC_TIERS_GLOBAL_12M', 'INDIC_TIERS_GLOBAL_3M',
                  'INDIC_TIERS_GLOBAL_6M', 'INDIC_TIERS_GLOBAL_M', 'NBPP_LUC_GPT', 'NBPP_TOT_GPT',
                  'NB_JOURS_DPS_DEB_DEF_TIERS', 'NB_JOURS_DPS_FIN_DEF_TIERS',
                  'NOTE_MIN_CPTABLE_AGRI', 'NOTE_MIN_CPTABLE_PRO', 'NOTE_MOY_CPTABLE_AGRI', 'NOTE_MOY_CPTABLE_PRO',
                  'NOTE_MOY_POND_CPTABLE_AGR', 'NOTE_MOY_POND_CPTABLE_PRO', 'NUMERO_SIREN', 'RETOUR_SAIN_SCEN2',
                  'SEGMENT_NOTATION', 'SITUATION_FAMILIALE',
                  'SIT_PARTICULIERE_PAR', 'SIT_PARTICULIERE_PRO', 'TIERS_CLOS_M', 'TYPE_TIERS', 'Top_Forborne_sain',
                  'Top_defaut_contagion', 'Top_defaut_no_contagion', 'Typo_DFT', 'defaut_12_mois_tiers_old_def',
                  'perimetre_modele', 'top_exclu']
    data = data.drop(Pas_utiles, axis=1)

    rows = ['cohort_concat', 'nb_date_arrete', 'cohort_debut', 'cohort_fin', 'pct_top_defaut', 'Total',
            'AllocProportion',
            'SampleSize', 'ActualProportion', 'SelectionProb', 'SamplingWeight']
    data = data.drop(rows, axis=1)

    return data


def extreme_values(data, Missing=False):
    """Deals with extreme values (ex : NaN, or not filled)
    Creates (or not) a column signaling which values were missing

        :param pandas.Dataframe data:
            Data to clean
        :param Bool Missing:
            Whether or not to create a column signaling missing values
    """
    # Valeurs extremes quand valeurs manquantes
    extremes = [99999, 99999.99, 999999.0, 9.999999999999E10, 99999999999.99, 999, 'NR']
    for column in data.columns:
        data[column].replace(to_replace=extremes, value=np.NaN, inplace=True)

    if Missing:
        # Creation de colonnes de variables présentes/absentes
        for column in data.columns:
            values = data[column]
            if None in values or pd.isna(values).any():
                new_column_name = "Exists_" + str(column)
                new_column = []
                for i in range(len(data)):
                    if values.iloc[i] == None or pd.isna(values.iloc[i]):
                        new_column.append(0)
                    else:
                        new_column.append(1)
                data.insert(0, new_column_name, new_column)

    for column in data.columns:
        data.loc[:,column] = data[column].fillna(value=0)

    Dav_Null = ["DAV_Null_INDIC_TIERS_CONTENTIEUX", "DAV_Null_INDIC_TIERS_NEIERTZ", "DAV_Null_INDIC_PERS_FICP",
                "DAV_Null_PERS_SAISIE_ATTRIB",
                "DAV_Null_INDIC_PERS_TUTELLE", "DAV_Null_INDIC_PERS_CURATELLE", "DAV_Null_INDIC_PERS_NEIERTZ",
                "DAV_Null_INDIC_PERS_REDRST_JUDIC", "DAV_Null_INDIC_PERS_CONSTIT_STE",
                "DAV_Null_INDIC_PERS_LIQUID_JUDIC"]
    for column in Dav_Null:
        if column in data.columns:
            data[column].replace(['0', '1'], [0, 1], inplace=True)

    return data


def categorie_data_labels(data, data_val):
    X = data.copy()
    X_val = data_val.copy()
    categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
                   "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET", "CRED_Null_Group_Dest_fin_Conso",
                   "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers", "CRED_Null_Group_bien_fin_Conso",
                   "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers", "CRED_Null_Group_interv_Conso",
                   "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
                   "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option", "segment"]
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
        if col.dtypes not in ("int32", "float64"):
            X_num[column] = col.astype(np.int32)
            X_val_num[column] = col.astype(np.int32)

    return pd.concat([X_num, X_cat], axis=1), pd.concat([X_val_num, X_val_cat], axis=1)


def categorie_data_bin_train_test(data, data_val):
    X = data.copy()
    X_val = data_val.copy()
    categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
                   "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET", "CRED_Null_Group_Dest_fin_Conso",
                   "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers", "CRED_Null_Group_bien_fin_Conso",
                   "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers", "CRED_Null_Group_interv_Conso",
                   "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
                   "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option", "segment"]
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
        if col.dtypes not in ["int32", "float64"]:
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


def categorie_data_bin_train(data):
    """Traite les variables catégoriques des données en les binarisant (après avoir regroupé certaines modalités)
    Retourne les données traitées, l'encodeur et les labels des colonnes

        :param pandas.Dataframe data:
            Data with some variables being categorical
    """
    X = data.copy()
    categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
                   "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET", "CRED_Null_Group_Dest_fin_Conso",
                   "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers", "CRED_Null_Group_bien_fin_Conso",
                   "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers", "CRED_Null_Group_interv_Conso",
                   "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
                   "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option", "segment"]
    to_change = []
    merged_cat = {}
    for column in categorical:
        if column in X.columns:
            to_change.append(column)
            X.loc[:,column] = X[column].astype(str)
            X, dico = regroupement(X, column, "Defaut_12_Mois_contagion")
            merged_cat[column] = dico

    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = enc.fit_transform(X[to_change])
    labels_cat = enc.get_feature_names_out()
    X_cat = pd.DataFrame(X_cat)

    X_num = X.drop(to_change, axis=1)
    if "Defaut_12_Mois_contagion" in X_num.columns:
        X_num.drop(["Defaut_12_Mois_contagion"], axis=1, inplace=True)
    col_num = X_num.columns
    labels_num = []
    for column in col_num:
        labels_num.append(column)

    for column in X_num.columns:
        col = X_num[column]
        if col.dtypes not in ["int32", "float64"]:
            X_num.loc[:,column] = X_num[column].astype(np.int32)

    # Need to reset the index for the concat to work well
    X_cat = X_cat.reset_index(drop=True)
    X_num = X_num.reset_index(drop=True)

    X_train = pd.concat([X_num, X_cat], axis=1, ignore_index=True)
    labels = [*labels_num, *labels_cat]
    return X_train, labels, enc, merged_cat


def categorie_data_bin_test(data_val, enc, merged_cat):
    """Traite les variables catégoriques des données de test en les binarisant en utilisant l'encodeur utilisé pour binariser les données de train
    Retourne les données traitées

        :param pandas.Dataframe data_val:
            Data with some variables being categorical
        :param sklearn.preprocessing.OneHotEncoder enc:
            OneHotEncoder fitted on the training data
    """
    X_val = data_val.copy()
    categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
                   "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET", "CRED_Null_Group_Dest_fin_Conso",
                   "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers", "CRED_Null_Group_bien_fin_Conso",
                   "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers", "CRED_Null_Group_interv_Conso",
                   "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
                   "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option", "segment"]
    to_change = []
    for column in categorical:
        if column in X_val.columns:
            to_change.append(column)
            X_val.loc[:,column] = X_val[column].astype(str)
    # Merging the categories
    X_val.replace(merged_cat, inplace=True)
    X_val_cat = enc.transform(X_val[to_change])
    X_val_cat = pd.DataFrame(X_val_cat)
    X_val_num = X_val.drop(to_change, axis=1)

    for column in X_val_num.columns:
        col = X_val_num[column]
        if col.dtypes not in ["int32", "float64"]:
            X_val_num.loc[:,column] = X_val_num[column].astype(np.int32)

    # Need to reset the index for the concat to work well (when not same index)
    X_val_cat = X_val_cat.reset_index(drop=True)
    X_val_num = X_val_num.reset_index(drop=True)
    X_test = pd.concat([X_val_num, X_val_cat], axis=1, ignore_index=True)
    return X_test


def traitement_train_val(X, X_val):
    """Traite les données et les données de test en gérant les valeurs extremes, les variables catégoriques et en normalisant
    Retourne les données traitées et les labels des colonnes

        :param pandas.Dataframe X:
            Data with some variables being categorical
        :param pandas.Dataframe X_val:
            Validation data with some variables being categorical
    """
    X = extreme_values(X, Missing=False)
    X_val = extreme_values(X_val, Missing=False)

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


def traitement_val(data, enc, scaler, merged_cat):
    """Traite les données de test en gérant les valeurs extremes, les variables catégoriques et en normalisant
    Retourne les données traitées
        :param pandas.Dataframe data:
            Data with some variables being categorical
        :param sklearn.preprocessing.OneHotEncoder enc:
            OneHotEncoder fitted on the training data
        :param sklearn.preprocessing.StandardScaler scaler:
            StandardScaler fitted on the training data
    """
    X_val = data.copy()
    if "Defaut_12_Mois_contagion" in X_val.columns:
        X_val.drop(["Defaut_12_Mois_contagion"], axis=1, inplace=True)
    X_val = extreme_values(X_val, Missing=False)
    X_val = categorie_data_bin_test(X_val, enc, merged_cat)
    X_val = scaler.transform(X_val)
    return X_val


def traitement_train(data):
    """Traite les données en gérant les valeurs extremes, les variables catégoriques et en normalisant
    Retourne les données traitées, les labels des colonnes, l'encodeur pour les données catégoriques et le scaler

        :param pandas.Dataframe data:
            Data with some variables being categorical
    """

    X = data.copy()
    if "Defaut_12_Mois_contagion" in X.columns:
        X["Defaut_12_Mois_contagion"].replace(["N", "O"], [int(0), int(1)], inplace=True)
    X = extreme_values(X, Missing=False)
    X, labels, enc, merged_cat = categorie_data_bin_train(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, labels, enc, scaler, merged_cat


def green_clust(X, var, var_predite, num_clusters):
    """ GreenClust algorithm to group modalities
        :param pandas.Dataframe X:
            Data with some variables being categorical
        :param str var:
            Column for which we apply the algorithm
        :param str var:
            Column we aim to predict
        :param int num_clusters:
            Number of modalities we want
     """
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
                    reductions[count1, count2] = "{:.2e}".format((1 - g / chi0))
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
                matrix[ind[0], count] = "{:.2e}".format((1 - g / chi0))
            if count < ind[0]:
                matrix[count, ind[0]] = "{:.2e}".format((1 - g / chi0))
        return matrix

    df = X.copy()
    # Necessary to be able to compare (and unique) categories
    df[var] = df[var].astype(str)
    # Get the modalities
    labels = df[var].unique()
    clustered = df[var].copy()
    # Square matrix (each line/column = a category), each cell represents the impact on chi2 if we merge the row & column categories
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

    final_clusters = []
    dico_regroupement = {}
    for group in clusters:
        cluster = []
        for ind in group:
            cluster += [labels[ind]]
        final_clusters += [cluster]
        name_cluster = ' - '.join(cluster)
        for ind in group:
            dico_regroupement[labels[ind]] = name_cluster

    df.replace({var:dico_regroupement}, inplace=True)
    return df, dico_regroupement


def regroupement(X, var, var_predite, seuil=0.05):
    """ Chi2 independence algorithm to group modalities
        :param pandas.Dataframe X:
            Data with some variables being categorical
        :param str var:
            Column for which we apply the algorithm
        :param str var:
            Column we aim to predict
        :param float seuil:
            Value for the p-value other which we merge modalities
     """
    X_grouped = X.copy()

    def chi2_test(liste):
        try:
            return chi2_contingency(liste)[1]
        except Exception:
            # Happens when one modality of the variable has only 0 or only 1
            return 1

    # Necessary to be able to compare (and unique) categories
    X_grouped[var] = X_grouped[var].astype(str)
    initial_categories = np.unique(X_grouped[var])
    p_value = 1
    while p_value > seuil:
        # Counts the number of 0/1 by modality
        freq_table = X_grouped.groupby([var, var_predite]).size().reset_index()
        # Order by the mean of the predicted variable
        modalities = np.unique(X_grouped[var])
        for moda in modalities:
            partial_table = freq_table.iloc[np.in1d(freq_table[var], moda)]

        # All the combinations of values
        liste_paires_modalities = list(itertools.combinations(np.unique(X_grouped[var]), 2))
        # Chi2 between the pairs
        liste_chi2 = [chi2_test([freq_table.iloc[np.in1d(freq_table[var], pair[0]), 2],
                                 freq_table.iloc[np.in1d(freq_table[var], pair[1]), 2]]) for pair in
                      liste_paires_modalities]
        p_value = max(liste_chi2)
        if p_value > 0.05 and len(np.unique(X_grouped[var])) > 1:
            # Updates the modality to the new concatenated value
            X_grouped[var].iloc[
                np.in1d(X_grouped[var], liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))])] = \
                liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))][0] + ' - ' + \
                liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))][1]
            # print('Feature ' + var + ' - levels merged : ' + str(liste_paires_modalities[np.argmax(np.equal(liste_chi2, p_value))]))
        else:
            break
    new_categories = np.unique(X_grouped[var])

    # Dictionary of the correspondence old category : merged category
    dico_regroupement = {}
    for regrouped_cate in new_categories:
        for cat in regrouped_cate.split(' - '):
            dico_regroupement[cat] = regrouped_cate

    print("Feature " + var + " went from ", len(initial_categories), " to ", len(new_categories), " modalities")

    return X_grouped, dico_regroupement
