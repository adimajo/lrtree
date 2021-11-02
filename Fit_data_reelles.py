import numpy as np
import glmtree
import pandas as pd
import tikzplotlib
import cProfile, pstats
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import linear_model
import matplotlib.pyplot as plt
from glmtree.fit import fit_parralized
import time

def clean_data(data):
    """Retire les colonnes inutiles ou qui sont déjà des prédictions
    Gère les valeurs manquantes en remplacant par 0, et créant une colonne pour absence/présence de données"""
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

    # Valeurs répétitives et trop corrélées
    # Repetitions = ["Top_incident_New_def_M", "Top_incident_New_def_3M", "Top_incident_New_def_12M", "Top_expo_M",
    #                "Top_Expo_3M", "Top_Expo_12M", "Top_Inact_M", "Top_Inact_3M", "Top_Inact_12M", "Top_DETTES_CT_M",
    #                "Top_DETTES_CT_3M", "Top_DETTES_CT_12M",
    #                "Top_Inact_inv_M", "Top_Inact_inv_3M", "Top_Inact_12M", "Top_Inact_M", "Top_Inact_3M",
    #                "Top_Inact_inv_12M", "Top_option_M", "Top_option_3M", "Top_option_12M", "Top_activite_1M",
    #                "Top_activite_3M", "Top_activite_12M", "flag_recent_6", "flag_recent_12", "flag_recent_24",
    #                "DAV_Null_SOLDE_MOYEN_M", "DAV_Null_SOLDE_MOYEN_6M", "DAV_Null_SOLDE_MOYEN_12M",
    #                "DAV_Null_SOLDE_MOY_CREDIT_3M", "DAV_Null_SOLDE_MOY_CREDIT_12M", "DAV_Null_SOLDE_MINIMUM_M",
    #                "DAV_Null_SOLDE_MINIMUM_3M", "DAV_Null_SOLDE_MINIMUM_12M",
    #                "DAV_Null_NBTOT_JOURS_DEBIT_3M", "DAV_Null_NBTOT_JOURS_DEBIT_12M", "DAV_Null_NBPREL_ORG_FINANC_3M",
    #                "DAV_Null_NBPREL_ORG_FINANC_12M", "DAV_Null_NB_TOT_PMT_CARTE_M", "DAV_Null_NB_TOT_PMT_CARTE_3M",
    #                "DAV_Null_NB_TOT_PMT_CARTE_12M",
    #                "DAV_Null_NB_TOT_JOURS_DEP_M", "DAV_Null_NB_TOT_JOURS_DEP_3M", "DAV_Null_NB_TOT_JOURS_DEP_12M",
    #                "DAV_Null_NB_REFUS_PAIEMT_M", "DAV_Null_NB_REFUS_PAIEMT_3M", "DAV_Null_NB_REFUS_PAIEMT_12M",
    #                "DAV_Null_NB_PMT_CARTE_DD_3M", "DAV_Null_NB_PMT_CARTE_DD_12M",
    #                "DAV_Null_NB_OPE_DEBIT_M", "DAV_Null_NB_OPE_DEBIT_3M", "DAV_Null_NB_OPE_DEBIT_12M",
    #                "DAV_Null_NB_OPE_CREDIT_M", "DAV_Null_NB_OPE_CREDIT_3M", "DAV_Null_NB_OPE_CREDIT_12M",
    #                "DAV_Null_NB_DAV_DEP_M", "DAV_Null_NB_DAV_DEP_3M", "DAV_Null_NB_DAV_DEP_12M",
    #                "DAV_Null_NB_DAV_DEBIT_M", "DAV_Null_NB_DAV_DEBIT_3M", "DAV_Null_NB_DAV_DEBIT_12M",
    #                "DAV_Null_MNT_REFUS_PAIEMENT_M", "DAV_Null_MNT_REFUS_PAIEMT_3M", "DAV_Null_MNT_REFUS_PAIEMT_12M",
    #                "DAV_Null_MNT_PREL_ORG_FINAN_M", "DAV_Null_MNT_PREL_ORG_FINA_3M", "DAV_Null_MNT_PRE_ORG_FINA_12M",
    #                "DAV_Null_MNT_PMT_CARTE_DD_M", "DAV_Null_MNT_PMT_CARTE_DD_3M", "DAV_Null_MNT_PMT_CARTE_DD_12M",
    #                "DAV_Null_MNT_PAIEMT_CARTE_M", "DAV_Null_MNT_PAIEMT_CARTE_3M", "DAV_Null_MNT_PAIEMT_CARTE_12M",
    #                "DAV_Null_MAXJOUR_CONS_DEP_3M", "DAV_Null_MAXJOUR_CONS_DEP_12M",
    #                "DAV_Null_MAXJOUR_CONS_DBT_3M", "DAV_Null_MAXJOUR_CONS_DBT_12M", "DAV_Null_FLX_CRED_DOM_MOY_3M",
    #                "DAV_Null_FLX_CRED_DOM_MOY_12M", "CRED_Null_NBMAX_IMP_12M", "CRED_Null_NBMAX_IMP_12M_CON",
    #                "CRED_Null_NBMAX_IMP_12M_HAB", "CRED_Null_NBMAX_IMP_12M_REV",
    #                "CRED_Null_NBMAX_IMP_3M", "CRED_Null_NBMAX_IMP_3M_CON", "CRED_Null_NBMAX_IMP_3M_HAB",
    #                "CRED_Null_NBMAX_IMP_3M_REV", "CRED_Null_NBFOIS_1_IMP_12M", "CRED_Null_NBFOIS_1_IMP_12M_CON",
    #                "CRED_Null_NBFOIS_1_IMP_12M_HAB", "CRED_Null_NBFOIS_1_IMP_12M_REV",
    #                "CRED_Null_MAXJOUR_CONS_RET_3M", "CRED_Null_MAXJOUR_CONS_RET_12M"]
    # data = data.drop(Repetitions, axis=1)
    return data


def extreme_values(data, Missing=True):
    # Valeurs extremes quand valeurs manquantes
    extremes = [99999, 99999.99, 999999.0, 9.999999999999E10, 99999999999.99, 999, 'NR']
    for column in data.columns:
        data[column].replace(to_replace=extremes, value=np.NaN, inplace=True)

    if Missing:
        # Creation de colonnes de variables présentes/absentes
        for colonne in data.columns:
            values = data[colonne]
            if None in values or pd.isna(values).any():
                new_column_name = "Exists_" + str(colonne)
                new_column = []
                for i in range(len(data)):
                    if values.iloc[i] == None or pd.isna(values.iloc[i]):
                        new_column.append(0)
                    else:
                        new_column.append(1)
                data.insert(0, new_column_name, new_column)

    for colonne in data.columns:
        data[colonne] = data[colonne].fillna(value=0)

    Dav_Null = ["DAV_Null_INDIC_TIERS_CONTENTIEUX", "DAV_Null_INDIC_TIERS_NEIERTZ", "DAV_Null_INDIC_PERS_FICP",
                "DAV_Null_PERS_SAISIE_ATTRIB",
                "DAV_Null_INDIC_PERS_TUTELLE", "DAV_Null_INDIC_PERS_CURATELLE", "DAV_Null_INDIC_PERS_NEIERTZ",
                "DAV_Null_INDIC_PERS_REDRST_JUDIC", "DAV_Null_INDIC_PERS_CONSTIT_STE",
                "DAV_Null_INDIC_PERS_LIQUID_JUDIC"]
    for column in Dav_Null:
        if column in data.columns:
            data[column].replace(['0', '1'], [0, 1], inplace=True)

    return data


def categorie_data_labels2(data):
    """Traitement le plus simple des variables catégoriques : Assigne un label (0, 1, ...) à chaque catégorie
    Vérifie que toutes ces colonnes sont au format int"""

    columns = data.columns
    if "Type_Option" in columns:
        options = ['Pause Mensualite', 'Modulation Echeance', 'Reduction echeance', 'Suspension echeance',
                   'Double mensualite', 'autre']
        data["Type_Option"].replace(options, [1, 2, 3, 4, 5, 6], inplace=True)
    if "CAPACITE_JURIDIQUE" in columns:
        capa_jur = ['00', '01', '02', '03', '04']
        data["CAPACITE_JURIDIQUE"].replace(capa_jur, [0, 1, 2, 3, 4], inplace=True)
        data["CAPACITE_JURIDIQUE"] = data["CAPACITE_JURIDIQUE"].astype('int32')
    if "REGIME_MATRIMONIAL" in columns:
        reg_matr = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '99']
        data["REGIME_MATRIMONIAL"].replace(reg_matr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
    if "Regroup_CSP_Initiale" in columns:
        CSP = [np.NaN, 0, 'Sans activite', 'Chomeur', 'Etudiant', 'Retraite', 'Agriculteur', 'Artisan', 'Commercant',
               'Employes',
               'Fonct pub armée', 'Prof intermediaire', 'Profession liberale', 'Cadre', "Chef d'entreprise"]
        data["Regroup_CSP_Initiale"].replace(CSP, [i for i in range(len(CSP))], inplace=True)
    if "regroup_categ_juridique" in columns:
        cat_jur = [np.NaN, 0, 'Personne Physique', 'Association', 'Cooperative', 'EXPLOITANT AGRICOLE',
                   'ARTISAN-COMMERCANT',
                   'PROFESSION LIBERALE', 'Societe', 'Officier public', 'Coll pub et orga', 'Banque et mutuelle',
                   'Personne Morale']
        data["regroup_categ_juridique"].replace(cat_jur,
                                                [i for i in range(len(cat_jur))], inplace=True)

    Int_tiers = [np.NaN, 0, 'Logement', 'Financement part', 'Agricole', 'Foncier', 'Professionnel', 'Calamité',
                 'Divers']
    if "CRED_Null_Group_interv_tiers" in columns:
        data["CRED_Null_Group_interv_tiers"].replace(Int_tiers,
                                                     [i for i in range(
                                                         len(Int_tiers))], inplace=True)
    if "CRED_Null_Group_interv_Hab" in columns:
        data["CRED_Null_Group_interv_Hab"].replace(Int_tiers, [i for i in range(
            len(Int_tiers))], inplace=True)
    if "CRED_Null_Group_interv_Conso" in columns:
        data["CRED_Null_Group_interv_Conso"].replace(Int_tiers,
                                                     [i for i in range(
                                                         len(Int_tiers))], inplace=True)

    Bien_finance = [np.NaN, 0, 'Logement', 'Matériel PRO', 'Energie', 'Terrain', 'Bâtiment PRO', 'Amenagement foncier',
                    'Part sociale', 'Social', 'Véhicule', 'Animaux', 'Plant', 'Stock Agricole', 'Electronique menager',
                    'Divers']
    if "CRED_Null_Group_bien_fin_tiers" in columns:
        data["CRED_Null_Group_bien_fin_tiers"].replace(Bien_finance,
                                                       [i for i in
                                                        range(
                                                            len(Bien_finance))], inplace=True)
    if "CRED_Null_Group_bien_fin_Hab" in columns:
        data["CRED_Null_Group_bien_fin_Hab"].replace(Bien_finance, [i for i in
                                                                    range(
                                                                        len(Bien_finance))], inplace=True)
    if "CRED_Null_Group_bien_fin_Conso" in columns:
        data["CRED_Null_Group_bien_fin_Conso"].replace(Bien_finance,
                                                       [i for i in
                                                        range(
                                                            len(Bien_finance))], inplace=True)

    Destination = [np.NaN, 0, 'Logement', 'Foncier Logement', 'Tresorerie', 'Matériel PRO', 'Foncier Agricole',
                   'Bâtiment PRO',
                   'Animaux', 'Agricole', 'Plants', 'Part sociale', 'Calamité', 'Rien', 'Divers']
    if "CRED_Null_Group_Dest_fin_tiers" in columns:
        data["CRED_Null_Group_Dest_fin_tiers"].replace(Destination, [i for i in
                                                                     range(
                                                                         len(Destination))], inplace=True)
    if "CRED_Null_Group_Dest_fin_Hab" in columns:
        data["CRED_Null_Group_Dest_fin_Hab"].replace(Destination, [i for i in
                                                                   range(
                                                                       len(Destination))], inplace=True)
    if "CRED_Null_Group_Dest_fin_Conso" in columns:
        data["CRED_Null_Group_Dest_fin_Conso"].replace(Destination, [i for i in
                                                                     range(
                                                                         len(Destination))], inplace=True)

    Cat_pret = [np.NaN, 0, 'Immobilier', 'Professionnel', 'Prêt personnel', 'Ressources Propres', 'Agricole',
                'Calamite',
                'Autre']
    if "CRED_Null_Regroup_CATEG_PRET" in columns:
        data["CRED_Null_Regroup_CATEG_PRET"].replace(Cat_pret, [i for i in range(
            len(Cat_pret))], inplace=True)
    if "CRED_Null_Regroup_CATEG_HAB" in columns:
        data["CRED_Null_Regroup_CATEG_HAB"].replace(Cat_pret,
                                                    [i for i in
                                                     range(len(Cat_pret))], inplace=True)
    if "CRED_Null_Regroup_CATEG_CONSO" in columns:
        data["CRED_Null_Regroup_CATEG_CONSO"].replace(Cat_pret,
                                                      [i for i in range(
                                                          len(Cat_pret))], inplace=True)
    if "CRED_Null_Regroup_CATEG_REV" in columns:
        data["CRED_Null_Regroup_CATEG_REV"].replace(Cat_pret,
                                                    [i for i in
                                                     range(len(Cat_pret))], inplace=True)

    Cat_pro = [np.NaN, 0, 'Immobilier', 'Professionnel', 'Prêt personnel', 'Ressources Propres', 'Agricole', 'Calamite',
               'Autre', 'Culture céréales', 'Culture et élevage', 'Construction', 'Divers', 'Agriculture', 'Commerce',
               'Administratif', 'Culture de la vigne', 'Élevage de porcins', 'Bovins et ruminant', 'Santé', 'Social',
               'Autre activite PRO', 'Restauration', 'Banque assurance', 'Vaches laitières', 'Élevage volailles',
               'Transport', 'Forêt', 'Industrie', 'Fruits et légumes', 'Energie', 'Aquaculture peche', 'Eau, dechet',
               'Etude']
    if "Categ_NAF_Pro_Agri" in columns:
        data["Categ_NAF_Pro_Agri"].replace(Cat_pro, [i for i in range(len(Cat_pro))], inplace=True)

    for column in data.columns:
        col = data[column]
        if col.dtypes not in ("int32", "float64"):
            data[column] = col.astype(np.int32)

    return data

def categorie_data_labels(data, data_val):
    X = data.copy()
    X_val = data_val.copy()
    categorical = ["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO",
                   "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET", "CRED_Null_Group_Dest_fin_Conso",
                   "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers", "CRED_Null_Group_bien_fin_Conso",
                   "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers", "CRED_Null_Group_interv_Conso",
                   "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique",
                   "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option"]
    to_change = []
    X_cat=[]
    X_val_cat=[]
    for column in categorical:
        if column in X.columns:
            to_change.append(column)
            X[column].replace(to_replace=0, value=" ", inplace=True)
            X_val[column].replace(to_replace=0, value=" ", inplace=True)
            enc = LabelEncoder()
            X_val_cat.append(enc.transform(X_val[column]))
            X_cat.append(enc.fit_transform(X[column]))

    X_cat = pd.DataFrame(X_cat)
    X_val_cat = pd.DataFrame(X_val_cat)

    X_num = X.drop(to_change, axis=1)
    X_val_num = X_val.drop(to_change, axis=1)
    for column in X_num.columns:
        col = X_num[column]
        if col.dtypes not in ("int32", "float64"):
            X_num[column] = col.astype(np.int32)
            X_val_num[column] = col.astype(np.int32)

    return pd.concat([X_num, X_cat], axis = 1), pd.concat([X_val_num, X_val_cat], axis = 1)

def categorie_data_bin(data, data_val):
    X=data.copy()
    X_val=data_val.copy()
    categorical=["Categ_NAF_Pro_Agri", "CRED_Null_Regroup_CATEG_REV", "CRED_Null_Regroup_CATEG_CONSO", "CRED_Null_Regroup_CATEG_HAB", "CRED_Null_Regroup_CATEG_PRET", "CRED_Null_Group_Dest_fin_Conso", "CRED_Null_Group_Dest_fin_Hab", "CRED_Null_Group_Dest_fin_tiers", "CRED_Null_Group_bien_fin_Conso", "CRED_Null_Group_bien_fin_Hab", "CRED_Null_Group_bien_fin_tiers", "CRED_Null_Group_interv_Conso", "CRED_Null_Group_interv_Hab", "CRED_Null_Group_interv_tiers", "regroup_categ_juridique", "Regroup_CSP_Initiale", "REGIME_MATRIMONIAL", "CAPACITE_JURIDIQUE", "Type_Option"]
    to_change=[]
    for column in categorical:
        if column in X.columns:
            to_change.append(column)
            X[column].replace(to_replace=0, value=" ", inplace=True)
            X_val[column].replace(to_replace=0, value=" ", inplace=True)
    enc = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    X_cat=enc.fit_transform(X[to_change])
    X_val_cat=enc.transform(X_val[to_change])
    X_cat=pd.DataFrame(X_cat)
    X_val_cat=pd.DataFrame(X_val_cat)

    X_num=X.drop(to_change, axis=1)
    X_val_num=X_val.drop(to_change, axis=1)

    for column in X_num.columns:
        col = X_num[column]
        if col.dtypes not in ("int32", "float64"):
            X_num[column] = col.astype(np.int32)
            X_val_num[column] = col.astype(np.int32)

    return pd.concat([X_num, X_cat], axis = 1), pd.concat([X_val_num, X_val_cat], axis = 1)

Oups = ["DATE_DEBUT_OPT", "Date_fin_option", "Max_depass_retard_new_def"]
Used = ["TOP_SEUIL_New_def", "Top_Forborne_sain", "NB_Credit", "top_exclu",
        "Top_incident_3M", "Top_incident_6M", "Top_incident_12M", "Top_expo_M",
        "Priorite_Option", "Type_Option", "Top_option", "Top_Debut_option", "Top_Inact_inv_12M", "Top_option_12M",
        "NB_MOIS_CREATION_TIERS", "DAV_Null_SOLDE_MOYEN_M", "DAV_Null_NB_OPE_DEBIT_12M",
        "CRED_Null_NB_JOURS_MAX_RETARD", "INDIC_PERS_INTERDIT_BANC", "DAV_Null_NB_TOT_JOURS_DEP_6M",
        "DAV_Null_CPT_JOURS_CONS_DEP_M", "CRED_Null_NB_JOURS_CONS_RETARD", "CRED_Null_MAXJOUR_CONS_RET_12M",
        "DAV_Null_NB_TOT_JOURS_DEP_12M", "DAV_Null_NB_TOT_JOURS_DEP_M", "DETTES_TOT_FLUX_CRED_TIERS",
        "DAV_Null_MNT_REFUS_PAIEMT_12M", "DETTES_CT_DETTES_TOT_TIERS", "ENCOURS_RETARD_SUP90",
        "DAV_Null_PLAFOND_DA_PERMANENT", "DAV_Null_SOLDE_MINIMUM_12M", "DAV_Null_SOLDE_MINIMUM_M",
        "DAV_Null_NB_TOT_PMT_CARTE_12M", "DAV_Null_NB_REFUS_PAIEMT_M", "DAV_Null_NB_TOT_JOURS_DEP_3M",
        "DAV_Null_SLD_MOY_CREDITEUR_M", "INTERETS_DEBIT_12M_TIERS", "DAV_Null_NB_REFUS_PAIEMT_6M",
        "ENCOURS_RETARD_INF90", "NB_MOIS_CREATION_ENTREP", "DAV_Null_FLX_DBT_NBOPE_DEB_12",
        "DAV_Null_MNT_REFUS_PAIEMENT_M", "DAV_Null_NBTOT_JOURS_DEBIT_6M", "DAV_Null_SOLDE_MOYEN_FLUX_12M",
        "CAPACITE_JURIDIQUE", "EPARGNE_TOTALE", "INTERETS_DAV_FLUX", "ANCIEN_RELATION_G_RISQUE", "EPARGNE_LIVRET_GR",
        "Regroup_CSP_Initiale", "ENGAGEMENTS_HORS_BILAN_GR_Calc", "Categ_NAF_Pro_Agri", "top_DAV", "Top_PP_GPP_Part"]


data = pd.read_pickle("data_app.pkl")
data_val = pd.read_pickle("data_val.pkl")


# X = data[Used].copy()
# X_val=data_val[Used].copy()
# X = extreme_values(X)
# X_val=extreme_values(X_val)

# X_bin, X_bin_val = categorie_data_bin(X, X_val)
# print(X_bin)
# print(X_bin_val)
# X_bin.to_pickle("X_bin.pkl")
# X_bin_val.to_pickle("X_val_bin.pkl")

# X_l, X_val_l = categorie_data_labels(X, X_val)
# print(X_l)
# print(X_val_l)
# X_l.to_pickle("X_label.pkl")
# X_val.to_pickle("X_val_label.pkl")

# X = data[Used].copy()
# X = extreme_values(X)
# X_train = categorie_data_labels(X)
# X_train.to_pickle("X_train.pkl")

# X = data_val[Used].copy()
# X = extreme_values(X)
# X_test = categorie_data_labels(X)
# X_test.to_pickle("X_test.pkl")

X_train = pd.read_pickle("X_bin.pkl")
X_train=X_train[:10000]
print(X_train)


X_test = pd.read_pickle("X_val_bin.pkl")
X_test = X_test.drop("Exists_ENGAGEMENTS_HORS_BILAN_GR_Calc", axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


y = data["Defaut_12_Mois_contagion"].replace(["N", "O"], [0, 1])
y_train  = y.astype(np.int32)
y_train = y_train[:10000]

y = data_val["Defaut_12_Mois_contagion"].replace(["N", "O"], [0, 1])
y = y.astype(np.int32)
y_test = y




time0=time.time()
model=fit_parralized(X_train, y_train, algo='EM', nb_init=1, nb_jobs=1, max_iter=100, tree_depth=4, class_num=10)
time1=time.time()
print("Time_fit :", time1-time0)
y_proba = model.predict_proba(X_test)
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.show()
plt.close()


# if __name__ == "__main__":
#     data = pd.read_pickle("clean_data.pkl")
#     data = categorie_data_labels(data)
#
#     y = data["Defaut_12_Mois_contagion"].to_numpy()
#
#     X = data.drop('Defaut_12_Mois_contagion', axis=1)
#     print(X.shape)
#
#     # Retire les varibles constantes, de variance 0
#     selector = VarianceThreshold()
#     X = selector.fit_transform(X)
#     print(X.shape)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
#     # Normalisation
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     print("Régression logistique :")
#     modele_regLog = linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto',
#                                                     max_iter=100)
#     modele_regLog.fit(X_train, y_train)
#     proba = modele_regLog.predict_proba(X_test)
#     y_proba = [proba[i][1] for i in range(len(proba))]
#     RocCurveDisplay.from_predictions(y_test, y_proba)
#     plt.title("Courbe ROC pour la régression logistique")
#     tikzplotlib.save("ROC_reg_log.tex")
#     plt.show()
#     plt.close()
#     # precision = modele_regLog.score(X_test, y_test)
#
#     print("Arbre de décision :")
#     model_tree = DecisionTreeClassifier(min_samples_leaf=100, random_state=0)
#     model_tree.fit(X_train, y_train)
#     proba = model_tree.predict_proba(X_test)
#     y_proba = [proba[i][1] for i in range(len(proba))]
#     RocCurveDisplay.from_predictions(y_test, y_proba)
#     plt.title("Courbe ROC pour l'arbre de décision")
#     tikzplotlib.save("ROC_decision_tree.tex")
#     plt.show()
#     plt.close()
#     # precision = model_tree.score(X_test, y_test)
#
#     print("Random forest :")
#     model = RandomForestClassifier(n_estimators=500, min_samples_leaf=100, random_state=0)
#     model.fit(X, y)
#     proba = model.predict_proba(X_test)
#     y_proba = [proba[i][1] for i in range(len(proba))]
#     RocCurveDisplay.from_predictions(y_test, y_proba)
#     plt.title("Courbe ROC pour Random Forest")
#     tikzplotlib.save("ROC_rand_forest.tex")
#     plt.show()
#     plt.close()
#     # precision = model.score(X_test, y_test)
#
#     print("Gradient Boosting :")
#     model = GradientBoostingClassifier(min_samples_leaf=100, random_state=0)
#     model.fit(X, y)
#     proba = model.predict_proba(X_test)
#     y_proba = [proba[i][1] for i in range(len(proba))]
#     RocCurveDisplay.from_predictions(y_test, y_proba)
#     plt.title("Courbe ROC pour Gradient Boosting")
#     tikzplotlib.save("ROC_grad_boost.tex")
#     plt.show()
#     plt.close()
#     # precision = model.score(X_test, y_test)
#
#     print("Réseau de neurones :")
#     model = Sequential()
#     model.add(Dense(12, input_dim=472, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=50, batch_size=10)
#     y_proba = model.predict(X_test)
#     RocCurveDisplay.from_predictions(y_test, y_proba)
#     plt.title("Courbe ROC pour le réseau de neurones")
#     tikzplotlib.save("ROC_neural.tex")
#     plt.show()
#     plt.close()
#     # _, precision = model.evaluate(X_test, y_test)

# print("GlmTree SEM :")
# model = glmtree.Glmtree(algo='SEM', test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=10,
#                         max_iter=100)
# model.fit(X_train, y_train, nb_init=1, tree_depth=4)
# y_proba = model.predict_proba(X_test)
# AUC = roc_auc_score(y_test, y_proba)
# RocCurveDisplay.from_predictions(y_test, y_proba)
# plt.title("Courbe ROC pour le Glmtree")
# tikzplotlib.save("ROC_glmtree.tex")
# plt.show()
# plt.close()
# precision = model.precision(X_test, y_test)
