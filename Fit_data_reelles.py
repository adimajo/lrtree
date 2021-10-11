import numpy as np
import glmtree
from glmtree.predict import score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import tree
import matplotlib.pyplot as plt


def clean_data(data):
    """Retire les colonnes inutiles ou qui sont déjà des prédictions
    Gère les valeurs manquantes en remplacant par 0, et créant une colonne pour absence/présence de données"""
    data["Defaut_12_Mois_contagion"] = data["Defaut_12_Mois_contagion"].replace(["N", "O"],[int(0), int(1)])

    #A ne pas utiliser
    Pas_utiles=['ACTIVITE_ECO_ENTREP_APE', 'ACTIVITE_ECO_ETABLIST_NAF', 'CATEGORIE_JURIDIQUE', 'CLASSE_NOTA_TIERS', 'CODE_AGENCE', 'CODE_COMMUNE_ADRESSE', 'CODE_RESIDENT_FISCALITE',
                'CRED_Null_BIEN_FINANCE', 'CRED_Null_BIEN_FINANCE_CONSO', 'CRED_Null_BIEN_FINANCE_HAB', 'CRED_Null_CATEGORIE_PRET', 'CRED_Null_CATEGORIE_PRET_CONSO',
                'CRED_Null_CATEGORIE_PRET_HAB', 'CRED_Null_CATEGORIE_PRET_REVLG', 'CRED_Null_DEST_FINANCE', 'CRED_Null_DEST_FINANCE_CONSO', 'CRED_Null_DEST_FINANCE_HAB',
                'CRED_Null_DOMAINE_INTERV', 'CRED_Null_DOMAINE_INTERV_CONSO', 'CRED_Null_DOMAINE_INTERV_HAB', 'CRED_Null_NB_EC_DEF_HS_CONTAG', 'CSP_INITIALE', 'DATE_ARRETE',
                'DATE_CLOS_GROUPE_RISQUE', 'DATE_CREATION_ENTREPRISE', 'DATE_CREATION_ETABLIST', 'DATE_CREATION_TIERS', 'DATE_DEBUT_OPT', 'DATE_INSTAL_DIRIGEANT', 'DATE_MODIF_GROUPE_RISQUE',
                'DATE_MODIF_SEGMENT_GR', 'DATE_MODIF_SEGMENT_TIERS', 'DATE_NAISSANCE_DIRIGEANT', 'DATE_NAISSANCE_TIERS', 'DATE_SORTIE_DEFAUT_TIERS', 'DATE_SURVENUE_DEFAUT_New', 'DATE_SURVENUE_DEFAUT_TIERS',
                'Date_fin_option', 'Defaut_12_Mois_No_contagion', 'Defaut_12_mois_New', 'FIN_TYPO_DFT_SCEN2', 'GRADE_ACTUEL','GRADE_PD_PA_SIM_New', 'IDENTIFIANT_COMMERCIAL', 'ID_CR',
                'ID_GROUPE_RISQUE', 'ID_TIERS', 'INDIC_DA_NON_PERM_TIERS', 'INDIC_DA_PERMANENT_TIERS', 'INDIC_GPT', 'INDIC_PERS_DECES', 'INDIC_PERS_GLOBAL_12M', 'INDIC_PERS_GLOBAL_3M', 'INDIC_PERS_GLOBAL_6M', 'INDIC_PERS_GLOBAL_M',
                'INDIC_PERS_INTERDIT_BANC', 'INDIC_TIERS_DOUTEUX', 'INDIC_TIERS_GLOBAL_12M', 'INDIC_TIERS_GLOBAL_3M', 'INDIC_TIERS_GLOBAL_6M', 'INDIC_TIERS_GLOBAL_M', 'NBPP_LUC_GPT', 'NBPP_TOT_GPT', 'NB_JOURS_DPS_DEB_DEF_TIERS', 'NB_JOURS_DPS_FIN_DEF_TIERS',
                'NOTE_MIN_CPTABLE_AGRI', 'NOTE_MIN_CPTABLE_PRO', 'NOTE_MOY_CPTABLE_AGRI', 'NOTE_MOY_CPTABLE_PRO', 'NOTE_MOY_POND_CPTABLE_AGR', 'NOTE_MOY_POND_CPTABLE_PRO', 'NUMERO_SIREN', 'RETOUR_SAIN_SCEN2', 'SEGMENT_NOTATION', 'SITUATION_FAMILIALE',
                'SIT_PARTICULIERE_PAR', 'SIT_PARTICULIERE_PRO', 'TIERS_CLOS_M', 'TYPE_TIERS', 'Top_Forborne_sain', 'Top_defaut_contagion', 'Top_defaut_no_contagion', 'Typo_DFT', 'defaut_12_mois_tiers_old_def', 'perimetre_modele', 'top_exclu']

    data = data.drop(Pas_utiles, axis=1)

    #Valeurs répétitives et trop corrélées
    Repetitions=["Top_incident_New_def_M", "Top_incident_New_def_3M", "Top_incident_New_def_12M", "Top_expo_M", "Top_Expo_3M", "Top_Expo_12M", "Top_Inact_M", "Top_Inact_3M", "Top_Inact_12M", "Top_DETTES_CT_M", "Top_DETTES_CT_3M", "Top_DETTES_CT_12M",
                 "Top_Inact_inv_M", "Top_Inact_inv_3M", "Top_Inact_12M", "Top_Inact_M", "Top_Inact_3M", "Top_Inact_inv_12M", "Top_option_M", "Top_option_3M", "Top_option_12M", "Top_activite_1M", "Top_activite_3M", "Top_activite_12M", "flag_recent_6", "flag_recent_12", "flag_recent_24",
                 "DAV_Null_SOLDE_MOYEN_M", "DAV_Null_SOLDE_MOYEN_6M", "DAV_Null_SOLDE_MOYEN_12M", "DAV_Null_SOLDE_MOY_CREDIT_3M", "DAV_Null_SOLDE_MOY_CREDIT_12M", "DAV_Null_SOLDE_MINIMUM_M", "DAV_Null_SOLDE_MINIMUM_3M", "DAV_Null_SOLDE_MINIMUM_12M",
                 "DAV_Null_NBTOT_JOURS_DEBIT_3M", "DAV_Null_NBTOT_JOURS_DEBIT_12M", "DAV_Null_NBPREL_ORG_FINANC_3M", "DAV_Null_NBPREL_ORG_FINANC_12M", "DAV_Null_NB_TOT_PMT_CARTE_M", "DAV_Null_NB_TOT_PMT_CARTE_3M", "DAV_Null_NB_TOT_PMT_CARTE_12M",
                 "DAV_Null_NB_TOT_JOURS_DEP_M", "DAV_Null_NB_TOT_JOURS_DEP_3M", "DAV_Null_NB_TOT_JOURS_DEP_12M", "DAV_Null_NB_REFUS_PAIEMT_M", "DAV_Null_NB_REFUS_PAIEMT_3M", "DAV_Null_NB_REFUS_PAIEMT_12M", "DAV_Null_NB_PMT_CARTE_DD_3M", "DAV_Null_NB_PMT_CARTE_DD_12M",
                 "DAV_Null_NB_OPE_DEBIT_M", "DAV_Null_NB_OPE_DEBIT_3M", "DAV_Null_NB_OPE_DEBIT_12M", "DAV_Null_NB_OPE_CREDIT_M", "DAV_Null_NB_OPE_CREDIT_3M", "DAV_Null_NB_OPE_CREDIT_12M", "DAV_Null_NB_DAV_DEP_M", "DAV_Null_NB_DAV_DEP_3M", "DAV_Null_NB_DAV_DEP_12M",
                 "DAV_Null_NB_DAV_DEBIT_M", "DAV_Null_NB_DAV_DEBIT_3M", "DAV_Null_NB_DAV_DEBIT_12M", "DAV_Null_MNT_REFUS_PAIEMENT_M", "DAV_Null_MNT_REFUS_PAIEMT_3M", "DAV_Null_MNT_REFUS_PAIEMT_12M", "DAV_Null_MNT_PREL_ORG_FINAN_M", "DAV_Null_MNT_PREL_ORG_FINA_3M", "DAV_Null_MNT_PRE_ORG_FINA_12M",
                 "DAV_Null_MNT_PMT_CARTE_DD_M", "DAV_Null_MNT_PMT_CARTE_DD_3M", "DAV_Null_MNT_PMT_CARTE_DD_12M", "DAV_Null_MNT_PAIEMT_CARTE_M", "DAV_Null_MNT_PAIEMT_CARTE_3M", "DAV_Null_MNT_PAIEMT_CARTE_12M", "DAV_Null_MAXJOUR_CONS_DEP_3M", "DAV_Null_MAXJOUR_CONS_DEP_12M",
                 "DAV_Null_MAXJOUR_CONS_DBT_3M", "DAV_Null_MAXJOUR_CONS_DBT_12M", "DAV_Null_FLX_CRED_DOM_MOY_3M", "DAV_Null_FLX_CRED_DOM_MOY_12M", "CRED_Null_NBMAX_IMP_12M", "CRED_Null_NBMAX_IMP_12M_CON", "CRED_Null_NBMAX_IMP_12M_HAB", "CRED_Null_NBMAX_IMP_12M_REV",
                 "CRED_Null_NBMAX_IMP_3M", "CRED_Null_NBMAX_IMP_3M_CON", "CRED_Null_NBMAX_IMP_3M_HAB", "CRED_Null_NBMAX_IMP_3M_REV", "CRED_Null_NBFOIS_1_IMP_12M", "CRED_Null_NBFOIS_1_IMP_12M_CON", "CRED_Null_NBFOIS_1_IMP_12M_HAB", "CRED_Null_NBFOIS_1_IMP_12M_REV",
                 "CRED_Null_MAXJOUR_CONS_RET_3M", "CRED_Null_MAXJOUR_CONS_RET_12M"]
    data = data.drop(Repetitions, axis=1)

    #Valeurs extremes quand valeurs manquantes
    extremes=[99999, 99999.99, 999999.0, 9.999999999999E10, 99999999999.99, 999]
    for column in data.columns :
        data[column] = data[column].replace(to_replace=extremes, value=np.NaN)

    # Creation de colonnes de variables présentes/absentes
    for colonne in data.columns:
        values=data[colonne]
        if None in values or pd.isna(values).any():
            new_column_name="Exists_"+str(colonne)
            new_column=[]
            for i in range(len(data)):
                if values.iloc[i]==None or pd.isna(values.iloc[i]) :
                    new_column.append(0)
                else :
                    new_column.append(1)
            data.insert(0, new_column_name, new_column)

    for colonne in data.columns:
        data[colonne] = data[colonne].fillna(value=0)

    return data


def categorie_data_labels(data):
    """Traite le plus simple des variables catégoriques : Assigne un label (0, 1, ...) à chaque catégorie"""

    options = ['Pause Mensualite', 'Modulation Echeance', 'Reduction echeance', 'Suspension echeance', 'Double mensualite', 'autre']
    data["Type_Option"] = data["Type_Option"].replace(options, [1, 2, 3, 4, 5, 6])

    capa_jur = ['00', '01', '02', '03', '04']
    data["CAPACITE_JURIDIQUE"] = data["CAPACITE_JURIDIQUE"].replace(capa_jur, [0, 1, 2, 3, 4])
    data["CAPACITE_JURIDIQUE"] = data["CAPACITE_JURIDIQUE"].astype('int32')

    reg_matr = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '99']
    data["REGIME_MATRIMONIAL"] = data["REGIME_MATRIMONIAL"].replace(reg_matr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    Dav_Null = ["DAV_Null_INDIC_TIERS_CONTENTIEUX", "DAV_Null_INDIC_TIERS_NEIERTZ", "DAV_Null_INDIC_PERS_FICP",
                "DAV_Null_PERS_SAISIE_ATTRIB",
                "DAV_Null_INDIC_PERS_TUTELLE", "DAV_Null_INDIC_PERS_CURATELLE", "DAV_Null_INDIC_PERS_NEIERTZ",
                "DAV_Null_INDIC_PERS_REDRST_JUDIC", "DAV_Null_INDIC_PERS_CONSTIT_STE",
                "DAV_Null_INDIC_PERS_LIQUID_JUDIC"]
    for column in Dav_Null:
        data[column] = data[column].replace(['0', '1'], [0, 1])

    CSP=[0, 'Sans activite', 'Chomeur', 'Etudiant', 'Retraite', 'Agriculteur', 'Artisan', 'Commercant', 'Employes', 'Fonct pub armée', 'Prof intermediaire', 'Profession liberale', 'Cadre', "Chef d'entreprise"]
    data["Regroup_CSP_Initiale"]=data["Regroup_CSP_Initiale"].replace(CSP, [i for i in range(len(CSP))])

    cat_jur=[0, 'Personne Physique', 'Association', 'Cooperative', 'EXPLOITANT AGRICOLE', 'ARTISAN-COMMERCANT', 'PROFESSION LIBERALE', 'Societe', 'Officier public', 'Coll pub et orga', 'Banque et mutuelle', 'Personne Morale']
    data["regroup_categ_juridique"]=data["regroup_categ_juridique"].replace(cat_jur, [i for i in range(len(cat_jur))])

    Int_tiers=[0, 'Logement',  'Agricole', 'Financement part', 'Foncier', 'Professionnel', 'Calamité', 'Divers']
    data["CRED_Null_Group_interv_tiers"]=data["CRED_Null_Group_interv_tiers"].replace(Int_tiers, [i for i in range(len(Int_tiers))])

    data["CRED_Null_Group_interv_Hab"] = data["CRED_Null_Group_interv_Hab"].replace('Logement', 1)
    data["CRED_Null_Group_interv_Conso"] = data["CRED_Null_Group_interv_Conso"].replace('Financement part', 1)

    Bien_finance=[ 0, 'Logement', 'Matériel PRO', 'Energie', 'Terrain', 'Bâtiment PRO', 'Amenagement foncier', 'Part sociale', 'Social', 'Animaux', 'Plant', 'Véhicule', 'Stock Agricole', 'Electronique menager', 'Divers']
    data["CRED_Null_Group_bien_fin_tiers"]=data["CRED_Null_Group_bien_fin_tiers"].replace(Bien_finance, [i for i in range(len(Bien_finance))])
    Bien_finance_hab=[0, 'Logement', 'Matériel PRO', 'Energie', 'Terrain', 'Part sociale', 'Divers']
    data["CRED_Null_Group_bien_fin_Hab"]=data["CRED_Null_Group_bien_fin_Hab"].replace(Bien_finance_hab, [i for i in range(len(Bien_finance_hab))])
    Bien_finance_conso =[0, 'Logement', 'Matériel PRO', 'Energie', 'Social', 'Véhicule', 'Electronique menager', 'Divers']
    data["CRED_Null_Group_bien_fin_Conso"]=data["CRED_Null_Group_bien_fin_Conso"].replace(Bien_finance_conso, [i for i in range(len(Bien_finance_conso))])

    Destination=[0, 'Logement', 'Foncier Logement', 'Matériel PRO', 'Foncier Agricole', 'Tresorerie', 'Bâtiment PRO', 'Animaux', 'Plants', 'Part sociale', 'Calamité', 'Rien', 'Divers']
    data["CRED_Null_Group_Dest_fin_tiers"]=data["CRED_Null_Group_Dest_fin_tiers"].replace(Destination, [i for i in range(len(Destination))])
    Destination_hab=[0, 'Logement', 'Foncier Logement', 'Tresorerie', 'Divers']
    data["CRED_Null_Group_Dest_fin_Hab"]=data["CRED_Null_Group_Dest_fin_Hab"].replace(Destination_hab, [i for i in range(len(Destination_hab))])
    Destination_conso=[0, 'Logement', 'Matériel PRO', 'Tresorerie', 'Rien', 'Divers']
    data["CRED_Null_Group_Dest_fin_Conso"]=data["CRED_Null_Group_Dest_fin_Conso"].replace(Destination_conso, [i for i in range(len(Destination_conso))])

    Cat_pret=[0, 'Immobilier', 'Professionnel', 'Prêt personnel', 'Ressources Propres', 'Agricole', 'Calamite', 'Autre']
    data["CRED_Null_Regroup_CATEG_PRET"]=data["CRED_Null_Regroup_CATEG_PRET"].replace(Cat_pret, [i for i in range(len(Cat_pret))])
    data["CRED_Null_Regroup_CATEG_HAB"]=data["CRED_Null_Regroup_CATEG_HAB"].replace(Cat_pret, [i for i in range(len(Cat_pret))])
    data["CRED_Null_Regroup_CATEG_CONSO"]=data["CRED_Null_Regroup_CATEG_CONSO"].replace(Cat_pret, [i for i in range(len(Cat_pret))])
    data["CRED_Null_Regroup_CATEG_REV"]=data["CRED_Null_Regroup_CATEG_REV"].replace(Cat_pret, [i for i in range(len(Cat_pret))])

    Cat_pro=[0, 'Immobilier', 'Professionnel', 'Prêt personnel', 'Ressources Propres', 'Agricole', 'Calamite', 'Autre', 'Culture céréales', 'Culture et élevage', 'Construction', 'Divers', 'Agriculture', 'Commerce', 'Administratif', 'Culture de la vigne', 'Élevage de porcins', 'Bovins et ruminant', 'Santé', 'Social', 'Autre activite PRO', 'Restauration', 'Banque assurance', 'Vaches laitières', 'Élevage volailles', 'Transport', 'Forêt', 'Industrie', 'Fruits et légumes', 'Energie', 'Aquaculture peche', 'Eau, dechet', 'Etude']
    data["Categ_NAF_Pro_Agri"]=data["Categ_NAF_Pro_Agri"].replace(Cat_pro, [i for i in range(len(Cat_pro))])

    return data

# data = pd.read_pickle("donnees_tiers.pkl")
# data = clean_data(data)
# data.to_pickle("clean_data.pkl")


data = pd.read_pickle("clean_data.pkl")
data = categorie_data_labels(data)

y = data["Defaut_12_Mois_contagion"].to_numpy()

X = data.drop('Defaut_12_Mois_contagion', axis=1)
print(X.shape)

#Retire les varibles constantes, de variance 0
selector = VarianceThreshold()
X=selector.fit_transform(X)
print(X.shape)

#Selection de variables suivant leur dépendence
X_select = SelectPercentile(mutual_info_classif, percentile=10).fit_transform(X, y)
print(X_select.shape)

X_train, X_test, y_train, y_test = train_test_split(X_select, y, random_state = 0)

#Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# modele_regLog = linear_model.LogisticRegression(random_state = 0, solver = 'liblinear', multi_class = 'auto', max_iter=100)
# modele_regLog.fit(X_train,y_train)
# precision = modele_regLog.score(X_test,y_test)
# print(precision)

model = glmtree.Glmtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=10, max_iter=100)
model.fit(X_train, y_train, nb_init=3)
precision=score(model, X_test, y_test)
print(precision)
text_representation = tree.export_text(model.best_link)
print(text_representation)