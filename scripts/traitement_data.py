import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def cacf_data(data,
              dates_to_duration=None,
              dates_to_duration_with_na=None,
              str_fill_na=None,
              num_fill_na=None):
    """

    """
    if num_fill_na is None:
        num_fill_na = ['MT_LOYER', 'MT_CHRG', 'MT_PENS_DU']
    if str_fill_na is None:
        str_fill_na = ['HABIT', 'SITFAM', 'CSP', 'CSPCJ', 'TOP_COEMP', 'CPCL', 'PROD', 'SPROD', 'CPROVS', 'NBENF',
                       'ECJCOE', 'NATB', 'CVFISC', 'grscor2']
    if dates_to_duration_with_na is None:
        dates_to_duration_with_na = ['DEMBA', 'AMEMBC', 'DCLEM', 'AMCIRC']
    if dates_to_duration is None:
        dates_to_duration = ['DNAISS', 'DNACJ']

    def to_decode(bbyte):
        return bbyte.decode('UTF-8')

    def format_to_date(date):
        return np.datetime64(str(date[0:4]) + '-' + str(date[4:6]))

    for col in dates_to_duration:
        data[col].fillna(np.datetime64('2022-01-01'), inplace=True)
        data[col] = (np.datetime64('2022-01-01') - data[col]).dt.days

    for col in dates_to_duration_with_na:
        data[col].fillna(b'202201', inplace=True)
        data[col] = data[col].apply(to_decode)
        data[col].replace(to_replace=['01', '0 0 0 0', '00000000'], value='202201', inplace=True)
        data[col].replace(to_replace=['10190201', '11111101'], value='197001', inplace=True)
        data[col] = data[col].apply(format_to_date)
        data[col] = (np.datetime64('2022-01-01') - data[col]).dt.days

    for col in str_fill_na:
        data[col].fillna(b'0', inplace=True)
        data[col] = data[col].apply(to_decode)

    for col in num_fill_na:
        data[col].fillna(0, inplace=True)

    return data


def gca_data(data: pd.DataFrame, target="Defaut_12_Mois_contagion",
             to_remove=None) -> pd.DataFrame:
    """
    Removes the columns that are not useful for the prediction, pr which were already predictions

    :param pandas.DataFrame data: data to clean
    :param str target: name of the target variable
    :param list to_remove: list of features to drop
    :return: clean data
    :rtype: pandas.DataFrame
    """
    if to_remove is None:
        to_remove = ['ACTIVITE_ECO_ENTREP_APE', 'ACTIVITE_ECO_ETABLIST_NAF', 'CATEGORIE_JURIDIQUE', 'CLASSE_NOTA_TIERS',
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
                     'INDIC_PERS_INTERDIT_BANC', 'INDIC_TIERS_DOUTEUX', 'INDIC_TIERS_GLOBAL_12M',
                     'INDIC_TIERS_GLOBAL_3M',
                     'INDIC_TIERS_GLOBAL_6M', 'INDIC_TIERS_GLOBAL_M', 'NBPP_LUC_GPT', 'NBPP_TOT_GPT',
                     'NB_JOURS_DPS_DEB_DEF_TIERS', 'NB_JOURS_DPS_FIN_DEF_TIERS',
                     'NOTE_MIN_CPTABLE_AGRI', 'NOTE_MIN_CPTABLE_PRO', 'NOTE_MOY_CPTABLE_AGRI', 'NOTE_MOY_CPTABLE_PRO',
                     'NOTE_MOY_POND_CPTABLE_AGR', 'NOTE_MOY_POND_CPTABLE_PRO', 'NUMERO_SIREN', 'RETOUR_SAIN_SCEN2',
                     'SEGMENT_NOTATION', 'SITUATION_FAMILIALE',
                     'SIT_PARTICULIERE_PAR', 'SIT_PARTICULIERE_PRO', 'TIERS_CLOS_M', 'TYPE_TIERS', 'Top_Forborne_sain',
                     'Top_defaut_contagion', 'Top_defaut_no_contagion', 'Typo_DFT', 'defaut_12_mois_tiers_old_def',
                     'perimetre_modele', 'top_exclu', 'cohort_concat', 'nb_date_arrete', 'cohort_debut', 'cohort_fin',
                     'pct_top_defaut', 'Total',
                     'AllocProportion', 'SampleSize', 'ActualProportion', 'SelectionProb', 'SamplingWeight']
    data[target] = data[target].replace(["N", "O"], [int(0), int(1)])
    data = data.drop(to_remove, axis=1)
    return data
