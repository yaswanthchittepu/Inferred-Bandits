import io
import zipfile

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import ipdb, sys, os

def load_drugs_data():
    """
    Load Drug Consumption dataset from drug+consumption+quantified.zip.

    Target variable
    ---------------
    overall_drug_use : most recent drug use across all 18 substances, collapsed
                       to 4 integer classes:
        0 = '10 years'   (CL0–CL1: never or >10 years ago)
        1 = '5 years'    (CL2: last decade)
        2 = 'past year'  (CL3)
        3 = 'past month' (CL4–CL6: last month/week/day)

    Sensitive attributes
    --------------------
    gender_label    : 'M' / 'F'   (raw float < 0 → Male)
    education_label : 'above college' / 'below college'  (threshold 0.45468)

    Notes
    -----
    - All context features (age, country, ethnicity, NEO scores, impulsivity, SS)
      are already z-score normalised in the UCI release; no further scaling needed.
    - Raw gender and education floats are dropped after deriving sensitive labels,
      so the policy cannot reconstruct the sensitive attribute from features.
    """
    path = './data-uci/drug+consumption+quantified.zip'
    columns = [
        'id', 'age', 'gender', 'education', 'country', 'ethnicity', 'nscore',
        'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'alcohol',
        'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc', 'coke', 'crack',
        'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd', 'meth', 'mushrooms',
        'nicotine', 'semer', 'vsa',
    ]
    drug_col_names = columns[13:]

    # Ordinal rank used only to find the most-recent usage across substances
    class_rank = {'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6}
    # Collapse 7 CL levels → 4 bins (consistent with Group Fairness paper)
    class_mappings = {
        'CL0': '10 years', 'CL1': '10 years',
        'CL2': '5 years',
        'CL3': 'past year',
        'CL4': 'past month', 'CL5': 'past month', 'CL6': 'past month',
    }

    data = pd.read_csv(path, names=columns, index_col='id')

    # ── Sensitive attributes ───────────────────────────────────────────────
    # gender: encoded float, -0.48246 = Male, 0.48246 = Female
    data['gender_label']    = np.where(data['gender'] < 0, 1, 0)
    # education: threshold 0.45468 separates below/above university level
    data['education_label'] = np.where(data['education'] >= 0.45468, 1, 0)
    # Drop raw floats — they directly encode the sensitive attributes
    data.drop(columns=['gender', 'education'], inplace=True)

    # ── Target: most-recent drug use across all substances ─────────────────
    data['overall_drug_use'] = data[drug_col_names].apply(
        lambda row: max(row, key=lambda x: class_rank[x]), axis=1
    )

    is_verified = data.apply(lambda row: row['overall_drug_use'] in row[drug_col_names].values, axis=1)
    if not is_verified.all():
        print(f"Warning: {(~is_verified).sum()} rows failed verification.")
    else:
        print("Verification passed: All overall_drug_use labels exist in source drug columns.")

    # Collapse all drug columns to 4-bin labels, then drop them
    for drug in drug_col_names + ['overall_drug_use']:
        data[drug] = data[drug].map(class_mappings)
    data.drop(columns=drug_col_names, inplace=True)

    # Map text bins → integer indices
    class_names = ['10 years', '5 years', 'past year', 'past month']
    data['overall_drug_use'] = data['overall_drug_use'].map(
        {name: i for i, name in enumerate(class_names)}
    )

    meta = {
        'target_col':     'overall_drug_use',
        'sensitive_cols': ['gender_label', 'education_label'],
        'class_names':    class_names,
        'idx_to_class':   {i: name for i, name in enumerate(class_names)},
    }
    return data, meta

def load_student_data(subject='por'):
    """
    Load Student Performance dataset from student+performance.zip.

    subject : 'mat' (Mathematics), 'por' (Portuguese), or 'both' (concatenated).
              Defaults to 'por' (Portuguese only) to avoid the student-overlap
              issue — 382 students appear in both mat and por files.

    Target variable
    ---------------
    G3 (0-20) is binned into 5 ordered classes:
        'fail'       : 0-9
        'sufficient' : 10-11
        'satisfact'  : 12-14
        'good'       : 15-16
        'excellent'  : 17-20


    Sensitive attribute
    -------------------
    parental_edu_label : 'above college' if either parent has higher education
                         (Medu==4 or Fedu==4), else 'below college'.

    Notes
    -----
    G1 and G2 (intermediate grades) are dropped to avoid data leakage.
    All categorical features are numerically encoded; multi-class nominals
    are one-hot encoded.
    """

    path = './data-uci/student+performance.zip'

    with zipfile.ZipFile(path) as outer:
        with outer.open('student.zip') as ib:
            with zipfile.ZipFile(io.BytesIO(ib.read())) as inner:
                dfs = []
                if subject in ('mat', 'both'):
                    dfs.append(pd.read_csv(inner.open('student-mat.csv'), sep=';'))
                if subject in ('por', 'both'):
                    dfs.append(pd.read_csv(inner.open('student-por.csv'), sep=';'))

    df = pd.concat(dfs, ignore_index=True)

    # Drop intermediate grades (leakage)
    df = df.drop(columns=['G1', 'G2'])

    # ── Sensitive attribute ────────────────────────────────────────────────
    # Medu / Fedu scale: 0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher
    # parental_edu_label : 1 = above college (either parent has higher edu), 0 = below college
    df['parental_edu_label'] = ((df['Medu'] == 4) | (df['Fedu'] == 4)).astype(int)
    # Drop raw parental education scores — they directly encode the sensitive
    # attribute, so keeping them would let a policy reconstruct it from features.
    df = df.drop(columns=['Medu', 'Fedu'])

    # ── Target: bin G3 using Portuguese/French grading scheme (5 classes) ──
    # 0=fail (0-9) | 1=sufficient (10-11) | 2=satisfactory (12-13) | 3=good (14-15) | 4=excellent (16-20)
    bins   = [-1, 9, 11, 13, 15, 20]
    labels = [0, 1, 2, 3, 4]
    df['grade_bin'] = pd.cut(df['G3'], bins=bins, labels=labels).astype(int)
    df = df.drop(columns=['G3'])

    # ── Feature encoding ──────────────────────────────────────────────────
    # Binary yes/no → 1/0
    for col in ['schoolsup', 'famsup', 'paid', 'activities',
                'nursery', 'higher', 'internet', 'romantic']:
        df[col] = (df[col] == 'yes').astype(int)

    # Binary nominal → 0/1
    df['school']   = (df['school']   == 'MS').astype(int)   # GP=0, MS=1
    df['sex']      = (df['sex']      == 'M').astype(int)    # F=0,  M=1
    df['address']  = (df['address']  == 'U').astype(int)    # R=0,  U=1
    df['famsize']  = (df['famsize']  == 'GT3').astype(int)  # LE3=0, GT3=1
    df['Pstatus']  = (df['Pstatus']  == 'T').astype(int)    # A=0,  T=1

    # Multi-class nominal → one-hot
    df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'],
                        drop_first=False, dtype=int)

    # ── Normalize numeric columns (z-score) ───────────────────────────────
    # Ordinal/continuous features with varying scales; binary columns are
    # already in {0,1} and do not need normalization.
    numeric_cols = ['age', 'traveltime', 'studytime', 'failures',
                    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    class_names = ['fail', 'sufficient', 'satisfactory', 'good', 'excellent']
    meta = {
        'target_col':     'grade_bin',
        'sensitive_cols': ['parental_edu_label'],
        'class_names':    class_names,
        'idx_to_class':   {i: name for i, name in enumerate(class_names)},
    }
    return df, meta


def load_adult_data():
    """
    Load Adult Income dataset from adult.zip (combines the provided train/test
    splits and performs our own 70/30 split downstream).

    Target variable
    ---------------
    income_label : 1 if annual income >$50K, else 0.  (2-arm bandit)

    Sensitive attributes
    --------------------
    gender_label : 'Male' / 'Female'
    race_label   : 'White' / 'Non-White'

    Notes
    -----
    - Rows with missing values (encoded as '?') are dropped.
    - fnlwgt (census sampling weight) and education (text duplicate of
      education-num) are removed.
    - sex and race are stored as sensitive labels then dropped from features.
    - Remaining categorical columns are one-hot encoded.
    """
    path = './data-uci/adult.zip'
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income',
    ]

    with zipfile.ZipFile(path) as z:
        train_df = pd.read_csv(z.open('adult.data'), names=columns,
                               skipinitialspace=True)
        # Test file has one header comment line; labels end with '.'
        test_df  = pd.read_csv(z.open('adult.test'), names=columns,
                               skipinitialspace=True, skiprows=1)

    df = pd.concat([train_df, test_df], ignore_index=True)

    # Normalise income labels (test set has trailing '.')
    df['income'] = df['income'].str.strip().str.rstrip('.')

    # Drop rows with missing values (encoded as '?')
    df = df.replace('?', np.nan).dropna().reset_index(drop=True)

    # ── Sensitive attributes (stored before encoding) ──────────────────────
    # gender_label : 1 = Male, 0 = Female
    df['gender_label'] = (df['sex'].str.strip() == 'Male').astype(int)
    # race_label   : 1 = White, 0 = Non-White
    df['race_label']   = (df['race'].str.strip() == 'White').astype(int)

    # ── Target ────────────────────────────────────────────────────────────
    df['income_label'] = (df['income'] == '>50K').astype(int)

    # ── Drop non-feature columns ──────────────────────────────────────────
    # fnlwgt : census sampling weight, not predictive of individual income
    # education : text version; education-num captures the same info numerically
    df = df.drop(columns=['fnlwgt', 'education', 'income', 'sex', 'race'])

    # ── One-hot encode remaining categorical columns ───────────────────────
    cat_cols = ['workclass', 'marital-status', 'occupation',
                'relationship', 'native-country']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # ── Normalize numeric columns (z-score) ───────────────────────────────
    # These have very different scales (e.g. capital-gain up to 99k vs age 17-90).
    numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    class_names = ['<=50K', '>50K']
    meta = {
        'target_col':     'income_label',
        'sensitive_cols': ['gender_label', 'race_label'],
        'class_names':    class_names,
        'idx_to_class':   {i: name for i, name in enumerate(class_names)},
    }
    return df, meta


def load_loan_data():
    """
    Load Statlog German Credit dataset from statlog+german+credit+data.zip.

    Uses german.data (the categorical version with 20 attributes).

    Target variable
    ---------------
    credit_risk_label : 1 = good credit risk, 0 = bad credit risk.  (2-arm bandit)
                        Original encoding: 1 = Good, 2 = Bad.

    Sensitive attributes
    --------------------
    sex_label : 'Male' / 'Female'   (derived from Attr 9: personal_status_sex)
    age_label : 'young' / 'old'     (age < 25 = young, age >= 25 = old)

    Notes
    -----
    - Attr 9 (personal_status_sex) conflates sex with marital status.
      Sex is extracted as a sensitive label; marital status is retained as a
      separate feature so no credit-relevant information is discarded.
    - personal_status_sex is then dropped so the policy cannot reconstruct
      sex directly from the feature vector.
    - Numeric features are z-score normalised.
    - Categorical features are one-hot encoded.
    """
    path = './data-uci/statlog+german+credit+data.zip'

    columns = [
        'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
        'housing', 'num_credits', 'job', 'num_dependents', 'telephone',
        'foreign_worker', 'credit_risk',
    ]

    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(z.open('german.data'), names=columns, sep=' ')

    # ── Sensitive attributes ───────────────────────────────────────────────
    # Attr 9 codes: A91=male divorced/sep, A92=female div/sep/married,
    #               A93=male single,       A94=male married/widowed,
    #               A95=female single
    # sex_label  : 1 = Male, 0 = Female
    sex_map = {'A91': 1, 'A92': 0, 'A93': 1, 'A94': 1, 'A95': 0}
    df['sex_label'] = df['personal_status_sex'].map(sex_map)

    # age_label  : 1 = old (age >= 25), 0 = young (age < 25)
    df['age_label'] = (df['age'] >= 25).astype(int)

    # ── Marital status feature (retain credit-relevant info from Attr 9) ──
    marital_map = {'A91': 'divorced', 'A92': 'married', 'A93': 'single',
                   'A94': 'married',  'A95': 'single'}
    df['marital_status'] = df['personal_status_sex'].map(marital_map)
    df = df.drop(columns=['personal_status_sex'])

    # ── Target ────────────────────────────────────────────────────────────
    # Original: 1=Good, 2=Bad  →  label 0='good' (idx 0), 1='bad' (idx 1)
    df['credit_risk_label'] = (df['credit_risk'] == 2).astype(int)
    df = df.drop(columns=['credit_risk'])

    # ── One-hot encode categorical columns ────────────────────────────────
    cat_cols = [
        'checking_status', 'credit_history', 'purpose', 'savings',
        'employment', 'other_debtors', 'property', 'other_installments',
        'housing', 'job', 'telephone', 'foreign_worker', 'marital_status',
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # ── Normalize numeric columns (z-score) ───────────────────────────────
    numeric_cols = ['duration', 'credit_amount', 'installment_rate',
                    'residence_since', 'age', 'num_credits', 'num_dependents']
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    class_names = ['good', 'bad']
    meta = {
        'target_col':     'credit_risk_label',
        'sensitive_cols': ['sex_label', 'age_label'],
        'class_names':    class_names,
        'idx_to_class':   {i: name for i, name in enumerate(class_names)},
    }
    return df, meta

def load_dataset(dset_id):
    if dset_id == 'drugs':
        return load_drugs_data()
    elif dset_id == 'students':
        return load_student_data()
    elif dset_id == 'adult':
        return load_adult_data()
    elif dset_id == 'loan':
        return load_loan_data()
    else:
        raise ValueError(f"Unknown dataset ID: {dset_id}")

if __name__ == "__main__":
    load_student_data()