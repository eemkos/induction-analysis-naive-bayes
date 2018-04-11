import pandas as pd
import os
from os.path import dirname, abspath


def load_iris(path=dirname(abspath(__file__)) + '/../../data/iris/raw/iris.data'):
    columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
    df = pd.read_csv(path, names=columns)
    classes_dict = {cls: i for i, cls in enumerate(df.Class.unique())}
    df['ClassIndex'] = df.Class.apply(lambda x: classes_dict[x])
    return df


def load_glass(path=dirname(abspath(__file__)) + '/../../data/glass/raw/glass.data'):
    columns = ['RefractiveIndex', 'Sodium',
               'Magnesium', 'Aluminum', 'Silicon',
               'Potassium', 'Calcium', 'Barium',
               'Iron', 'Class']
    df = pd.read_csv(path, names=columns, index_col=0)
    classes_dict = {cls: i for i, cls in enumerate(df.Class.unique())}
    df['ClassIndex'] = df.Class.apply(lambda x: classes_dict[x])
    return df


def load_wine(path=dirname(abspath(__file__)) + '/../../data/wine/raw/wine.data'):
    columns = ['Class', 'Alcohol', 'MalicAcid', 'Ash',
               'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols',
               'Flavanoids', 'NonflavanoidPhenols',
               'Proanthocyanins', 'ColorIntensity', 'Hue',
               'OD280/OD315 of diluted wines', 'Proline']
    df = pd.read_csv(path, names=columns)
    classes_dict = {cls: i for i, cls in enumerate(df.Class.unique())}
    df['ClassIndex'] = df.Class.apply(lambda x: classes_dict[x])
    return df


def load_diabetes(path=dirname(abspath(__file__)) + '/../../data/diabetes/processed/diabetes.csv'):
    df = pd.read_csv(path)
    df.DateTime = pd.to_datetime(df.DateTime)
    return df


def load_diabetes_raw(dirpath=dirname(abspath(__file__)) + '/../../data/diabetes/raw/Diabetes-Data/'):
    dfs = []
    for filename in os.listdir(dirpath):
        if filename.startswith('data-'):
            df = pd.read_csv(dirpath + filename, delimiter='\t', names=['Date', 'Time', 'Code', 'Value'])
            df['namefile'] = filename
            dfs.append(df)
    return pd.concat(dfs)


def load_pima_diabetes(path=dirname(abspath(__file__)) + '/../../data/pima_diabetes/raw/pima-indians-diabetes.data'):
    columns = ['NbPregnancies', 'PlasmaGlucoseConcentration',
               'DiastolicBloodPressure', 'TricepsSkinFoldThickness',
               'TwoHourSerumInsulin', 'BMI',
               'DiabetesPedigreeFunction', 'Age', 'Class']
    df = pd.read_csv(path, names=columns)
    classes_dict = {cls: i for i, cls in enumerate(df.Class.unique())}
    df['ClassIndex'] = df.Class.apply(lambda x: classes_dict[x])
    return df
