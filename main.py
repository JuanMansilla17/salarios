#import inline as inline
#import matplotlib
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import colors
#import seaborn as sb

#matplotlib
#inline
#plt.rcParams['figure.figsize'] = (16, 9)
#plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest


def dataset():
    dataframe = pd.read_csv(r"encuesta-salarios-Argentina.csv")
    #print(dataframe.head(10))

    cols=['lugar_trabajo', 'tipo_contrato', 'salario_mensual_bruto', 'salario_mensual_neto', 'trabajo_de', 'anios_experiencia', 'anios_empresa_actual', 'anios_puesto_actual', 'personal_a_cargo', 'plataformas', 'lenguajes_programacion','framework_librerias_herramientas','bases_de_datos','qa_testing','ides','cantidad_personal_organizacion','actividad_principal','nivel_estudios','estado','carrera','universidad','realizaste_especializacion','edad']




    print(dataframe.shape)
    print(dataframe.groupby(dataframe.tipo_contrato).size())

    #Eliminar registros vacíos con campos vacíos
    for col in dataframe.columns:
        print(f'Hay {dataframe[col].isnull().sum()} filas vacías en la columna {col}')
    dataframe = dataframe.dropna()

    print(dataframe.shape)

    #agrupar salarios
    ranges_limits = [dataframe['salario_mensual_neto'].quantile(0),
                     dataframe['salario_mensual_neto'].quantile(0.2),
                     dataframe['salario_mensual_neto'].quantile(0.4),
                     dataframe['salario_mensual_neto'].quantile(0.6),
                     dataframe['salario_mensual_neto'].quantile(0.8),
                     dataframe['salario_mensual_neto'].quantile(1)]
    ranges_names = [1,2,3,4,5]

    dataframe['nivel_gastos'] = pd.cut(dataframe['salario_mensual_neto'], bins=ranges_limits, labels=ranges_names)
    print(dataframe.groupby(dataframe.nivel_gastos).size())

    dataframe.describe()

    # histograma
    #dataframe.drop(['comprar'], axis=1).hist()
    #plt.show()

    """
      procesamos algunas de estas columnas. 
      Por ejemplo, podríamos agrupar los diversos gastos. 
      También crearemos una columna llamada financiar que será la resta del precio de la vivienda con los ahorros de la familia.
      """
    #dataframe['gastos'] = (dataframe['gastos_comunes'] + dataframe['gastos_otros'] + dataframe['pago_coche'])
    #dataframe['financiar'] = dataframe['vivienda'] - dataframe['ahorros']
    #dataframe.drop(['gastos_comunes', 'gastos_otros', 'pago_coche'], axis=1).head(10)

    """
    Resumen estadístico que nos brinda la librería Pandas con describe():
    """
    #reduced = dataframe.drop(['gastos_comunes', 'gastos_otros', 'pago_coche'], axis=1)
    #reduced.describe()



    """
    En vez de utilizar las 11 columnas de datos de entrada que tenemos,
     vamos a utilizar una Clase de SkLearn llamada SelectKBest con la que seleccionaremos las 5 mejores características
      y usaremos sólo esas.
    """

    y = dataframe['nivel_gastos']
    X = dataframe.drop(['nivel_gastos','salario_mensual_neto'], axis=1)

    best = SelectKBest(k=5)
    X_new = best.fit_transform(X, y)
    X_new.shape
    selected = best.get_support(indices=True)
    print(X.columns[selected])


    #Las que “más aportan” al momento de clasificar.
    #Veamos qué grado de correlación tienen:

    """
    used_features = X.columns[selected]
    colormap = plt.cm.viridis
    plt.figure(figsize=(12, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sb.heatmap(dataframe[used_features].astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
               linecolor='white', annot=True)

    return dataframe,used_features

"""

if __name__== "__main__":

    #DataSet
    dataset()
    #dataframe,used_features = dataset()
"""
    #Split dataset in training and test datasets
    X_train, X_test = train_test_split(dataframe, test_size=0.2, random_state=6)
    y_train = X_train["comprar"]
    y_test = X_test["comprar"]

    #Instantiate the classifier
    gnb = GaussianNB()
    # Train classifier
    gnb.fit(
        X_train[used_features].values,
        y_train
    )
    y_pred = gnb.predict(X_test[used_features])

    print('Precisión en el set de Entrenamiento: {:.2f}'
          .format(gnb.score(X_train[used_features], y_train)))
    print('Precisión en el set de Test: {:.2f}'
          .format(gnb.score(X_test[used_features], y_test)))


    #Probando el clasificador
    #['ingresos', 'ahorros', 'hijos', 'trabajo', 'financiar']
    a=[6000, 34000, 2, 5, 320000]
    b=[2000, 5000, 0, 5, 200000]
    rta = gnb.predict([b])
    # Resultado esperado 0-Alquilar, 1-Comprar casa
    if (rta[0]==0):
        print("Se recomienda Alquilar")
    else:
        print("Se recomienda Comprar")
"""