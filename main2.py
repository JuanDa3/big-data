import pandas as pd
# se importan las bibliotecas necesarias
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# se importan las librerías
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree
import graphviz
import pydotplus
import pandas as pd
import numpy as np
import seaborn as sns
# se importan las bibliotecas necesarias
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors, datasets, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import neighbors, datasets, preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("arrhythmia_csv.csv")

# 3. Agregue una nueva columna llamada clase2, que contemple 2 opciones
# únicamente Normal o Arritmia


def AgregarClase2(row):
    if row['class'] == 1:
        return 'Normal'
    else:
        return 'Arritmia'


df['clase2'] = df.apply(AgregarClase2, axis=1)

# se eliminan filas que tienen datos nulos
df = df.dropna()

# # Elimina las columnas con datos faltantes
df = df.dropna(axis=1)

# # Elimina las filas duplicadas
df = df.drop_duplicates()

# Reemplazamos los valores atípicos por la media de cada clase o la moda
num_columns = df.select_dtypes(include=['float', 'int']).columns
cat_columns = df.select_dtypes(include=['object', 'category']).columns


for col in num_columns:
    # Identificamos los datos atípicos
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    # Reemplazamos los valores atípicos por la media de cada clase o la moda
    for cls in df['class'].unique():
        cls_outliers = outliers[outliers['class'] == cls]
        if cls_outliers.empty:
            continue
        if col in cat_columns:
            mode = df[df['class'] == cls][col].mode()[0]
            df.loc[cls_outliers.index, col] = mode
        else:
            mean = df[df['class'] == cls][col].mean()
            df.loc[cls_outliers.index, col] = mean


df.to_csv('arrhythmia_clean2.csv', index=False)

null_values = df.isnull().sum()
print(null_values)
null_values.to_csv('null_values.txt', header=None, sep='\t')


# -------------------Bayes ------------------------------------

X = df.drop('class', axis=1)
y = df['class']

le = LabelEncoder()
obj_cols = X.select_dtypes(include=['object']).columns
for col in obj_cols:
    X[col] = le.fit_transform(X[col].astype(str))
# este es un metodo de seleccion de caracteristicas
# le da puntaje a cada caracteristica y borra todas las demas
# excepto las que tienen puntaje mas alto

best = SelectKBest(k=20)
print(y)
X_new = best.fit_transform(X, y)
selected = best.get_support(indices=True)
caracteristicasUsadas = X.columns[selected]
print(caracteristicasUsadas)

# Split dataset in training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

gnb = GaussianNB()

# Train classifier
gnb.fit(X_train.iloc[:, selected], y_train)
# se predice con las variables seleccionadas
y_pred = gnb.predict(X_test[caracteristicasUsadas])
# Devuelve la precisión media en los datos y etiquetas de prueba dados.
print('Precision con el set de prueba: {:.2f}'.format(
    gnb.score(X_test[caracteristicasUsadas], y_test)))
print('Se obtiene la precision con el dataset de entrenamiento: {:.2f}'.format(
    gnb.score(X_train[caracteristicasUsadas], y_train)))

#
print("se hace prediccion")
print(gnb.predict([[10, 102, 40, 1, 6, 100, 90, 67,
      29, 15, 13, 44, 129, 77, -44, 53, 2, 9, 5, 71]]))


# Create Decision Tree classifer object
# verificar si usa Gini o usa entropia
clf = DecisionTreeClassifier(criterion='entropy')
# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)
clf2 = clf


# -------------------------------------------------------------------------------
# La precisión se define como el número de verdaderos positivos
#  dividido por el número total de predicciones positivas, mientras
# que el recall se define como el número de verdaderos positivos dividido
# por el número total de verdaderos positivos y falsos negativos.
# El puntaje F1 es una medida de la precisión y el recall, y se
#  define como la media armónica de la precisión y el recall.
# La función devuelve un diccionario que contiene los valores
# de las métricas calculadas, es decir, la precisión (accuracy),
#  la precisión (precision), el recall y el puntaje F1.
def obtener_performance_classification(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred, normalize=True)
    prec = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1_score1 = f1_score(y_true, y_pred, average='macro')

    return {'accuracy': accuracy,
            'precision': prec,
            'recall': recall,
            'f1_score': f1_score1}

# --------------------------------------

# Este código define una función llamada construir_modelo, que toma cuatro parámetros:
# dataset, classifier_fn, atributos_seleccionados y label_clase.
# La función toma el conjunto de datos (dataset) y realiza una partición de 70/30 
# entre los datos de entrenamiento (X_train, y_train) y los datos de prueba (X_test, y_test).
# Luego, utiliza la función classifier_fn para construir un modelo clasificador a partir de 
# los datos de entrenamiento. Una vez construido el modelo, hace predicciones utilizando 
# los datos de entrenamiento y los datos de prueba.
# La función utiliza la función obtener_performance_classification 
# para calcular las métricas de rendimiento de clasificación para los
#  datos de entrenamiento y de prueba. Luego, genera una matriz de confusión 
# a partir de las predicciones realizadas en los datos de prueba.


def construir_modelo(dataset, classifier_fn,
                     atributos_seleccionados,
                     label_clase):

    X = dataset[atributos_seleccionados]
    Y = dataset[label_clase]
    # se particiona el dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)  # 70% training and 30% test
    # se aplica el clasificador
    model = classifier_fn(X_train, y_train)
    # se hacen las predicciones en el set de entramiento y testeo
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    consolidado_entrenamiento = obtener_performance_classification(
        y_train, y_pred_train)
    consolidado_testeo = obtener_performance_classification(y_test, y_pred)

    pred_results = pd.DataFrame({'y_test': y_test,
                                 'y_pred': y_pred})

    model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)

    return {'training': consolidado_entrenamiento, 'test': consolidado_testeo, 'confusion_matrix': model_crosstab}


# ---------------------------------------------------------
# Este código define una función llamada compare_results() que se encarga 
# de imprimir los resultados de las métricas de desempeño obtenidas para
#  cada técnica de clasificación en el conjunto de entrenamiento y en el conjunto de prueba.
def compare_results():
    for key in result_dict:
        print('Tecnica de clasificación: ', key)

        print()
        print('Training data')
        for score in result_dict[key]['training']:
            print(score, result_dict[key]['training'][score])

        print()
        print('Test data')
        for score in result_dict[key]['test']:
            print(score, result_dict[key]['test'][score])

        print()


# -------------------------------
#  una función llamada decision_tree_fn que construye y entrena un modelo de
#  árbol de decisión utilizando la biblioteca sklearn. La función toma los siguientes argumentos:
# x_train: las características (atributos) del conjunto de datos de entrenamiento
# y_train: la variable objetivo (clase) del conjunto de datos de entrenamiento
# criterion: el criterio utilizado para medir la calidad de la división en el árbol (por defecto es entropy)
# max_depth: la profundidad máxima del árbol (por defecto es None, lo que significa que el 
# árbol se expande hasta que todas las hojas contengan menos de min_samples_split muestras)
# max_features: el número máximo de características que se utilizarán en cada división del 
# árbol (por defecto es None, lo que significa que se utilizan todas las características)
def decision_tree_fn(x_train, y_train, criterion='entropy', max_depth=None, max_features=None):
    model = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, max_features=max_features)
    model.fit(x_train, y_train)
    return model


# La función naive_bayes_fn es una función que construye y entrena un modelo de 
# clasificación Naive Bayes usando la clase GaussianNB de la librería sklearn.
# Toma como entrada los datos de entrenamiento x_train y las etiquetas y_train,
#  y devuelve el modelo entrenado.
# Esta función no utiliza los argumentos opcionales max_depth y max_features
def naive_bayes_fn(x_train, y_train, max_depth=None, max_features=None):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model


# Recibe como entrada un conjunto de datos de entrenamiento (x_train e y_train) y
# el parámetro n_neighbors que indica el número de 
# vecinos a considerar en la clasificación. La función devuelve el modelo KNN entrenado.
def knn_fn(x_train, y_train, n_neighbors=5):
    model = neighbors.KNeighborsClassifier(n_neighbors)
    model.fit(x_train, y_train)

    return model


# Random forest

def random_forest_fn(x_train, y_train, criterion='entropy', max_depth=None, max_features=None):
    model = RandomForestClassifier(
        n_estimators=10, criterion=criterion, random_state=0)
    model.fit(x_train, y_train)
    return model


# Support Vector Machines (SVM)
def svm_fn(x_train, y_train, max_depth=None, max_features=None):
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model

# kernel_svm

def kernel_svm_fn(x_train, y_train, n_neighbors=5):
    model = SVC(kernel='rbf', random_state=0)
    model.fit(x_train, y_train)

    return model


result_dict = {}
print(df)

label_encoding = preprocessing.LabelEncoder()
# se usa label encoding
df['clase2'] = label_encoding.fit_transform(df['clase2'].astype(str))
#df['Arritmia'] = label_encoding.fit_transform(df['Arritmia'].astype(str))

feature_cols = list(df.columns[1:])
#criterion = 'entropy'

result_dict['Decision_tree'] = construir_modelo(
    df, decision_tree_fn, feature_cols, 'class')
print("1")
compare_results()
result_dict['Naive_bayes'] = construir_modelo(
    df, naive_bayes_fn, feature_cols, 'class')
compare_results()


result_dict['knn'] = construir_modelo(df, knn_fn, feature_cols, 'class')
compare_results()

result_dict['Random_forest'] = construir_modelo(
    df, random_forest_fn, feature_cols, 'class')
compare_results()


result_dict['Svm'] = construir_modelo(df, svm_fn, feature_cols, 'class')
compare_results()

result_dict['kernel_svm'] = construir_modelo(
    df, kernel_svm_fn, feature_cols, 'class')
compare_results()


# se graba el árbol
# se utiliza la función export_graphviz de la biblioteca scikit-learn para generar una
# representación en formato DOT del árbol de decisión (clf2), utilizando los nombres de
# las características (feature_cols) y los nombres de las clases (class_names). Luego, 
# la función graph_from_dot_data de Pydotplus crea un objeto de gráfico a partir de los datos DOT. 
# Finalmente, el método write_pdf se utiliza para escribir el gráfico en un archivo PDF.
dot_data = export_graphviz(clf2, out_file=None, filled=True, feature_names=feature_cols, class_names=[
                           str(x) for x in sorted(y_train.unique())])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('./Diagrama/arbol.pdf')
