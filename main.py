import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("arrhythmia_csv.csv")

for col in df.columns:
    print(col)


#3. Agregue una nueva columna llamada clase2, que contemple 2 opciones 
# únicamente Normal o Arritmia
def AgregarClase2(row):
    if row['class'] == 1:
        return 'Normal'
    else:
        return 'Arritmia'

df['clase2'] = df.apply(AgregarClase2, axis=1)

print(df.head)
#4Construya un gráfico de cajas y bigotes para 
# la variable edad, en relación con clase2.  
# Realice otro gráfico de cajas y bigotes para otra
#  variable numérica. Concluya al respecto

sns.boxplot(x='age', y='clase2', data=df)
plt.title("Grafico cajas y bigotes para la variable edad,en relación con clase2")
plt.show()

sns.boxplot(x='weight', y='clase2', data=df)
plt.title("Gráfico de cajas y bigotes para otra variable numérica")
plt.show()


#5 5. Solucione los problemas referentes a calidad de datos presentes en al menos 6 
# variables, al menos dos de ellas deben categóricas. Elimine columnas irrelevantes.
#  Identifique los datos atípicos reemplace esos valores por la media de cada clase o 
# la moda, según corresponda, este proceso debe realizarse en python.
#  Debe agregar un párrafo, describiendo de manera ejecutiva, cuál fue el proceso seguido. 


# Eliminar columnas irrelevantes
df = df.drop(['J', 'chDI_Qwave', 'chDI_RPwaveExists', 'QRST','chAVR_SPwaveAmp'], axis=1)

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

# Guardar el resultado en un nuevo archivo CSV
data_limpio=df.to_csv('arrhythmia_clean.csv', index=False)

