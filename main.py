import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# se importan las bibliotecas necesarias
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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

# sns.boxplot(x='age', y='clase2', data=df)
# plt.title("Grafico cajas y bigotes para la variable edad,en relación con clase2")
# plt.show()

# sns.boxplot(x='weight', y='clase2', data=df)
# plt.title("Gráfico de cajas y bigotes para otra variable numérica")
# plt.show()


# 5. Solucione los problemas referentes a calidad de datos presentes en al menos 6 
# variables, al menos dos de ellas deben categóricas. Elimine columnas irrelevantes.
#  Identifique los datos atípicos reemplace esos valores por la media de cada clase o 
# la moda, según corresponda, este proceso debe realizarse en python.
#  Debe agregar un párrafo, describiendo de manera ejecutiva, cuál fue el proceso seguido. 

# Eliminar columnas irrelevantes
df = df.drop(['J', 'chDI_Qwave', 'chDI_RPwaveExists', 'QRST','chAVR_SPwaveAmp','chV1_SPwave'], axis=1)


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
df.to_csv('arrhythmia_clean.csv', index=False)

print('Dataset limpio con datos removidos \n',pd.read_csv('arrhythmia_clean.csv'))

def ponerNumerosBarra (ax):
  for bar in ax.patches:
        height = bar.get_height()
        width = bar.get_width()
        x = bar.get_x()
        y = bar.get_y()
        
        label_text = height
        label_x = x + width / 2
        label_y = y + height / 2
         

        ax.text(label_x, label_y, '{:,.1f}'.format(label_text), ha='center',    
                va='center')    
# df2 = df.iloc[0:10]
# ax = df2.plot(kind='bar',x='weight',y='age',width=0.9 )
# ax.set(xlabel='peso', ylabel='Count')
# ponerNumerosBarra (ax)
# plt.title("Grafico cajas y bigotes para la variable edad,en relación con clase2")
# plt.show()

# ax= df.groupby('clase2')['age'].nunique().plot(kind='bar')
# ponerNumerosBarra(ax)
# plt.title("Grafico cajas y bigotes para la variable edad,en relación con clase2")
# plt.show()


grouped = df.groupby('clase2')['weight'].count()

# Crea el gráfico de barras
fig, ax = plt.subplots()
grouped.plot(kind='bar', ax=ax)

ponerNumerosBarra (ax)
plt.title("Diagrama de barras de personas con arritmia")
plt.show()

print("Se construye un grafico de barras sexo clase apilado'")
ax = df.groupby(['sex','clase2']).size().unstack().plot(kind='bar',stacked=True)
ponerNumerosBarra(ax)
plt.title("Diagrama de barras de personas con arritmia basadas en su sexo\n0=Masculino 1=Femenino")
plt.show() 

ax = df[['age']].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8)

ponerNumerosBarra(ax)
plt.title("Diagrama de barras de personas con arritmia basadas en su sexo\n0=Masculino 1=Femenino")
plt.show() 


# ax= df.pivot(columns='clase2').heartrate.plot(kind='hist',bins=[40,60,80,100,120,140],rwidth=0.8, stacked=True)
# ax.set(xlabel='Ritmo cardiaco', ylabel='cantidad')  
# plt.grid(True,'both', 'y')
# ax.set_axisbelow(True)
# ponerNumerosBarra(ax)
# plt.title("Diagrama de barras de personas con arritmia basadas en el ritmo cardiaco ")
# plt.show() 
# print (df2['edad'])
# px.line(df2,y='edad',title='Valor Total')
# query = df.query("edad==25")
# print (df.dtypes)
# px.scatter(query, x="edad", y="tarifa")

# px.scatter(df, x="age", y="weight", color="clase2").s
# ponerNumerosBarra(px)
# plt.title("Diagrama de barras de personas con arritmia basadas en el ritmo cardiaco ")
# plt.show()