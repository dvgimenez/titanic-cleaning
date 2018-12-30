import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
# Càrrega del dataset.
data = pd.read_csv("\train_titanic.csv")


### 2. Integració i selecció de les dades d’interès a analitzar
# Variables importants:
# SURVIVED
# PCLASS
# SEX
# AGE
# EMBARKED
# FARE
data_vars = data[["Pclass","Sex","Age","Embarked", "Fare","Survived"]]


###3. Neteja de les dades
###3.1. Les dades contenen zeros o elements buits? Com gestionaries aquests casos
# Recompte de valors nuls.
for var in data_vars:
    print("La variable",var,"conté",data_vars[var].isna().sum(),"valors nuls")

# Reomplim els NA a partir de la mediana de les edats per Sexe.
data_vars['Age'].fillna(data_vars.groupby('Sex')['Age'].transform("median"), inplace=True)
# Elimino les dues observacions amb NA de "Embarked"
data_vars = data_vars.dropna()



### 3.2. Conversió de valors (punt afegit)
#Conversió "male" - "famale" en enters:
data_clean = data_vars.replace("female",2)
data_clean = data_clean.replace("male", 1)

# Conversió dels tres valors d'embarked:
data_clean["Embarked"] = data_clean["Embarked"].replace("C", 1)
data_clean["Embarked"] = data_clean["Embarked"].replace("Q", 2)
data_clean["Embarked"] = data_clean["Embarked"].replace("S", 3)

# Conversió de "Age" a enters:
data_clean["Age"] = data_clean["Age"].round().astype(int)


### 3.3. Identificació i tractament de valors extrems
# Valors extrems
print(data_clean.describe())

fig=plt.figure(figsize=(15,15))
i = 1

for var in data_vars:   
    ax=fig.add_subplot(3,3,i) 
    ax.boxplot(data_clean[var])
    ax.set_title(var)
    i = i + 1

### 3.4. Emmagatzemar dataframe net (punt afegit)
# Guardo el dataset net:
data_clean.to_csv("data_clean.csv")


### 4. ANÀLISI DE LES DADES
# Càrrega del dataset.
col = ["Pclass", "Sex", "Age", "Embarked", "Fare", "Survived"]
data = pd.read_csv("data_clean.csv", usecols = col )


### 4.2. Comprovació de la normalitat i homogeneïtat de la variància
# Revisem normalitat
# Nomès analitzem les variables conínues: Age i Fare:
k21, p1 = stats.normaltest(data["Age"])
k22, p2 = stats.normaltest(data["Fare"])

print("El p-value de la variable Age és",p1)
print("El p-value de la variable Fare és",p2)


import statsmodels.api as sm
from statsmodels.formula.api import ols

mod1 = ols('Age ~ Survived', data = data).fit()
aov_table1 = sm.stats.anova_lm(mod1, typ = 2)
print("ANOVA agrupació Survived - Age\n",aov_table1)

mod2 = ols('Fare ~ Survived', data = data).fit()
aov_table2 = sm.stats.anova_lm(mod2, typ = 2)
print("\nANOVA agrupació Survived - Fare\n",aov_table2)



### 4.3. Aplicació de proves estadístiques per comparar els grups de dades. 
###En funció de les dades i de l’objectiu de l’estudi, aplicar proves de contrast d’hipòtesis, 
###correlacions, regressions, etc.

# Correlació entre variables.
corr = data.corr()
corr = abs(corr) # Obvio el signe de els correlacions.
print(corr)


# Regressió lineal.
# Creo 3 models:
#   1 - Amb totes les variables triades.
#   2 - Amb les variables més importants segons l'estudi de correlació: Sex, Pclass i Fare
#   3-  Amb la variable amb màxima correlació: Sex

def reg_lin(data_X, data_Y):
    data_X_train = data_X[:-200]
    data_Y_train = data_Y[:-200]

    data_X_test = data_X[-200:]
    data_Y_test = data_Y[-200:]
    
    regr = linear_model.LinearRegression()
    regr.fit(data_X_train, data_Y_train)

    survived_pred = regr.predict(data_X_test)

    # Càlculs:
    MSE = mean_squared_error(data_Y_test, survived_pred)
    acc = accuracy_score(data_Y_test, survived_pred.round())
    cm = confusion_matrix(data_Y_test, survived_pred.round())
    return(MSE, acc, cm, regr)

data_X_m1 = data[["Pclass", "Sex", "Age", "Embarked", "Fare"]]
data_X_m2 = data[["Sex", "Pclass", "Fare"]]
data_X_m3 = data[["Sex"]]
data_Y = data[["Survived"]]


MSE1, acc1, cm1, regr1 = reg_lin(data_X_m1,data_Y)
MSE2, acc2, cm2, regr2 = reg_lin(data_X_m2,data_Y)
MSE3, acc3, cm3, regr3 = reg_lin(data_X_m3,data_Y)



### 5. Representació dels resultats a partir de taules i gràfiques.
# Correlació de les variables:
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=0, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
print(corr)


# Matrius de Confusió dels models:
def representa_mtx(cm):
    labels = ['MORTS', 'VIUS']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title("Matriu de confusió (vius, morts)")
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicció')
    plt.ylabel('Real')
    plt.show()
    print("La taula amb els valors és la següent:\n",cm)

representa_mtx(cm1)
representa_mtx(cm2)
representa_mtx(cm3)


# Valors de MES i precissió:
print("MODEL 1:\n    Error quadràtic mig (MSE) =",MSE1,"\n    Precissió =",acc1)
print("MODEL 2:\n    Error quadràtic mig (MSE) =",MSE2,"\n    Precissió =",acc2)
print("MODEL 3:\n    Error quadràtic mig (MSE) =",MSE3,"\n    Precissió =",acc3)


### 6. Resolució del problema. A partir dels resultats obtinguts, quines són les conclusions? 
### Els resultats permeten resoldre el problema?

# Càrrega del dataset.
data_test= pd.read_csv("test_titanic.csv")

data_test_vars = data_test[["Sex","Pclass","Fare"]]

print ("El dataset conté:",len(data_test_vars),"registres")

for var in data_test_vars:
    print("La variable",var,"conté",data_test_vars[var].isna().sum(),"valors nuls")

#### NETEJA
# Reomplim els NA a partir de la mediana de les tarifes per Sexe.
data_test_vars['Fare'].fillna(data_test_vars.groupby('Sex')['Fare'].transform("median"), inplace=True)

#### CONVERSIÓ
#Conversió "male" - "famale" en enters:
data_test_clean = data_test_vars.replace("female",2)
data_test_clean = data_test_clean.replace("male", 1)

pred = regr2.predict(data_test_clean)
resultat = data_test.assign(Survived_PRED = pred.round())

print(resultat[0:10])

# Recompte de morts i supervivents.
sumari = resultat["Survived_PRED"].value_counts()
print("En total moren", sumari[0], "passatgers i sobreviuen", sumari[1],"passatgers")

# Emmagatzemo
resultat.to_csv("resultat.csv")