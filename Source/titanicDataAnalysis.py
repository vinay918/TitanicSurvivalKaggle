import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import pylab

def main():
    
    #this script was used to perform exploratory data analysis, and can be edited to do whatever is needed
    #nothing specific to be followed, as it was used for feature engineering decisions
    
    #Average survival rate depending on socio-economic class
    dftrain  = pd.read_csv('C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\\ProcessedTrain.csv')
    
    print(dftrain[['Title','Age']].groupby('Title').mean())
    
    print(dftrain.describe())
    
    print(dftrain.isnull().sum())
    
    print(dftrain[['Title','Survived']].groupby('Title').mean())
    
    print(dftrain[['Pclass','Survived']].groupby('Pclass').mean())

    #Average survival rate depending on sex
    print(dftrain[['Sex','Survived']].groupby('Sex').mean())
    
    #Average survival rate depending on sex
    print(dftrain[['Embarked','Survived']].groupby('Embarked').mean())
    
    
    dfTemp=dftrain.groupby('Embarked').count()
    print(dfTemp)
    
    g = sns.FacetGrid(dftrain, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    pylab.show()

main()