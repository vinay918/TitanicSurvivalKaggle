import pandas as pd
import math

def main():
    dftrain  = pd.read_csv('C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\\train.csv')
    dftest = pd.read_csv('C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\\test.csv')
    datasets = [dftrain,dftest]
    i=0
    for dataset in datasets:
        print(dataset.isnull().sum())
  
        #extracting title from name and remapping
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset=myAvgImpute('Age','Title','Master',6,'nothing',0,dataset)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Sir', 'Jonkheer', 'Dona'], 'Upper')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')       
        dataset['Title'] = dataset['Title'].replace('Rev', 'Mr')      
        titleMap={'Mrs':4,'Miss':3,'Master':2,'Upper':1,'Mr':0}    
        dataset['Title'] = dataset['Title'].map(titleMap) 
        
        #Average age of Men and Women to impute age
        avgAgeF=condAverage('Sex','female','Age',dataset)
        avgAgeM=condAverage('Sex','male','Age',dataset)
        
        #filling age value based on gender
        dataset=myAvgImpute('Age','Sex','female',avgAgeF,'male',avgAgeM,dataset)
        
        #mapping embarked to ordinal numerical representation
        embark={'C':2,'Q':1,'S':1}
        dataset['Embarked']=dataset['Embarked'].map(embark)
        
        
        #mapping gender to binary
        gender={'male':0,'female':1}
        dataset['Sex']=dataset['Sex'].map(gender) 
    
        #extract deck level from cabin number
        dataset=deck(dataset)
    
        #converting Sex, Embarked and Deck to their Categorical representations using Dummy Variables
        dataset['Deck']=dataset['Deck'].fillna('N')
        dataset=pd.get_dummies(dataset, columns=["Deck"])
        
        #creating number of family members
        dataset['FamMem']=dataset.Parch+dataset.SibSp
        
        #dropping unnecessary columns
        dataset=dataset.drop(['Name','Ticket','Parch','SibSp','Cabin','Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_N'],axis=1)
        
        dataset['Embarked']=dataset['Embarked'].fillna('0')
        dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].median())
        
        #checking number of missing values
        #print(dataset.isnull().sum())
        
        #normalizing data
        dataset['Age']=dataset.Age/100
        
        
        datasets[i]=dataset
        i=i+1
        print(dataset.isnull().sum())
    
    
    datasets[0]=datasets[0].drop(['Deck_T'],axis=1)

    #saving resulting dataset to csv
    datasets[0].to_csv('C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\\ProcessedTrain.csv',index=False)
    datasets[1].to_csv('C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\\ProcessedTest.csv',index=False)

#Conditional Imputation-To impute missing values based on the value of another column
def myAvgImpute(targetColumn,sourceColumn,criteria1,value1,criteria2,value2,dataFrame):
    for index, row in dataFrame.iterrows():
        if row[sourceColumn] == criteria1 and math.isnan(row[targetColumn]) == True:
            dataFrame.loc[index,targetColumn]=value1
        elif row[sourceColumn] == criteria2 and math.isnan(row[targetColumn]) == True:
            dataFrame.loc[index,targetColumn]=value2
    return dataFrame         
            
#To extract the deck from cabin number   
def deck(df):
    for index, row in df.iterrows():
        compoundString=row['Cabin']
        if pd.isnull(compoundString)!=True:
            trimmed=compoundString.strip()
            df.loc[index,'Deck']=trimmed[0]
    return df
        
#To determine average value of a target column dependent on
#information from another column.
#This helps to deal with missing data points using an educated guess
def condAverage(sourceColumn,criteria,targetColumn,dataFrame):
    x=[];
    for index, row in dataFrame.iterrows():
        if row[sourceColumn] == criteria and math.isnan(row[targetColumn]) != True:
            x.append(row[targetColumn])
      
    return sum(x) / float(len(x))

main()