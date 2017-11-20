import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def main():
    #reading in data
    dfTrain = pd.read_csv('C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\\ProcessedTrain.csv')
    dfPredict = pd.read_csv('C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\\ProcessedTest.csv')
    
    #just the list of features in string form
    featuresTrain = list(dfTrain.columns[2:8])
    label = "Survived"
    featuresPredict = list(dfPredict.columns[1:7])
    
    #making sure my columns will be in the right order
    print(featuresPredict)
    print(featuresTrain)
    
    #needed dataframes
    xPredict = dfPredict[featuresPredict]
    x=dfTrain[featuresTrain]
    y=dfTrain[label]
    
    #splitting data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    #Models and Evaluation Metrics

    #Decision Tree
    myTree = tree.DecisionTreeClassifier()
    myTree = myTree.fit(x_train,y_train)
    ys = myTree.predict(x_test)
    print("------------For Decision Tree------------")
    print("Accuracy is %.2f%%"%metrics.accuracy_score(y_test, ys))
    print("Precision is %.2f%%"%metrics.precision_score(y_test, ys))
    print("Recall is %.2f%%"%metrics.recall_score(y_test, ys))
    
    #Random Forest
    paras={'bootstrap': False,'min_samples_leaf': 3,'n_estimators': 50,'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    myForest = RandomForestClassifier(**paras)
    myForest = myForest.fit(x_train,y_train)
    ys = myForest.predict(x_test)
    print("------------For Random Forest------------")
    print("Accuracy is %.2f%%"%metrics.accuracy_score(y_test, ys))
    print("Precision is %.2f%%"%metrics.precision_score(y_test, ys))
    print("Recall is %.2f%%"%metrics.recall_score(y_test, ys))
    
    #SVM
    mySVM = SVC()
    mySVM = mySVM.fit(x_train,y_train)
    ys = mySVM.predict(x_test)
    print("---------------For SVM----------------")
    print("Accuracy is %.2f%%"%metrics.accuracy_score(y_test, ys))
    print("Precision is %.2f%%"%metrics.precision_score(y_test, ys))
    print("Recall is %.2f%%"%metrics.recall_score(y_test, ys))
    
    #Neural Network
    myNN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    myNN = myNN.fit(x_train,y_train)
    ys = myNN.predict(x_test)
    print("---------------For Neural Network----------------")
    print("Accuracy is %.2f%%"%metrics.accuracy_score(y_test, ys))
    print("Precision is %.2f%%"%metrics.precision_score(y_test, ys))
    print("Recall is %.2f%%"%metrics.recall_score(y_test, ys))
    
    #Compiling test predictions into my Results dataset
    dfPredict['Survived']=myForest.predict(xPredict)
    output=dfPredict[["PassengerId","Survived"]]
    output.to_csv("C:\Users\Vinay\Desktop\Data Science\Titanic Survival\Data\Results.csv",index=False)
    

main()