#import the required libraries
import pandas  
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier

#Load training data (csv file stored in the same directory as the program) 
train = pandas.read_csv("titanic_train.csv")

#Features selected for testing and training from the dataset)
features = ["Pclass","Sex","Age","SibSp","Parch"]

#Load the features to be tested
test = pandas.read_csv("titanic_test.csv")

#Select features and parse the result to pandas dataframe
X_test = pandas.DataFrame(test.loc[:,features].values)
print(X_test,"\n")

#Load test targets
targets = pandas.read_csv("titanic_gender_targets.csv")

#Select which column to target
Y_test = targets.loc[:,"Survived"].values
print(Y_test,"\n")

#Split train data into features and targets for training
X_train = pandas.DataFrame(train.loc[:,features].values)
print(X_train)
Y_train = train.loc[:,"Survived"].values
print(Y_train,"\n")

#Data encoding 
le = preprocessing.LabelEncoder()
X_train = X_train.apply(le.fit_transform)	
X_test = X_test.apply(le.fit_transform)	

#Create a Ada Boost Classifier instance 
classifier = AdaBoostClassifier()

#Fit the classifier
classifier.fit(X_train,Y_train)

#Calculate the score (Accuracy)
accuracy = classifier.score(X_test,Y_test)
print('Accuracy of the AdaBoost Classifier on the dataset = ',accuracy)
