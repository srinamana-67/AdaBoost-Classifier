# AdaBoost-Classifier

For this assignment, I have applied the AdaBoost Classifier method on a dataset that I have downloaded based on the Titanic Ship Crash. The dataset has 891 samples for training the model and 418 samples for testing. Overall there are 13 features in the dataset, out of which I only selected 5 for the classification.

For implementing the mentioned classification method, I have used Python. I have made use of two libraries in Python, namely pandas and sklearn. For using hte given libraries, we will need to first execute the following commands:

1. sudo pip install pandas 
2. sudo pip install -U scikit-learn

The steps required for implementing the given classifier are:

1. Import the required libraries.
2. Load the training data into the "train" variable from the csv file "titanic_train.csv"
3. Store the features selected for classification in the "features" variable
4. Load the testing data into the "test" variable from the csv file "titanic_test.csv"
5. Load the data of whether the person survived or not from the csv file "titanic_gender_targets.csv"
6. Store only the data of the selected features from the testing data in the X_test variable.
7. Store the target column of the testing data in Y_test variable.
8. Split training data into features and targets for training. Store features in X_train and targets in Y_train.
9. Encode the features data into machine-understandable form and now store in X_train and X_test.
10. Perform the AdaBoost classification on X_train and Y_train to train the model.
11. Test the classifier on X_test and Y_test variables to find the accuracy of the classifier.
12. Higher the value of accuracy, the better is the fit of the training model.
