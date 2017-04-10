import numpy
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load data
dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=",")

# print dataset
#
# print '-'*10

# split data into X and y
# X is data set from columns 0 to 7 (i.e first 8 columns)
X = dataset[:,0:8]
# Y is classified outcome (0 or 1), column 9 (as index starts at 0)
Y = dataset[:,8]

# split data into train and test sets

# The seed number is the starting point used in the generation of a sequence of random numbers
# Pseudo-random number generator state used for random sampling.
# Does not really matter unless want to have reproducible result
seed = 1
# Train with 2/3, test with 1/3
test_size = 0.33
# train_test_split - Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model on training data
# model is now a class 
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
