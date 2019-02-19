# Importing numpy for the mathematical functions
# Importing tree from sklearn to use decision tree classifier
# Importing the iris database from sklearn datasets
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

# Loading the iris data set
iris_data_set = load_iris()

print("Hello human!")
print("\nI'm in need of your help.")
print("\nI have great knowledge of Iris flowers but I can't seem to understand how this testing device works.")
print("\nFor years I have been predicting what type of Iris flowers these are, but if only I could use this testing device....")
print("\nI would know for sure if I've been right all this time.")
print("\nAre you up to the task human?")
wait = input("\nPress enter to help the machine.")
print("-" * 40)

print("\nGreat! Let's begin then.")
print("\nFirst you need to view the data the testing device uses to help you understand the end result.")
wait = input("Press enter to view data.")
print("-" * 40)

# Print out iris dataset features
print(" " * 40)
print("Iris features:")
print(iris_data_set.feature_names)


# Print out the iris labels
print(" " * 40)
print("Iris labels:")
print(iris_data_set.target_names)

# Prints the label legend
print(" " * 40)
print("Label legend:")
print("0 = Setosa 1 = Versicolor 2 = Virginica")

# Print out row 0 of the data set, then show the label for the row for reference
print("-" * 40)
print("Row example(row 0):")
print(iris_data_set.data[0])
print("Row 0 label =", iris_data_set.target[0])

print("_" * 40)
print("Dataset functionality: ")
print("- Classifies 3 different types of flowers")
print("\n- 150 rows")
print("\n- 50 rows per flower")
print("\n- Rows 0-49 = Setosa")
print("\n- Rows 50-99 = Versicolor")
print("\n- Rows 100-149 = Virginica")

# Prints out the whole data set and structures it
print("_" * 40)
print("Full dataset:")
for i in range(len(iris_data_set.target)):
    print("Row %d: Label %s: Features %s:" % (i, iris_data_set.target[i], iris_data_set.data[i]))

# Creating an index of 1 row for each flower type
# These rows wont be used as training
test_index = [0, 50, 100]


# Using the remaining 147 out of 150 rows as training data
training_target = np.delete(iris_data_set.target, test_index)
training_data = np.delete(iris_data_set.data, test_index, axis=0)

# Setting the 3 rows we indexed as tests
test_target = iris_data_set.target[test_index]
test_data = iris_data_set.data[test_index]

# Creating a classifier
# Using the decision tree classifier from tree
# Fitting the training data and target
dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(training_data, training_target)

print(" " * 40)
print("\nNow that you have seen all the data, are you ready to tell me if my prediction is right or......")
print("\nwrong...")
wait = input("Press enter when you're ready")

# Printing out the test data labels
# Printing out the prediction
print("-" * 40)
print("Testing device results: ")
print(test_target)
print("-" * 40)
print("My prediction is this.")
print(dt_clf.predict(test_data))
print("Am I right human?!")


