from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import tree, neighbors, ensemble

import matplotlib.pyplot as plt

wdbc = datasets.load_breast_cancer()

X = wdbc.data
y = wdbc.target

print("Data shape: ", X.shape)
print("Target shape: ", y.shape)
print("Target names: ", wdbc.target_names)
print("First 5 feature names: ",wdbc.feature_names[:5])

model = svm.SVC()
model.fit(X,y)
predict = model.predict(X)

print("Accuracy: ", metrics.accuracy_score(y, predict))
print("Precision: ", metrics.precision_score(y, predict))
print("Recall: ",metrics.recall_score(y, predict))


models = {
    "SVM": svm.SVC(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "KNN": neighbors.KNeighborsClassifier(),
    "Random Forest": ensemble.RandomForestClassifier()
}
results={}

best_accuracy = 0
best_model = None
best_name = ""

for name, model in models.items():
    model.fit(X,y)
    predict = model.predict(X)
    print(name)
    print("Accuracy: ", metrics.accuracy_score(y, predict))
    print("Precision: ", metrics.precision_score(y, predict))
    print("Recall: ",metrics.recall_score(y, predict))
    print()

    results[name] = metrics.accuracy_score(y, predict)

    if results[name] > best_accuracy:
        best_accuracy = results[name]
        best_model = model
        best_name = name

print("Best classifier: ", best_name)
print("Best accuracy: ", best_accuracy)
print("Best model: ", best_model)

conf_matx = metrics.confusion_matrix(y, predict)
conf_disp = metrics.ConfusionMatrixDisplay(conf_matx, display_labels=wdbc.target_names)
conf_disp.plot()
plt.savefig("wdbc_classification_matrix.png")

plt.figure(figsize=(8,6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap='coolwarm'
)

plt.xlabel(wdbc.feature_names[0])
plt.ylabel(wdbc.feature_names[1])

plt.title("Breast Cancer Dataset Scatter Plot")

plt.savefig("wdbc_classification_scatter.png")

plt.show()

