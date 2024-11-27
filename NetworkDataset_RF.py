from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('datasets/UNSW_NB15_training-set.csv')

df = df.drop(columns=['proto','service','state','attack_cat'])

x_train = df.iloc[:, :-1]
y_train = df.label

clf = RandomForestClassifier()

print("Training has started....")
clf.fit(x_train, y_train)

df_test = pd.read_csv('datasets/UNSW_NB15_testing-set.csv')

df_test = df_test.drop(columns=['proto','service','state','attack_cat'])

x_test = df_test.iloc[:, :-1]
y_test = df_test.label

print(clf.score(x_test, y_test))

y_pred = clf.predict(x_test)

accuracy = clf.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1}")

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision}")
print(f"Recall: {recall}")

y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2, 3])

fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()