from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(x_test, y_test, model):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))