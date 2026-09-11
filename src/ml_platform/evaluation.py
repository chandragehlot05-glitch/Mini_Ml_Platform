from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    return accuracy, matrix

def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0)
