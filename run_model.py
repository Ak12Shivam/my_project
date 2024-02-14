import sys
import preprocess
import train_model
import evaluate_model

if len(sys.argv) < 2:
    print("Usage: python run_model.py /Users/hp/OneDrive/Desktop/heart disease model/indian_liver_patient.csv")
    sys.exit(1)

data_file = sys.argv[1]
x_train, x_test, y_train, y_test = preprocess.preprocess(data_file)
model = train_model.train_model(x_train, y_train)
evaluate_model.evaluate_model(x_test, y_test, model)