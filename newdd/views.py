from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    data_path = r"C:\Users\sudha\Downloads\diadataset.csv"  # Update with your dataset path
    data = pd.read_csv(data_path)
    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)

    if request.method == 'GET':
        try:
            val1 = float(request.GET.get('n1'))
            val2 = float(request.GET.get('n2'))
            val3 = float(request.GET.get('n3'))
            val4 = float(request.GET.get('n4'))
            val5 = float(request.GET.get('n5'))
            val6 = float(request.GET.get('n6'))
            val7 = float(request.GET.get('n7'))
            val8 = float(request.GET.get('n8'))

            pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

            if pred == 1:
                result = "Positive"
            else:
                result = "Negative"

        except ValueError:
            result = "Enter correct numeric values."

        return render(request, "predict.html", {"result2": result})
