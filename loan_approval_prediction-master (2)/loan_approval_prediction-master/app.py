import math
import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index.html')
def main_page():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    dependents = int(request.form['dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self-employed'])
    applicant_income = int(request.form['applicant-income'])
    co_applicant_income = int(request.form['co-applicant-income'])
    total_income = int(applicant_income+co_applicant_income)
    log_total_income = math.log(int(total_income))
    loan_amt = int(request.form['loan-amount'])
    loan_amt_term = int(request.form['loan-amount-term'])
    emi = float(loan_amt/loan_amt_term)
    credit_history = int(request.form['credit-history'])
    property_area = int(request.form['property-area'])
    log_loan_amt = math.log(int(loan_amt))
    balance = float(total_income - (emi*1000))

    final_features = np.array([[gender, married, dependents, education, self_employed, credit_history, property_area, log_loan_amt, total_income, log_total_income,  emi, balance]])
    expected_emi = ((loan_amt *1000) + (loan_amt *80 )) /loan_amt_term
    data = pd.DataFrame(final_features)
    print(final_features)
    prediction = model.predict(data)
    output = prediction[0]
    print(output)
    if output == 1:
        return render_template('Congrats.html', monthly_emi='Your emi would be $ {} with a interest of 8%'.format(expected_emi))
    else:
        return render_template('Denied.html')


if __name__ == "__main__":
    app.run(debug=True)