import streamlit as st
import pickle
import numpy as np 

model = pickle.load(open('model.pkl','rb'))
education_dict = {'graduate school':1,'university':2,'high school':3,'others':4}
marriage_dict = {'married':1,'single':2,'others':3}
PAY_1_dict = {'account started that month with a zero balance, and never used any credit':-2,'account had a balance that was paid in full':-1,'atleast the minimum payment was made, but the entire balance was not paid':0, 'payment delay for 1 month':1, 'payment delay for 2 months':2,'payment delay for 3 months':3, 'payment delay for 4 months':4,'payment delay for 5 months':5,'payment delay for 6 months':6,'payment delay for 7 months':7,'payment delay for 8 months':8}

def predict(LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6):
    EDUCATION = education_dict[EDUCATION]
    MARRIAGE = marriage_dict[MARRIAGE]
    PAY_1 = PAY_1_dict[PAY_1]
    input_features = np.array([LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]).astype(np.float64).reshape(1,-1)
    prediction = model.predict_proba(input_features)
    pred='{0:.{1}f}'.format(prediction[0][0],4)
    return float(pred)

def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card Default Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    LIMIT_BAL = st.number_input("Limited Balance (in New Taiwanese (NT) dollar)",)
    EDUCATION = st.selectbox("Select Education",("graduate school","university","high school","others"))
    MARRIAGE = st.selectbox("Marrital Status",("married","single","others"))
    AGE = st.slider("Age (in years)",1,100)
    PAY_1 = st.selectbox("Last Month Payment Status",('account started that month with a zero balance, and never used any credit','account had a balance that was paid in full','atleast the minimum payment was made, but the entire balance was not paid','payment delay for 1 month','payment delay for 2 months','payment delay for 3 months','payment delay for 4 months','payment delay for 5 months','payment delay for 6 months','payment delay for 7 months','payment delay for 8 months'))
    BILL_AMT1 = st.number_input("Last month Bill Amount (in New Taiwanese (NT) dollar)")
    BILL_AMT2 = st.number_input("Last 2nd month Bill Amount (in New Taiwanese (NT) dollar)")
    BILL_AMT3 = st.number_input("Last 3rd month Bill Amount (in New Taiwanese (NT) dollar)")
    BILL_AMT4 = st.number_input("Last 4th month Bill Amount (in New Taiwanese (NT) dollar)")
    BILL_AMT5 = st.number_input("Last 5th month Bill Amount (in New Taiwanese (NT) dollar)")
    BILL_AMT6 = st.number_input("Last 6th month Bill Amount (in New Taiwanese (NT) dollar)")
    PAY_AMT1 = st.number_input("Amount paid in Last Month (in New Taiwanese (NT) dollar)")
    PAY_AMT2 = st.number_input("Amount paid in Last 2nd Month (in New Taiwanese (NT) dollar)")
    PAY_AMT3 = st.number_input("Amount paid in Last 3rd Month (in New Taiwanese (NT) dollar)")
    PAY_AMT4 = st.number_input("Amount paid in Last 4th Month (in New Taiwanese (NT) dollar)")
    PAY_AMT5 = st.number_input("Amount paid in Last 5th Month (in New Taiwanese (NT) dollar)")
    PAY_AMT6 = st.number_input("Amount paid in Last 6th Month (in New Taiwanese (NT) dollar)")

    if st.button("Predict"):
        output=predict(LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6)
        st.success('The probability of not making credit card default is {0:.{1}f}%'.format(output * 100,2))


if __name__=='__main__':
    main()
