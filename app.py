from extract_features import get_features
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

def convertEncodingToPositive(data):
    mapping = {-1: 2, 0: 0, 1: 1}
    i = 0
    for col in data:
        data[i] = mapping[col]
        i+=1
    return data

def make_prediction(url):

    features = get_features(url)
    #print(features)
    features_extracted = convertEncodingToPositive(features)
    #print(features_extracted)

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)

    one_hot_enc = pickle.load(open("One_Hot_Encoder", "rb"))
    transformed_point = one_hot_enc.transform(np.array(features_extracted).reshape(1, -1))

    model = pickle.load(open("RF_Final_Model.pkl", "rb"))
    prediction = model.predict(transformed_point)[0]

    return prediction




@app.route('/')
def first():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    
    url = request.form['url']
    
    output = make_prediction(url)

    return render_template('home.html', prediction_text=output)




if __name__ == "__main__" :

    #Updates app dynamically
    app.run(debug=True)




# url = "https://www.github.com/karans-15"

# def make_prediction(url):

#     features = get_features(url)
#     #print(features)
#     features_extracted = convertEncodingToPositive(features)
#     #print(features_extracted)

#     from sklearn.preprocessing import OneHotEncoder
#     encoder = OneHotEncoder(sparse=False)

#     one_hot_enc = pickle.load(open("One_Hot_Encoder", "rb"))
#     transformed_point = one_hot_enc.transform(np.array(features_extracted).reshape(1, -1))

#     model = pickle.load(open("RF_Final_Model.pkl", "rb"))
#     prediction = model.predict(transformed_point)[0]

#     if(prediction==1):
#         print("Website is SAFE!")
#     elif(prediction==2):
#         print("DANGER!! This appears to be a phishing website")
#     else:
#         print("Proceed with CAUTION, this seems Suspicious")

# make_prediction(url)