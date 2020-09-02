# # app.py - a minimal flask api using flask_restful
# from flask import Flask
# from flask_restful import Resource, Api
#
# app = Flask(__name__)
# api = Api(app)
#
# class HelloWorld(Resource):
#     def get(self):
#         return {'hello': 'world'}
#
# api.add_resource(HelloWorld, '/')
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

# original code for the API
# load the Flask
import flask
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import json
from itertools import chain
app = flask.Flask(__name__)
app.config["DEBUG"] = True
# veriables
model_file = './lib/model'
tokenizer = model_file+'/tokenizer.json'
model = model_file+'/sentiment_dnn.h5'
vocab_size = 10000
max_review_length = 100
oov_tok = "<OOV>"
embedding_dim = 16
@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404
# function to convert score to negative, nutral and positive
def sentiment(score):
    if score >= 0.60:
        return('positive')
    elif (0.40 <= score < 0.60):
        return('neutral')
    else:
        return('negative')
# open tokenizer
with open(tokenizer) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
model = load_model(model)
# define the predict end point
@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}
    # get the request parameter
    params = flask.request.json
    if (params == None):
        params = flask.request.args

    if (params != None):
        x = str(params.get("text"))
        if (x=='None'):
            return page_not_found(404)
        else:
            print(x)
            validate_seq = tokenizer.texts_to_sequences([x])
            validate_padding = pad_sequences(validate_seq, truncating='post', padding='post', maxlen=max_review_length)
            print(validate_seq)
            score = model.predict(validate_padding)
            data["text"] = x
            data["response"] = str(sentiment(score))
            data["score"] = str(list(chain.from_iterable(score)))
            print(str(model.predict(validate_padding)))
            data["success"] = True

    # return the response
    return flask.jsonify(data)

# start the app
app.run(host='0.0.0.0', port=5000, threaded=True)

# curl -X POST -H "Content-Type: application/json" -d "{ \"text\":\"It is so boring restaurant with really bad food\" }" http://localhost:5000/predict
# GET http://103.86.177.136:5000/v1/predict?text='that movie was bad'
# GET http://localhost:5000/v1/predict?text="I've been around the area for quite some time to see this particular location change from one restaurant to another but finally I was tempted enough by the new owner's menu. I've walked by the place a couple of times to see it go from nothing to what I saw today. The place inside looked so inviting that it really made me want to come again(after quarantine is over)just for the ambiance. The menu is simple yet creative and spoke to my cravings. I decided to get the fried calamari, the springtime salad with salmon, the watermelon passion drink and the emerald pasta. EVERYTHING IS SO GREAT ABOUT THIS PLACE!!! Like omfggggggggggggggg djrifhdifjcusifuchdjfi is exactly how I feel about this place!! I'm coming back-for-sure-hands-down-no-doubt-about-it. The service is GREAT from the phone to in person. The food quality, quantity, and price are all 10/10 in my book. And let me tell you my book is exclusive, I'm usually not this impressed by a restaurant to really want to even write a review for it."
