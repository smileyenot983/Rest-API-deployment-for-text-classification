# import numpy as np
import torch

import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

import Preprocessor

from models import RNNModel

app = Flask(__name__)
api = Api(app)




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model_path = 'rnn_pretrained.pt'


model = RNNModel()
model.load_state_dict(torch.load(model_path,map_location=device))
model.to(device)

preproc = Preprocessor.Preprocessor()

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        sentence = posted_data['text']
        # print(f"sentence: {sentence}")
        user_tokens = preproc.prep(sentence)
        encoded_tokens = torch.tensor([preproc.encode(token) for token in user_tokens]).unsqueeze(0).to(device)

        # to get probabilities(makes results easier to interpret)
        softmax = torch.nn.Softmax()

        pred = model(encoded_tokens)
        pred_probas = softmax(pred)
        pred_score = float(max(pred_probas[0])) 

        pred_class = 'python' if pred[0][0]>pred[0][1] else 'data science'
        
        return jsonify({
            'Class': pred_class,
            'Score': pred_score
        })

api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run('127.0.0.1',port=5500,debug=True)