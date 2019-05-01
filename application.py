from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore
import datetime

cred = credentials.Certificate("./ServiceAccountKey.json")
firebase_app = firebase_admin.initialize_app(cred)
db = firestore.client()

application = app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')

single_parser = api.parser()
single_parser.add_argument('file', location='files',
  type=FileStorage, required=True)

model = load_model('my_model.h5')
graph = tf.get_default_graph()

@ns.route('/prediction')
class CNNPrediction(Resource):
  """Uploads your data to the CNN"""
  @api.doc(parser=single_parser, description='Upload an mnist image')
  def post(self):
    args = single_parser.parse_args()
    image_file = args.file
    image_file.save('last_image.png')
    img = Image.open('last_image.png')
    image_red = img.resize((28, 28)).convert('1')
    image = img_to_array(image_red)
    print("Image shape:", image.shape)
    x = image.reshape(1, 28, 28, 1)
    x = x/255
    with graph.as_default():
      out = model.predict(x)
    print(out[0])
    print(np.argmax(out[0]))
    r = np.argmax(out[0])

    pred = str(r)
    savePredictionToDB(pred)

    return {'prediction': pred}

def savePredictionToDB(pred):
  time = datetime.datetime.now()
  title = "pred " + time.strftime("%I:%M%p on %B %d, %Y")
  doc_ref = db.collection(u'userlogs').document(title)
  doc_ref.set({
    u'prediction': pred,
    u'time': time,
  })

if __name__ == '__main__':
  # app.run(host='0.0.0.0', port=8000)
  app.run(debug=True)


