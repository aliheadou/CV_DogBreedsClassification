import os
import flask
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from flask import Flask , render_template  , request , send_file

from preprocess_input_image import *

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(os.path.join(BASE_DIR , 'best_Xcep.hdf5'))


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(filename , model):
    IMAGE_SIZE = 128
    CHANNELS   = 3
    images_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    # Load image
    img = keras.preprocessing.image.load_img(
        filename, target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    # Create batch axis
    img_array = tf.expand_dims(img_array, 0)  
    # Equalize pixels
    img_cv = prep_images(img_array)
    # Rescale image
    img_res = data_rescale(img_array, images_shape)

    # Inference
    predictions = model.predict(img_res)
    predictions = predictions[0]

    prob_result = []
    class_result = []
    for i in range(3):
        rank = np.argsort(predictions)[::-1][i]
        prob_result.append((predictions[rank]*100).round(2))
        breed = class_names(rank)
        class_result.append(' '.join(breed))

    return class_result , prob_result


@app.route('/')
def home():
        return render_template("home.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':            
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('new_predict.html' , img  = img , predictions = predictions)
            else:
                return render_template('home.html' , error = error)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug = True)
