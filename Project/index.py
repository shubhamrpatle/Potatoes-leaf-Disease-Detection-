#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Model Loading
# filepath = './model.h5'
# model = load_model(filepath)

saved_model_path = "./my_model"
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.keras.models.load_model(saved_model_path, options=load_options)



def pred_potato_disease(potato_leaf):
    #print("@@ Got Image for prediction")
    #test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    #test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
    class_names = ["Early_Blight", "Healthy", "Late_Blight"]
    
    image = load_img(potato_leaf, target_size = (256, 256)) # load image
    # image = tf.keras.preprocessing.image.load_img(potato_leaf)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)

    tem = predictions[0][0]
    index = 0
    curr_index = 0
    for i in predictions[0]:
        if i > tem:
            tem = i
            index = curr_index
        curr_index += 1
    
    # print(tem, index)
    
    predicted_class = class_names[index]
    confidence = round(100 * (tem), 2)

    predicted_class.replace("_", " ")
    
    # except:
    #     predicted_class = False
    #     confidence = 0
    # print(predictions)



    # print(predicted_class, confidence)
    # predicted_class = class_names[np.argmax(predictions[0])]
    # confidence = round(100 * (np.max(predictions[0])), 2)
    # print('@@ Raw result = ', predicted_class, confidence)

    return predicted_class, confidence

    # print(predictions)
    # return predictions
  


    

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        # print("@@ Input posted = ", filename)
        
        file_path = os.path.join('./static/upload', filename)
        file.save(file_path)

        # print("@@ Predicting class......")

        prediction, confidence = pred_potato_disease(potato_leaf=file_path)

        if not prediction:
              prediction = "Connot Predict"
        if confidence < 20:
            prediction = "Connot Predict"
            confidence = 0

        return render_template("output.html", prediction_output=prediction, confidence=confidence, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
    
    
