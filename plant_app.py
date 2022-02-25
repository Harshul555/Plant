from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)





def pred(k):
	plant_class = {0:'diseased cotton leaf',
	 1:'diseased cotton plant',
	 2:'fresh cotton leaf',
	 3:'fresh cotton plant'}
	model_path2 = 'plant_model.h5'
	cnn_model=load_model(model_path2)
	img = image.load_img(k,target_size=(150,150))
	img = np.expand_dims(img,axis = 0)
	r = list(cnn_model.predict(img)[0])
	return plant_class[	r.index(1)]


@app.route('/')

def home():
	return render_template('index.html')

@app.route('/predict', methods = ["GET","POST"])
def predict():
	p = "None"
	if request.method == "POST":
		f = request.files['file']
		sfname = "static/images/"+str(secure_filename(f.filename))
		f.save(sfname)
		p = pred(sfname)
		return render_template("index.html", p = p, imgpath = sfname)

if __name__ == '__main__':
	#app.debug = True
	app.run()