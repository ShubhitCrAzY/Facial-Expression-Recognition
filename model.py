from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.15
session=tf.compat.v1.Session(config=config)

class FacialExpressionModel(object):

    EMOTIONS_LIST=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

    def __init__(self,model_json_file,model_weights_file):
        with open(model_json_file,'r') as json_file:
            loaded_model=json_file.read()
            self.loaded=model_from_json(loaded_model)

        self.loaded.load_weights(model_weights_file)
        #self.loaded._make_predict_function()

    def predict_emotion(self,img):
        self.pred=self.loaded.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.pred)]
