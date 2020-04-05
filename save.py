# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:10:08 2019

@author: Bobykhani
"""

model_json = final.to_json()
with open("save/FirstTry2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
final.save_weights("save/FirstTry2.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")