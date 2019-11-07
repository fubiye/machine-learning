from PIL import Image
import numpy as np,os
import tensorflow as tf
model = tf.keras.models.load_model('D:\\workspace\\mnist\\mnist.h5')
for file in os.listdir('img'):
    if not file.startswith('origin'):
        continue
    if not file.endswith('.png'):
        continue
    im = Image.open('img/' + file)
    imArray = np.array(im)
    predictions = model.predict([[imArray]])
    print("Prediction of image:" + file)
    print(np.argmax(predictions[0]))
    print('\n')
print('Done!')