from PIL import Image
import os

for file in os.listdir('img'):
    if not file.startswith('origin'):
        continue
    if not file.endswith('.jpg'):
        continue
    im = Image.open('img/' + file)
    imGrey = im.convert('L').resize((28,28), Image.ANTIALIAS)
    (filename, extension) = os.path.splitext(file)
    imGrey.save('img/' + filename + '.png')