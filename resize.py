from PIL import Image                                              
import os, sys

def resize(data_path, sizeX, sizeY):
    dirs = os.listdir( data_path )
    for item in dirs:
        print(item)
        if os.path.isfile(data_path+item):
            im = Image.open(data_path+item)
            f, e = os.path.splitext(data_path+item)
            imResize = im.resize((sizeX,sizeY), Image.ANTIALIAS)
            # imResize.save(f + ' resized.png', 'PNG', quality=90)
            imResize.save(f + '.png', 'PNG', quality=90)

def delete(data_path):
    dirs = os.listdir( data_path )
    for item in dirs:
        if not item.endswith(".png"):
            os.remove(os.path.join(data_path, item))

data_path = 'D:\learning_data\\train\\1\\'
resize(data_path, 256, 256)
# delete(data_path)