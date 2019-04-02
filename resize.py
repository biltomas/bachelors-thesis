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

# data_path = 'D:\Downloads\\3471833\RawImage\RawImage\TrainingData'
# resize(data_path, 256, 256)
data_path = 'D:\learning_data2\\train\\0\\'
resize(data_path, 224, 224)
data_path = 'D:\learning_data2\\train\\1\\'
resize(data_path, 224, 224)
data_path = 'D:\learning_data2\\train\\2\\'
resize(data_path, 224, 224)
data_path = 'D:\learning_data2\\train\\3\\'
resize(data_path, 224, 224)
data_path = 'D:\learning_data2\\train\\4\\'
resize(data_path, 224, 224)
data_path = 'D:\learning_data2\\train\\5\\'
resize(data_path, 224, 224)
data_path = 'D:\learning_data2\\train\\6\\'
resize(data_path, 224, 224)
# delete(data_path)