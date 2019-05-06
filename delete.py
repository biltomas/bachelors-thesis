from PIL import Image                                              
import os, sys
# smaze obrazky z datasetu se spatnou priponou
def delete(data_path):
    dirs = os.listdir( data_path )
    for item in dirs:
        if not item.endswith(".png"):
            os.remove(os.path.join(data_path, item))

data_path = 'D:\Downloads\\3471833\RawImage\RawImage\TrainingData'