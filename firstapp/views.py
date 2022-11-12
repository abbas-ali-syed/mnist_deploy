from django.shortcuts import render

# Create your views here.
from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array 
from tensorflow.python.keras import Sequential
import tensorflow as tf
from tensorflow import Graph
img_height=28
img_width=28
model=load_model('./models/mni.h5')

model_graph= Graph()
with model_graph.as_default():
 tf_session=tf.compat.v1.Session()
 with tf_session.as_default():
    # with tf_session.graph.as_default():
     model=load_model('./models/mni.h5')


def index(request):
    context={'a':1}
    return render(request,'index.html',context)


def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName =fs.url(filePathName)
    testimage='.'+filePathName
    img=tf.keras.utils.load_img(testimage,target_size=(img_height,img_width))
    x=tf.keras.utils.img_to_array(img)
    x=x/255
    x=x.reshape(-1,img_height,img_width)
    with model_graph.as_default():
     with tf_session.as_default():
      predictions=model.predict(x)
        
    import numpy as np 

    predictions=str(np.argmax(predictions))
    #predictedLabel=model.predict(x)
    #,'predictedLabel':predictedLabel
    context={'FilePathName':filePathName,'Predictions':predictions}
    return render(request,'index.html',context)
