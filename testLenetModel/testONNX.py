import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt
import onnxruntime as rt
from onnx import shape_inference
import onnx
import onnx.helper
import numpy as np
import onnxruntime as ort
import cv2 

def preprocess(inputImage, imSize):

    img = cv2.imread(inputImage)
    cv2.imshow("input",img)
    cv2.waitKey(0)
    imResized = cv2.resize(img, (imSize,imSize))
  
    img_out = np.expand_dims(imResized[:,:,0],0) # 1,28,28
    img_out = np.expand_dims(img_out,0) # 1,1,28,28 : NCHW

    return img_out

if __name__ == '__main__':

    modelFileName = "../onnxModels/lenetTF.onnx"
    model = onnx.load(modelFileName)
    model_graph = model.graph
    print (onnx.helper.printable_graph(model_graph))
    # model_out = shape_inference.infer_shapes(model)
    # print (model_out)
    
    imSize = 28
    class_names = [0,1,2,3,4,5,6,7,8,9]
    imagePath = "../images/im_1.png"
 
    inTensor = preprocess(imagePath, imSize)
    inputTensor = np.asarray(inTensor,dtype=np.float32)  
    print ("input shape:",inputTensor.shape, inputTensor.dtype)

    sess = ort.InferenceSession(modelFileName)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print ("in/out tensor names:",input_name,output_name)

    res = sess.run([output_name], {input_name: inputTensor})[0]
    output = res[0]
    # print ("output tensor:",output)
    maxIndex = np.argmax(output)
    print("prediction result:")
    print ("input image is \"%s\" in ONNX lenet model"%class_names[maxIndex])