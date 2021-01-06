
import cvlib as cv
import numpy as np
import dlib
import cv2
import os
import numpy as np
from PIL import Image
import distutils.dir_util

root_path = "E:/Data_Science/Gender_Age/Datasets/CACD/"
fault=[];
keep_picture = []
one=root_path + 'res/'
print("one",one)
count=0;
destination_path="E:/Data_Science/Gender_Age/Datasets/CACD/destination_folder"
#fol=1;
#out_path = "E:/Data_Science/Gender_Age/Datasets/CACD/1"
for picture_name in os.listdir(one):
##    if(count>=10000):
##        out_path = "E:/Data_Science/Gender_Age/Datasets/CACD/"+str(fol)
##        print(out_path)        
##        distutils.dir_util.mkpath(out_path)
##        count=0
##        fol=fol+1
##    print("image  before :",picture_name)
##    picture_name = picture_name.replace("å", "a")
##    print("image after :",picture_name)
    print("count:",count)
    count=count+1
    path1= root_path + 'CACD2000/'+ picture_name
    #print("image",path1)
    try:
        img = cv2.imread(path1)

        face, confidence = cv.detect_face(img) 
        

            # loop through detected faces
        for idx, f in enumerate(face):

            # get corner points of face rectangle       
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]


            # crop the detected face region
            tmp = np.copy(img[startY:endY,startX:endX])
                       
            try:
                tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((120, 120), Image.ANTIALIAS))

                cv2.imwrite( destination_path+ '/' + picture_name, tmp)
                keep_picture.append(picture_name)
            except ValueError:
                pass
            break;
    except:
        fault.append(path1)
        continue

with open("output.txt", "w") as txt_file:
    for line in fault:
        txt_file.write("".join(line) + "\n") # works with any number of elements in a lin
