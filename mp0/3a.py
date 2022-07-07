from PIL import Image
import numpy as np

def imageStack(img,name):
    image = Image.open(img)
    I_array = np.array(image)
    h,w = I_array.shape
    height = int(h/3)

    b = I_array[:height,::]
    g = I_array[height:2*height,::]
    r = I_array[2*height:3*height,::]

    combine = np.dstack([r,g,b])
    output = Image.fromarray(combine)

    name = './image_origin/origin_'+name+'.jpg'
    output.save(name)
    output.show()


imagelist = ['./prokudin-gorskii/00125v.jpg','./prokudin-gorskii/00149v.jpg','./prokudin-gorskii/00153v.jpg','./prokudin-gorskii/00351v.jpg',
                                        './prokudin-gorskii/00398v.jpg','./prokudin-gorskii/01112v.jpg']
name = ['125','149','153','351','398','1112']

for i in range(len(imagelist)):
    imageStack(imagelist[i],name[i])
