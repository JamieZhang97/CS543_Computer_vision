from PIL import Image
import numpy as np


def ssd(a, b):
    max = 9999999999
    for i in range(-15,15):
        for j in range(-15,15):
            b1 = np.roll(b,[i,j],axis=(0,1))
            ssd_temp = np.sum((a-b1)**2)
            if ssd_temp < max:
                max = ssd_temp
                offset = [i,j]
    img_align = np.roll(b,offset,axis=(0,1))
    return (img_align,offset)



def imageAlign(img,name):
    image = Image.open(img)
    I_array = np.array(image)
    h,w = I_array.shape
    height = int(h/3)
    b = I_array[:height,::]
    g = I_array[height:2*height,::]
    r = I_array[2*height:3*height,::]

    # fixed blue channel
    (green, offset1) = ssd(b,g)
    (red, offset2) = ssd(b,r)
    print(offset1,offset2)
    combine = np.dstack([red,green,b])
    output = Image.fromarray(combine)

    name = './image_ssd/ssd_'+name+'.jpg'
    output.save(name)
    output.show()

imagelist = ['./prokudin-gorskii/00125v.jpg','./prokudin-gorskii/00149v.jpg','./prokudin-gorskii/00153v.jpg','./prokudin-gorskii/00351v.jpg',
                                        './prokudin-gorskii/00398v.jpg','./prokudin-gorskii/01112v.jpg']
name = ['125','149','153','351','398','1112']

for i in range(len(imagelist)):
    imageAlign(imagelist[i],name[i])
