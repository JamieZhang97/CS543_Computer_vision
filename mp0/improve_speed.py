from PIL import Image
import numpy as np
import time

def ncc(img1,img2):
    h,w = img1.shape
    h1 = int(h/2)
    w1 = int(w/2)
    a = img1[h1:h1+50,w1:w1+50]
    b = img2[h1:h1+50,w1:w1+50]
    min = -1
    for i in range(-15,15):
        for j in range(-15,15):
            b1 = np.roll(b,[i,j],axis=(0,1))
            ncc_temp = np.sum((((a-a.mean(axis=0))/np.linalg.norm(a-a.mean(axis=0))) * ((b1-b1.mean(axis=0))/np.linalg.norm(b1-b1.mean(axis=0)))))
            if ncc_temp > min:
                min = ncc_temp
                offset = [i,j]
    img_align = np.roll(img2,offset,axis=(0,1))
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
    (green, offset1) = ncc(b,g)
    (red, offset2) = ncc(b,r)
    print('green offset, red offset: ',offset1,offset2)
    combine = np.dstack([red,green,b])
    output = Image.fromarray(combine)

    name = './image_ncc/ncc_'+name+'.jpg'
    output.save(name)
    output.show()

imagelist = ['./prokudin-gorskii/00125v.jpg','./prokudin-gorskii/00149v.jpg','./prokudin-gorskii/00153v.jpg','./prokudin-gorskii/00351v.jpg',
                                        './prokudin-gorskii/00398v.jpg','./prokudin-gorskii/01112v.jpg']
name = ['125','149','153','351','398','1112']

for i in range(len(imagelist)):
    time_start = time.time()
    imageAlign(imagelist[i],name[i])
    time_end = time.time()
    print('time cost',time_end-time_start,'s')
