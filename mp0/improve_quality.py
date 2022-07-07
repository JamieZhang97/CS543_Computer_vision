from PIL import Image
import numpy as np
import time


def remove_white_border(img):
    x = img.shape[1]
    y = img.shape[0]
    left_x = []
    right_x = []
    top_y = []
    bottom_y = []

    binary = np.copy(img)
    for i in range(x):
        for j in range(y):
            if int(img[j][i]) <= 80:
                binary[j][i] = 0
            elif int(img[j][i]) >= 220:
                binary[j][i] = 255
            else:
                binary[j][i] = img[j][i]

    for i in range(x-1):
        for j in range(y-1):
            if int(binary[j][i]) == 255 and binary[j][i+1] == 0 and i<0.05*x:
                left_x.append(i)
            elif int(binary[j][i]) == 0 and binary[j][i+1] == 255 and i>0.95*x:
                right_x.append(i)
            elif int(binary[j][i]) == 255 and binary[j+1][i] == 0 and j<0.01*y:
                top_y.append(j)
            elif int(binary[j][i]) == 0 and binary[j+1][i] == 255 and j>0.97*y:
                bottom_y.append(j)


    left = max(left_x)
    right = min(right_x)
    top = max(top_y)
    bottom = min(bottom_y)
    output = img[top:bottom,left:right]
    return output

def ncc(a,b):
    min = -1
    for i in range(-15,15):
        for j in range(-15,15):
            b1 = np.roll(b,[i,j],axis=(0,1))
            ncc_temp = np.sum((((a-a.mean(axis=0))/np.linalg.norm(a-a.mean(axis=0))) * ((b1-b1.mean(axis=0))/np.linalg.norm(b1-b1.mean(axis=0)))))
            if ncc_temp > min:
                min = ncc_temp
                offset = [i,j]
    img_align = np.roll(b,offset,axis=(0,1))
    return (img_align,offset)


def imageAlign(img,name):
    image = Image.open(img)
    I_array = np.array(image)
    I_array = remove_white_border(I_array)
    h,w = I_array.shape

    height = int(h/3)
    b = I_array[:height,::]
    g = I_array[height:2*height,::]
    r = I_array[2*height:3*height,::]

    # fixed blue channel
    (green, offset1) = ncc(b,g)
    (red, offset2) = ncc(b,r)
    print(offset1,offset2)
    combine = np.dstack([red,green,b])
    output = Image.fromarray(combine)

    name = './image_ncc/hq_ncc_'+name+'.jpg'
    output.save(name)
    output.show()

imagelist = ['./prokudin-gorskii/00125v.jpg','./prokudin-gorskii/00149v.jpg','./prokudin-gorskii/00153v.jpg','./prokudin-gorskii/00351v.jpg',
                                        './prokudin-gorskii/00398v.jpg','./prokudin-gorskii/01112v.jpg']
name = ['125','149','153','351','398','1112']

for i in range(len(imagelist)):
    time_start = time.time()
    imageAlign(imagelist[i],name[i])
    time_end = time.time()
    # print('time cost',time_end-time_start,'s')
