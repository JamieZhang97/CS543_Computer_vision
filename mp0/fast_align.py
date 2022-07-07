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


def nccalign(img,res,name,off1,off2):
    I_array = np.array(img)
    h,w = I_array.shape
    height = int(h/3)
    b = I_array[:height,::]
    g = I_array[height:2*height,::]
    r = I_array[2*height:3*height,::]

    g = np.roll(g,off1,axis=(0,1))
    r = np.roll(r,off2,axis=(0,1))
    # fixed blue channel

    (green, offset1) = ssd(b,g)
    (red, offset2) = ssd(b,r)

    combine = np.dstack([red,green,b])
    output = Image.fromarray(combine)
    print(offset1, offset2)
    name = './image_pyramid/pyramid_'+name+'_'+res+'.jpg'
    output.save(name)
    output.show()
    return (offset1, offset2)


def imageAlign(img,name):
    image = Image.open(img)
    I_array = np.array(image)
    h,w = I_array.shape

    half = image.resize((int(w/2),int(h/2)))
    half_array = np.array(half)
    h1, w1 = half_array.shape

    (offset1,offset2) = nccalign(half,'low',name,[0,0],[0,0])
    offset1 = [2*offset1[0],2*offset1[1]]
    offset2 = [2*offset2[0],2*offset2[1]]
    (offset3,offset4) = nccalign(image,'high',name,offset1,offset2)


imagelist = ['seoul_tableau.jpg','vancouver_tableau.jpg']
name = ['seoul','vancouver']
for i in range(len(imagelist)):
    imageAlign(imagelist[i],name[i])
