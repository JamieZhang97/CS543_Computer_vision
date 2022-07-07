# Code from Lana Lazebnik.
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage.filters import gaussian_laplace, rank_filter, generic_filter
import os, sys, numpy as np, cv2
import time
from skimage import transform

initial = 1.5
k = 1.5
level = 10


def show_all_circles(image, cx, cy, rad, out_dir,imname, mode, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    if mode == 'up':
        out_file_name = os.path.join(out_dir, str(imname)+'increase.png')
    elif mode == 'down':
        out_file_name = os.path.join(out_dir, str(imname)+'downsample.png')
    plt.savefig(out_file_name)

    # plt.show()


def nms(array):
    mid = array[int(3*3*3/2)]
    if mid >= max(array):
        return mid
    else:
        return 0

def rank_nms(scale_space):
    h, w, n = scale_space.shape
    nms_2d = np.zeros((h,w,level))
    nms_3d = np.zeros((h,w,level))
    # rank filter
    for i in range(level):
        nms_2d[:,:,i] = rank_filter(scale_space[:,:,i], -1, (3,3))
    for i in range(h):
        for j in range(w):
            max_val = max(nms_2d[i,j,:])
            max_idx = np.argmax(nms_2d[i,j,:])
            nms_3d[i, j, max_idx] = max_val
    for i in range(h):
        for j in range(w):
            nms_3d[i,j,:] = np.where(nms_3d[i,j,:] == scale_space[i,j,:], scale_space[i,j,:], 0 )

    return nms_3d

def blob_detector_increase(image):
    threshold = 0.01
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)/255.

    h, w = gray.shape
    scale_space = np.zeros((h,w,level))
    # Laplacian Gaussian filter
    for i in range(level):
        sigma = initial*k**i
        scale_normalize = sigma**2*gaussian_laplace(gray, sigma=sigma)
        scale_space[:,:,i] = scale_normalize**2

    # generic filter
    nms_3d = generic_filter(scale_space, nms, size = (3,3,3))
    # nms_3d = rank_nms(scale_space)
    # nms_3d = nms_3d/np.max(nms_3d)
    cx = []
    cy = []
    radius = []

    for i in range(level):
        sigma = initial*k**i
        cx.append(list(np.where(nms_3d[:, :, i] > threshold)[1]))
        cy.append(list(np.where(nms_3d[:, :, i] > threshold)[0]))
        radius.append([np.sqrt(2)*sigma]*len(cx[i]))

    cx = np.concatenate(cx)
    cy = np.concatenate(cy)
    radius = np.concatenate(radius)

    return gray,cx,cy,radius

def blob_detector_downsample(image):
    threshold = 0.003
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)/255.

    h, w = gray.shape
    scale_space = np.zeros((h,w,level))
    for i in range(level):
        scale = k**i
        scale_gray = transform.resize(gray,(int(h/scale), int(w/scale)), mode = 'reflect')
        square_lg = gaussian_laplace(scale_gray, sigma=initial)**2
        scale_space[:,:,i] = transform.resize(square_lg, (h,w), mode = 'reflect')

    nms_3d = generic_filter(scale_space, nms, size = (3,3,3))
    # nms_3d = rank_nms(scale_space)
    # nms_3d = nms_3d/np.max(nms_3d)
    cx = []
    cy = []
    radius = []

    for i in range(level):
        sigma =  initial*k**i
        cx.append(list(np.where(nms_3d[:, :, i] > threshold)[1]))
        cy.append(list(np.where(nms_3d[:, :, i] > threshold)[0]))
        radius.append([np.sqrt(2)*sigma]*len(cx[i]))

    cx = np.concatenate(cx)
    cy = np.concatenate(cy)
    radius = np.concatenate(radius)

    return gray,cx,cy,radius


if __name__ == '__main__':
    IMAGE_DIR = 'blobs-data'
    imagelist = ['101085','119082','182053','210088','butterfly','einstein','fishes','sunflowers']
    # imagelist = ['fishes']
    out_dir = 'blobs_output'
    for i in range(len(imagelist)):
        imname = imagelist[i]
        I = cv2.imread(os.path.join(IMAGE_DIR, imname + '.jpg'))
        start = time.time()
        gray,cx,cy,radius = blob_detector_increase(I)
        show_all_circles(gray, cx, cy, radius, out_dir, imname,'up')
        end = time.time()
        print("running time for up sample is:" , end - start, "s.")

        start = time.time()
        gray,cx,cy,radius = blob_detector_downsample(I)
        # gray,cx,cy,radius = Downsample(I)
        show_all_circles(gray, cx, cy, radius, out_dir, imname,'down')
        end = time.time()
        print("running time for down sample is:" , end - start, "s.")
