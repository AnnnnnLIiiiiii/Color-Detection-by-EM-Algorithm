import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import em


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


images = load_images_from_folder("yellow")
# image = cv2.imread("/home/an/Desktop/673/Project_3/green/Train/6.jpg")

hist_bgr = np.zeros((256, 256, 256))
for img in images:
    bgr_update = cv2.calcHist([images[0]], [0, 1, 2], None, [256, 256, 256], [50, 256, 50, 256, 50, 256])
    hist_bgr += bgr_update
hist_bgr /= len(images)
hist_bgr /= hist_bgr.sum()
num_data = 2
train = np.zeros((1, 3))  # Initialize training data
print('hist max, ', hist_bgr.max())
for scale in range(1, num_data):
    scale /= num_data
    scale *= hist_bgr.max()
    train_tmp = np.stack(np.where(hist_bgr >= scale), axis=1)
    if scale == 1/num_data:
        train = train_tmp
    else:
        train = np.concatenate((train, train_tmp))
train /= 255  # Training data normalization
print('train shape, ', train.shape)


yellow_gmm = em.GaussianMixture(train, 1)
hist_br = np.sum(hist_bgr, axis=1)
# br_gmm = em.GaussianMixture(hist_br, 1)
iteration = 200
for _ in range(iteration):
    yellow_gmm.train()
print("likelihood: ", yellow_gmm.getLikelihood()-len(train)*3*np.log(255))

# Video Reader
vid = cv2.VideoCapture("detectbuoy.avi")
success, img = vid.read()
count = 0
while success:
    # img = cv2.imread("buoy/frame196.jpg")
    img_result = img.copy()
    img_shape = np.shape(img)
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    tmp = np.reshape(blur, (-1, 3)) / 255  # Testing data normalization

    print('Gaussian mean: ', yellow_gmm.getModel()[0][0]*255)
    prob = yellow_gmm.getPdf(tmp)
    prob = np.reshape(prob, (img_shape[0], img_shape[1]))

    print('prob max, ', prob.max())
    print("prob sum:", np.sum(prob)/(255**3))
    print("prob mean", np.mean(prob))

    #_, img_thresh = cv2.threshold(prob, prob.mean()*2, 255, cv2.THRESH_BINARY)
    _, img_thresh = cv2.threshold(prob, 1, 255, cv2.THRESH_BINARY)
    _, img_thresh_slack = cv2.threshold(prob, 10e-3, 255, cv2.THRESH_BINARY)
    img_thresh = img_thresh.astype(np.uint8)
    print(img_thresh.max())
    seg = cv2.bitwise_and(img, img, mask=img_thresh)  # segmentaion

    #  Find bonding box
    contours, hierarchy = cv2.findContours(img_thresh, 1, 2)
    x = np.shape(img)[0]-1
    y = np.shape(img)[1]-1
    w = h = 0
    if contours:
        area = 100
        for cnt in contours:
            if cv2.contourArea(cnt) > area:
                area = cv2.contourArea(cnt)
                print('area:', area)
                x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(seg, (x-int(0.5*w), y-int(0.5*h)), (x + int(1.5*w), y + int(1.5*h)), (0, 255, 0), 2)

    mask = np.zeros((img_shape[0], img_shape[1])).astype(np.uint8)
    cv2.rectangle(mask, (x - int(0.5 * w), y - int(1.5*h)), (x + int(1.5 * w), y + int(1.5 * h)), 255, -1)
    mask = cv2.bitwise_and(mask, img_thresh_slack.astype(np.uint8))
    img = cv2.bitwise_and(img, img, mask=mask)

    # Fit circle
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    if contours:
        area = 0
        center = (0, 0)
        radius = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > area:
                area = cv2.contourArea(cnt)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
        cv2.circle(img_result, center, radius, (0, 255, 0), 2)
    cv2.imshow("preprocess", seg)
    cv2.imshow("crop", img)
    cv2.imshow("Result", img_result)
    cv2.waitKey(0)

    success, img = vid.read()

cv2.destroyAllWindows()
vid.release()

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(img_gray, 100, 200)

# x = y = np.arange(0, 256, 1)
# X, Y = np.meshgrid(x, y)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, hist_br, cmap='binary')
# plt.contourf(X, Y, hist_br, cmap='hot')
# plt.show()

