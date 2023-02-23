# In[]
import cv2
import numpy as np
# In[]
def Gaussian_Filter(image,size=3,sigma=1.0):
    image=np.asarray(np.uint8(image))
    height,width=image.shape[:2]
    #Zero padding
    pad=size//2
    result=np.zeros((height+pad*2,width+pad*2),dtype=np.float)
    result[pad:pad+height,pad:pad+width]=image.copy().astype(np.float)
    #kernel
    kernel=np.zeros((size,size),dtype=np.float)
    for x in range(-pad,-pad+size):
        for y in range(-pad, -pad+size):
            kernel[y+pad,x+pad]=np.exp(-(x**2+y**2)/(2*(sigma**2)))
    kernel/=(2*np.pi*sigma*sigma)
    kernel/=kernel.sum()
    buffer=result.copy()
    #filtering
    for y in range(height):
        for x in range(width):
            result[pad+y,pad+x]=np.sum(kernel*buffer[y:y+size,x:x+size])
    result=np.clip(result,0,255)
    result=result[pad:pad+height,pad:pad+width].astype(np.uint8)
    return result
# In[]
def Sobel(image):
    image = image.astype(np.uint8)
    # create temp image container
    sobel_image_x = np.zeros(shape=image.shape)
    sobel_image_y = np.zeros(shape=image.shape)
    # create sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(image.shape[0]-2):
        for j in range(image.shape[1]-2):
            sobel_image_x[i, j] = np.abs(np.sum(image[i:i+3, j:j+3] * sobel_x))
            sobel_image_y[i, j] = np.abs(np.sum(image[i:i+3, j:j+3] * sobel_y))
    sobel_image = np.sqrt(sobel_image_x*sobel_image_x + sobel_image_y*sobel_image_y).astype(np.uint8)
    angle = np.arctan2(sobel_image_y,  sobel_image_x)
    angle=np.rad2deg(np.arctan2(sobel_image_y,  sobel_image_x))
    return sobel_image, angle
# In[]
def non_maximum_suppression(gradient,angle):
    after_NMS = gradient.copy()
    height,width=gradient.shape[:2]
    for j in range(1, height-1):
        for i in range(1, width-1):
            #0~22.5度，157.5~180度
            if (angle[j, i] >= 0 and angle[j, i] < 22.5 ) or (angle[j, i] >= 157.5  and angle[j, i] <= 180):
                if gradient[j, i] < gradient[j, i-1] or gradient[j, i] < gradient[j, i+1]:
                   after_NMS[j, i] = 0
            #22.5~67.5度 
            elif angle[j, i] >= 22.5  and angle[j, i] < 67.5 :
                if gradient[j, i] < gradient[j-1, i-1] or gradient[j, i] < gradient[j+1, i+1]:
                    after_NMS[j, i] = 0
            #67.5~112.5度 
            elif angle[j, i] >= 67.5 and angle[j, i] < 112.5:
                if gradient[j, i] < gradient[j-1, i] or gradient[j, i] < gradient[j+1, i]:
                    after_NMS[j, i] = 0
            #112.5~157.5度 
            elif angle[j, i] >= 112.5  and angle[j, i] < 157.5 :
                if gradient[j, i] < gradient[j+1, i-1] or gradient[j, i] < gradient[j-1, i+1]:
                    after_NMS[j, i] = 0             
    return after_NMS
# In[]
def Double_Thresholding(img, minimum, maximum):
    result = img.copy()
    result[(result<maximum)*(result>=minimum)]=maximum
    result[result>=maximum]=255
    result[result<minimum] = 0   
    return result
# In[]
def Connect(img,minimum,maximum):
    temp=img.copy()
    height,width=temp.shape[:2]
    total_strong=np.sum(temp==255)
    t_max=0
    while(True):
        for i in range(1,img.shape[0]- 1) :
            for j in range(1, img.shape[1]- 1) :
                if(img[i, j] >=minimum and img[i,j]<maximum) :
                    t_max = max(img[i-1, j-1], img[i-1, j], img[i-1, j+1], img[i, j-1],
                               img[i, j+1], img[i+1, j-1], img[i+1, j], img[i+1, j+1])
                   # t_max = max(img[i-1, j],img[i, j-1],img[i, j+1], img[i+1, j])
                    if(t_max == 255) :
                        temp[i, j] = 255
        if(total_strong==np.sum(temp==255)):
            break
        total_strong=np.sum(temp==255)
    for i in range (height)  :
        for j in range(width) :
           if(temp[i, j] >=minimum and temp[i,j]<maximum) :
               temp[i, j] = 0
    return temp
# In[]
def use_opencv_canny(img,k_size,sigma,minimum,maximum):
    blur_gray = cv2.GaussianBlur(img,(k_size, k_size), sigma)
    edges = cv2.Canny(blur_gray, minimum, maximum)
    return edges
# In[]
def use_opencv_erosion_dilation(img):
    kernel=np.ones((2,2),np.uint8)
    dilation=cv2.dilate(img,kernel,iterations=1)
    erosion=cv2.erode(dilation,kernel,iterations=1)
    return erosion
# In[]
#以灰階方式讀取
image=cv2.imread('b.jpg',0)
#resize image
image=cv2.resize(image,(512,512))
#Gaussian_Blurring
k_size=3
sigma=1
Gaussian_img=Gaussian_Filter(image)
#Sobel
Sobel_img,angle=Sobel(Gaussian_img)
#NMS
NMS_img=non_maximum_suppression(Sobel_img, angle)
minimum=50
maximum=150
#Double Thresholding
DT_img = Double_Thresholding(NMS_img,minimum,maximum)
#Connect weak edge
result=Connect(DT_img,minimum,maximum)
result1=use_opencv_canny(image, k_size, sigma, minimum, maximum)
result2=use_opencv_erosion_dilation(result)
# In[]
#cv2.imshow('original',image)
#cv2.imshow('after Gaussian',Gaussian_img)
#cv2.imshow('after Sobel',Sobel_img)
#cv2.imshow('after non_maximum_suppression',NMS_img)
#cv2.imshow('after Double_Threshold ',DT_img)
cv2.imshow('Final',result)
#cv2.imshow('canny from opencv',result1)
#cv2.imshow('opening from opencv',result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

