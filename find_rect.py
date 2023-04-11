import os
import cv2
import webp
import matplotlib.pyplot as plt
import numpy as np
import logging

# logging.basicConfig()


logger = logging.getLogger("nomlBoxDetector")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
formatter_critical = logging.Formatter("!!!!!%(asctime)s %(levelname)s:%(message)s")
handler_critical = logging.FileHandler("log_event.log")
handler_critical.setLevel(logging.CRITICAL)
handler_critical.setFormatter(formatter_critical)
logger.addHandler(handler)
logger.addHandler(handler_critical)

def find_rect(p):
    
    image = cv2.imread(p)
    img = image
    result = image.copy()
 
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian filter
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)

    edged = cv2.Canny(dilation, 30, 30)
    cv2.imshow('edged', edged)
    cv2.waitKey()

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=lambda  x:x[1],reverse=True)


    i = -1
    j = 1
    y_old = 5000
    x_old = 5000
    print(len(rects))
    for i,rect in enumerate(rects):
        x,y,w,h = rect
        area = w * h
        # print(rect)
        # if 1000> area or area>10000:
            # continue
        if i%50 != 2:
            continue
        img = cv2.rectangle(img, (rect[0], rect[1]),(rect[2], rect[3]) ,(255, 0, 0), 5, cv2.LINE_4)
        # img = cv2.drawContours(img, [rect], -1, (0,255,0), 3)

        # if ratio >= 0.9 and ratio <= 1.1:
        #     img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
        #     # cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # else:
        #     # cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #     img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
        # if area > 47000 and area < 70000:
            
        #     if (y_old - y) > 200:
        #         i += 1
        #         y_old = y

        #     if abs(x_old - x) > 300:
        #         x_old = x
        #         x,y,w,h = rect

        #         out = img[y+10:y+h-10,x+10:x+w-10]
        #         filename = os.path.basename(p)
        #         cv2.imwrite('cropped\\' + filename + '_' + str(j) + '.jpg', out)

        #         j+=1
    
    filename = os.path.basename(p)
    outpath = os.path.join('out','opencv',filename)
    cv2.imwrite(outpath, img)
    return
 
 
 
 
 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
   
   
    # ret,thresh = cv2.threshold(gray,50,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    print("Number of contours detected:", len(contours))


    threshold_area = 100
    for cnt in contours:
        print(cnt)
        area = cv2.contourArea(cnt)         
        if area < threshold_area:
            continue

        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.9 and ratio <= 1.1:
                print(cnt)
                img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
                # cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

    # # Fill rectangular contours
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

    # # Morph open
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # # Draw rectangles
    # cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
    
    filename = os.path.basename(p)
    outpath = os.path.join('out','opencv',filename)
    
    cv2.imwrite(outpath, img)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('opening', opening)
    # cv2.imshow('image', image)
    # cv2.waitKey()

def find_rect_v1(p,erosion_iter = 5,canny_threshold = 70):
    image = cv2.imread(p)
    img = image

    img_x,img_y,img_channel = img.shape
    img_size = img_x * img_y
    threshold_area = img_size / 10000
    
    # logger.info('imgsize img_x img_y : {},{},{}'.format(img_size, img_x,img_y))

    result = image.copy()

    # cv2.imshow('original', image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('gray', gray)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = erosion_iter)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = erosion_iter)

    sigma = 1
    dilation = cv2.GaussianBlur(dilation, (0, 0), sigma)
    # cv2.imshow('gray-dilation', dilation)
    
    

    cany = cv2.Canny(dilation, canny_threshold, canny_threshold)

    
    cany_with_border = np.full((img_x+4,img_y+4), 255, dtype=np.uint8)
    cany_with_border[2:img_x+2,2:img_y+2] = cany


    sobelx = cv2.Sobel(dilation,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(dilation,cv2.CV_64F,0,1,ksize=5)  # y
    
    # cv2.imshow('cany', cany_with_border)
    # cv2.imshow('sobelx', sobelx)
    # cv2.imshow('sobely', sobely)

    # cv2.imshow('cany', cany_with_border)
    # cv2.waitKey()
    
    
    return image,dilation,cany,sobelx,sobely
    #### Countour
    contours, hierarchy = cv2.findContours(back, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # print(cnt)
        area = cv2.contourArea(cnt)   
        # print(area)
        if area < threshold_area:
            continue
        
        
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
        if len(approx) != 4:
            continue

        
        

        rect = cv2.boundingRect(cnt)
        logger.debug(rect)
        img = cv2.rectangle(img, (rect[0], rect[1]),(rect[2], rect[3]) ,(255, 0, 0), 5, cv2.LINE_4)
    
    cv2.imshow('boxes', img)
    cv2.waitKey()
if __name__ == '__main__':
    
    p1='D:\\data\\KD\\oper1'
    lst_files = os.listdir('D:\\data\\KD\\oper1')

    for file_ in lst_files:
        find_rect_v1(os.path.join(p1,file_))    

    # img = webp.load_image('D:\\data\\KD\\oper2\\053-023-010-1673489928-299533812-3d43a10d-10010-84FA6C-F9AE57-EB4688-78B05C-1AC4806-0C9F.webp', 'RGBA')
    # # print(cv2)
    # plt.imshow(img)
    # plt.show()

