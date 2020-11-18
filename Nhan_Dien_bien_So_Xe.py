#python opencv Identify license plate numbers

import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

im = cv2.imread("./car_demo1.jpg",1)
(h, w, d) = im.shape 
center = (w // 2, h // 2) 
M = cv2.getRotationMatrix2D(center, -4, 1.0) 
im = cv2.warpAffine(im, M, (w, h))

cv2.imshow("Anh goc",im)
cv2.waitKey(1)


#lấy ảnh xám
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow("Anh xam" ,im_gray)
cv2.waitKey(1)


#lấy ảnh lọc nhiễu 
noise_removal = cv2.bilateralFilter(im_gray,9,75,75)
cv2.imshow("Anh loc nhieu" ,noise_removal)
cv2.waitKey(1)

#cân bằng histogram
equal_histogram = cv2.equalizeHist(noise_removal);
cv2.imshow("Anh can bang Histogram" ,equal_histogram)
cv2.waitKey(1)

#giảm cạnh nhiễu,và làm cạnh thật thêm sắc nhọn 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)
cv2.imshow("Anh giam canh nhieu, tang canh that" ,morph_image)
cv2.waitKey(1)

#xóa phông không cần thiết
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
cv2.imshow("Anh xoa phong" ,sub_morp_image)
cv2.waitKey(1)

#dùng threshold OTSU đưa ảnh về trắng đen tách biệt nền và vùng cần quan tâm
ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
cv2.imshow("Anh den trang tach nen" ,thresh_image)
cv2.waitKey(1)

#dùng thuật toán canny để nhận biết canh
canny_image = cv2.Canny(thresh_image,250,255)
kernel = np.ones((3,3), np.uint8)
cv2.imshow("Anh tach canh Canny" ,canny_image)
cv2.waitKey(1)

#dùng thuật toán Dilate để tăng độ bén cho cạnh
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
cv2.imshow("Anh tang canh " ,dilated_image)
cv2.waitKey(1)

#DUNG Contour để lấy ra biên số xe hinh chữ nhật
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True) 
    if len(approx) == 4:
            screenCnt = approx
            break
if screenCnt is None:
    print ("Không thấy biển số xe")     

        
#lấy ảnh biển số

plate_im = cv2.drawContours(im,[screenCnt] ,0 ,(0,255,0) ,2)
cv2.imshow("Anh chua bien so" ,plate_im)
cv2.waitKey(1)

#Cắt ảnh

''' Tạo ảnh kích thước bằng ảnh xám với các điểm ảnh 0'''
mask = np.zeros(im_gray.shape, np.uint8)
'''Trích rút ảnh chỉ gồm biển số'''
new_img1 = cv2.drawContours(mask,[screenCnt],0,255,-1,) 
new_img = cv2.bitwise_and(im,im,mask=mask)
cv2.imshow("Mask Img co bien so" ,new_img)

'''Lấy tọa độ biển số xe và cắt ảnh'''
(x,y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = im_gray[topx:bottomx, topy:bottomy]
Cropped = cv2.resize(Cropped , (470,110))

cv2.imshow("Cropped" ,Cropped)

#Nhận diện ký tự quang học

text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Chương trình nhận diện biển số xe\n")
print("Biển số xe được tìm thấy:",text)
