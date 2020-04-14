import numpy as np
import xlsxwriter


data = np.load("loss_logger.npz")
data = np.transpose(np.array(data['loss_logger']))

workbook = xlsxwriter.Workbook('logger.xlsx')
worksheet = workbook.add_worksheet()


row = 0

for col, data in enumerate(data):
    worksheet.write_column(row, col, data)

workbook.close()



import cv2

img = cv2.imread("Untitled.png")

dim = (300, 80)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

cv2.imwrite("resized.jpg", resized)








