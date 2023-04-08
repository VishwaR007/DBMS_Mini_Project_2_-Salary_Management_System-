import os
import shutil
import cv2
import sys
import json


DIR_run_seg_prediction = "E:\AIML\ZIP_TAG_PROJECT\\attempt_06\\runs\segment\predict"
pred_img_path = os.path.join(DIR_run_seg_prediction, 'image0.jpg')

pred_img = cv2.imread(pred_img_path)
pred_img_resize = cv2.resize(pred_img, (700, 700))
cv2.imshow("abc", pred_img_resize)
cv2.waitKey(0)


# sys.stdout = open('declare.js', 'w')
# pred_img_path = os.path.join(DIR_run_seg_prediction, 'image0.jpg')
# img_rep_path = pred_img_path.replace('\\', '/')
# jsonobj = json.dumps(img_rep_path)
# print("var jsonstr = '{}'".format(jsonobj))


