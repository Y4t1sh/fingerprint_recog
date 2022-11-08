import cv2
import os 
import numpy as np


#Reading the Image which we will check with the other images
img = cv2.imread("a.jpg")

#defining the variables
best_score = 0
filename = None 
image = None

#keypoints and match points
kp1, kp2, mp = None, None, None

fingerprint_image = cv2.imread("b.jpg", cv2.IMREAD_GRAYSCALE)

  #main function starts here

sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
  
  #using KD tree algorithm
matches = cv2.FlannBasedMatcher({'algorithm' : 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k = 2)

match_points = []

for p, q in matches:
  if p.distance < 0.1 * q.distance:
    match_points.append(p)

keypoints = 0
if len(keypoints_1) < len(keypoints_2):
  keypoints = len(keypoints_1)
else:
  keypoints = len(keypoints_2)


best_score = len(match_points) / keypoints * 100
image = fingerprint_image
kp1, kp2, mp = keypoints_1, keypoints_2, match_points

#showing the resuls
print("SCORE: " +str(best_score))

result = cv2.drawMatches(img, kp1, image, kp2, mp, None) #to show the matching in the diagram
result = cv2.resize(result, None, fx = 4, fy = 4)
cv2.imshow("Result",result)
cv2.waitKey(0)
