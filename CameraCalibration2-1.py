#%%
import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((15*19,3), np.float32)
objp[:,:2] = np.mgrid[0:19,0:15].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#%%
images = glob.glob('C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Calib Images\LEFT\*.bmp')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findCirclesGrid(img, (19,15), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (19,15), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1)
    cv.destroyAllWindows()

#%%
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img_calib = cv.imread('C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Calib Images\LEFT\LEFT1.bmp',0)
cv.imshow('Uncalibrated Image',img_calib)
cv.waitKey(2000)
cv.destroyAllWindows()

h,  w = img_calib.shape[:2]
newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv.undistort(img_calib, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

#%%
# save undistorted image and check for errors
cv.imwrite('C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\calibresult.bmp',dst)
if not cv.imwrite('C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\calibresult.bmp', dst):
    raise Exception("Could not write image")