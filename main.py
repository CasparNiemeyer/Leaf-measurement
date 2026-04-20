import cv2 as cv
import numpy as np


########################Settings#########################
MODE = True # True=Camera|False=Image
#####################Image Settings######################
IMAGE_PATH = "images/IMG_0101.jpeg"
###################CROP/ALIGN SETTINGS###################
PHYSWIDTH = 13.4      #Width of physical Square
PHYSHEIGHT = 13.4     #Height of physical Square
DIGWIDTH = 700       #Number of pixels to crop to
######################Edge detection#####################
KERNELSIZE = 6                     #noise removal area
LOWER = np.array([0, 60, 60])      #color range for mask
UPPER = np.array([355, 200, 200])
##########################DEBUG##########################
FULLFRAME = True      #Show the full frame with markers
CROPPED = True        #Show the cropped Square
DRAWMARKER = True     #Show detected markers
DRAWBOUND = True      #Show crop boundary
CONTOURS = True       #Show contours of edge detection
CONVEXHULL = True     #Show convex hull
SHOWMASK = True       #Show mask used to find damage
#########################################################

#Generell var
pixelarea = (PHYSHEIGHT/DIGWIDTH)*(PHYSWIDTH/DIGWIDTH)  #Area of a pixel based on real size of Square and Width of crop used to translate from pixel to cm^2
unsorted = np.zeros((4, 1, 2), dtype=np.int32) #Array later used for unsorted corner centeroids

if MODE:
    #basic setup for cam capture
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


while True:
    if MODE:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    else:
        frame = cv.imread(IMAGE_PATH)


    """
        ARUCO DETECTION.
        - uses the 4x4 library top 50 as the square only uses 0,1,2,3
    """

    # prepare for aruco detection
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()

    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect the markers
    corners, ids , rejected = detector.detectMarkers(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    
    displayframe = frame.copy()

    #draw markers
    if DRAWMARKER: cv.aruco.drawDetectedMarkers(displayframe,corners,ids)

    if ids is not None and len(ids) == 4:
        corners_array = np.array(corners)

        #convert all four corners of each marker into one centeroid
        for i in range(len(ids)):
            x = [p[0] for p in corners_array[i,0]]
            y = [p[1] for p in corners_array[i,0]]
            centroid = (sum(x) / 4, sum(y) / 4)

            unsorted[i,0,0] = centroid[0]
            unsorted[i,0,1] = centroid[1]


        #sort list to tr tl br bl
        sorted = np.zeros((4,2),np.int32)

        s = np.sum(unsorted, axis=2)
        dif = np.diff(unsorted, axis=2)

        sorted[0] = unsorted[np.argmin(s)]
        sorted[2] = unsorted[np.argmax(s)]

        sorted[1] = unsorted[np.argmin(dif)]
        sorted[3] = unsorted[np.argmax(dif)]

        #draw boundry of all markers
        if DRAWBOUND: cv.polylines(displayframe,[sorted],True,(0,255,255))

        #Crop and warp image to size
        corners = np.concatenate(sorted).tolist()
        corners = np.array([[corners[0],corners[1]],[corners[2],corners[3]],[corners[4],corners[5]],[corners[6],corners[7]]])
    
        destination_corners = [[0, 0], [DIGWIDTH, 0], [DIGWIDTH, DIGWIDTH], [0, DIGWIDTH]]

        M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

        cropped = cv.warpPerspective(frame, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv.INTER_LINEAR)

        #show cropped
        if CROPPED: cv.imshow("Cropped", cropped)

        #chroma key
        hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)

        # Create mask
        mask = cv.inRange(hsv, LOWER, UPPER)
        
        #filter out noise
        kernel = np.ones((KERNELSIZE,KERNELSIZE),np.uint8)
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        #Apply mask
        result = cv.bitwise_and(cropped, cropped, mask=mask)
        
        #run edge detection
        edged = cv.Canny(result, 30, 200)

        contours, hierarchy = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        try:
            contour = max(contours, key=cv.contourArea)
            size = round(cv.contourArea(contour)*pixelarea,3)

            hull = cv.convexHull(contour)
            sizeconvex = round(cv.contourArea(hull)*pixelarea,3)

            kernel = np.ones((KERNELSIZE*2,KERNELSIZE*2),np.uint8)
            opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)


            mask = np.zeros_like(cv.cvtColor(cropped, cv.COLOR_BGR2GRAY))
            cv.drawContours(mask,[contour],-1, (255,255,255) ,thickness=-1)
            mask = cv.bitwise_and(cropped,cropped,mask=mask)
            mask = cv.erode(mask,kernel,iterations=1)
            graymask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)

            contours, hierarchy = cv.findContours(graymask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            


            if contours is not None:
                try:
                    largest_contour = max(contours, key=cv.contourArea)

                    area = 0

                    for cnt in contours:
                        if cnt is largest_contour:
                            pass

                        area += cv.contourArea(cnt)
                        
                    print(area*pixelarea)
                    cv.drawContours(result,contours,-1,(0,0,255),3)
                except:
                    print("error")

            if SHOWMASK: cv.imshow("Mask",mask)

            if CONVEXHULL: cv.drawContours(result, [hull], -1, (255,0,0),5)

            if CONTOURS: cv.drawContours(result, [contour], -1, (0, 255, 0), 5)

            cv.putText(result, f"Area: {size} cm2", (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(result, f"Area convex hull: {sizeconvex} cm2", (40, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except TypeError:
            print("move object error tracking")

        cv.imshow("Result",result)
    
    cv.namedWindow("Fullframe", cv.WINDOW_NORMAL)

    #show full frame
    if FULLFRAME: cv.imshow("Fullframe",displayframe)

    if MODE:   
        if cv.waitKey(1) == ord('q'):
            break
    else:
        cv.waitKey()
        break
 
# When everything done, release the capture
if MODE: cap.release()
