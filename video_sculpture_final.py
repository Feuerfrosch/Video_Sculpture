# import all the modules we need
import cv2
import numpy as np
import operator
import os
import math
import time
from moviepy.editor import *
from moviepy.video.tools.credits import credits1
import moviepy.video.fx.all as vfx
import moviepy.editor as mpe
from random import randint



# some initial variables for validity checking of possible characters ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

# The Alphabet Variables #############################################################################################

A = "skillset.mp4"
B = "escalated.mp4"
C = "madness.mp4"
D = "only_one.mp4"
E = "disappointed.mp4"
F = "bond.mp4"
G = "english.mp4"
H = "trap.mp4"
I = "shaken.mp4"
J = "force.mp4"
K = "vather.mp4"
L = "yippie.mp4"
M = "nein.mp4"
N = "never_late.mp4"
O = "rules.mp4"

P = "future.mp3"
Q = "needle.mp3"
R = "psycho.mp3"
S = "benny.mp3"
T = "forrest.mp3"
U = "busters.mp3"
V = "wilhelm.mp3"
W = "chariots.mp3"

X = "blackwhite"
Y = "crop"
Z = "colorx"

# Dictionary mapping Alphabet Strings to Variables #######################################################################

varDict = {"A" : A,
           "B" : B,
           "C" : C,
           "D" : D,
           "E" : E,
           "F" : F,
           "G" : G,
           "H" : H,
           "I" : I,
           "J" : J,
           "K" : K,
           "L" : L,
           "M" : M,
           "N" : N,
           "O" : O,
           "P" : P,
           "Q" : Q,
           "R" : R,
           "S" : S,
           "T" : T,
           "U" : U,
           "V" : V,
           "W" : W,
           "X" : X,
           "Y" : Y,
           "Z" : Z}
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rectangle for contour
    intRectX = 0                # bounding rectangle top left corner x location
    intRectY = 0                # bounding recta top left corner y location
    intRectWidth = 0            # bounding rectangle width
    intRectHeight = 0           # bounding rectangle height
    fltArea = 0.0               # area of contour
    intBoundingRectArea = 0     # area of bounding rectangle
    fltAspectRatio = 0          # aspect ratio

    def calculateBoundingRectInfo(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight
        self.intBoundingRectArea = self.intRectWidth * self.intRectHeight # this is mainly important for validity checking
        self.fltAspectRatio = float(self.intRectWidth) / float(self.intRectHeight) # this one too

    def checkIfContourIsValid(self):                            # this is possibly still not detailed enough
        if (self.fltArea > MIN_CONTOUR_AREA and self.intBoundingRectArea > MIN_PIXEL_AREA and
            self.intRectWidth > MIN_PIXEL_WIDTH and self.intRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < self.fltAspectRatio and self.fltAspectRatio < MAX_ASPECT_RATIO):
            return True        # better validity checking might still be necessary for use in unoptimized conditions (random background / bad lighting)
        else:
            return False

# the function for detecting the characters used in the interface ##################################################################################################
def main():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # these will be filled shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read training classifications
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return


    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read training images
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return


    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    #imgScene = cv2.imread("blocks4.png")          # this is for use with pictures (for reference)

    #if imgScene is None:
    #    print "error: image not read from file \n\n"
    #    os.system("pause")
    #    return
    # end if

    video_capture = cv2.VideoCapture(0)                     # instantiate video capture object -> 0 for integrated laptop webcam
    while(True):                                            # start loop that keeps checking for the initiator block
        current_frame = video_capture.read() [1]           # gets the current frame of the video capture
        imgScene = current_frame                   # reads it in for further use


        #mask = np.zeros(imgScene.shape[:2], np.uint8)                 # this block was an experiment
        #bgdModel = np.zeros((1,65),np.float64)                                 # extracting the foreground led to more accurate results
        #fgdModel = np.zeros((1,65),np.float64)                                 # but it took way too long with pictures and was unusable with a video-feed

        #rect = (10,200, 1100, 450)
        #cv2.grabCut(imgScene,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

        #mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        #imgScene = imgScene*mask2[:,:,np.newaxis]


        imgGray = cv2.cvtColor(imgScene, cv2.COLOR_BGR2GRAY)       # get grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur


        imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                          255,                                  # turn pixels that pass the threshold white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                          3,                                   # size of a pixel neighborhood used to calculate threshold value
                                          2)

        #imgCanny = cv2.Canny(imgBlurred,100,200)               # I tried to use the canny edge detection algorithm instead of thresholding, but it didn't produce as compelling results

        imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, because findContours modifies the image

        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image
                                                     cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                     cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

        for npaContour in npaContours:                             # for each contour
            contourWithData = ContourWithData()                                             # instantiate a contour with data object (found further up)
            contourWithData.npaContour = npaContour                                         # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rectangle
            contourWithData.calculateBoundingRectInfo()                                     # get bounding rect info and info for validity checking
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
            allContoursWithData.append(contourWithData)                                     # add contour with data object to the list of all contours with data


        for contourWithData in allContoursWithData:                 # for all contours
            if contourWithData.checkIfContourIsValid():             # check if valid character
                validContoursWithData.append(contourWithData)       # if so, append it to the valid contour list


        validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

        strFinalString = ""         # this string will have the detected characters in it, which will be used to determine which characters will be used in the moviepy part of the program

        # parts of this are for debugging purposes
        for contourWithData in validContoursWithData:            # for each contour
                                                    # draw a green rectangle around the current character
            cv2.rectangle(imgScene,                                        # draw rectangle on original image
                          (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                          (0, 255, 0),              # green
                          2)                        # thickness

            imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop character out of threshold image
                               contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage (like in the learning program)

            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call K-Nearest-Neighbor function find_nearest

            strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

            strFinalString = strFinalString + strCurrentChar            # append current char to string


        if "7" in strFinalString:                                       # if the initiator block is found among the detected characters

            strFinalStringSplit = list(strFinalString)                  # split the final character string, so it can be read into the dictionary

            charDict = {}                              # create an empty dictionary
            invCharDict = {}
            increment = 1
            for char in strFinalStringSplit:           # fill the dictionary with characters and their place in the list
                if char in charDict:                   # exempt previously encountered characters
                    pass
                elif char not in charDict and char is not "7":      # excempt 7, as it will not be used in moviepy section
                    charDict[char] = increment                      #this is used for avoiding fasely read repeated characters
                    invCharDict[increment] = char               #THIS is for actual use later on
                    increment += 1
                else:
                    pass

            varList = []                            # empty list
            for number in invCharDict:
                stringy = invCharDict[number]
                variably = varDict[stringy]
                varList.append(variably)            # fill list with variables in order from left to right

            filmList = []                           # empty lists
            musicList = []
            fxList = []
            for media in varList:                   # fill lists with either video snippets, audio snippets or visual effects variables
                if media is A or media is B or media is C or media is D or media is E or media is F or media is G or media is H or media is J or media is K or media is L or media is M or media is N or media is O:
                    filmList.append(media)
                elif media is P or media is Q or media is R or media is S or media is T or media is U or media is V or media is W:
                    musicList.append(media)
                elif media is X or media is Y or media is Z:
                    fxList.append(media)
                else:
                    pass

            video_capture.release()                 # relieve the webcam from it's duty

            VideoFileList = []                          # read in all the video files from filmlist
            for film in filmList:
                film = mpe.VideoFileClip(film)
                film.fps = 24
                VideoFileList.append(film)



            # credit_clip = credits1("credits.txt", 10, stretch=30, color='white', stroke_color='black', stroke_width=2, font='Impact-Normal', fontsize=60)
            # this was an experiment for adding a credit sequence at the end of the video

            final_clip = mpe.concatenate_videoclips(VideoFileList)      # cut the videos after one another


            AudioFileList = []                          # read in all the audio from music list
            for music in musicList:
                music = mpe.AudioFileClip(music)
                AudioFileList.append(music)

            final_AudioClip = mpe.CompositeAudioClip(AudioFileList) # overlay all the music found

            Composite_Clip = mpe.CompositeAudioClip([final_clip.audio, final_AudioClip])
            final_clip.audio = Composite_Clip                              #final_clip has now sound + soundtrack


            for fx in fxList:                               # add any found visual effects from fxlist

                if fx is X:

                    final_clip = vfx.blackwhite(final_clip, RGB=[1,1,1], preserve_luminosity=True)

                elif fx is Z:

                    final_clip = vfx.clip1 = vfx.colorx(final_clip, randint(2, 10))


                elif fx is Y:

                    final_clip = vfx.crop(final_clip, x1=200, y1=300, x2=500, y2=600)

                else:
                    pass

            if "benny.mp3" in musicList:                        # speed up or slow down footage when 2 special music pieces are found

                final_clip = vfx.speedx(final_clip, factor=1.5)

            else:
                pass

            if "chariots.mp3" in musicList:

                final_clip = vfx.speedx(final_clip, factor=0.75)

            else:
                pass

            final_clip.write_videofile("CLICK_THIS!!!.mp4")                   # write the final video
            os.rename("/Users/fynnmollenhauer/Desktop/TECHSTUFF!TOP_SECRET/CLICK_THIS!!!.mp4", "/Users/fynnmollenhauer/Desktop/CLICK_THIS!!!.mp4")


            print "the raw string: %s" % strFinalStringSplit                           # show the full string split, for debugging
            print "the dictionary with the strings: %s" % charDict                      # show the dictionaries
            print "the inverse string-number dictionary in correct order: %s" % invCharDict
            print "the list with all detected variables in order from left to right: %s" % varList

            print "film list: %s" % filmList
            print "music list: %s" % musicList
            print "fx list: %s" % fxList

            cv2.imshow("imgScene", imgScene)                    # for debugging
            cv2.imshow("imgThreshtest", imgThresh)              # for debugging
            cv2.imshow("imgContourtest", imgContours)           # for debugging
            cv2.waitKey(0)                                          # wait for user key press

            cv2.destroyAllWindows()             # WE GET TO DESTROY SOME  WINDOWS!!!
            break                               # stop the loop, as it has fulfilled it's purpose

        else:
            strFinalString = ""                 # empty the string again
            time.sleep(20)                       # wait some time to avoid misjudging a hand or something like that for a (the initiator) character
    video_capture.release()                     # finally stop capturing video
###################################################################################################
if __name__ == "__main__":                      # boilerplate
    main()
# end if
