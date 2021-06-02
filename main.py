import os
import face_recognition
import numpy as np
import cv2
from datetime import datetime


class FaceRecognition:
    # getting images name and storing them in the list
    path = 'images'
    images = []
    nameOfStudents = []
    imageDirectoryList = os.listdir(path)

    # extracting images using the image name list and storing them
    for imageName in imageDirectoryList:
        currentImage = cv2.imread(f'{path}/{imageName}')
        images.append(currentImage)
        nameOfStudents.append(os.path.splitext(imageName)[0])  # taking name only
    print(nameOfStudents)

    # changing to rgb
    def findEncodings(images):
        encodeList = []
        i = 0
        for image in images:
            i = i + 1
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faceLocationForLoaded = face_recognition.face_locations(image)
            encode = face_recognition.face_encodings(image, faceLocationForLoaded)[0]
            encodeList.append(encode)
        return encodeList

    def markingAttendence(name):
        with open('Attendence.csv', 'r+') as f:
            attendenceList = f.readlines()
            nameList = []
            for detail in attendenceList:
                entry = detail.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dateNow = now.strftime('%H:%M:%S')  # time
                f.writelines(f'\n{name},{dateNow}')

    listEncodingsForKnown = findEncodings(images)

    captureVideo = cv2.VideoCapture(0)

    while True:

        success, image = captureVideo.read()
        imageResized = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # resizing image to 1/4
        imageResized = cv2.cvtColor(imageResized, cv2.COLOR_BGR2RGB)

        faceLocCurrentFrame = face_recognition.face_locations(imageResized)

        encodeCurrentFrame = face_recognition.face_encodings(imageResized, faceLocCurrentFrame)

        # grabs one face and one encoding at a time
        for encodeFace, faceLocation in zip(encodeCurrentFrame, faceLocCurrentFrame):
            matchedImages = face_recognition.compare_faces(listEncodingsForKnown, encodeFace, tolerance=0.6)
            faceDistance = face_recognition.face_distance(listEncodingsForKnown, encodeFace)  # gives euclidean distance
            print(f"-->{faceDistance}")
            matchIndexes = np.argmin(faceDistance)  # it returns the index having minimum value

            # generating rectangle

            if matchedImages[matchIndexes]:
                name = nameOfStudents[matchIndexes].upper()
                y1, x2, y2, x1 = faceLocation
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
                markingAttendence(name)

            cv2.imshow('WebCamera', image)
            cv2.waitKey(1)
