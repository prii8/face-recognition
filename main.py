import cv2
import numpy as np
from PIL import Image
import os
import json
import webbrowser


choice = input("LOGIN OR SIGNUP?").lower()

if choice == "signup":
    id = input("ENTER YOUR ID OR NAME:")
    dictobj = {}
    f = open("data.json", "r")
    dictobj = json.load(f)
    print(dictobj)
    if id in dictobj.keys():
        print("id already exists...")


    # add sample code here
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # create a video capture object which is helpful to capture videos through webcam
    cam.set(3, 640)  # set video FrameWidth
    cam.set(4, 480)  # set video FrameHeight
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Taking samples, look at camera ....... ")
    count = 0
    path = "samples"
    print(id)
    r_path = os.path.join(path, id)
    os.mkdir(r_path)
    while True:

        ret, img = cam.read()
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(converted_image, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(os.path.join(r_path, "face." + "1" + '.' + str(count) + ".jpg"),
                        converted_image[y:y + h, x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 100:
            break

    print("Samples are taken!!")
    cam.release()
    cv2.destroyAllWindows()

    # as well as training code
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def Images_And_Labels(r_path):  # function to fetch the images and labels

        imagePaths = [os.path.join(r_path, f) for f in os.listdir(r_path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:  # to iterate particular image path

            gray_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_arr = np.array(gray_img, 'uint8')  # creating an array

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_arr)

            for (x, y, w, h) in faces:
                faceSamples.append(img_arr[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids


    print("wait training the sample...")

    faces, ids = Images_And_Labels(r_path)
    recognizer.train(faces, np.array(ids))
    t_path = "trainer"
    recognizer.write(os.path.join(t_path, id) + ".yml")

    print("Model is trained")


    #creation of json file
    #dict={id,os.path.join(t_path, id) + ".yml"}


    dictobj = {}
    f = open("data.json", "r")
    dictobj = json.load(f)
    print(dictobj)
    if id not in dictobj.keys():
        dictobj[id] = os.path.join(t_path, id) + ".yml"
        print(dictobj)
        abc = json.dumps(dictobj)
        f = open("data.json", "w")
        f.write(abc)
        f.close()
    else:
        print("id already exsits")



elif choice == "login":
    idd = input("ENTER YOUR ID OR NAME: ")
    dictobj = {}
    f = open("data.json", "r")
    dictobj = json.load(f)

    if idd not in dictobj.keys():
        print("id is incorrect")
    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns Histograms
        p = dictobj[idd]
        recognizer.read(p)  # load trained model
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)  # initializing haar cascade for object detection approach

        font = cv2.FONT_HERSHEY_SIMPLEX  # denotes the font type



        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW to remove warning
        cam.set(3, 640)  # set video FrameWidht
        cam.set(4, 480)  # set video FrameHeight

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        # flag = True
        exiting = 0
        while True:

            ret, img = cam.read()  # read the frames using the above created object

            converted_image = cv2.cvtColor(img,
                                           cv2.COLOR_BGR2GRAY)  # The function converts an input image from one color space to another

            faces = faceCascade.detectMultiScale(
                converted_image,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # used to draw a rectangle on any image

                id, accuracy = recognizer.predict(converted_image[y:y + h, x:x + w])  # to predict on every single image

                # Check if accuracy is less them 100 ==> "0" is perfect match
                if (accuracy < 100):

                    confidence = 100 - accuracy
                    accuracy = "  {0}%".format(round(100 - accuracy))

                    if confidence > 50:
                        cv2.putText(img, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('camera', img)
                        exiting = 1
                        url = 'https://ecommerce.priyanshikhippa.repl.co/index.html'
                        webbrowser.register('chrome', None,webbrowser.BackgroundBrowser("C://Program Files (x86)//Google//Chrome//Application//chrome.exe"))
                        webbrowser.get('chrome').open(url)

                    else:
                        cv2.putText(img, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('camera', img)
                else:
                    id = "unknown"
                    accuracy = "  {0}%".format(round(100 - accuracy))
                    cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                    cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
                    cv2.imshow('camera', img)
                    break

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27 or exiting == 1:
                break



        cam.release()
        cv2.destroyAllWindows()



    # check if the idd in the json file
    # if id is there in the json file than go for recognition code

else:
    print("wrong choice.....")
