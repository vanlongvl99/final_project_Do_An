import tkinter as tk
from tkinter import Message, Text, messagebox
import cv2
import os
import csv
import numpy as np
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
import joblib
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import datetime
from numpy import load
from os import listdir
import os.path


window = tk.Tk()
window.title("Face_Recogniser")
window.configure(background ='white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
    window, text ="Face-Recognition-System",
    bg ="green", fg = "white", width = 50,
    height = 3, font = ('times', 30, 'bold'))

message.place(x = 200, y = 20)


lbl2 = tk.Label(window, text ="Name of new person",
width = 20, fg ="green", bg ="white",
height = 2, font =('times', 15, ' bold '))
lbl2.place(x = 400, y = 200)

txt2 = tk.Entry(window, width = 20,
bg ="white", fg ="green",
font = ('times', 15, ' bold ')  )
txt2.place(x = 700, y = 215)


model_facenet = load_model('./model_file_main/facenet_keras.h5')
model_facenet.load_weights('./model_file_main/facenet_keras_weights.h5')
# Take train image of new user.
def TakeImages():
    name =(txt2.get())
    if os.path.exists("./dataset/data_new/" + name) == True:
        messagebox.showinfo(title = "Error", message = "Tên người dùng đã có trong tập dữ liệu!")
    else:
        os.mkdir( "./dataset/data_new/" + name)
        print("./dataset/data_new/" + name)
        cam = cv2.VideoCapture(0)
        detector = MTCNN()
        count = 0
        while(True):
            name_rand = np.random.rand(1)[0]
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame,(640,int(frame.shape[0]/frame.shape[1]*640)))
            faces = detector.detect_faces(frame)
            # get face of new person by mtcnn
            if len(faces) == 1:
                for person in faces:
                    bounding_box = person['box']
                    im_crop = frame[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
                    if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                        cv2.rectangle(frame,(bounding_box[0],bounding_box[1]),(bounding_box[0] + bounding_box[2],bounding_box[1] + bounding_box[3]),(0,155,255),3)
                        count += 1
                        cv2.imwrite("./dataset/data_new/" + name + "/" + name +'_'+ str(count) + ".jpg", im_crop)
            cv2.imshow('frame', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is more than 300
            elif count == 150:
                break
        cam.release()
        cv2.destroyAllWindows()

# Training the images saved in training image folder
def TrainImages():
    path = './dataset/data_new'
    label_to_index = {}
    cnt = 0
    for forder_name in listdir(path):
        label_to_index[forder_name] = cnt
        cnt += 1
    # print(label_to_index)
    X_train = []
    y_train = []
    print(datetime.datetime.now())
    for forder_name in listdir(path):
        count = 0
        for file_name in listdir(path + '/' + forder_name):
            path_of_image = path + '/' + forder_name + '/' + file_name
            image = cv2.imread(path_of_image)
            # print(count)
            count += 1
            if count == 200:
                break
            # image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
            image = cv2.resize(image,(160,160))
            image = image/255
            X_train.append(image)
            y_train.append(label_to_index[forder_name])
        print( forder_name,count)
    print(datetime.datetime.now())
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # convert each face in the train set to an embedding
    newX_train = list()
    for face_pixels in X_train:
    	embedding = get_embedding(model_facenet, face_pixels)
    	newX_train.append(embedding)
    newX_train = np.array(newX_train)
    print(newX_train.shape)
    print(datetime.datetime.now())
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    newX_train = in_encoder.transform(newX_train)
    # fit model
    model_svm = SVC(kernel='linear', probability=True)
    model_svm.fit(newX_train, y_train)
    # predict
    yhat_train = model_svm.predict(newX_train)
    print(datetime.datetime.now(), "finished")
    #save model svm
    filename = './model_file_main/new_face_recognition_model_svm_1.sav'
    joblib.dump(model_svm, filename)
    messagebox.showinfo(title="Thông báo", message="Hoàn tất training!")

# For testing phase
# get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = cv2.resize(face_pixels,(160,160))
    face_pixels = (face_pixels/255)
	# scale pixel values
    face = face_pixels.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    samples = expand_dims(face, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


def face_recognition(index_to_labels, svm_file):
    loaded_model_SVM = joblib.load(svm_file)
    detector = MTCNN()
    cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture('http://192.168.1.16:8080/video')
    while True:
        ret, image = cam.read()
        # print("mtcnn")
        # print(datetime.datetime.now())
        image = cv2.flip(image, 1)
        image = cv2.resize(image,(640,int(image.shape[0]/image.shape[1]*640)))
        faces = detector.detect_faces(image)
        # print(datetime.datetime.now(),"\n")
        for person in faces:
            bounding_box = person['box']
            # Specifying the coordinates of the image as well
            im_crop = image[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
            if im_crop.shape[0] > 0 and im_crop.shape[1] > 0:
                # print("facenet")
                im_embedding = get_embedding(model_facenet, im_crop)
                # print(datetime.datetime.now())

                im_embedding = expand_dims(im_embedding,axis = 0)
                in_encoder = Normalizer(norm='l2')
                im_test = in_encoder.transform(im_embedding)
                print("====")
                # print(loaded_model_SVM.decision_function(im_test))
                # print("svm")
                pre_face = loaded_model_SVM.predict_proba(im_test)[0]
                # print(datetime.datetime.now())
                for i in range(len(index_to_labels)):
                    print(index_to_labels[i] + ':','{0:0.2f}'.format(pre_face[i]))
                max_index = np.argmax(pre_face)
                print("=====")
                print(index_to_labels[max_index],pre_face[max_index])
                cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
                if pre_face[max_index] > 0.65:
                    cv2.putText(image, index_to_labels[max_index] + ': ' + str('{0:0.2f}'.format(pre_face[max_index])), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)

                else:
                    cv2.putText(image, 'unknown', (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2, cv2.LINE_AA)
        cv2.imshow('image',image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


def button_test_video():
    label_to_index = {'manh_hung': 0, 'Mai_ly': 1, 'van_long': 2, 'Tien_dung': 3, 'bich_lan': 4, 'nguyen': 5, 'trong_nghia': 6, 'tien_thang': 7, 'Truong Thanh Sang_1712945': 8, 'Phung Tuan Hung_1611444': 9, 'Nguyen Thi Tuyet_1613947': 10, 'Truong Minh Tuan_1613935': 11, 'Dinh Manh Cuong_1510352': 12, 'Dao Duy Hanh_1711205': 13, 'Hoang Vu Nam_1512062': 14, 'Luu Anh Khoa_1611610': 15, 'Huynh Vuong Vu_1514097': 16, 'Le Phuc Thanh_1613178': 17, 'Dinh Giang Nam_1512059': 18, 'Nguyen Phu Cuong_1710725': 19, 'Ngo Trong Huu_1511438': 20, 'duc_thinh': 21, 'duy_quang': 22, 'hoang_hiep': 23, 'ngoc_anh': 24, 'phuoc_loc': 25, 'hieu_nguyen': 26, 'xuan_tien': 27}
    index_to_labels = {}
    for key,value in label_to_index.items():
        index_to_labels[value] = key
    print(index_to_labels)
    svm_file = './model_file_main/face_recognition_model_svm_1.sav'
    face_recognition(index_to_labels, svm_file)

def new_button_test_video():
    new_svm_file = './model_file_main/new_face_recognition_model_svm_1.sav'
    path = './dataset/data_new'
    cnt = 0
    index_to_labels = {}
    for forder_name in listdir(path):
        index_to_labels[cnt] = forder_name
        cnt += 1
    print(index_to_labels)
    face_recognition(index_to_labels, new_svm_file)



#Take image from camera
takeImg = tk.Button(window, text ="Take new face",      
command = TakeImages, fg ="white", bg ="green",
# kích thước của các button  
width = 15, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 50, y = 400) 

# trainning model
trainImg = tk.Button(window, text ="Training",  
command = TrainImages, fg ="white", bg ="green",  
width = 15, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trainImg.place(x = 250, y = 400)

# Test model with video
trackImg = tk.Button(window, text ="Testing",  
command = button_test_video, fg ="white", bg ="green",  
width = 15, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trackImg.place(x = 450, y = 400)

# New Test model with video
trackImg = tk.Button(window, text ="New Testing",  
command = new_button_test_video, fg ="white", bg ="green",  
width = 15, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trackImg.place(x = 650, y = 400)

# Quit GUI
quitWindow = tk.Button(window, text ="Quit",  
command = window.destroy, fg ="white", bg ="green",  
width = 15, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
quitWindow.place(x = 900, y = 400) 
window.mainloop()