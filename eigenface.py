import cv2
import matplotlib.pyplot as plt
import numpy as np

people_train = ['lhg', 'kih'] # folder name
people_test = 'test'
IMG_SIZE = 64
TRAIN_IMG_NUM = 5
TEST_IMG_NUM = 1
DEMO_SIZE = 320

def detect_face_crop(src):
    if src is None:
        print('Image load failed')
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(src)
    faces_arr = []
    for (x1, y1, w1, h1) in faces:
        faceROI = src[y1:y1+h1, x1:x1+w1]
        faces_arr.append(faceROI)
    #assert len(faces_arr) > 0, 'Cannot find face on some images'
    bbox = faces
    return np.array(faces_arr), bbox

def visualize_face(img):
    img = np.uint8((img-img.min())/img.max() * 255)
    t = cv2.resize(img, (DEMO_SIZE, DEMO_SIZE))
    cv2.equalizeHist(t, t)
    cv2.imshow('test', t)
    cv2.waitKey()

def load_imgs(prefix, num):
    img_arr = []
    for i in range(0, num):
        img = plt.imread(prefix + '/' + str(i+1) + ".png")
        img_arr.append(img)
    return np.array(img_arr)

def load_faces_with_image(img):
    face_arr = []
    bbox_arr = []
    idx_arr = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face, bbox = detect_face_crop(img)
    for p in range(len(face)):
        t_img = cv2.resize(face[p], (IMG_SIZE,IMG_SIZE))
        cv2.equalizeHist(t_img, t_img)
        face_arr.append(t_img)
        bbox_arr.append(bbox[p])
        idx_arr.append(i)
    face_arr = np.array(face_arr)
    return face_arr, bbox_arr, idx_arr

def load_faces_with_file(prefix, num):
    face_arr = []
    bbox_arr = []
    idx_arr = []
    for i in range(0, num):
        img = cv2.imread(prefix + '/' + str(i+1) + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face, bbox = detect_face_crop(img)
        for p in range(len(face)):
            t_img = cv2.resize(face[p], (IMG_SIZE,IMG_SIZE))
            cv2.equalizeHist(t_img, t_img)
            face_arr.append(t_img)
            bbox_arr.append(bbox[p])
            idx_arr.append(i)
    face_arr = np.array(face_arr)
    return face_arr, bbox_arr, idx_arr

def get_eigenface(faces):
    faces = faces.reshape(faces.shape[0], -1)
    mean, eigenVectors = cv2.PCACompute(faces, mean=None, maxComponents=1)
    eigen_mat = eigenVectors.reshape((IMG_SIZE,IMG_SIZE))
    mean_mat = mean.reshape((IMG_SIZE,IMG_SIZE))
    visualize_face(eigen_mat)
    return eigen_mat, mean_mat

def classify_person(target, classes, means):
    min_distance = 99999999
    best_label = None
    for label in classes:
        comp_A = classes[label]
        comp_B = target - np.mean(target)
        #comp_B = target - means[label]
        distance = np.sum((comp_A - comp_B) * (comp_A - comp_B))
        if(min_distance > distance):
            min_distance = distance
            best_label = label
    return best_label

if __name__ == '__main__':
    eigen_vectors = {}
    eigen_means = {}
    for person in people_train:
        faces, bbox, idx_arr = load_faces_with_file(person, TRAIN_IMG_NUM)
        label = person
        eigenVectors, mean = get_eigenface(faces)
        eigen_vectors[label] = eigenVectors
        eigen_means[label] = mean

    print("Train Complete !")

    src_img_arr = load_imgs(people_test, TEST_IMG_NUM)
    test_faces, bbox, idx_arr = load_faces_with_file(people_test, TEST_IMG_NUM)
    for i in range(len(test_faces)):
        test_person = test_faces[i]
        test_person_batch = []
        test_person_batch.append(test_person)
        test_person_batch = np.array(test_person_batch)
        test_eigenVector, test_mean = get_eigenface(test_person_batch)
        prediction = classify_person(test_person / 255, eigen_vectors, eigen_means)
        srcimg = src_img_arr[idx_arr[i]]
        bb = bbox[i]
        x = bb[0]
        y = bb[1]
        r = x + bb[2]
        b = y + bb[3]
        cv2.rectangle(srcimg, (x,y), (r, b), (255,0,255),2)
        srcimg = cv2.putText(srcimg, prediction, (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 4)
        
    '''
    while True:
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()
        test_faces, bbox, idx_arr = load_faces_with_image(image)
        srcimg = image
        for i in range(len(test_faces)):
            test_person = test_faces[i]
            test_person_batch = []
            test_person_batch.append(test_person)
            test_person_batch = np.array(test_person_batch)
            test_eigenVector, test_mean = get_eigenface(test_person_batch)
            prediction = classify_person(test_person / 255, eigen_vectors, eigen_means)
            srcimg = image
            bb = bbox[i]
            x = bb[0]
            y = bb[1]
            r = x + bb[2]
            b = y + bb[3]
            cv2.rectangle(srcimg, (x,y), (r, b), (255,0,255),2)
            srcimg = cv2.putText(srcimg, prediction, (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 4)
        cv2.imshow('test', srcimg)
        cv2.waitKey(10)
    '''
    

    for i in src_img_arr:
        plt.figure()
        plt.imshow(i)
    plt.show()
