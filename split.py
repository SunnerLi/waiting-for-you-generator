import numpy as np
import cv2
import os

wait_counter = 0
store_index = 0

def saveImg(img, output_dir = './wait/', store_period = 10):
    global wait_counter
    global store_index
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if wait_counter < store_period:
        wait_counter += 1
    else:
        wait_counter = 0
        print(output_dir + str(store_index) + '.png')
        cv2.imwrite(output_dir + str(store_index) + '.png', img)
        store_index += 1

if __name__ == '__main__':
    cap = cv2.VideoCapture("waiting_for_you.mp4")
    if (cap.isOpened() == False):
        print("Error to open the video capture object...")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            height, width, _ = np.shape(frame)
            frame = cv2.resize(frame, (width // 2, height // 2))
            saveImg(frame)
        else:
            break

    cap.release()
    cv2.destroyWindow()
