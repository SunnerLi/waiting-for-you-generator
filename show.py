import numpy as np
import cv2

def getMask(img, threshold = 230):
    # Generate make
    r_idx = img[:, :, 0] > threshold
    g_idx = img[:, :, 1] > threshold
    b_idx = img[:, :, 2] > threshold
    filter_idx = r_idx * g_idx * b_idx
    mask = np.copy(frame)
    mask[:, :, 0] = img[:, :, 0] * filter_idx
    mask[:, :, 1] = img[:, :, 1] * filter_idx
    mask[:, :, 2] = img[:, :, 2] * filter_idx
    
    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel)
    return mask

if __name__ == '__main__':

    cap = cv2.VideoCapture("waiting_for_you.mp4")

    if (cap.isOpened() == False):
        print("Error to open the video capture object...")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            # Resize image
            height, width, _ = np.shape(frame)
            frame = cv2.resize(frame, (width // 2, height // 2))
            print(width // 2, height // 2)

            # Get mask
            mask = getMask(frame)

            # Visualize
            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyWindow()
