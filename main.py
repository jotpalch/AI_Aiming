import time
import cv2
import mss
import numpy
import win32gui
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pyautogui

# interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')
interpreter.allocate_tensors()


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))[:7]

    y1, x1, c1 = shaped[-2]
    y2, x2, c2 = shaped[-1]
    y3, x3, c3 = shaped[0]

    if (c1 > confidence_threshold) & (c2 > confidence_threshold):
        cv2.circle(frame, (int(x1), int(y1)), 4, (0,255,0), -1)
        cv2.circle(frame, (int(x2), int(y2)), 4, (0,255,0), -1)
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

    if c3 > confidence_threshold+0.04 :
        pyautogui.moveTo(x3-450+1280, y3-420+719)
        pyautogui.click(clicks=3, interval=0.05)
        cv2.circle(frame, (int(x3), int(y3)), 4, (0,255,0), -1)
        cv2.line(frame, (int(x1), int(y1)), (int(x3), int(y3)), (0,0,255), 2)
        cv2.line(frame, (int(x3), int(y3)), (int(x2), int(y2)), (0,0,255), 2)

monitor = {"top": 270, "left": 830, "width": 900, "height": 900}

with mss.mss() as sct:

    while "Screen capturing":
        last_time = time.time()

        img = numpy.array(sct.grab(monitor))[:, :, :3]
        img = numpy.clip(img,0,255).astype(numpy.uint8)

        # Reshape image
        fimg = img.copy()
        fimg = tf.image.resize_with_pad(np.expand_dims(fimg, axis=0), 192,192)
        input_image = tf.cast(fimg, dtype=tf.uint8)
        # input_image = tf.cast(fimg, dtype=tf.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        draw_keypoints(img, keypoints_with_scores, 0.34)

        # show fps
        cv2.putText(img, "fps: {}".format(numpy.round(1 / (time.time() - last_time),2) ), (10,20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("OpenCV/Numpy normal", img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
