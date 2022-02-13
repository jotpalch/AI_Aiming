# AI_Aiming

利用 AI 技術實現槍戰遊戲中輔助瞄準敵人的功能  

## 使用技術

` MoveNet ` google推出的人體姿勢檢測模型  
[movenet/singlepose/lightning - TensorFlow Hub](https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4)  
  
` Python MSS ` 用來擷取螢幕的輸出   
[python-mss - github](https://github.com/BoboTiG/python-mss)  
  
` pyautogui ` 控制滑鼠動作(指向敵人、點擊射擊)    
[PyAutoGUI - PyPI](https://pypi.org/project/PyAutoGUI/)  
  
## 實作步驟

1. 載入模型
```python
interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')
interpreter.allocate_tensors()
```
2. 取得視窗內容
```python
with mss.mss() as sct:
    while "Screen capturing":
        img = numpy.array(sct.grab(monitor))[:, :, :3]
        img = numpy.clip(img,0,255).astype(numpy.uint8)

        cv2.imshow("OpenCV/Numpy normal", img)
```
3. 針對擷取內容進行預測
```python
        fimg = img.copy()
        fimg = tf.image.resize_with_pad(np.expand_dims(fimg, axis=0), 192,192)
        input_image = tf.cast(fimg, dtype=tf.uint8)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        draw_keypoints(img, keypoints_with_scores, 0.34)
```
4. 標示出身體部位並連線
```python
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))[:7]

    y1, x1, c1 = shaped[-2] # 左肩
    y2, x2, c2 = shaped[-1] # 右肩
    y3, x3, c3 = shaped[0]  # 鼻子

    if (c1 > confidence_threshold) & (c2 > confidence_threshold):
        cv2.circle(frame, (int(x1), int(y1)), 4, (0,255,0), -1)
        cv2.circle(frame, (int(x2), int(y2)), 4, (0,255,0), -1)
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

    if c3 > confidence_threshold+0.04 :
        cv2.circle(frame, (int(x3), int(y3)), 4, (0,255,0), -1)
        cv2.line(frame, (int(x1), int(y1)), (int(x3), int(y3)), (0,0,255), 2)
        cv2.line(frame, (int(x3), int(y3)), (int(x2), int(y2)), (0,0,255), 2)
```
5. 將鼠標移至預測點位
```python
        pyautogui.moveTo(x3-450+1280, y3-420+719)
        pyautogui.click(clicks=3, interval=0.05)
```

### :movie_camera: 拿一個MV做測試，試著將滑鼠指到人物的鼻頭上
左下視窗為擷取視窗，當檢測到人物會顯現由鼻子、左肩、右肩組成的三角形  
![](https://github.com/jotpalch/AI_Aiming/blob/2807a0ea21ee98b9ef8ee04ae863f03ca20dab6d/mv.gif)

## 成果展示
本來要拿 Apex Legend 作演示，但無奈執行起來極為卡頓，只好改為 CS1.6  
從慢動作放大的視窗觀察，這個程式對於遊戲中的角色還是可以偵測到  
在跑動往前的情況下，沒有對於滑鼠有任何外力作用下，還是能穩穩把敵人打死，起到輔助瞄準的作用  

![](https://github.com/jotpalch/AI_Aiming/blob/2807a0ea21ee98b9ef8ee04ae863f03ca20dab6d/kill.gif)  

## 改進方向

- 現行方案讓程式與遊戲跑在同一台裝置上，大量的運算導致效能低落，可以感受到明顯的卡頓感。如將遊戲跑在另一台電腦上，利用擷取卡把影像傳回執行程式的電腦，可以同時提高程式與遊戲效能。
  
- python的擷取畫面 fps 在 15-20 左右徘徊，如果可以進一步將 fps 提升，相信成效會更佳。   
  ( `mss` 已經是目前最佳方案，經過測試`Pillow`擷取的 fps 只有在 10 左右 )  
  
- 槍戰遊戲中開槍會模擬現實後座力，造成準心上飄，如果要練續開槍必須考慮到壓槍問題。
  
- 如果再加入遊戲人物的 dataset 下去訓練，相信準確率會有顯著提升。
