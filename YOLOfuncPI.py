import cv2
import numpy as np

#  Use this function after the Cat detection was done on the original image and the results were saved.
#  Crop the original image using this function and pass the cropped image to detectCats function for new results
def crop_center(img, cropx, cropy):
       _, y, x = img.shape
       startx = x // 2 - (cropx // 2)
       starty = y // 2 - (cropy // 2)
       return img[:, starty:starty + cropy, startx:startx + cropx]

def detectCats(net,img,classes,layer_names,output_layers,colors):
    # Loading image
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    objectsDetected = []
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            objectsDetected.append(label)
    return objectsDetected

# Load Yolo (These should be loaded into the memory when we start the program for efficiency reasons)
# Adjust the path of the files

net = cv2.dnn.readNet("V:\Cat_Or_Not\YOLO\yolov3.weights", "V:\Cat_Or_Not\YOLO\yolov3.cfg")
classes = []
with open("V:\Cat_Or_Not\YOLO\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Image must be read using cv2.imread
img = cv2.imread("V:/Cat_Or_Not/cat-or-not-1.0/cat-or-not-1.0/Cat_Or_NotCat/Training set/Cat/image22(1).jpg")

objectsDetected = detectCats(net,img,classes,layer_names,output_layers,colors)
print(objectsDetected)