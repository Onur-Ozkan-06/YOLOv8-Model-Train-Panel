import cv2
from ultralytics import YOLO

def main(model_path, img_path):

    PPE_classes = [0,1,2,3]
    image_path = str(img_path)
    img=cv2.imread(image_path)

    PPE_model= YOLO(model_path)

    pred_result = PPE_model.predict(
        source=img,
        classes=PPE_classes,
        conf=0.4,       
        iou=0.45,        
        save=False,      
        show=False,   
    )

    result_img = pred_result[0].plot()

    cv2.imshow("YOLO Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



