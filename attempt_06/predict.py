from ultralytics import YOLO

model = YOLO("yolov8m-seg-custom.pt")

prediction = model.predict(source = "1.png", show = True, save = True, hide_labels = False, hide_conf = False, conf = 0.5, save_txt = False, save_crop = False, line_thickness = 2)

if prediction[0].names[0] == "head":
    print("Head is there")