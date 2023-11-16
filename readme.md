# Engineering Diagrams Automated Information Extraction

## Installatie

* python -m venv env
* env\scripts\activate.bat
* pip install -r requirements.txt

## Gebruik

* Detecteer objecten  met de getrainde modellen
  * yolo detect predict model=yolov8n/train/weights/best.pt source=testafbeeldingen/p_id_6_0.png imgsz=3008 save_txt=true
  * yolo detect predict model=yolov8m-seg/train/weights/best.pt source=testafbeeldingen/p_id_6_0.png imgsz=3008 save_txt=true
* open *connection_detection.py*
* scroll naar beneden onderaan de pagina
* wijzig path, labels en labels_segmentation
* run het bestand

## YOLOv8 Documentation and more

* Repository: [https://github.com/ultralytics/ultralytics/]()
* Official page: [https://yolov8.com/]()
* Documentation: [https://docs.ultralytics.com/]()

## Cuda support

* [https://pytorch.org/get-started/locally/]()
* pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Roboflow

* Official page: [https://roboflow.com/]()

## Video tutorial

* [https://youtu.be/f6bn96WMgh4]()