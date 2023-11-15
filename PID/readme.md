# Engineering Diagrams Automated Information Extraction

## Installatie

* python -m venv env
* env\scripts\activate.bat
* pip install -r requirements.txt

## Cuda support

* [https://pytorch.org/get-started/locally/]()
* pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Gebruik

* Detecteer objecten  met getrainde modellen
  * yolo detect predict model=yolov8n/train/weights/best.pt source=testafbeeldingen/p_id_6_0.png imgsz=3008 save_txt=true
  * yolo detect predict model=yolov8m-seg/train/weights/best.pt source=testafbeeldingen/p_id_6_0.png imgsz=3008 save_txt=true
* open *connection_detection.py*
* scroll naar beneden onderaan de pagina
* wijzig path, labels en labels_segmentation
* run het bestand
