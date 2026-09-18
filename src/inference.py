from ultralytics import YOLO
from pathlib import Path
# import torch
# from PIL import Image
# import numpy
from src.config import load_config

class YOLOv11Inference:
    def __init__(self, model_name, device='cpu'):
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(self.device)

        # loading config from default.yaml
        config = load_config()
        self.conf_threshold = config["model"]["conf_threshold"]
        self.extensions = config["data"]["image_extension"]

    def process_image(self, image_path):

        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device
        )

        # process results
        detection = []
        class_counts = {}

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()

                detection.append({
                    'class' : cls,
                    'confidence' : conf,
                    'bbox' : bbox,
                    'count' : 1
                })

                class_counts[cls] = class_counts.get(cls, 0) + 1

        for det in detection:
            det['count'] = class_counts[det['class']]

        return {
            'image_path' : str(image_path),
            'detections' : detection,
            'total_objects' : len(detection),
            'unique_class' : list(class_counts.keys()), # [0, 1, 2]
            'class_counts' : class_counts # {0 : 3, 1 : 10, 2, : 1}
        }


    def process_directory(self, directory):
        metadata = []

        directory = Path(directory).resolve()

        patterns = [f"*{ext}" for ext in self.extensions]

        image_paths = []
        for pattern in patterns:
            image_paths.extend(directory.glob(pattern))

        for img_path in image_paths:
            try:
                img_path = img_path.resolve()
                metadata.append(self.process_image(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        return metadata
