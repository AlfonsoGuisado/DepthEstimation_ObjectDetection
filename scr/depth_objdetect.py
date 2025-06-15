import numpy as np
from PIL import ImageSequence, ImageDraw, ImageFont
import torch
from torchvision import transforms
import torchvision.models.detection as detection
from transformers import pipeline

class DepthEst_ObjectDetect:
    def __init__(self, pipe, mask_rcnn, score_threshold=0.7):
        self.pipe = pipe
        self.mask_rcnn = mask_rcnn
        self.score_threshold = score_threshold

    def object_filter(self, detections, depth_array, threshold=0.7):
        objetos_filtrados = []

        for i in range(len(detections['boxes'])):
            score = detections['scores'][i].item()
            if score >= threshold:
                mask1 = detections['masks'][i, 0].cpu().numpy() > 0.5
                if np.any(mask1):
                    depth1 = np.mean(depth_array[mask1])

                    reemplazado = False
                    for j, (mask2, depth2) in enumerate(objetos_filtrados):
                        if np.any(np.logical_and(mask1, mask2)):
                            if depth1 > depth2:
                                objetos_filtrados[j] = (mask1, depth1)
                            reemplazado = True
                            break

                    if not reemplazado:
                        objetos_filtrados.append((mask1, depth1))

        return objetos_filtrados

    def object_draw(self, draw, object_filter):
        for mask, depth in object_filter:
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                
                x1, y1, x2, y2 = int(x.min()), int(y.min()), int(x.max()), int(y.max())
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 10), f"Object Depth: {depth:.1f}", fill="red", font=ImageFont.load_default())

    def frame_processor(self, frame, pipe, mask_rcnn):
        frame = frame.convert("RGB")

        result = pipe(frame)
        depth_img = result["depth"].convert("RGB")
        depth_array = result["predicted_depth"].squeeze().cpu().numpy()

        input_tensor = transforms.ToTensor()(frame)
        with torch.no_grad():
            detections = mask_rcnn([input_tensor])[0]

        draw = ImageDraw.Draw(depth_img)
        objetos = self.object_filter(detections, depth_array)
        self.object_draw(draw, objetos)

        return depth_img