#  Depth & Object Detection; Deep Learning, CNN, Transformers
Explore how to apply deep learning models to estimate depth and segment moving objects frame by frame in videos. A powerful combination of computer vision, transformers, and object segmentation demonstration to enhance 2D scenes with implicit 3D understanding.

🚩 **Key Features:**

  • **🖼️ Depth Estimation:** Leverage transformer-based models to generate depth maps from 2D images.  
  • **🧩 Object Segmentation:** Accurately detect and segment objects giving the proximity of the object.  
  • **🧠 Smart Filtering Logic:** Prioritizes closer objects, ignoring those partially hidden, ensuring visual coherence.  
  
🛠️ **Tools & Dependencies:**

[![Python](https://img.shields.io/badge/Python-3.12.10-blue)](https://www.python.org/downloads/release/python-31210/) [![NumPy](https://img.shields.io/badge/NumPy-2.2.4-green)](https://numpy.org/) [![Transformers](https://img.shields.io/badge/Transformers-4.52.4-green)](https://huggingface.co/docs/transformers/es/index) [![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-green)](https://pytorch.org/) [![Torchvision](https://img.shields.io/badge/Torchvision-0.22.1-green)](https://docs.pytorch.org/vision/stable/index.html)

🤖 **Models:**

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Intel/dpt_large-purple)](https://huggingface.co/Intel/dpt-large) [![PyTorch](https://img.shields.io/badge/PyTorch-Intel/maskrcnn_resnet50_fpn-purple)](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn)

🎞️ **Visual Demo:**

| **Original** | **Processed (Depth + Masks)** |
|---|---|
| ![Original](notebook/driving_highway.gif) | ![Processed](notebook/gifHighway_depth_mask.gif) |


🌍 **Language:**

![Idioma](https://img.shields.io/badge/🇪🇸-Spanish-red)