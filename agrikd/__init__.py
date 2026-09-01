"""AGRI-KD: multi-task rice disease diagnosis for mobile deployment.

A single lightweight model produces five outputs from one forward pass:
disease classification, lesion localisation, leaf-area infected percentage,
severity grade, and lesion count.

Basic use::

    from agrikd import StudentModel
    from agrikd.data import eval_transform
    import cv2, torch

    model = StudentModel(pretrained=False)
    model.load_state_dict(torch.load("A6_Full_AGRIKD_best.pth", map_location="cpu"))

    image = cv2.cvtColor(cv2.imread("leaf.jpg"), cv2.COLOR_BGR2RGB)
    tensor = eval_transform()(image=image)["image"].unsqueeze(0)
    prediction = model.predict(tensor)
"""

__version__ = "1.0.0"

from agrikd.config import ABLATION, DataConfig, LeafSegConfig, TrainConfig
from agrikd.models import LeafSegmenter, StudentModel, TeacherEnsemble, TeacherModel

__all__ = ["StudentModel", "TeacherModel", "TeacherEnsemble", "LeafSegmenter",
           "DataConfig", "TrainConfig", "LeafSegConfig", "ABLATION", "__version__"]
