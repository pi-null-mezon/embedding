import torch
import cv2
import torchvision.models
from torchvision import transforms
import os

os.environ['TORCH_HOME'] = '../weights'
weights_path = "./weights" if os.path.exists("./weights") else "../weights"


def crop_rect_from_center(img):
    width = img.shape[0]
    height = img.shape[1]
    if width >= height:
        diff = (width - height) // 2
        return img[diff:(height + diff), 0:height]
    else:
        diff = (height - width) // 2
        return img[0:width, diff:(width + diff)]


class PhotoEmbedder:
    """
    Base class for embedders
    Any descendants should provide embedding(img) method that should return torch.Tensor with img embedding
    and embedder uid
    """

    def __init__(self, uid, input_size, transform=None):
        self.uid = uid
        self.input_size = input_size
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.model = None

    def photo2tensor(self, cv_bgr_img, target_size):
        crop = cv2.resize(crop_rect_from_center(cv_bgr_img), target_size)
        # cv2.imshow("probe", crop)
        # cv2.waitKey(0)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(crop)

    def prepare_input_tensor(self, cv_bgr_img):
        # FUTURE - add image crops here
        return self.photo2tensor(cv_bgr_img, self.input_size).unsqueeze(0)

    def embedding(self, img) -> tuple:
        # img - opencv image with BGR channels order
        if self.model:
            with torch.no_grad():
                t_input = self.prepare_input_tensor(img)
                prediction = self.model(t_input.to(self.device))
                return self.uid, prediction
        raise NotImplementedError


class TorchvisionResNet50(PhotoEmbedder):
    def __init__(self, device):
        super().__init__("TorchvisionResNet50", (224, 224))
        self.device = device
        self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Identity()  # replace last layer connections by identity
        self.model.to(device)
        self.model.eval()
        self.model(torch.rand(1, 3, 224, 224).to(device))


class TorchvisionEfficientNetB5(PhotoEmbedder):
    def __init__(self, device):
        super().__init__("TorchvisionEfficientNetB5", (224, 224))
        self.device = device
        self.model = torchvision.models.efficientnet_b5(weights=torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()  # replace last layer connections by identity
        self.model.to(device)
        self.model.eval()
        self.model(torch.rand(1, 3, 224, 224).to(device))


class TorchvisionMobileNetV2(PhotoEmbedder):
    def __init__(self, device):
        super().__init__("TorchvisionMobileNetV2", (224, 224))
        self.device = device
        self.model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model.classifier = torch.nn.Identity()  # replace last layer connections by identity
        self.model.to(device)
        self.model.eval()
        self.model(torch.rand(1, 3, 224, 224).to(device))


similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def normalize(t: torch.Tensor) -> torch.Tensor:
    return t / torch.max(t.abs())


def embedding(embedders, img):
    embeddings = []
    uid = ""
    for embedder in embedders:
        emb = embedder.embedding(img)
        uid += f"{emb[0]}" if len(uid) == 0 else f"+{emb[0]}"
        embeddings.append(normalize(emb[1]))
    return uid, torch.concat(embeddings, 1)
