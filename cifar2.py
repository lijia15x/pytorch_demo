import torch
import torchvision.transforms
from PIL import Image
from torch import nn
 
img_path = "./data/3.PNG"
image = Image.open(img_path)
# print(image) # <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=405x303 at 0x2834CC62F48>
 
image = image.convert("RGB") # 该代码的作用：
# 因为png格式的图片是四个通道，除了RGB三通道外，还有一个透明度通道。
# 所以我们调用image = image.convert("RGB")，保留其颜色通道。
 
# print(image) # <PIL.Image.Image image mode=RGB size=405x303 at 0x2834CD7AB88>
 
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
# print(image.size)
 
# CIFAR10网络模型结构
class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 展平
            nn.Linear(1024, 64),  # 1024=64*4*4
            nn.Linear(64, 10)
        )
 
    def forward(self, x):
        x = self.model(x)
        return x
 
# 加载网络模型参数
# model = torch.load("./pth/M_29.pth", map_location=torch.device("cpu"))
model = torch.load("./pth/M_20.pth")
# map_location=torch.device("cpu") 将GPU版本的模型对应到CPU上
# print(model)
#['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))