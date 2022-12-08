import torchvision.transforms as transforms
from question8 import cnn
from torchvision.transforms import GaussianBlur, ToTensor, Normalize

if __name__ == '__main__':
    # transform = transforms.Compose([
    #     GaussianBlur(kernel_size=3),
    #     ToTensor()
    # ])
    transform = transforms.Compose([
        ToTensor(),
        Normalize(0.5, 0.5)
    ])
    cnn(transform=transform)
