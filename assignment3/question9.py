import torchvision.transforms as transforms
from question8 import cnn, load_dataset, get_batch_size
from torchvision.transforms import GaussianBlur, ToTensor, Normalize

if __name__ == '__main__':
    transform = transforms.Compose([
        GaussianBlur(kernel_size=3),
        ToTensor()
    ])
    # transform = transforms.Compose([
    #     ToTensor(),
    #     Normalize(0.5, 0.5)
    # ])
    batch_size = get_batch_size()
    train_transformed_batches, val_tensor_batches = load_dataset(transform=transform, batch_size=batch_size)
    cnn(train=train_transformed_batches, val=val_tensor_batches, batch_size=batch_size)
