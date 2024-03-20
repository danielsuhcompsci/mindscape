import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale, ToPILImage
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class Dummyset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, picture_dimensions=(224,224), matrix_dimensions=(32, 32), size=None):
        
        print("initing")
        
        self.size = size
        self.cifar = CIFAR10(root=root, train=train, target_transform=target_transform, download=download)
        self.matrix_transform = Compose([
            Resize(matrix_dimensions),
            Grayscale(),
            ToTensor(),
            lambda x: x.float() / 255.0,
            lambda x: x.squeeze(0).flipud()
        ])
        self.picture_transform = Compose([
            Resize(picture_dimensions),
            ToTensor()
        ])
        if transform is not None:
            self.picture_transform = Compose([self.picture_transform, transform])
        print('data loaded')

    def __len__(self):
        if self.size is not None:
            return self.size
        return len(self.cifar)

    def __getitem__(self, idx):
        image, _ = self.cifar[idx]
        transformed_image = self.picture_transform(image)
        matrix = self.matrix_transform(image)
        return transformed_image, matrix




def show_predictions(test_set, get_matrix, model, N=5):

    test_loader = DataLoader(test_set, batch_size=N, shuffle=True)  # Ensure data is shuffled

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    test_images, matrix_images, predictions = [], [], []

    with torch.no_grad():
        for images, _ in test_loader:
            if len(test_images) >= N:
                break

            images = images.to(device)
            outputs = model.forward(images).cpu()  # Ensure outputs are moved to CPU

            test_images.extend(images.cpu())
            predictions.extend(outputs)

    # Now, generate the matrix representations for the original test images
    for img in test_images:
        img_pil = ToPILImage()(img)  # Convert tensor to PIL Image
        matrix_img = get_matrix(img_pil)  # Apply get_matrix method
        matrix_images.append(matrix_img)

    # Display images, their matrix representations, and their reconstructions
    fig, axs = plt.subplots(N, 3, figsize=(15, 2*N))
    for i in range(N):
        axs[i, 0].imshow(test_images[i].permute(1, 2, 0))  # Original Image
        axs[i, 0].title.set_text('Original Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(matrix_images[i].squeeze(), cmap='gray')  # Matrix representation
        axs[i, 1].title.set_text('Matrix Representation')
        axs[i, 1].axis('off')

        pred_image = predictions[i].squeeze()  # Predicted Image
        axs[i, 2].imshow(pred_image, cmap='gray')
        axs[i, 2].title.set_text('Predicted Image')
        axs[i, 2].axis('off')

    plt.show()
