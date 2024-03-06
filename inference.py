from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from MyCNN.MyCNN import MyCNN

cnn = MyCNN()
cnn.load_model()

test = FashionMNIST(
    root="data",
    download=True,
    train=False,
    transform=transforms.ToTensor()
)

def pre_process_x(image_tensor):
    normalized = image_tensor.float() / 255
    padded = F.pad(normalized, (2, 2, 2, 2), mode="constant", value=0.0)
   
    output_tensor = padded
    return output_tensor

def pre_process_y(int_label):
    return torch.tensor(int_label, dtype=torch.long)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

x_test = [pre_process_x(image[0]) for image in test]
x_test = torch.stack(x_test)

batch_size=16
test_set = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)

original_batch = [image for image in test_set]
original_batch = original_batch[:16]
torch.stack(original_batch)
    

predictions = cnn.inference(test_set)



figure1 = plt.figure(figsize=(8,8))
figure2 = plt.figure(figsize=(8,8))

cols = 5
rows = 5

fig1, axes1 = plt.subplots(rows, cols, figsize=(8, 8))  # For original images
fig2, axes2 = plt.subplots(rows, cols, figsize=(8, 8))  # For predicted images

for i in range(cols):
    for j in range(rows):
        sample_index = torch.randint(batch_size, size=(1,)).item()

        # Get the original and predicted images
        original_image = original_batch[0][sample_index]
        predicted_image = predictions[2][sample_index]

        # Display original image
        # if rows > 1:
        #     ax1 = axes1[j, i]
        # else:
        #     ax1 = axes1[i]
        # # ax1.imshow(original_image.squeeze(), cmap='gray')
        # ax1.axis('off')

        # Display predicted image
        if rows > 1:
            ax2 = axes2[j, i]
        else:
            ax2 = axes2[i]
        ax2.imshow(predicted_image.squeeze(), cmap='gray')
        ax2.axis('off')


plt.show()