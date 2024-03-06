from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch 
from MyCNN.MyCNN import MyCNN

cnn = MyCNN()

train = FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test = FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

def pre_process_x(image_tensor):
    normalized = image_tensor.float() / 255
    padded = F.pad(normalized, (2, 2, 2, 2), mode="constant", value=0.0)
   
    output_tensor = padded
    return output_tensor
    

def pre_process_y(int_label):
    return torch.tensor(int_label, dtype=torch.long)


x_train = [pre_process_x(image[0]) for image in train]
x_train = torch.stack(x_train)

batch_size=16
training_set = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)


cnn.training_loop(100, training_set, batch_size)





# figure = plt.figure(figsize=(8,8))
# training_data_length = len(train)
# cols = 3
# rows = 3

# for i in range(1, cols * rows + 1):
#     sample_index = torch.randint(training_data_length, size=(1,)).item()
#     image, label = train[sample_index]
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.title(labels_map[label])
#     plt.imshow(image.squeeze(), cmap="grey")
    
# plt.show()