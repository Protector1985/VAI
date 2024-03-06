import torch.nn as nn
import torch.optim as optim
import torch
from classes.Sampling import Sampling
import torch.nn.functional as F

sampling = Sampling()

class MyCNN(nn.Module):
    
    def __init__(self):
        super(MyCNN, self).__init__()
        self.lr = 1e-4
        self.kernel_size = 4
        self.padding = ((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2)
        ##encoder 
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=self.kernel_size, stride=2,padding=self.padding)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=2, padding=self.padding)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel_size, stride=2, padding=self.padding)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, stride=2, padding=self.padding)
        self.relu4 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, stride=2, padding=self.padding)
        self.relu5_encode = nn.ReLU()
        self.dense1 = nn.Linear(in_features=32, out_features=2)
        self.dense2 = nn.Linear(in_features=32, out_features=2)
     
        #decoder
        
        self.decoder_linear = nn.Linear(in_features=2, out_features=128 * 7 * 7)
        self.transpose_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=396, kernel_size=4, stride=2, padding=self.padding)
        self.relu5 = nn.ReLU()
        self.transpose_conv2 = nn.ConvTranspose2d(in_channels=396, out_channels=128, kernel_size=4, stride=2, padding=self.padding)
        self.relu6 = nn.ReLU()
        self.transpose_conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=self.padding)
        self.relu7 = nn.ReLU()
        self.transpose_conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=self.padding)
        self.relu8 = nn.ReLU()
        self.transpose_conv5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=self.padding)
        
        self.sigmoid = nn.Sigmoid()
        
        #training loop
        
        self.loss = nn.BCELoss()
    
    def load_model(self):
        self.load_state_dict(torch.load("./trained/model-20"))
        
    def forward(self, x):
        # The forward method should define the feedforward behavior of the model
        z_mean, z_log_var = self.encoder(x)
        z = sampling.call([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction
     
    def training_loop(self, epochs, training_set, batch_size):  
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        for epoch in range(epochs):
            total_loss = 0
            total_reconstruction_loss = 0
            total_kl_loss = 0
            for batch in training_set:
                # Assuming `batch` is a tensor with shape [batch_size, channels, height, width]
                # and the data is scaled to [0, 1] for binary cross-entropy.
                batch = batch.to(next(self.parameters()).device)  # Ensure batch is on the same device as the model
                optimizer.zero_grad()

                z_mean, z_log_var, reconstructed = self.forward(batch)
                reconstruction_loss = F.binary_cross_entropy(reconstructed, batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = reconstruction_loss + kl_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_kl_loss += kl_loss.item()
                

            # Adjust learning rate based on the total loss if needed
            # [your learning rate adjustment logic]

            average_loss = total_loss / len(training_set.dataset)
            average_reconstruction_loss = total_reconstruction_loss / len(training_set.dataset)
            average_kl_loss = total_kl_loss / len(training_set.dataset)
            print(f"Epoch {epoch+1}, Total Loss: {average_loss}, Reconstruction Loss: {average_reconstruction_loss}, KL Divergence: {average_kl_loss}")
            torch.save(self.state_dict(), f"./trained/model-{epoch}")
        
    
    def inference(self, inference_set):
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in inference_set:
                z_mean, z_log_var = self.encoder(batch)
                z = sampling.call([z_mean, z_log_var])
                decoder_output = self.decoder(z)
                predictions.append(decoder_output)
        return predictions
    
    
            
    def decoder(self, location):
        x = self.decoder_linear(location)
       
        x = x.view(x.size(0), 128, 7, 7)
        
        x = self.transpose_conv1(x)
        x = self.relu5(x)
        x = self.transpose_conv2(x)
        x = self.relu6(x)
        x = self.transpose_conv3(x)
        x = self.relu7(x)
        x = self.transpose_conv4(x)
        x = self.relu8(x)
        x = self.transpose_conv5(x)
    
        output = self.sigmoid(x)

        
        return output
        
        
    def encoder(self, image):
        
        x = self.input_layer(image)
      
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.relu4(x)
        x = self.conv4(x)
        x = self.relu5_encode(x)
        
        #first dim = batch size so x.size(0), the rest is inferred so -1 which multiplies the remaining vals
       
        x = x.view(x.size(0), -1)
      
        z_mean = self.dense1(x)
        z_log_var = self.dense2(x)
        
        return z_mean, z_log_var
        