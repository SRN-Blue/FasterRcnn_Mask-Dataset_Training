import torchvision
import torch
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter

from Dataset.data_loader import get_trainer_dataloader
from Model.model import get_model

# Create a SummaryWriter (logs will be saved in the 'runs' directory)
writer = SummaryWriter()

# Load data and model
data_loader = get_trainer_dataloader()
model = get_model()

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 10
model.to(device)

# Define optimizer, learning rate scheduler, and other parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
len_dataloader = len(data_loader)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (imgs, annotations) in enumerate(data_loader):
        imgs = [img.to(device) for img in imgs]
        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in annotations]

        # Forward pass
        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Print and log loss
        epoch_loss += losses.item()
        print(f'Epoch: {epoch + 1}/{num_epochs}, Iteration: {i +
              1}/{len_dataloader}, Loss: {losses.item()}')

    # Update the learning rate
    scheduler.step()
    print(f'Epoch {epoch + 1} Loss: {epoch_loss}')

# Save the trained model
save_path = r'.\Trained_Model\model.pt'
torch.save(model, save_path)
print('Trained model saved!')
