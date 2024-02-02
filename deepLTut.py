import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backend.mps.is_available()
    else "cpu"
)

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(8,5))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    if predictions is not None: 
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size":14})
    plt.show()



class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
torch.manual_seed(42)



# model_0 = LinearRegressionModel().to(device)

model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(X_test)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)

epochs = 5000


train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):

    ### TRAINING

    # Put the model in training mode(This is the deafult state of the model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method 
    y_pred = model_0(X_train)

    # 2. Calculate loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer 
    optimizer.step()

    ### TESTING

    #Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = model_0(X_test)

        # 2. Calculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisions need to be done with tensors of the same type

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            # print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
            # plt.plot(epoch_count, train_loss_values, label="Train loss")
            # plt.plot(epoch_count, test_loss_values, label="Test loss")
            # plt.title("Training and test loss curves")
            # plt.ylabel("Loss")
            # plt.xlabel("Epochs")
            # plt.legend()
            # plot_predictions()

    model_0.eval()

    with torch.inference_mode():
        y_preds = model_0(X_test)
y_preds
# plot_predictions(predictions=y_preds)

##Saving a Pytorch model's state_dict()

#1. Create models directory
MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

#2 Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
# print(f"Saving model to: {MODEL_SAVE_PATH}")
# print(model_0.state_dict())
# torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

##Loading a model

#Instatiate a new instance of our model(This will be instatiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved mdoel(this will updated the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

#1 Put the loaded model into evaluation mode
loaded_model_0.eval()

#2 Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_mode_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model
    #Compare previous model predictions with loaded model predictions(should be the same)
    print(y_preds == loaded_mode_preds)