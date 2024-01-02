# %%
# Package imports
import torch
import os
import numpy as np
import loss_function
import monai
from inference import infer
# %%
# Set data preprocess params
USE_DUMMY = False
# %%
# Data preprocess
if not USE_DUMMY : 
    data_dir = '../data/train_multi_frame_dataset/data'
    data_npy_list = os.listdir(data_dir)

    train_X = list()
    for itr,npy_files in enumerate(data_npy_list,1):
        print('\r', f"processed : {itr}", end='')
        data = np.load(os.path.join(data_dir, npy_files))
        data = data
        data = np.transpose(data, (0,3,1,2))
        data = torch.Tensor(data)
        data = torch.unsqueeze(data,0)
        train_X.append(data)

    train_X = torch.cat(train_X,0)
    print('\n',train_X.shape)

    label_dir = '../data/train_multi_frame_dataset/label'
    label_npy_list = os.listdir(label_dir)

    train_Y = list()
    for itr,npy_files in enumerate(label_npy_list,1):
        print('\r', f"processed : {itr}", end='')
        data = np.load(os.path.join(label_dir, npy_files))
        data = np.transpose(data, (2,0,1))
        data = torch.FloatTensor(data)
        data = torch.unsqueeze(data,0)
        train_Y.append(data)
    train_Y = torch.cat(train_Y,0)
    print('\n',train_Y.shape)

if USE_DUMMY :
   train_X = torch.randn(100,5,3,720,1280)
   train_Y = torch.randn(100,1,720,1280)
# %% 
# get loader
train_set = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
# %%
# define model and loss function
from model import LaneNet
model = LaneNet()
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA: model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# criterion = torch.nn.BCELoss(weight=torch.tensor([10.0]).to(device))
criterion = monai.losses.DiceCELoss(weight=torch.tensor([10.0]).to(device))
# %%
# train session
EPOCH = 2000
train_loss_list = list()

for epoch in range(EPOCH):
    print(f"---------------Epoch : {epoch+1}/{EPOCH}--------------------")
    train_loss = 0.0
  
    for train_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        print('\r',f"training {train_idx+1}/{len(train_loader)}, train_loss: {train_loss:0.4f}",end=" ")
        inputs, labels = data
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    if epoch % 10 == 0:
        infer(model, device, seed=1, epoches=(epoch+1), plot_img=False, save_img=True)
    train_loss_list.append(train_loss)
    print('')
# %%
# save train session log
train_loss_list = np.array(train_loss_list)
np.save("../data/loss_diceCE_weight100.npy",train_loss_list)
torch.save(model.state_dict(), "../data/conv_lstm_diceCE_weight100.pth")
# %%
# inference
from inference import infer
infer(model, device, seed=1480, epoches=1)
# %%
# infer data from train set
test_input = np.load('/home/data/train_multi_frame_dataset/data/clips_0313-1_120_road_clips.npy')
test_input = np.transpose(test_input,(0,3,1,2))
test_input = np.expand_dims(test_input,0)
test_input = torch.FloatTensor(test_input)

test_label = np.load('/home/data/train_multi_frame_dataset/label/clips_0313-1_120_lane.npy')
test_label = np.transpose(test_label, (2,0,1))
# %%
# draw prediction
import matplotlib.pyplot as plt
with torch.no_grad():

    y_pred = model(test_input.to(device))
    # y_pred = torch.sigmoid(y_pred)
    y_pred =y_pred.cpu().detach().numpy()
plt.imshow(y_pred[0][0], "gray")
# %%
# draw label
plt.imshow(test_label[0], "gray" )
# %%
# histo of values in prediction 
plt.hist(np.ravel(y_pred[0]))
# %%
# load saved model
model = LaneNet()
model.load_state_dict(torch.load("/home/data/conv_lstm_weight20.pth"))
model.to(device)
# %%
# draw
# setting values to rows and column variables
import torchvision
import matplotlib.pyplot as plt
INFERENCE_PATH = '/home/data/inference'
img_list = os.listdir(INFERENCE_PATH)
img_list.sort()
RESIZER = torchvision.transforms.Resize((720,1080))

img_array=[]
for idx,filename in enumerate(img_list):
    img = plt.imread(os.path.join(INFERENCE_PATH, filename))
    img = np.transpose(img, (2,0,1))
    # img_array.append(img)
    img_tensor = torch.FloatTensor(img)
    resized_img = RESIZER(img_tensor)
    img_array.append(resized_img)

img_array = tuple(img_array)

torch_array = torch.stack(img_array, dim=0)
inference_input = torch.unsqueeze(torch_array, 0)
# %%
inference_img = torch.unsqueeze(resized_img, 0)
inference_img = torch.unsqueeze(inference_img, 0)
inference_img = inference_img.to(device)
# %%
y_pred = model(inference_img)
# %%
import matplotlib.pyplot as plt
rows = 1
columns = 3

fig = plt.figure(figsize=(10, 7))
# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 1) 

# showing image
a = plt.imread('/home/data/000027.jpg')
plt.imshow(a)
plt.axis('off')
plt.title("Input") 

# Adds a subplot at the 3rd position 
fig.add_subplot(rows, columns, 2) 

# showing image 
plt.imshow(y_pred[0][0], "gray")
plt.axis('off') 
plt.title("prediction")