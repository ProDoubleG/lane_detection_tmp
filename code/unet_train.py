# %%
import torch
import os
import numpy as np
import monai
import loss_function
import matplotlib.pyplot as plt
# %%
def model_test():
    dummy_input = torch.randn(1,3,720,1280)
    model = monai.networks.nets.UNet(spatial_dims=2,in_channels=3,out_channels=1,channels=(4, 8, 16),strides=(2, 2))
    x = model(dummy_input)
    assert x.shape == (1,1,720,1280)

model_test()
# %%
data_dir = '../data/train_one_frame_dataset/data'
data_npy_list = os.listdir(data_dir)

train_X = list()
for itr,npy_files in enumerate(data_npy_list,1):
    print('\r', f"processed : {itr}", end='')
    data = np.load(os.path.join(data_dir, npy_files))
    data = data/255.0
    data = np.transpose(data, (2,0,1))
    data = torch.Tensor(data)
    data = torch.unsqueeze(data,0)
    train_X.append(data)

train_X = torch.cat(train_X,0)
print(train_X.shape)
# %%
label_dir = '../data/train_one_frame_dataset/label'
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
print(train_Y.shape)
# %%
train_set = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
# %%
model = monai.networks.nets.SegResNet(spatial_dims=2, init_filters=16, in_channels=3, out_channels=1)
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

if USE_CUDA: model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

criterion = torch.nn.BCELoss(weight=torch.tensor([10.0]).to(device))

sigmoid = torch.nn.Sigmoid()

def save_pic(model, epoches):
    test_input = np.load('/home/data/train_one_frame_dataset/data/clips_0313-1_60_road.npy')
    test_input = np.transpose(test_input,(2,0,1))
    test_input = np.expand_dims(test_input,0)
    test_input = torch.FloatTensor(test_input)

    test_input  = test_input.to(device)
    test_label = np.load('/home/data/train_one_frame_dataset/label/clips_0313-1_60_lane.npy')
    test_label = np.transpose(test_label,(2,0,1))

    y_pred = model(test_input.to(device))
    y_pred = torch.sigmoid(y_pred[0])
    y_pred = y_pred.cpu().detach().numpy()
    
    # Adds a subplot at the 1st position 
    figure, axis = plt.subplots(1, 3) 
    input_img = plt.imread('../data/TUSimple/train_set/clips/0313-1/60/20.jpg')

    axis[0].imshow(input_img)
    axis[0].set_title("Input")
    
    axis[1].imshow(test_label[0], "gray")
    axis[1].set_title("label")

    axis[2].imshow(y_pred[0], "gray")
    axis[2].set_title("prediciton")
    
    fname = f'{epoches}'
    fname = f"../data/checkpoint/{fname.zfill(6)}"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
# %%
EPOCH = 1000
train_loss_list = list()
for epoch in range(EPOCH):
    print(f"---------------Epoch : {epoch+1}/{EPOCH}--------------------")
    train_loss = 0.0

    for train_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        print('\r',f"training {train_idx+1}/{len(train_loader)}, train_loss: {train_loss:0.4f}",end=" ")
        inputs, labels = data

        outputs = model(inputs.to(device))

        loss = criterion(sigmoid(outputs), labels.to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss_list.append(train_loss)
    
    if epoch % 5 == 0:
        save_pic(model, epoch)
    else:
        pass
        
    print('')
# %%
train_loss_list = np.array(train_loss_list)
np.save("../data/loss.npy",train_loss_list)
torch.save(model.state_dict(), "../data/segnet.pth")
print("finished!")
# %%
def infer():
    assert False
    model = monai.networks.nets.UNet(spatial_dims=2,in_channels=3,out_channels=1,channels=(4, 8, 16, 32, 64),strides=(2, 2, 2, 2))
    model.load_state_dict(torch.load("../data/unet.pth"))
    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    if USE_CUDA: model.to(device)
    model.to(device)

    test_input = np.load('/home/data/train_one_frame_dataset/data/clips_0313-1_60_road.npy')
    test_input = np.transpose(test_input,(2,0,1))
    test_input = np.expand_dims(test_input,0)
    test_input = torch.FloatTensor(test_input)

    test_input  = test_input.to(device)
    test_label = np.load('/home/data/train_one_frame_dataset/label/clips_0313-1_60_lane.npy')
    test_label = np.transpose(test_label,(2,0,1))

    import matplotlib.pyplot as plt
    with torch.no_grad():

        y_pred = model(test_input.to(device))
        y_pred = torch.sigmoid(y_pred[0])
        y_pred = y_pred.cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 7)) 
  
# setting values to rows and column variables 
    rows = 1
    columns = 3
    
    # Adds a subplot at the 1st position 
    fig.add_subplot(rows, columns, 1) 
    
    # showing image
    a = plt.imread('../data/TUSimple/train_set/clips/0313-1/60/20.jpg')
    plt.imshow(a)
    plt.axis('off')
    plt.title("Input") 
    
    # Adds a subplot at the 2nd position 
    fig.add_subplot(rows, columns, 2) 
    
    # showing image 
    plt.imshow(test_label[0], "gray" ) 
    plt.axis('off') 
    plt.title("label") 
    
    # Adds a subplot at the 3rd position 
    fig.add_subplot(rows, columns, 3) 
    
    # showing image 
    plt.imshow(y_pred[0], "gray")
    plt.axis('off') 
    plt.title("prediction")
# %%
