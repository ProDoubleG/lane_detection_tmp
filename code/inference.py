import torch
import numpy as np
import matplotlib.pyplot as plt
import os
def infer(model, device, epoches=0, seed=1, plot_img=True, save_img=False):

    npy_dir = "/home/data/train_multi_frame_dataset"
    raw_dir = "/home/data/TUSimple/train_set"

    npy_data_dir = os.path.join(npy_dir,"data")
    npy_label_dir = os.path.join(npy_dir,"label")

    clip_npy = os.listdir(npy_data_dir)[seed]
    sample_id = clip_npy.split('road_clips.npy')[0]

    print("\n","save inference for : ", clip_npy)
    lane_npy = sample_id + "lane.npy"
    lane_dir = os.path.join(npy_label_dir, lane_npy)

    substr = sample_id.split('_')
    raw_sample = os.path.join(raw_dir,substr[0],substr[1],substr[2],"20.jpg")

    # test_input = np.load('/home/data/train_multi_frame_dataset/data/clips_0313-1_60_road_clips.npy')
    test_input = np.load(os.path.join(npy_data_dir, clip_npy))
    test_input = np.transpose(test_input,(0,3,1,2))
    test_input = np.expand_dims(test_input,0)
    test_input = torch.FloatTensor(test_input)
    test_input  = test_input.to(device)

    # test_label = np.load('/home/data/train_multi_frame_dataset/label/clips_0313-1_60_lane.npy')
    test_label = np.load(os.path.join(npy_label_dir, lane_npy))
    test_label = np.transpose(test_label, (2,0,1))
    y_pred = model(test_input.to(device))
    y_pred = torch.sigmoid(y_pred[0])
    y_pred = y_pred.cpu().detach().numpy()
    
    # Adds a subplot at the 1st position 
    figure, axis = plt.subplots(1, 3) 
    input_img = plt.imread(raw_sample)
    # input_img = plt.imread('/home/data/TUSimple/train_set/clips/0313-1/60/20.jpg')

    axis[0].imshow(input_img)
    axis[0].set_title("Input")
    
    axis[1].imshow(test_label[0], "gray")
    axis[1].set_title("label")

    axis[2].imshow(y_pred[0], "gray")
    axis[2].set_title("prediciton")
    
    fname = f'{epoches}'
    fname = f"/home/data/checkpoint/{fname.zfill(6)}"
    plt.tight_layout()
    if plot_img:
        plt.show()
    if save_img:
        plt.savefig(fname)
    plt.close()