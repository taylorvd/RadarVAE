#!/usr/bin/env python3
from cv_bridge import CvBridge
import numpy as np
import torch
from rosbag import Bag
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from depth_image_dataset import DepthImageDataset
import random



def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS bags.")
    parser.add_argument("bag_files", nargs='+', help="Input ROS bags.")
    parser.add_argument("image_topic", help="Image topic.")
    parser.add_argument("data_type", help= "normal = divide 80/20 train/test, train = just training data, test = just testing data")

    args = parser.parse_args()

    tensor_list = []
    for bag_file in args.bag_files:
        bag = Bag(bag_file, "r")
        bridge = CvBridge()
        for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
            valid, tensor = ros_depth_image_to_torch(msg, bridge)
            if valid:
                tensor_list.append(tensor)

        # num_samples = 10000
        # if len(tensor_list)>num_samples:
        #     inds = list(range(len(tensor_list)))
        #     test_inds = random.sample(inds, num_samples)
        #     temp_list = [tensor_list[i] for i in test_inds]
        #     tensor_list = temp_list

    if (args.data_type == "normal"):

        #split data, 80% training
        train_end_ind = round(len(tensor_list)*0.8)
    
        train_list = tensor_list[0:train_end_ind]
        depth_image_dataset_train = DepthImageDataset(train_list)

        print("Length of training set: ", len(train_list))
        with open('./data/input/train_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_train, f)
    
        #20% testing
        test_list = tensor_list[train_end_ind:]
        depth_image_dataset_test = DepthImageDataset(test_list)
        print("Length of test set: ", len(test_list))
        with open('./data/input/test_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_test, f)
    
    elif (args.data_type == "train"):
        num_samples = 20000
        if len(tensor_list)>num_samples:
            inds = list(range(len(tensor_list)))
            test_inds = random.sample(inds, num_samples)
            temp_list = [tensor_list[i] for i in test_inds]
            tensor_list = temp_list
        depth_image_dataset_train = DepthImageDataset(tensor_list)
        print("Length of training set: ", len(tensor_list))
        with open('./data/input/train_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_train, f)

    elif (args.data_type == "test"):
        num_samples = 5000
        if len(tensor_list)>num_samples:
            inds = list(range(len(tensor_list)))
            test_inds = random.sample(inds, num_samples)
            temp_list = [tensor_list[i] for i in test_inds]
            tensor_list = temp_list
        depth_image_dataset_test = DepthImageDataset(tensor_list)
        print("Length of test set: ", len(tensor_list))
        with open('./data/input/test_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_test, f)



#https://github.com/qboticslabs/rostensorflow/blob/master/image_recognition.py
#depth_image = ROS Image message
def ros_depth_image_to_torch(depth_image, bridge):
    
    # convert ROS image to numpy array
    #https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
    cv_bridge = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')

    # convert to float and normalize to be between 0 and 1
    depth_array = np.array(cv_bridge, dtype=np.float32)
    

    #assume min value of array = 0
    #https://stackoverflow.com/questions/70783357/how-do-i-normalize-the-pixel-value-of-an-image-to-01
    max_depth = np.max(depth_array)
    if (max_depth != 0):

        norm_depth_array = depth_array /255  #np.max(depth_array)
        
        
   # else:
        #norm_depth_array = depth_array * 0
       
   

    # convert to torch tensor and add batch dimension (1 channel)
        depth_tensor = torch.from_numpy(norm_depth_array).unsqueeze(0)

        return 1, depth_tensor
    else:
        return 0,0

"""
#https://github.com/qboticslabs/rostensorflow/blob/master/image_recognition.py
#depth_image = ROS Image message
def ros_depth_image_to_torch(depth_image, bridge):
    
    # convert ROS image to numpy array
    #https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
    cv_bridge = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')

    # convert to float and normalize to be between 0 and 1
    depth_array = np.array(cv_bridge, dtype=np.float32) 

    #assume min value of array = 0
    #https://stackoverflow.com/questions/70783357/how-do-i-normalize-the-pixel-value-of-an-image-to-01
    max_depth = np.max(depth_array)
    if (max_depth != 0):

        norm_depth_array = depth_array / np.max(depth_array)
    else:
        norm_depth_array = depth_array * 0

    for i in norm_depth_array:
        for j in i: 
            if (j > 1 or j < 0):
                print(j)

    # convert to torch tensor and add batch dimension (1 channel)
    depth_tensor = torch.from_numpy(norm_depth_array).unsqueeze(0)

    return depth_tensor



#https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9
def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("image_topic", help="Image topic.")
    parser.add_argument("data_type", help= "normal = divide 80/20 train/test, train = just training data, test = just testing data")

    args = parser.parse_args()

    bag = Bag(args.bag_file, "r")
    bridge = CvBridge()
    tensor_list = []
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        tensor = ros_depth_image_to_torch(msg, bridge)
        tensor_list.append(tensor)

    if (args.data_type == "normal"):

        #split data, 80% training
        train_end_ind = round(len(tensor_list)*0.8)
    
        train_list = tensor_list[0:train_end_ind]
        depth_image_dataset_train = DepthImageDataset(train_list)

        print("Length of training set: ", len(train_list))
        with open('./data/input/train_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_train, f)
    
        #20% testing
        test_list = tensor_list[train_end_ind:]
        depth_image_dataset_test = DepthImageDataset(test_list)
        print("Length of test set: ", len(test_list))
        with open('./data/input/test_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_test, f)
    
    elif (args.data_type == "train"):
        depth_image_dataset_train = DepthImageDataset(tensor_list)
        print("Length of training set: ", len(tensor_list))
        with open('./data/input/train_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_train, f)

    elif (args.data_type == "test"):
        depth_image_dataset_test = DepthImageDataset(tensor_list)
        print("Length of test set: ", len(tensor_list))
        with open('./data/input/test_dataset.pkl', 'wb') as f:
            pickle.dump(depth_image_dataset_test, f)

    # Display image 
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
  
    # train_features = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(len(train_features))
    # img = train_features[0].squeeze()

    # plt.imshow(img, cmap="gray")
    # plt.show()
    # img = train_features[1].squeeze()

    # plt.imshow(img, cmap="gray")
    # plt.show()
    


    """

if __name__ == '__main__':
    main()