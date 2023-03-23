#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
import numpy as np
import torch
import rospy
from rosbag import Bag
from sensor_msgs.msg import Image
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from depth_image_dataset import DepthImageDataset


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
    #using max float value for np.float16
    norm_depth_array = depth_array / np.max(depth_array)

    # convert to torch tensor and add batch dimension (1 channel)
    depth_tensor = torch.from_numpy(norm_depth_array).unsqueeze(0)

    return depth_tensor



"""
def tensor_to_csv(tensor, file_path):
    # convert tensor to numpy array
    array = tensor.numpy().reshape(tensor.shape[1], tensor.shape[2])

    # save array to CSV file
    np.savetxt(file_path, array, delimiter=",")
"""

#https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9
def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    bag = Bag(args.bag_file, "r")
    bridge = CvBridge()
    tensor_list = []
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        tensor = ros_depth_image_to_torch(msg, bridge)
        tensor_list.append(tensor)

    """
    #print output to check
    # assuming depth_tensors is a list of depth tensors
    for i, depth_tensor in enumerate(tensor_list):
        file_path = f"depth_tensor_{i}.csv"
        tensor_to_csv(depth_tensor, '/home/taylorlv/RadarVAE/input/test.txt')
    """

    depth_image_dataset = DepthImageDataset(tensor_list)
    print(len(tensor_list))
    with open('my_dataset.pkl', 'wb') as f:
        pickle.dump(depth_image_dataset, f)
    
    # Display image 
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    train_features = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(len(train_features))
    img = train_features[0].squeeze()

    plt.imshow(img, cmap="gray")
    plt.show()
    img = train_features[1].squeeze()

    plt.imshow(img, cmap="gray")
    plt.show()
    """
if __name__ == '__main__':
    main()