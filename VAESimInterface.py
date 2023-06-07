from VAE import *
import rospy
from sensor_msgs.msg import Image
import torch
import time
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from data_parser import ros_depth_image_to_torch


device = "cpu"
IMAGE_TOPIC = "/output"
LATENT_SPACE_TOPIC_NAME = "/latent_space"
IMAGE_HEIGHT = 8
IMAGE_WIDTH = 8


#decode_img_publisher = rospy.Publisher('/decoded_image', Image, queue_size=1)
        # Subscribe to image
#image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=1)




x = np.zeros((8,8), dtype=np.float32)
x[3:6, 3:6] = 1.0

# x_torch = torch.from_numpy(x).to(device)

# recon, *_ = model(x_torch)

# print(x_torch)

# print("------")

# print(recon)



class VAESimInterface():
    def __init__(self):
        self.model = VAE(image_height=8, image_width=8, latent_size=20, hidden_size=212, beta=0.0001).to(device)

        dict = torch.load("/home/taylorlv/RadarVAE/model/vae.pth")#"/home/arl/workspaces/learning_ws/verifiable_learning/vernav/vernav/resources/weights/EVO/dragvoll_dataset_test_6_LD_128_epoch_39.pth")
        self.model.load_state_dict(dict, strict=True)

   
        #self.net_interface = VAENetworkInterface(LATENT_SPACE, "cpu")
       
        self.decode_img_publisher = rospy.Publisher('/decoded_image', Image, queue_size=1)
        # Subscribe to image
        self.image_sub = rospy.Subscriber(
            IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
     
        # Publish latent space as Float32MultiArray
        self.latent_space_publisher = rospy.Publisher(LATENT_SPACE_TOPIC_NAME, Float32MultiArray, queue_size=2)
        print("Publishing latent space to {}", LATENT_SPACE_TOPIC_NAME)

        
    def image_callback(self, data):
        bridge = CvBridge()
        valid, input_arr = ros_depth_image_to_torch(data, bridge)
    
        recon_data, z, means, log_var = self.model(input_arr)
        
        print(recon_data)
        img_filtered_uint8 = ((recon_data[0].detach().numpy()) * 255).astype(np.uint8)
        print("img_filtered_uint8", img_filtered_uint8)
        msg_filtered = Image()
        msg_filtered.height = IMAGE_HEIGHT # hardcoded and not data.height
        msg_filtered.width = IMAGE_WIDTH # hardcoded and not data.width
        msg_filtered.encoding = "8UC1"
        msg_filtered.is_bigendian = 0
        msg_filtered.step = IMAGE_WIDTH # 1 byte for each pixel
        msg_filtered.data = np.reshape(img_filtered_uint8, (IMAGE_WIDTH*IMAGE_HEIGHT,)).tolist()
        self.decode_img_publisher.publish(msg_filtered)


        latent_space_msg = Float32MultiArray()
        latent_space_msg.data = means.flatten().tolist()
        self.latent_space_publisher.publish(latent_space_msg)


        # print("input_image characteristics", input_image.shape, IMAGE_MAX_DEPTH* np.max(input_image), IMAGE_MAX_DEPTH*np.min(input_image), IMAGE_MAX_DEPTH*np.mean(input_image))
        # print("reconstruction characteristics", reconstruction.shape, IMAGE_MAX_DEPTH* np.max(reconstruction), IMAGE_MAX_DEPTH*np.min(reconstruction), IMAGE_MAX_DEPTH*np.mean(reconstruction))
        
        # if args.show_cv:
        #     # Display images in a CV window
        #     cv2.imshow("Image window", np_image)
        #     cv2.waitKey(3)
        #     if CALCULATE_RECONSTRUCTION:
        #         cv2.imshow("Reconstruction window", reconstruction)
        #         cv2.waitKey(3)
        # # publish filtered image
        #print('Compute time:', compute_time)
        print("yay")


if __name__ == "__main__":
    rospy.init_node("vae_interface")
    print("Node Initialized. Loading weights.")
    nav_interface = VAESimInterface()
    # nav_interface = VAESimInterface()
    # print("Loaded weights. Lets gooooooooooo.......")
    rospy.spin()
