#!/usr/bin/env python3

import cv2
import time
import psutil
import rospy
import numpy as np
import time
from threading import Thread
from typing import Tuple, cast
from collections import namedtuple
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from dt_class_utils import DTReminder


class VideoFilter(DTROS):
    def __init__(self, node_name):
        super(VideoFilter,self).__init__(node_name=node_name,node_type=NodeType.PERCEPTION)
        self.bridge = CvBridge()

        self.sub_img= rospy.Subscriber(
            "/perry/camera_node/image/compressed", CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )

        self.pub_img = rospy.Publisher(
            "~debug/videoFilter/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.pub_img2 = rospy.Publisher(
            "~debug/videoFilter2/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )


    def image_cb(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return
        
        frame_out_circle = self.bridge.cv2_to_compressed_imgmsg(cv2.circle(img, (50,50), 20, (255,0,0), 2))
        frame_out_circle2 = self.bridge.cv2_to_compressed_imgmsg(cv2.circle(img, (100,100), 20, (255,255,0), 10))
        self.pub_img.publish(frame_out_circle)
        self.pub_img2.publish(frame_out_circle2)

if __name__ == '__main__':
    node = VideoFilter('VideoFilterNode')
    rospy.spin()

