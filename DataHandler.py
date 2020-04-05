# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math
from datetime import datetime
from scipy.spatial import distance as dist

class DataHandler:
	def __init__(self):

		self.TENNIS_BALL_WIDTH = 32.85
		self.video_start_time = -1

		self.center_pts = []
		self.angular_change = []
		self.frame_time = []
		self.distance_change = []
		self.speed = []

	def calculate_angle_change(self):
		prev_position = self.center_pts[-2]
		current_position = self.center_pts[-1]
		angle_radians = math.atan2(prev_position[1]-current_position[1], prev_position[0]-current_position[0])
		angle_degrees = int(math.degrees(angle_radians))
		return angle_degrees

	def calculate_distance(self, mm_per_pixel):
		prev_position = self.center_pts[-2]
		current_position = self.center_pts[-1]
		euclid_pixel_distance = dist.euclidean(prev_position, current_position)
		euclid_mm_distance = euclid_pixel_distance * mm_per_pixel
		return euclid_mm_distance

	def calculate_speed(self):
		time_delta = self.frame_time[-1] - self.frame_time[-2]
		return self.distance_change[-1] / time_delta

	def get_mm_per_pixel(self, radius, known_width):
		return known_width / radius