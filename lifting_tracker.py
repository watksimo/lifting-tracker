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

class ObjectTracker:
	def __init__(self, lower_color, upper_color, video_file = None, buffer = 64):

		self.TENNIS_BALL_WIDTH = 32.85
		self.lower_color = lower_color
		self.upper_color = upper_color

		self.video_file = video_file
		self.buffer = buffer

		self.video_start_time = -1

		self.vs = None

		self.center_pts = []
		self.angular_change = []
		self.frame_time = []
		self.distance_change = []
		self.speed = []


	def init_video(self):
		# if a video path was not supplied, grab the reference
		# to the webcam
		if not self.video_file:
			self.vs = VideoStream(src=0).start()
		# otherwise, grab a reference to the video file
		else:
			self.vs = cv2.VideoCapture(self.video_file)

		self.video_start_time = int(round(time.time() * 1000))

	def create_color_mask(self, original_frame):
		# resize the frame, blur it, and convert it to the HSV
		# color space
		blurred = cv2.GaussianBlur(original_frame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		# construct a mask for the color "green", then perform
		# a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		return mask

	def get_largest_contour(self, original_frame):
		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(original_frame.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		center = None

		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			
			return {
				'enclosing_circle': {'x': int(x), 'y': int(y), 'radius': int(radius)},
				'center': center
			}
		else:
			None

	def draw_circles(self, frame, contour_info):
		enclosing_circle = contour_info['enclosing_circle']
		if enclosing_circle['radius'] > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (enclosing_circle['x'], enclosing_circle['y']), enclosing_circle['radius'],
				(80, 255, 255), 2)
			cv2.circle(frame, contour_info['center'], 5, (80, 255, 255), -1)
			return True
		else:
			return False

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

	def end_video(self):
		# if we are not using a video file, stop the camera video stream
		if not self.video_file:
			self.vs.stop()
		# otherwise, release the camera
		else:
			self.vs.release()

	def get_mm_per_pixel(self, radius, known_width):
		return known_width / radius


	def process_video(self):
		if self.vs is None:
			print("ERROR: No video stream initialized.");
			exit(1)

		# keep looping
		while True:
			# grab the current frame
			frame = self.vs.read()
			frame_time = int(round(time.time() * 1000))
			# handle the frame from VideoCapture or VideoStream
			frame = frame[1] if self.video_file else frame
			# if we are viewing a video and we did not grab a frame,
			# then we have reached the end of the video
			if frame is None:
				break

			frame = imutils.resize(frame, width=1000)
			masked_frame = self.create_color_mask(frame)

			detection_contour = self.get_largest_contour(masked_frame)

			if detection_contour:	# No object detected
				draw_circles = self.draw_circles(frame, detection_contour)
				detection_center = detection_contour['center']

				self.center_pts.append(detection_center)
				self.frame_time.append(frame_time - self.video_start_time)

				# Calculations need to be done after above inserts
				if len(self.center_pts) > 1:
					angle_change = self.calculate_angle_change()
					self.angular_change.append(angle_change)

					mm_per_pixel = self.get_mm_per_pixel(detection_contour['enclosing_circle']['radius'], self.TENNIS_BALL_WIDTH)
					self.distance_change.append(self.calculate_distance(mm_per_pixel))

					self.speed.append(self.calculate_speed())
			
			# Show the frame
			cv2.imshow("mask", frame)

			# if the 'q' key is pressed, stop the loop
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

		self.end_video()

		# close all windows
		cv2.destroyAllWindows()

	def start_tracking(self):
		self.init_video()

		# allow the camera or video file to warm up
		time.sleep(2.0)

		self.process_video()


def create_arg_parser():
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help="path to the (optional) video file")
	ap.add_argument("-b", "--buffer", type=int, default=64,
		help="max buffer size")
	return vars(ap.parse_args())

if __name__ == "__main__":
	# construct the argument parse and parse the arguments
	args = create_arg_parser();

	tracker = ObjectTracker((20, 80, 80), (80, 255, 255))

	tracker.start_tracking()