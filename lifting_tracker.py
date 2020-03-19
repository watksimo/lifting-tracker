# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video",
# 	help="path to the (optional) video file")
# ap.add_argument("-b", "--buffer", type=int, default=64,
# 	help="max buffer size")
# args = vars(ap.parse_args())
# # define the lower and upper boundaries of the "green"
# # ball in the HSV color space, then initialize the
# # list of tracked points
# greenLower = (20, 80, 80)
# greenUpper = (80, 255, 255)
# pts = deque(maxlen=args["buffer"])
# # if a video path was not supplied, grab the reference
# # to the webcam
# if not args.get("video", False):
# 	vs = VideoStream(src=0).start()
# # otherwise, grab a reference to the video file
# else:
# 	vs = cv2.VideoCapture(args["video"])
# # allow the camera or video file to warm up
# time.sleep(2.0)

# # keep looping
# while True:
# 	# grab the current frame
# 	frame = vs.read()
# 	# handle the frame from VideoCapture or VideoStream
# 	frame = frame[1] if args.get("video", False) else frame
# 	# if we are viewing a video and we did not grab a frame,
# 	# then we have reached the end of the video
# 	if frame is None:
# 		break
# 	# resize the frame, blur it, and convert it to the HSV
# 	# color space
# 	frame = imutils.resize(frame, width=600)
# 	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
# 	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
# 	# construct a mask for the color "green", then perform
# 	# a series of dilations and erosions to remove any small
# 	# blobs left in the mask
# 	mask = cv2.inRange(hsv, greenLower, greenUpper)
# 	mask = cv2.erode(mask, None, iterations=2)
# 	mask = cv2.dilate(mask, None, iterations=2)

	# cv2.imshow("mask", mask)

	# # find contours in the mask and initialize the current
	# # (x, y) center of the ball
	# cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)
	# center = None
	# # only proceed if at least one contour was found
	# if len(cnts) > 0:
	# 	# find the largest contour in the mask, then use
	# 	# it to compute the minimum enclosing circle and
	# 	# centroid
	# 	c = max(cnts, key=cv2.contourArea)
	# 	((x, y), radius) = cv2.minEnclosingCircle(c)
	# 	M = cv2.moments(c)
	# 	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	# 	# only proceed if the radius meets a minimum size
	# 	if radius > 10:
	# 		# draw the circle and centroid on the frame,
	# 		# then update the list of tracked points
	# 		cv2.circle(frame, (int(x), int(y)), int(radius),
	# 			(0, 255, 255), 2)
	# 		cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# # update the points queue
	# pts.appendleft(center)

	# # loop over the set of tracked points
	# for i in range(1, len(pts)):
	# 	# if either of the tracked points are None, ignore
	# 	# them
	# 	if pts[i - 1] is None or pts[i] is None:
	# 		continue
	# 	# otherwise, compute the thickness of the line and
	# 	# draw the connecting lines
	# 	thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
	# 	cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	# # show the frame to our screen
	# cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF
# 	# if the 'q' key is pressed, stop the loop
# 	if key == ord("q"):
# 		break
# # if we are not using a video file, stop the camera video stream
# if not args.get("video", False):
# 	vs.stop()
# # otherwise, release the camera
# else:
# 	vs.release()
# print(pts);
# # close all windows
# cv2.destroyAllWindows()

class ObjectTracker:
	def __init__(self, lower_color, upper_color, video_file = None, buffer = 64):
		self.lower_color = lower_color
		self.upper_color = upper_color

		self.video_file = video_file
		self.buffer = buffer

		self.pts = deque(maxlen=self.buffer)
		self.vs = None

		self.angular_change = []

	def init_video(self):
		# if a video path was not supplied, grab the reference
		# to the webcam
		if not self.video_file:
			self.vs = VideoStream(src=0).start()
		# otherwise, grab a reference to the video file
		else:
			self.vs = cv2.VideoCapture(self.video_file)

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

	def process_video(self):
		if self.vs is None:
			print("ERROR: No video stream initialized.");
			exit(1)

		# keep looping
		while True:
			# grab the current frame
			frame = self.vs.read()
			# handle the frame from VideoCapture or VideoStream
			frame = frame[1] if self.video_file else frame
			# if we are viewing a video and we did not grab a frame,
			# then we have reached the end of the video
			if frame is None:
				break

			frame = imutils.resize(frame, width=600)
			masked_frame = self.create_color_mask(frame)

			mask_contour = self.get_largest_contour(masked_frame)
			if mask_contour:
				draw_circles = self.draw_circles(frame, mask_contour)

			# self.pts.append(mask_contour)
			
			cv2.imshow("mask", frame)

			key = cv2.waitKey(1) & 0xFF
			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break

		# if we are not using a video file, stop the camera video stream
		if not self.video_file:
			self.vs.stop()
		# otherwise, release the camera
		else:
			self.vs.release()
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



