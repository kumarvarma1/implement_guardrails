import rospy
from std_msgs.msg import String, Int32,Bool,Float32,Float32MultiArray
from sensor_msgs.msg import Image,CompressedImage
from geometry_msgs.msg import TwistWithCovarianceStamped 
import cv2, queue, threading, time
import numpy as np
import time
from cv_bridge import CvBridge
#from numba import cuda
import statistics
import math
import os,signal
import sys
import vpi
import time

class implement_detection:

	def __init__(self):
		self.img_beat = False
		self.rate = rospy.Rate(10)
		self.kernel = np.ones((3,3),np.uint8)
		self.background_sub = False
		self.yuv_threshold = 0.5
		self.gray_threshold = 0.5
		self.follow = False
		self.optical_init = True
		self.vinerow = False
		self.wheel_speed= 0
		self.viz = False
		self.bridge = CvBridge()
		self.implement_hist = 1
		rospy.Subscriber('/planning_manager/follow_me_activation', Bool, self.follow_activation)
		rospy.Subscriber('/planning_manager/vinerow_activation', Bool, self.vinerow_check)
		rospy.Subscriber('/camera/pto/image/compressed',CompressedImage,self.implement_cam)
		rospy.Subscriber('/Localization/wheel_odometry',TwistWithCovarianceStamped,self.localization_callback)
		self.img_publisher = rospy.Publisher('/implement_awareness/debug_image',Image,queue_size = 1)
		self.implement_publisher = rospy.Publisher('/implement_awareness/lock',Int32,queue_size = 1)

	def localization_callback(self,msg):
		self.wheel_speed = msg.twist.twist.linear.x
		self.wheel_speed = self.wheel_speed*3.6

	def follow_activation(self,msg):
		self.follow = msg.data

	def vinerow_check(self,msg):
		self.vinerow = msg.data

	def implement_cam(self,data):
		img_np = np.fromstring(data.data, np.uint8)
		self.hitch_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
		self.img_beat = True

	def erode(self,img):

#	''' Morphology erosion operation '''

		kernel = np.ones((7,7),np.uint8)
		erosion = cv2.erode(img,kernel,iterations = 1)
		return erosion

	def soiling_gray(self,img_list):

#	''' Function to preprocess images in gray scale for soiling detection '''

		gray_diff_images = []
		i = 1
		while i<22:
			a = random.randint(0,9)
			b = random.randint(0,9)
			if a!=b:
				i+=1
				result = self.pixel_diff_gray(img_list[a],img_list[b],i)
				gray_diff_images.append(result)
		return gray_diff_images

	def soiling_yuv(self,img_list):

#	''' Function to preprocess images in YUV scale for soiling detection '''
	
		yuv_diff_images = []
		i = 31
		while i<52:
			a = random.randint(0,9)
			b = random.randint(0,9)
			if a != b:
				i+=1
				result = self.pixel_diff_yuv(img_list[a],img_list[b],i)
				result = self.dilate(result)
				yuv_diff_images.append(result)
		return yuv_diff_images

	def background(self,imageset):
		backSub = cv2.createBackgroundSubtractorKNN(history=10,detectShadows=False,dist2Threshold = 10.0)
		for i in imageset:
			fgmask = backSub.apply(i)
		ret,mask = cv2.threshold(fgmask,254,255, 1)
		img = imageset[-1]
		img[mask == 0] = [255,255,255]
		img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret,mask = cv2.threshold(img1_gray,200,255, 1)
		result = self.erode(mask)
		try:
			contours, hier = cv2.findContours(result,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
			c = max(contours, key = cv2.contourArea)
			x,y,w,h = cv2.boundingRect(c)
		except:
			x,y,w,h = 0,0,0,0
		coords = [x,y,w,h]
		area = w*h
		return fgmask,area,img1_gray 

	def transform(self,mv):

#		''' Convert optical flow pixel wise data to hsv image '''

		with mv.rlock():
			flow = np.float32(mv.cpu())/(1<<5)  
		magnitude, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
		clip = 5.0
		cv2.threshold(magnitude, clip, clip, cv2.THRESH_TRUNC, magnitude)
		gray = np.ndarray([flow.shape[0], flow.shape[1]], np.float32)
		gray[:,:] = magnitude/clip
		gray = np.uint8(gray*255)
		gray_out = cv2.threshold(gray,200,254,0)[1]
		gray_out = 255 - gray_out
		return gray_out

	def main(self):
		backSub = cv2.createBackgroundSubtractorKNN(history=10,detectShadows=False,dist2Threshold = 100.0)
		backend = vpi.Backend.NVENC
		quality = vpi.OptFlowQuality.HIGH
		count = 0
		final = np.zeros((270,480),np.float32)
		self.frame_timer = time.time()
		next_frame = False
		time.sleep(0.5)
		while 1:

			if self.vinerow:
				self.rate.sleep()
				continue
			if self.follow:
				self.rate.sleep()
				continue

			if abs(self.wheel_speed) > 1:
				#print('have wheel')
				#print('img_beat is',self.img_beat)
				if self.img_beat:
					curFrame = vpi.asimage(self.hitch_image, vpi.Format.BGR8).convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA).convert(vpi.Format.NV12_ER_BL, backend=vpi.Backend.VIC)
					if time.time() - self.frame_timer > 1:
						next_frame = False
						#final = np.zeros((270,480),np.float32)
					self.img_beat = False
					#print('next frame is',next_frame)
					#print('doing optical')
					count +=1
					#print('count is',count)
					if next_frame:
						t1 = time.time()
						with backend:
							motion_vector = vpi.optflow_dense(prevFrame, curFrame, quality = quality)
							postprocessed_img = self.transform(motion_vector)
						#print('time taken is',time.time()-t1)

					if next_frame:
						img_show = cv2.resize(self.hitch_image,(1920//4,1080//4))
						if self.optical_init:
							#print('processing final')
							final = final + np.uint8(postprocessed_img/255)
							self.implement_publisher.publish(data=self.implement_hist)
						if self.viz:
							if count> 10:
								result = cv2.threshold(final,int(count/2.5),count,0)[1]
								cv2.imshow('frame4',result)
								cv2.imshow('frame5',img_show)
								if cv2.waitKey(25) & 0xFF == ord('q'):
									break

						if count > 250:
							count = 0
							final = cv2.threshold(final,100,255,0)[1]
							final[:,200:] = 0
							final = np.uint8(final)
							final = self.erode(final)
							try:
								contours,hier = cv2.findContours(final,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
								c = max(contours,key=cv2.contourArea)
								x,y,w,h = cv2.boundingRect(c)
								coords = [x,y,w,h]
								x,y,w,h = [i*4 for i in coords]
								if w*h > 8000:
									self.implement_publisher.publish(data=2)
									self.implement_hist = 2
									img = cv2.putText(self.hitch_image,'implement found',(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4,cv2.LINE_AA)
									img = cv2.rectangle(self.hitch_image,(x,y),(x+w,y+h),(0,0,255),2)
								else:
									self.implement_publisher.publish(data=1)
									self.implement_hist = 1
									img = cv2.putText(self.hitch_image,'No implement',(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4,cv2.LINE_AA)
								if self.viz:
									cv2.imshow('frame3',img)
									cv2.waitKey(0)
									cv2.destroyAllWindows()
							except:
								img = cv2.putText(self.hitch_image,'No implement',(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4,cv2.LINE_AA)
								self.implement_publisher.publish(data=1)
								self.implement_hist = 1
								if self.viz:
									cv2.imshow('frame3',img)
									cv2.waitKey(0)
									cv2.destroyAllWindows()
							self.publish_alive = True
							image_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
							self.img_publisher.publish(image_msg)				
							final = np.zeros((270,480),np.float32)

					prevFrame = curFrame
					self.frame_timer = time.time()
					next_frame = True
				if self.background_sub:
					fgMask = backsub.apply(self.hitch_image)
				self.rate.sleep()

			else:
				self.implement_publisher.publish(data=0)
				self.rate.sleep()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	rospy.init_node('implement_awareness',anonymous=False)
	implement_detection().main()
