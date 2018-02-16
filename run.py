
import utils 
import numpy as np
import random
import tensorflow as tf
from posenet import GoogLeNet as PoseNet
import cv2
from tqdm import tqdm
import math

path_ckpt = '/home/enroutelab/Amy/caffe/examples/img2gps/tensorflow_posenet/PoseNet.ckpt'

def test():
	image = tf.placeholder(tf.float32, [1, 224, 224, 3])
	datasource = utils.get_data("test")
	results = np.zeros((len(datasource.images),2))

	net = PoseNet({'data': image})

	p3_x = net.layers['cls3_fc_pose_xyz']
	p3_q = net.layers['cls3_fc_pose_wpqr']

	init = tf.initialize_all_variables()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# Load the data
		sess.run(init)
		saver.restore(sess, path_ckpt)

		for i in range(len(datasource.images)):
			np_image = datasource.images[i]
			feed = {image: np_image}
			predicted_x, predicted_q = sess.run([p3_x, p3_q], feed_dict=feed)
			predicted_x = np.squeeze(predicted_x)
			predicted_q = np.squeeze(predicted_q)

			pose_x= np.asarray(datasource.poses[i][0:3])
			pose_q= np.asarray(datasource.poses[i][3:7])			
			pose_x = np.squeeze(pose_x)
			pose_q = np.squeeze(pose_q)

			#Compute Individual Sample Error
			q1 = pose_q / np.linalg.norm(pose_q)
			q2 = predicted_q / np.linalg.norm(predicted_q)
			d = abs(np.sum(np.multiply(q1, q2)))
			theta = 2 * np.arccos(d) * 180 / math.pi
			error_x = np.linalg.norm(pose_x - predicted_x)
			results[i,:] = [error_x, theta]
			print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  \
					Error Q (degrees):  ', theta
	#Compute Median Sample Error
	median_result = np.median(results, axis=0)
	print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'

def train():
	batch_size = 75
	max_iterations = 3000

	images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
	poses_x = tf.placeholder(tf.float32, [batch_size, 3])
	poses_q = tf.placeholder(tf.float32, [batch_size, 4])
	datasource = utils.get_data("train")

	net = PoseNet({'data': images})

	p1_x = net.layers['cls1_fc_pose_xyz']
	p1_q = net.layers['cls1_fc_pose_wpqr']
	p2_x = net.layers['cls2_fc_pose_xyz']
	p2_q = net.layers['cls2_fc_pose_wpqr']
	p3_x = net.layers['cls3_fc_pose_xyz']
	p3_q = net.layers['cls3_fc_pose_wpqr']

	l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 0.3
	l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
	l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 0.3
	l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
	l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 1
	l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * 500

	loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q
	opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, 
			epsilon=0.00000001, use_locking=False, name='Adam').minimize(loss)
	
	# ---- create a summary to monitor cost tensor
	tf.summary.scalar("loss", loss)
	merged_summary_op = tf.summary.merge_all() # merge all summaries into a single op
	logs_path = './logs' # op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	print("Run the command line: --> tensorboard --logdir=./logs " \
			"\nThen open http://0.0.0.0:6006/ into your web browser")

	# ---- Set GPU options
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		# Load the data
		sess.run(init)
		net.load('posenet.npy', sess)

		data_gen = utils.gen_data_batch(datasource, batch_size)
		for i in range(max_iterations):
			np_images, np_poses_x, np_poses_q = next(data_gen)
			feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}

			sess.run(opt, feed_dict=feed) # run the optimizer
			np_loss = sess.run(loss, feed_dict=feed) #get the loss

			# ---- print the logs
			if i % 20 == 0:
				print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(np_loss))
			if i % 100 == 0:
				saver.save(sess, path_ckpt)
				print("Intermediate file saved at: " + path_ckpt)

			# ---- write logs at every iteration
			summary = merged_summary_op.eval(feed_dict=feed)
			summary_writer.add_summary(summary, i)
		
		saver.save(sess, path_ckpt)
		print("Intermediate file saved at: " + path_ckpt)

if __name__ == '__main__':
	train()