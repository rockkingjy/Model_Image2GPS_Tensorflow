# Datasets

 * Download the Cambridge Landmarks King's College dataset [from here.](https://www.repository.cam.ac.uk/handle/1810/251342)

 * The starting and trained weights (posenet.npy and PoseNet.ckpt respectively) for training were obtained by converting caffemodel weights [from here](http://3dvision.princeton.edu/pvt/GoogLeNet/Places/) and then training.

 * Change the path of dataset in utils.py; change the path of ckpt in run.py
 
 * If wanna test: change the main function in run.py to test()

 * If wanna train: change the main function in run.py to train()
