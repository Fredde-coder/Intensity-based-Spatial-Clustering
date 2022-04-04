from xmlrpc.client import boolean
from sFCM.sFCM import *
from deepClustering.DFC import DFC 
from utils import *
from experiment_util import *
from os import path
import os, tqdm, cv2, argparse

directory = path.dirname(path.abspath(path.abspath(__file__)))

data_storage_path_trials = path.join(directory, 'data_storage', 'trials')+ '/'
data_storage_path_images = path.join(directory, 'data_storage', 'images')+ '/'
data_storage_path_compiled = path.join(directory, 'data_storage', 'compiled')+ '/'
experiments_storage_path_images = path.join(directory, 'Experiments_data', 'Images') + '/'
experiments_storage_path_label_prob = path.join(directory, 'Experiments_data', 'ground_truths') + '/'
default_im_format = "jpeg"

parser = argparse.ArgumentParser(description='Perform clustering')

parser.add_argument('--name', type=str, help='Name of experiment', default='demo')
parser.add_argument('--cropping', nargs='+', default=['100', '100', '100', '100'], help = 'Cropping paramters, e.g., --cropping <top> <bot> <left> <right>')
parser.add_argument('--model', type=str, default='DFC', help='Type of model (DFC or sFCM). Note that this script runs with demo hyper-parameters.')
parser.add_argument('--dfc-args',nargs='+', default=[9,100,50,3], help='hyperparameters for the DFC model look at line 39 for a description of the different arguments')
parser.add_argument('--sfcm-args',nargs='+', default=[2, 9, 1, 1, 3], help='hyperparameters for the sFCM model, description can be found in sFCM folder in sFCM.py file')
parser.add_argument('--show-convergence', type=bool, default=True, help='Shows the changing clustering during modelling')
parser.add_argument('--show-afterimage', type=bool, default=True, help='Shows the image after completed clustering')


args = parser.parse_args()
if args.dfc_args[0] is not int:
    for i,arg in enumerate(args.dfc_args):
        args.dfc_args[i] = int(arg)
elif args.sfcm_args[0] is not int:
    for i,arg in enumerate(args.sfcm_args):
        args.sfcm_args[i] = int(arg)

experiments_name = args.name #"synthetic_brain"
images = load_imgdir(experiments_storage_path_images,"png")
cropping = {"top":int(args.cropping[0]),"bot":int(args.cropping[1]),"left":int(args.cropping[2]),"right":int(args.cropping[3])}#{"top":100,"bot":100,"left":100,"right":100}
cropped_image = simple_cropping(images[0],  cropp_args=cropping)

if args.model.lower() == 'dfc':
    model = DFC(minLabels=args.dfc_args[0], max_iters=args.dfc_args[1], nChannel=args.dfc_args[2], nConv=args.dfc_args[3])
    model.initialize_clustering(cropped_image)
elif args.model.lower() == 'sfcm':
    model = sFCM(*args.sfcm_args, cropped_image.shape)
    model.maxIters = model.MAX_ITER



labels_dict = load_experiments_data(experiments_storage_path_label_prob, "npy",item=True)
labels_names = [[key for key in sample] for sample in labels_dict]
labels_probs = [[sample[key] for key in sample] for sample in labels_dict]
sample_labels = [[np.random.binomial(1, x) for x in label_p] for label_p in labels_probs]


metric_stats = load_run(model,sample_labels, labels_names,preloaded_images=images,cropping=True, n_iter=model.maxIters, cropp_args=cropping,paths=False, n_trials=1, show_convergence=args.show_convergence,show_image=args.show_afterimage, save_trials=False,save_stats=False, verbose=True)