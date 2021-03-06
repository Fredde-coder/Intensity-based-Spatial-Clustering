import brainweb
from brainweb import volshow
import numpy as np
from os import path
from tqdm.auto import tqdm
import logging
import random
import cv2
from imutils import build_montages

class brainweb_images:
    def __init__(self):
        self.files = None

    def fetch(self):
        self.files = brainweb.get_files()

    
    def sample_mMR_dataset(self, parameters, nr_samples, seed = None):
        '''
            Samples samples, given a parameter setting
            parameters : a map with values petNoise, t1Noise, t2Noise, petSigma, t1Sigma, t2Sigma, PetClass
            nr_samples : number of samples
            seed       : seed for random number gen

            returns four collections, each of length nr_samples: pet, uMap, T1, T2
        '''
        if self.files == None: self.fetch()
        brainweb.seed(seed)
        np.random.seed(seed)

        #Compute images
        pet, uMap, T1, T2, ground_truth = [],[],[],[],[]

        #Sampling procedure
        for f in tqdm(random.sample(self.files, nr_samples), desc='mMR ground truths', unit='subject'):
            vol = brainweb.get_mmr_fromfile(f, **parameters) #volumes
            gt  = brainweb.get_label_probabilities(f, labels=brainweb.Act.all_labels) #ground truths
            gt_map = {}
            for i, label in enumerate(brainweb.Act.all_labels):
                gt_map[label] = gt[i,...]

            ground_truth.append(gt_map)
            pet.append(vol['PET'])
            uMap.append(vol['uMap'])
            T1.append(vol['T1'])
            T2.append(vol['T2'])
                
        return pet, uMap, T1, T2, ground_truth

    def _color_mmr_slice(self, mmr_slice, cmap = None):
        if cmap == None: return mmr_slice

        return cv2.applyColorMap(np.uint8(mmr_slice), cmap)

    def color_mmr_slice(self, mmr_slice, cmap = cv2.COLORMAP_HOT):
        assert len(mmr_slice.shape) == 2
        
        return self._color_mmr_slice(mmr_slice, cmap)
        

    def sample_as_montage(self, mmr_volume, montage_size = 1000, cmap = cv2.COLORMAP_HOT):
        assert len(mmr_volume.shape) == 3

        nr_images, height, width = mmr_volume.shape
        montage = [self._color_mmr_slice(mmr_volume[i, :, :], cmap) for i in range(nr_images)]
        
        order = int(nr_images**0.5)
        side  = montage_size//order


        return build_montages(montage, (side, side), (order, order))

        

if __name__ == "__main__":
    
    s = brainweb_images()
    s.fetch()
    parameters = {'petNoise' : 1, 
                  't1Noise' : 0.75, 
                  't2Noise' : 0.75,
                  'petSigma' : 1, 
                  't1Sigma' : 1, 
                  't2Sigma' : 1,
                  'PetClass' : brainweb.FDG}
    nr_images_T1 = 2
    nr_images_T2 = 2
    lower_bound = 60
    upper_bound = 80


    pet, uMap, T1, T2, ground_truth = s.sample_mMR_dataset(parameters, nr_images_T1+nr_images_T2, 1337)
    data_set = T1
    t_set = "T1"
    for i in range(nr_images_T1+nr_images_T2):
        if i>nr_images_T1-1:
            data_set = T2
            t_set="T2"

        for j in range(pet[0].shape[0]):
            cv2.imwrite("Project-Course-in-Data-Science/Experiments_data/Images/Volume/{}sample{}_slice{}.png".format(t_set,i,j),data_set[i][j])
        np.save("Project-Course-in-Data-Science/Experiments_data/ground_truths/Volume/{}sample{}_truth_probs.npy".format(t_set,i),ground_truth[i])

        
        brain_slice = np.random.randint(lower_bound,upper_bound)
        cv2.imwrite("Project-Course-in-Data-Science/Experiments_data/Images/{}sample{}_slice{}.png".format(t_set,i,brain_slice),data_set[i][brain_slice])
        brain_slice_dict = {key:ground_truth[i][key][brain_slice] for key in ground_truth[i]}
        np.save("Project-Course-in-Data-Science/Experiments_data/ground_truths/{}sample{}_truth_probs.npy".format(t_set,i),brain_slice_dict)


    s.show_sample_as_montage(T1[0])
