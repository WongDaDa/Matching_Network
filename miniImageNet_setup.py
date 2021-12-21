import pickle
import tqdm
import pandas as pd
import numpy as np

class MiniImageNetNShotDataset():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2021,use_cache=True):
        np.random.seed(seed)
        self.nClasses = 100
        self.train = self.retrieve_data('mini-imagenet-cache-train.pkl')
        self.test = self.retrieve_data('mini-imagenet-cache-test.pkl')
        self.val = self.retrieve_data('mini-imagenet-cache-val.pkl')
        self.x_train = self.processes_batch(self.train, np.mean(self.train), np.std(self.train))
        self.x_test = self.processes_batch(self.test, np.mean(self.test), np.std(self.test))
        self.x_val = self.processes_batch(self.val, np.mean(self.val), np.std(self.val))
        self.x = np.vstack([self.train,self.test,self.val])
        print(self.x.shape)

        self.batch_size = batch_size
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
            
    def processes_batch(self, x_batch, mean, std):
        return (x_batch - mean) / std
    
    def retrieve_data(self, file):
        pkl_file = open(file, 'rb')
        miniImageData  = pickle.load(pkl_file)
        pkl_file.close()
        
        data = []
        for key in list(miniImageData['class_dict'].keys()):
            image_index_list = miniImageData['class_dict'][key]
            data.append(miniImageData['image_data'][image_index_list])
            
        return np.array(data)
    
    def _sample_new_batch(self, data_pack):
        support_set_x = []
        support_set_y = []
        target_x = []
        target_y = []

        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)
            choose_sample = np.random.choice(samples_idx, size=self.samples_per_class, replace=False)

            x_temp = data_pack[choose_classes, choose_sample]
            y_temp = np.arange(self.classes_per_set)
            support_set_x.append(x_temp)
            support_set_y.append(np.expand_dims(y_temp[:], axis=1))
            target_x.append(data_pack[choose_classes[choose_label], np.random.randint(600)])
            target_y.append(y_temp[choose_label])

        return np.array(support_set_x).reshape((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],data_pack.shape[3], data_pack.shape[4])),np.array(support_set_y).reshape((self.batch_size, self.classes_per_set, self.samples_per_class)), np.array(target_x).reshape((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4])), np.array(target_y).reshape((self.batch_size, 1))
    
    def _rotate_data(self, image, k):
        return np.rot90(image, k)
    
    def _rotate_batch(self, batch_images, k):
        batch_size = batch_images.shape[0]
        for i in np.arange(batch_size):
            batch_images[i] = self._rotate_data(batch_images[i], k)
        return batch_images
    
    def _get_batch(self, dataset_name, augment=False):
        support_set_x, support_set_y, target_x, target_y = self._sample_new_batch(self.datatset[dataset_name])
        
        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                               support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, target_x, target_y
    
    def get_normal_train_batch(self,batch_size):
        choose_class = np.random.randint(64,size=batch_size)
        choose_image = np.random.randint(600,size=batch_size)
        train_samples = self.x_train[choose_class,choose_image,:,:,:]
        train_labels = choose_class
            
        return np.array(train_samples).reshape(batch_size,84,84,3),np.array(train_labels)
    
    def get_train_batch(self, augment=False):
        return self._get_batch("train", augment)

    def get_val_batch(self, augment=False):
        return self._get_batch("val", augment)

    def get_test_batch(self, augment=False):
        return self._get_batch("test", augment)
