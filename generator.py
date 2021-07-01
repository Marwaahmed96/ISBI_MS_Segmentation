import numpy as np

import h5py
import tensorflow as tf

class ISBIDataset(tf.keras.utils.Sequence):
    def __init__(self, data,options,patches, phase='train'):
        self.data= data
        self.patches= patches
        self.options= options
        self.batch_size= 100
    def __len__(self):
        return self.patches//self.batch_size
    
    def __getitem__(self, batch_index=1):
        
        x_data=[]
        y_data=[]
        # needed baches
        batch_size=self.batch_size
        
        for p in self.data:
            
            patient_x_data=self.data[p]['X'][()]
            patient_y_data=self.data[p]['Y'][()]
            iter_start=0
            while iter_start< patient_x_data.shape[0]:
                if iter_start+ batch_size < patient_x_data.shape[0]:
                    iter_end=  iter_start+ batch_size
                    if len(x_data)>0:
                        x_data=np.concatenate((x_data, patient_x_data[iter_start: iter_end]),axis=0)
                        y_data=np.concatenate((y_data, patient_y_data[iter_start: iter_end]),axis=0)
                    else:
                        x_data=patient_x_data[iter_start: iter_end]
                        y_data=patient_y_data[iter_start: iter_end]
                    
                else:
                    x_data=patient_x_data[iter_start:]
                    y_data=patient_y_data[iter_start: ]
                
                x=x_data
                y=y_data
                    
                if len(x_data) != self.batch_size:
                    #nedded batches
                    batch_size= self.batch_size- len(x_data)
                    iter_start+=batch_size
                else:
                    batch_size= self.batch_size
                    iter_start+=batch_size
                    x_data=[]
                    y_data=[]
                    #handle if last batch less than batch_size
                    x=np.stack(x).astype(np.float32)
                    y=np.stack(y).astype(np.float32)
                    yield  x, y
                    
        '''
        if len(x)<self.batch_size:
            x=np.concatenate(x,x[:self.batch_size-len(x)+1])
            print('last',x.shape)
            y=np.concatenate(y,y[:self.batch_size-len(y)+1])
            yield x,y
        '''
