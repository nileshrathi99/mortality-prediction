import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import scipy
from datetime import datetime, timedelta
import pandas as pd
import tqdm
import tqdm.notebook as n_tqdm
import concurrent.futures
from .data_utils import *


class ECG_Data(Dataset):
    def __init__(self, table, x, y, z = 'VStart', modality = ['II'], dim = 1):
        super(ECG_Data, self).__init__()
        self.table = table
        self.x = x
        self.y = y 
        self.z = z
        self.modality = modality
        self.dim = dim

        class_labels = table['label'].values
        class_counts = np.bincount(class_labels)
        class_weights = 1.0 / class_counts

        # Use class weights to create the WeightedRandomSampler
        self.samples_weights = [class_weights[label] for label in class_labels]
        self.sampler = WeightedRandomSampler(weights= self.samples_weights, num_samples= len(table), replacement=True)

    def __len__(self):
        return len(self.table)

    def get_table(self):
        return self.table

    def transform(self):
        return transforms.Compose([
                # transforms.ToTensor(),  # Convert to a PyTorch tensor #problematic becuase spectograms are not in range (0, 255)
                # transforms.Lambda(lambda x: np.log(x + 1e-8)),
                transforms.Lambda(lambda x: torch.tensor((x - np.min(x))/(np.max(x) - np.min(x) + 1e-8)).unsqueeze(0))
            ])
    
    def get_sample_weights(self):
        return self.samples_weights

    def preprocess(self, sig):
        ## Median Filtering removes very low frequency
        lowpassed = scipy.signal.medfilt(sig, kernel_size= 125)
        highpassed = sig - lowpassed

        # High Frequency removal
        lowpass = scipy.signal.butter(1, 10, btype='lowpass', fs= 125, output='sos')
        lowpassed = scipy.signal.sosfilt(lowpass, highpassed)

        sig = lowpassed
        # sig = highpassed
        f, t, sig = scipy.signal.stft(sig, nperseg = 125, noverlap = 70)
        sig = np.abs(sig)
        # sig = sig[:30, :]
        sig = self.transform()(sig)
        return sig

    def __getitem__(self, idx):
        chart = self.table
        patient_id = chart.loc[idx]['SUBJECT_ID']
        # start = datetime.strptime(chart.loc[idx]['INTIME'], '%Y-%m-%d %H:%M:%S.%f %Z')
        # end = start + timedelta(hours= self.z)
        # start = end - timedelta(seconds= self.x)

        end = datetime.strptime(chart.loc[idx]['VStart'], '%Y-%m-%d %H:%M:%S.%f %Z')
        start = end - timedelta(seconds= self.x)

        data = get_waveform_data(int(patient_id), (start, end), channels= self.modality)

        # missing value aware training
        data[np.isnan(data)] = 0.0
        # missing = ~(np.isnan(data))
        # data = np.concatenate([data, missing])
        # print(f'E: {np.max(data)}')
        # data = borrow(data[0, :])
        # print(f'X: {np.max(data)}')
        if self.dim == 2:
            data = self.preprocess(data[0])
        # if torch.sum(data) == 0:
        #     print(f'{patient_id}: has all 0 data')
        label = chart.loc[idx]['label']
        return data, label
    

class Data:
    def __init__(self, table, x, y, z = 'VStart', modality = ['II'], num_workers = 64):
        super(Data, self).__init__()
        self.x = x
        self.z = z
        self.y = y
        self.modality = modality
        self.num_workers = num_workers
        self.table = pd.read_csv(table)
        self.table = create_snapshot(self.table, 0, self.y, 'DEATH_MV')
        self.selected_table = None
    
    def get_table(self):
        return self.table
    
    def get_selected_table(self):
        return self.selected_table
    
    def select_patients(self, threshold = 0.5):
        self.selected_table = self.select_patients_fast(self.table, threshold)

    def select_patients_fast(self, table, threshold = 0.5):
        df = table
        selected = [] 

        def process_row(i):
            pid = df.loc[i]['SUBJECT_ID']
            # start = datetime.strptime(df.loc[i]['INTIME'], '%Y-%m-%d %H:%M:%S.%f %Z')
            # end = start + timedelta(hours= self.z)
            # start = end - timedelta(seconds= self.x)

            icutime = datetime.strptime(df.loc[i]['INTIME'], '%Y-%m-%d %H:%M:%S.%f %Z')
            mvtime = datetime.strptime(df.loc[i]['VStart'], '%Y-%m-%d %H:%M:%S.%f %Z')

            # patient doesn't have enough waveform data to be analyzed
            if mvtime < icutime or (mvtime - icutime).total_seconds() < self.x:
                return None

            end = datetime.strptime(df.loc[i]['VStart'], '%Y-%m-%d %H:%M:%S.%f %Z')
            start = end - timedelta(seconds= self.x)

            # get data availability stats
            availability = get_waveform_availability(pid, (start, end), channels= self.modality)
            # remove patients if availability < 50%
            if availability[0] < threshold:
                return None 
            
            data = get_waveform_data(int(pid), (start, end), channels= self.modality)
            nans = np.isnan(data[0]).sum()/len(data[0]) if data is not None else -1
            
            if nans < threshold:
                return pid
            

            return None

        def run(f, data):
            l = len(data)
            results= []
            with n_tqdm.tqdm(total=l) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers= self.num_workers) as executor:
                    futures = {executor.submit(f, arg): arg for arg in range(l)}
                    for future in concurrent.futures.as_completed(futures):
                        arg = futures[future]
                        results.append(future.result())
                        pbar.update(1)
            return results
        
        results = run(process_row, df)
        selected = df[df['SUBJECT_ID'].isin([pid for pid in results if pid is not None])].reset_index(drop= True)
        return selected