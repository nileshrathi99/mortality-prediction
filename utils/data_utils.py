import os, numpy as np
import wfdb
from datetime import datetime, timedelta
from collections import defaultdict
import csv, tqdm
from multiprocessing import Pool
# from numba import njit
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from .configs import base_db


# save any list as csv with a given column name and file name
def save_list_as_csv(_list, column_name, file_name):
    # Open the CSV file for writing
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([column_name])
        for item in _list:
            writer.writerow([item])

#returns overlapped window
def find_overlap(cst, cet, wst, wet):
    intervals = [(cst, cet), (wst, wet)]
    intervals = sorted(intervals, key= (lambda x: x[1]))
    if intervals[1][0] > intervals[0][1]:
        return 0, 0
    
    start = intervals[0][0] if intervals[0][0] > intervals[1][0] else intervals[1][0]
    return (start, intervals[0][1])

def get_record_list(db_dir):
    header_records = os.path.join(db_dir, 'RECORDS')
    with open(header_records, 'r') as file:
            headers = file.readlines()
    headers = [x.strip() for x in headers]
    return headers


def get_record_list_2(db_dir):
    header_records = os.path.join(db_dir, 'RECORDS')
    with open(header_records, 'r') as file:
            headers = file.readlines()
    records = []
    for x in headers:
        x = x.strip()
        if x.startswith('p') and not x.endswith('n'):
            records.append(x)
    return records

def create_splits(df, n_splits= 5):
        selected_table = df
        selected_table['fold'] = -1
        selected_table.reset_index(inplace= True, drop= True)
        
        # Stratified K-Fold splitting
        labels = selected_table['label'].values.tolist()
        skf = StratifiedKFold(n_splits, shuffle=True)
        for fold_index, (_, test_index) in enumerate(skf.split(np.zeros_like(labels), labels)):
            selected_table.loc[test_index, 'fold'] = fold_index

def get_fold_split(df, fold_index):
        selected_table = df
        train_mask = selected_table['fold'] != fold_index
        train_dataset = df[train_mask].reset_index(drop = True)
        test_mask = ~train_mask
        test_dataset = df[test_mask].reset_index(drop = True)
        return train_dataset, test_dataset

#patient_id: str Subject id. Eg. '000033'
#window: a tuple of str like (start time, end time). Eg. ('2116-11-24 12:30:06', '2116-12-24 12:38:40')
#date_format: str Default: '%Y-%m-%d %H:%M:%S'
#Return a dictionary having total time in secs for each available waveform signal in the given window

def get_waveform_availability(id: int, window: tuple, channels: list = ['II']) -> dict:
    string_value = str(id)
    if len(string_value) < 6:
        num_zeros_to_prepend = 6 - len(string_value)
        string_value = '0' * num_zeros_to_prepend + string_value

    patient_id = string_value
    patient_dir = 'p' + patient_id[:2] + '/p' + patient_id
    window_start_time, window_end_time = window
    duration = (window_end_time - window_start_time).total_seconds()

    #check the given window
    if duration == 0:
        print('Error: Window Size is zero')
        return {}

    records = get_record_list_2(db_dir= os.path.join(base_db, patient_dir))
    info = [0]*len(channels)#defaultdict(lambda: 0)
    flag= False

    # for record in records:
    for i in range(len(records)):
        record = records[i]
        try:
            # header = wfdb.rdheader(record_name= record, pn_dir= os.path.join(base_db, patient_dir), rd_segments= True)
            header = wfdb.rdheader(record_name= os.path.join(base_db, patient_dir, record), rd_segments= True)

            #calculate start & end time for the header
            wave_start_time = header.base_datetime
            time_delta = header.sig_len * (1/header.fs)
            time_delta = timedelta(seconds= time_delta)
            wave_end_time = wave_start_time + time_delta

            start, end = find_overlap(window_start_time, window_end_time, wave_start_time, wave_end_time)
            if start - end == 0:
                if flag:
                    break
                continue
            
            flag= True
            start_frame = header.get_frame_number(start)
            end_frame = header.get_frame_number(end)

            sig_names = channels #header.get_sig_name()
            fs= header.fs

            # for signal in sig_names:
            for idx in range(len(channels)):
                signal = channels[idx]
                contained_ranges = header.contained_ranges(sig_name= signal)
                for wsf, wef in contained_ranges:
                    x, y = find_overlap(wsf, wef, start_frame, end_frame)
                    if x-y== 0:
                        continue
                    info[idx] += (y - x)*1.0/fs
        except:
            continue

    # for key, val in info.items():
    #     info[key] = val/duration
    info = [x/duration for x in info]
    return info#dict(info)


#patient_id: str Subject id. Eg. '000033'
#window: a tuple of str like (start time, end time). Eg. ('2116-11-24 12:30:06', '2116-12-24 12:38:40')
#date_format: str Default: '%Y-%m-%d %H:%M:%S'
#Returns a numpy array where the missing data is represented as float('nan')

def get_waveform_data(id: int, window: tuple, channels: list = 'II'):
    string_value = str(id)

    if len(string_value) < 6:
        num_zeros_to_prepend = 6 - len(string_value)
        string_value = '0' * num_zeros_to_prepend + string_value

    patient_id = string_value
    patient_dir = 'p' + patient_id[:2] + '/p' + patient_id
    window_start_time, window_end_time = window ### [start, end)

    window_len = (window_end_time - window_start_time).total_seconds()
    if window_len == 0:
        print('Error: Window Size is zero')
        return {}

       
    #find all records
    records = get_record_list_2(db_dir= os.path.join(base_db, patient_dir))

    data = None
    n_sig = len(channels)
    fs = None
    
    for record in records:
        record_name = os.path.join(base_db, patient_dir, record)
        header = wfdb.rdheader(record_name= record_name, rd_segments= False)

        #calculate start & end time for the header
        wave_start_time = header.base_datetime
        time_delta = header.sig_len * (1/header.fs)
        time_delta = timedelta(seconds= time_delta)
        wave_end_time = wave_start_time + time_delta

        start, end = find_overlap(window_start_time, window_end_time, wave_start_time, wave_end_time)
        
        # print(f'init: {start, end, wave_start_time, wave_end_time, window_start_time, window_end_time}')
        # since intervals are sorted
        if start-end == 0:
            if data is not None:
                break
            continue
        
        # Initialize the data numpy array
        if fs is None:
            fs = header.fs
            data = np.full((n_sig, int(fs*window_len)), float('nan'))

        start_frame = int(header.get_frame_number(start))
        end_frame = int(header.get_frame_number(end))
        start_window_frame = int(header.get_frame_number(window_start_time))
        frame_diff = start_frame - start_window_frame

        # print(f'E: {start_frame, end_frame, end_frame - start_frame}')
        signals,_ = wfdb.rdsamp(record_name= record_name, channel_names= channels, sampfrom= start_frame, sampto= end_frame)
        # print(f'X: {signals.shape}')
        ## signals can be none, if the all the channels are not present
        duration = min(len(signals), int(fs*window_len))
        data[:n_sig, frame_diff: frame_diff + (end_frame - start_frame)] = signals.T[:n_sig, :duration]

    return data


# df: dataframe with subject id, ICU INTIME
# duration: length of window starting from INTIME, to check the waveform availability
# filter: boolean whether to filter the subject ids
# filter_list: list to select the patients from, if filter is true
# Returns a dictionary of each waveform signal and corressponding list of availabilities in the window provided
def get_time_distribution(df, duration: float = 1.0, filter_list: list = [], filter: bool = True):
    time_pdf = defaultdict(list)
    matched = filter_list
    for index in tqdm.tqdm(range(len(df))):
        record = df.loc[index]
        id = record['SUBJECT_ID']
        #filter for waveform matched db
        if filter and id not in matched:
            continue
        
        window_start_time = datetime.strptime(record['INTIME'], '%Y-%m-%dT%H:%M:%S')
        window_end_time = window_start_time + timedelta(hours= duration)

        try:
            availability =  get_waveform_availability(id, (window_start_time, window_end_time))
            for key, val in availability.items():
                time_pdf[key].append(val)
        except Exception as e:
            print(f'An exception occured for patient:{id} -> {e}')
            continue
    return time_pdf


def process_record(record: dict, duration: float):
    id = record['SUBJECT_ID']
    window_start_time = datetime.strptime(record['INTIME'], '%Y-%m-%dT%H:%M:%S')
    window_end_time = window_start_time + timedelta(hours=duration)

    try:
        availability = get_waveform_availability(id, (window_start_time, window_end_time))
        return availability, id
    except Exception as e:
        print(f'An exception occurred for patient:{id} -> {e}')
        return {}, id

def get_time_distribution_parallel(df, duration=1.0, num_workers= 8):
    time_pdf = defaultdict(list)
    # Split the DataFrame into chunks for parallel processing
    data_chunks = [(df.loc[x].to_dict(), duration) for x in range(len(df))]
    with Pool(num_workers) as pool:
        for chunk,_ in tqdm.tqdm(pool.starmap(process_record, data_chunks), total=len(data_chunks)):
            for key, val in chunk.items():
                time_pdf[key].append(val)

    return time_pdf


def give_patients_with_wave_threshold(df, duration = 1.0, signal = 'II', threshold = 0.5, num_workers = 8):
    patients = []
    data_chunks = [(df.loc[x].to_dict(), duration) for x in range(len(df))]
    with Pool(num_workers) as pool:
        for chunk, id in tqdm.tqdm(pool.starmap(process_record, data_chunks), total=len(data_chunks)):
            if signal in chunk.keys() and chunk[signal] > threshold:
                patients.append(id)

    return patients


# Assumption: There is enough data from left or right neighbour to borrow from
# Missing value imputation
def borrow(data):
    if np.isnan(data).sum() == 0:
        return data
    start = -1 
    end = -1
    # for idx, val in enumerate(signal):
    #     if start == -1 and np.isnan(val):
    #         start = idx 
    #     elif start != -1 and ~np.isnan(val):
    #         end = idx 
    #         curr_len = end - start

    #         if start - curr_len >= 0 and end + curr_len <= len(signal) and np.isnan(signal[end: end + curr_len]).sum() == 0: 
    #             signal[start: end] = (signal[start - curr_len : start] + signal[end: end + curr_len])/2 # average of both forward and backward window
    #         elif start - curr_len >= 0:
    #             signal[start: end] = signal[start - curr_len : start] # bFill
    #         elif end + curr_len <= len(signal) and np.isnan(signal[end: end + curr_len]).sum() == 0:
    #             signal[start: end] = signal[end: end + curr_len] # FFill
    #         else:
    #             signal
    #             # raise Exception('No window available to borrow for imputation!')
            
    #         start = -1

    def bfill(signal):
        start = -1
        end = -1
        for idx, val in enumerate(signal):
            if start == -1 and np.isnan(val):
                start = idx 
            elif start != -1 and ~np.isnan(val):
                end = idx 
                curr_len = end - start 
                if start - curr_len >= 0:
                    signal[start: end] = signal[start - curr_len : start] # bFill
        return signal
    
    data = bfill(data)
    data = bfill(data[::-1])[::-1]
    data[np.isnan(data)] = np.nanmean(data) #np.nanmedian(data)
    if np.isnan(data).sum() > 0:
        print(f' Number of nan values: {np.isnan(data).sum()}/{len(data)}')
    return data

# Cohort Creation
def create_snapshot(table, start_time, end_time, target = 'DEATH_T0'):
    data = table
    data['label'] = 0
    data['label'][(data[target] > start_time) & (data[target] <= end_time)] = 1
    data = data[data[target] > start_time]
    data.reset_index(inplace= True, drop= True)
    return data

# death distribution snapshot
def create_death_distribution(table, interval, target = 'DEATH_T0', range_= 21):
    house_deaths = np.array(table[target].values.tolist())
    hours = [i*interval for i in range(range_)]
    deaths_6= []
    for hrs in hours:
        deaths = len(house_deaths[(house_deaths > hrs - interval) & (house_deaths <= hrs)])
        deaths_6.append(deaths)
    
    return deaths_6, hours