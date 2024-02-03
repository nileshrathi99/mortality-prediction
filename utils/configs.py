base_db = '/scratch/rmahawar/mimic/waveform_db//mimic3wdb-matched-1.0.physionet.org/matched/matched'
base_dir = '/scratch/rmahawar/mimic/code'
n_folds = 3
classes = 2
X = 30
Y = 24
workers = 0
modality = ['II']
batch_size = 8

# main.py
fetch_new_window = False
# [LSTM, CNN]
which_model = 'CNN'
input_dim = 2