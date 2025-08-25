import numpy as np
from scipy import signal
from scipy.signal import stft
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import os
import sys
import time
from sklearn.model_selection import StratifiedKFold
from utils.mne_utils import setup_mne_for_processing
from utils.new_dataloader import load_epoch_data, read_all_events, create_balanced_epochs
from scipy.signal.windows import gaussian, hann
from scipy.signal import ShortTimeFFT
from sklearn import svm
from utils.new_pipeline import test_model
import logging
from utils.log import OutputCapture, StdoutCapture, StderrCapture
import joblib
from utils.new_analysis import analyze_classification


def generate_psd_feature(X_train, epoch_duration, samplerate):
    psd_feature_all_epochs = []
    for X_sample in X_train: #X_sample is a 2D array of shape (n_channels, n_samples), that is (19, 1024)
        # read every channel in X_sample.
        psd_feature_multi_channel = []
        for ch in range(X_sample.shape[0]):
            uni_ch = X_sample[ch,:]
            w = hann(epoch_duration*samplerate)
            SFT = ShortTimeFFT(w, hop=epoch_duration*samplerate+1, fs=samplerate, scale_to='psd') #edit mfft to adjust the frequency resolution
            # SFT.k_max = epoch_duration*samplerate        
            Sx = SFT.stft(uni_ch)#(513, 2)
            
            # t_lo, t_hi = SFT.extent(samplerate*epoch_duration)[:2] 
            # print("t_lo: ", t_lo, "t_hi: ", t_hi)
            psd_values = Sx[:, 0]#(513,2)

            if samplerate == 128:
                
                psd1 = psd_values[0:4]
                psd2 = psd_values[4:8]
                psd3 = psd_values[8:13]
                psd4 = psd_values[13:20]
                psd5 = psd_values[20:30]
                psd6 = psd_values[30:40]
                psd7 = psd_values[40:60]
                psd_feature_per_channel = [psd1, psd2, psd3, psd4, psd5, psd6, psd7]
                # psd 1-7 is array with different length, so we need to concatenate them into one array
                psd_feature_per_channel = np.concatenate(psd_feature_per_channel, axis=0)
                #vertically append psd_feature_per_channel to psd_feature_multi_channel
                psd_feature_multi_channel = psd_feature_per_channel if len(psd_feature_multi_channel) == 0 else np.concatenate((psd_feature_multi_channel, psd_feature_per_channel))
                # print("psd_feature_multi_channel.shape: ", psd_feature_multi_channel.shape)

        psd_feature_all_epochs.append(np.abs(psd_feature_multi_channel))
    psd_feature_all_epochs = np.array(psd_feature_all_epochs)

    return psd_feature_all_epochs

def main():

    with open('./config_fft.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['timestamp'] = datetime.now().strftime("%m%d_%H%M%S") # add unique timestamp to the config
        config['dataset'] = config['bids_root'].split('/')[-1]

    os.makedirs(config['preprocessed_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    # train_epochs, test_epochs = load_dataset(config)
    # load data
    setup_mne_for_processing(verbose=config.get('mne_verbose', False))

    print("\n=== Preprocess Pipeline with Balanced Epochs ===")
    
    # Read and label all events
    all_events, (seizure_subjects, bckg_subjects) = read_all_events(config)
    # train test split, using stratified k-fold cross validation
    split_seed = config.get('split_seed')
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_subjects = np.concatenate([seizure_subjects, bckg_subjects])
    all_labels = np.concatenate([np.ones(len(seizure_subjects)), np.zeros(len(bckg_subjects))])
    split_counter = 0
    for train_index, test_index in kf.split(all_subjects, all_labels):
        split_counter += 1
        train_subjects = all_subjects[train_index]
        test_subjects = all_subjects[test_index]
        print(f"Train subjects ids for split {split_counter}: {train_subjects}, train labels: {all_labels[train_index]}")
        print(f"Test subjects ids for split {split_counter}: {test_subjects}, test labels: {all_labels[test_index]}")


        train_events = all_events[all_events['subject'].isin(train_subjects)]
        test_events = all_events[all_events['subject'].isin(test_subjects)]

        if config.get('debug', False):
            print("\nReading and labeling events for split {}...".format(split_counter))
            if all_events.empty:
                print("No events found! Check your BIDS directory.")
                break
            print(f"Total events loaded for split {split_counter}: {len(all_events)}")
    
        # Create balanced epochs
        print(f"Creating balanced epochs for train set for split {split_counter}...")
        train_epochs = create_balanced_epochs(train_events, config)
        print(f"Creating balanced epochs for test set for split {split_counter}...")
        test_epochs = create_balanced_epochs(test_events, config)

        X_train, y_train = load_epoch_data(train_epochs, config, split_name='train', split_counter=split_counter, cv=True)
        # print("X_train.shape: ", X_train.shape) #(5651, 19, 1024)
        epoch_duration = config.get('epoch_duration')
        samplerate = config.get('sample_rate')
        psd_feature_all_epochs = generate_psd_feature(X_train, epoch_duration, samplerate)
        model = svm.SVC()
        start_training_time = time.time()
        model.fit(psd_feature_all_epochs, y_train)
        end_training_time = time.time()

        print(f"Model training completed in: {end_training_time - start_training_time:.2f} seconds")
    
        # save model
        model_name = "{}_{}_split_{}".format(config['model_name'], config['timestamp'], split_counter)
        model_path = os.path.join(config['model_dir'], f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            joblib.dump(model, f)
        print(f"Model saved to: {model_path}")
    
        # save model config to the same directory
        # TODO: we need configuration, number of epochs, performance(terminal output).
        config_path = os.path.join(config['model_dir'], f"{model_name}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Config saved to: {config_path}")
    
        print(f"\n ===== Training Model Completed ===== \n")
            # return np.array(psd_feature_all_epochs)
        X_test, y_test = load_epoch_data(test_epochs, config, split_name='test', split_counter=split_counter, cv=True)
        print(f"\n ===== Evaluating Model on test set ===== \n")
        psd_feature_test = generate_psd_feature(X_test, epoch_duration, samplerate)
        y_pred = model.predict(psd_feature_test)
        print(f"Test set size(X_test): {len(X_test)}")

        f1, precision, recall, accuracy, conf_mat, conf_mat_norm = analyze_classification(y_pred, y_test)

        print(f"\n ===== Evaluating Model Completed ===== \n")
        

    
                    

log_path = 'temp.txt'
output_capture = OutputCapture(log_path, also_console=True)
sys.stdout = StdoutCapture(output_capture)

if __name__ == "__main__":
    main()
    logging.shutdown()
    os.rename('temp.txt', log_path)