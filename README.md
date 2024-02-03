# mortality-prediction

## ABSTRACT
Predicting mortality poses a formidable challenge within modern ICUs, carrying significant implications for medical resource allocation and clinical decision-making. Our project endeavours to forecast the likelihood of sur- vival for ICU patients within a near future time frame, leveraging insights gained from analyzing data within a past look-back window. Inspired by recent strides in ECG-based arrhythmia classification, particularly high- lighted in the work by Ebrahimi Zahra et al. [3] on deep learning methods for ECG arrhythmia classification (Ex- pert Syst. Appl.: X, 7, 2020, Article 100033), our re- search work focuses on ECG waveform analysis to esti- mate a patient’s survival rate during ICU admission.
The significance of this project lies in its potential to enhance clinical interventions and optimize resource management within the ICU, ultimately contributing to improved patient outcomes in critical care settings. By harnessing the power of ECG waveform data, our ap- proach aims to provide a valuable tool for guiding more effective clinical decisions, ultimately elevating the stan- dards of care in critical care environments.

## INTRODUCTION
Electrocardiogram (ECG) data has been instrumen- tal in advancing the field of arrhythmia classification, demonstrating its potential in detecting complex heart conditions with remarkable accuracy. The utilization of ECG data, particularly through feature extraction from spectral images, has shown promising results in enhanc- ing the accuracy of medical diagnoses. This innovative approach has opened new avenues in the analysis and interpretation of ECG data, enabling a more nuanced understanding of cardiac health. However, despite these advancements, there remains a significant gap in the ap- plication of ECG data for predicting mortality, especially in critical care settings. This raises a pertinent question: Can ECG data alone be a reliable predictor of mortality? Furthermore, if ECG data in isolation is insufficient for this purpose, could it augment the predictive power of other methods, such as time-series analysis and the in- tegration of various clinical variables? This exploration could pave the way for more effective and personalized patient care strategies, especially in environments where accurate and timely predictions are crucial for patient outcomes, such as in Intensive Care Units (ICUs).

## BACKGROUND
Predicting mortality is a longstanding challenge in medical research, with extensive efforts directed towards leveraging clinical variables (such as diabetes) and time- series variables (including Spo2 measurements, heart rate, and blood pressure). Traditional scoring systems like APACHE-II, SOFA, and SAPS have historically been employed for mortality prediction in ICU patients, yield- ing AUC scores around 0.68 +- 0.02. Notably, Kim JH, Kwon YS, Baek MS [2] achieved notable success in pre- dicting 30-day mortality in mechanically ventilated ICU patients using machine learning models like extreme gra- dient boost and random forest.
The role of ECG data in predicting cardiovascular dis- eases, particularly arrhythmia, has been extensively ex- plored. Ebrahimi Zahra, Loni Mohammad, Daneshta- lab Masoud, Gharehbaghi Arash’s [3] comprehensive re- view underscores advancements in arrhythmia classifica- tion using ECG data, achieving remarkable AUC scores of approximately 99
In our research, we distinctly focus on the fundamen- tal question: Can ECG data alone serve as an effective predictor of mortality in ICU patients?

## DATASET
In our investigation, we employed the MIMIC-III Waveform Database Matched Subset, featuring 22,317 waveform records and 22,247 numerics records, as a valu- able resource for exploring the physiological dynamics of 10,282 distinct ICU patients. The dataset encom- passes digitized signals such as ECG, ABP, respiration, and PPG, as well as periodic measurements like heart rate, oxygen saturation, and blood pressure parameters. Notably, our focus was specifically on utilizing the ECG waveform data for predicting mortality rates. In contrast to the comprehensive MIMIC-III Waveform Database, our analysis honed in on patients with identified clini- cal records available in the MIMIC-III Clinical Database. By integrating the ECG waveform data with correspond- ing clinical information, our goal is to develop predictive models that leverage the intricate relationship between physiological signals and mortality outcomes. This cu- rated dataset, centered around ECG data and mortality prediction, provides a robust foundation for understand- ing the potential of observed phenomena as indicators of mortality in the ICU setting.

## METHODOLOGY 

A. Problem Formulation
The primary objective of this project is to predict the survival probability of patients in Intensive Care Units (ICUs) over a specific future time window. The binary classification problem is formulated as follows: Given the electrocardiogram (ECG) data (X), the task is to predict whether a patient will be classified as either ’dead’ or ’alive’ (y). (see Figure 1)

B. Data Preparation
The MIMIC dataset comprises ECG data obtained from patients in ICUs. The dataset prepared includes instances with a lookback window of either 10 seconds or 30 seconds from the onset of mechanical ventilation in patients.

C. Data Preprocessing
• Filtering Patients: Exclude patients if the data availability within the provided window is less than 50%.
• Handling Missing Values: Utilize missing value imputation strategies, including setting missing values to 0, employing nearest neighbour imputa- tion techniques(see Figure 2).
FIG. 3. Flow
• Noise Removal: To mitigate baseline wander in ECG signals, a 1D median filter was applied using a kernel size equal to the sampling frequency of the ECG signals. This approach effectively smoothed out fluctuations in the baseline.
Additionally, to address high-frequency noise in- troduced by factors such as patient shivering or tremors, a Butterworth filter was subsequently em- ployed. The Butterworth filter helped attenuate these high-frequency components, resulting in a cleaner ECG dataset for more accurate analysis.
• Short-time Fourier Transform (STFT): Uti- lize a technique called short-time Fourier trans- form (STFT). This method divides the signals into short segments and calculates Fourier transforms on each segment individually. The input to the STFT is a time-domain signal, and the output is a two-dimensional representation, known as a spec- trogram, which illustrates how the frequency com- ponents of the signal vary over time. The segment window used is equal to sampling frequency (125 hz) of ECG signals with overlap of 75%.

D. Prediction Scenarios
The project explores multiple prediction scenarios to assess the model’s performance under varying conditions:
• Lookback Window = 10 seconds, Prediction After 72 Hours. Class Distribution 200 instances of the ’dead’ class and 1400 instances of the ’alive’ class.
• Lookback Window = 30 seconds, Prediction After 24 Hours. Class Distribution 80 instances of the ’dead’ class and 1500 instances of the ’alive’ class.

E. Models
Several deep learning architectures were explored to find the best prediction on the given dataset. 1-D, 2-
 
3
 FIG. 4. Results
D CNNs or LSTMs as feature extractors followed by a fully connected layer were explored. 1-D CNNs were used to analyze signals in time domain while 2-D CNNs were used to extract features from the frequency domain of the signals. 2-D CNNs architecture were inspired from Resnet-18 architecture.
F. Evaluation Metrics
Given the imbalanced nature of the dataset, the Area Under the ROC Curve (AUC) is employed as a robust metric for evaluating the model’s discrimination ability between the ’dead’ and ’alive’ classes.

## RESULTS
The evaluation results of the models trained under dif- ferent scenarios are presented in the table depicted in 4.

## CONCLUSIONS AND LIMITATIONS
While 1D models (such as CNN1D and LSTM1D) gen- erally outperformed their 2D counterparts, it was ob- served that more sophisticated preprocessing methods did not yield improvements over basic missing value im- putation. This can be because of the way ecg data has been collected in the mimic-iii database. ECG signals collected were sampled 125 times per second, but at inter- vals that vary between 2 and 14 ms (averaging 8 ms). Be- cause of this reason we believe converting ecg signals from
time-domain to frequency-domain did not yield promis- ing results.
Despite these efforts, the obtained results do not show promise, as evidenced by the low Area Under the ROC Curve (AUC). This may be attributed to significant class imbalance and limited data availability, where the scarcity of instances for certain classes could have im- peded the model’s ability to generalize effectively.
It is plausible that relying solely on ECG data may not be sufficient for accurate mortality prediction.
In summary, although 1D models displayed potential, the overall limitations—such as imbalanced classes and data scarcity—underscore the challenges associated with predicting mortality within the ICU context.

## FUTURE SCOPE
Expanding the dataset by acquiring additional data is a crucial avenue for future exploration. Increasing the volume of available data can provide the model with a richer and more diverse set of instances, potentially im- proving its ability to generalize and make more accurate predictions.
Additionally, an avenue for experimentation involves extending the lookback window time. By lengthening the period considered in the lookback window, a more com- prehensive set of temporal features can be captured. This extension may offer insights into the temporal dynamics of patient data and potentially enhance the model’s pre- dictive capabilities.

## REFERENCES
1. Safdar, M.F.; Nowak, R.M.; Palka, P. A Denois- ing and Fourier Transformation-Based Spectrograms in ECG Classification Using Convolutional Neural Network. Sensors 2022, 22, 9576. https://doi.org/10.3390/ s22249576
2. Kim JH, Kwon YS, Baek MS. Machine Learning Models to Predict 30-Day Mortality in Mechanically Ven- tilated Patients. J Clin Med. 2021 May 18;10(10):2172. doi:10.3390/jcm10102172. PMID: 34069799; PMCID: PMC8157228.
3. Ebrahimi Zahra, Loni Mohammad, Daneshtalab Masoud, Gharehbaghi Arash A review on deep learning methods for ECG arrhythmia classification Expert Syst. Appl.: X, 7 (2020), Article 100033
4. Mishra, A.; Dharahas, G.; Gite, S.; Kotecha, K.; Koundal, D.; Zaguia, A.; Kaur, M.; Lee, H.-N. ECG Data Analysis with Denoising Approach and Customized CNNs. Sensors 2022, 22, 1928. https://doi.org/10. 3390/s22051928
