Human Activity Recognition Based on Motion Sensor Using U-Net

---

Y. Zhang, Z. Zhang, Y. Zhang, J. Bao, Y. Zhang and H. Deng, "Human Activity Recognition Based on Motion Sensor Using U-Net," in IEEE Access, vol. 7, pp. 75213-75226, 2019, doi: 10.1109/ACCESS.2019.2920969.

https://ieeexplore.ieee.org/document/8731875

- data
    - OPPGestureDataset
        - features_extraction_codes:provides the codes to conduct the features extractions for machine learing algorithms
        - extracted_features_dataset.txt:provides Baidu Netdisk link to download the extracted features dataset
    - SanitationDataset
        - readme.docx: introudce the Sanitation dataset
        - sanitation.csv:provides the raw dataset
        - Sanitation_processed_data.zip:provides the extracted features dataset
- HAR_dense_prediction_methods:includes the HAR codes of U-Net, FCN, SegNet, MaskRCNN
    - main.py: the main funciton
    - run.py: you can run it directly,configure the parameters
    - postcorrection.py:the source codes of the post correction algorithm
    - unet_info.py: setting the GPU
    - unet_data_load.py: the source codes of the generation of subsequences on the four datasets
    - unet_model.py: the source codes of HAR based on unet
    - FCN_model.py: the source codes of HAR based on FCN
    - segnet.py&layer.py: the source codes of HAR based on SegNet
    - maskrnn.py: the source codes of HAR based on Mask R-CNN
- HAR_sliding_window_prediction_methods: includes HAR based on machine learning and deep learning methods(CNN,lstm,cnnlstm)
    - com_main.py: the main function of HAR based on CNN,LSTM,CovLSTM
    - com_run.py: you can directly run it
    - win_data_load.py:the source codes of the generation of sling window data on the four datasets
    - deep_model_cnn_lstm_covlstm.py: the source codes of HAR based on CNN,LSTM,CovLSTM
    - SVM_nitin.py&decisionTress.py：the source codes fo HAR baesd on SVM and DT
    - common.py: calculate the confusion matrix and Fw-score
    - sameindex.py: convert the sliding window prediction index to the dense prediction index
