# 486FinalProject
How to run our program!

python program.py [TRAINING_DATA] [TESTING_DATA] [TRACK_ID_AND_YEAR] [CLASSIFICATION_METHOD] [optional: CLASSIFICATION_SPECIFIC_PARAMETERS]

Examples:

Rocchio Text Classification:
python program.py mxm_dataset_train.txt mxm_dataset_test.txt MSD_track_id_and_year.txt rocchio

KNN, k=25
python program.py mxm_dataset_train.txt mxm_dataset_test.txt MSD_track_id_and_year.txt knn 25

SVM, linear kernel
python program.py mxm_dataset_train.txt mxm_dataset_test.txt MSD_track_id_and_year.txt svm linear
