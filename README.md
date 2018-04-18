# 486FinalProject
How to run our program!

python3 program.py [TRAINING_DATA] [TESTING_DATA] [TRACK_ID_AND_YEAR] [CLASSIFICATION_METHOD] [optional: CLASSIFICATION_SPECIFIC_PARAMETERS]

Word Distribution / Top Words:
python3 song_insight.py [FILENAME] ['stats' | 'words'] [optional: max_songs | max_words]

Examples:

Rocchio Text Classification:
python3 program.py mxm_dataset_train.txt mxm_dataset_test.txt MSD_track_id_and_year.txt rocchio

KNN, k=25:
python3 program.py mxm_dataset_train.txt mxm_dataset_test.txt MSD_track_id_and_year.txt knn 25

SVM, linear kernel:
python3 program.py mxm_dataset_train.txt mxm_dataset_test.txt MSD_track_id_and_year.txt svm linear

LDA:
python3 program.py mxm_dataset_train.txt mxm_dataset_test.txt MSD_track_id_and_year.txt lda
