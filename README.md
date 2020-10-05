# Huawei_SearchRanking
The Deep Learning Classification model that I trained using Huawei Digix AI Challenge 2020 Search Ranking Task using Keras, Tensorflow, SckitLearn.

## Getting started with

Firstly you should have Huawei Digix AI 2020 Dataset in this case. Data is available for the competitors however, it could be public in the near future.
Simply, Data has 364 columns and roughly 6M rows. Columns mainly gives information about query, BM25 Score, users, time etc. Except Documents Id and Query Id, all of the features are integer and float. 

**More details I recommend to read comments is scripts**

**To clone:** git clone https://github.com/akalieren/HuwaeiDigixAI_SearchRanking.git

**To install requirements:** pip install -r requirements.txt

**Example Commands to train model**
python3 model.py --data /Path/to/TRAIN_DATA.csv --test /Path/to/TEST_DATA_B.csv --csv /PATH/to/submission.csv

## Output
Submission file in csv format.
* ids, docid, labels, where ids is unique query id, docid is document that search results of query, and finaly labels is relevancy between document and query. In this case label is integer between 0 and 4.

## Results
* Due to technical problems still we could not access results. After Huawei stuffs informs us, I will updated.
* There are 5 labels to measure relevancy between documents and query. In this sense; 4 is relevant, 0 is not.
