# Protein-Atlas-Image-Classification

- mean_std.py is the python script for calculating the mean and std of the image data. 
- train.py is for training model 
- eval.py evaluates the results on the train dataset
- test.py evaluates the results on the test dataset

- The weights of my trained model may be found on Google Drive.

The following command trains the model from a resumed checkpoint and still runs if I closed the SSH session:
'''
nohup python3 -u train.py --checkpoint 'save/out_1.pth' --resume_epoch 1 > aa.log </dev/null 2>&1&
echo $! > pid.txt
'''
