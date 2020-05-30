# path
'snow_detect_simple/' is the event folder
'harmo_patch/' saves new generated dataset for training
'harmo_harmo-ml-challenge-2019-0922/' saves original dataset
 path of the text file for program execution

set their paths in 'mypath.py'

# training
run for training
	python train.py

pretrained model is saved in 'run/' folder

# test
run for test pre-trained model on a part of published images
	python test.py

# detect
run for detection on undisclosed verification dataset
	python detector.py

'output.csv' is saved in event folder
