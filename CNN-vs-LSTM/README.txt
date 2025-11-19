(This is a copy of the last page from the report)


How to use the following program:


First, ensure that you have python installed by opening up your terminal and entering:


> py --version
	

If it is not installed, install it on windows using:


> winget install Python.python.3
	

After it is installed, you must install PyTorch by entering:


> pip install torch
	

or if you have an NVIDIA GPU:


> pip install torch --index-url https://download.pytorch.org/whl/cu121
	

After this is finished, extract the .py files into a folder and navigate there within your terminal.


To run the model that you want, enter the following:


> py main.py --network (nw) (others)
	

To specify hyperparameters, replace (others) with your choice of the following:


> --epochs (ep) --patience (p) --batch_size (bs) --lr (lr)
	

If you would like to find the most optimal hyperparameters yourself, there is a tuning mode that can be used by entering:


> py main.py --network (nw) --tuning True
	

This takes a long time so be prepared to wait.