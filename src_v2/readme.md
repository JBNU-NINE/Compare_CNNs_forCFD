 The code in SRC is getting unmanageable. As it was initial research. It's fine but I want to have a code which will be good to read and understand, properly organized too. 

Things I'll need for training neural networks: 
1. Callbacks (Early Stopping, Model Checkpointing)
2. Visualizations (Model Loss etc)


*All of these need a proper place where the model's training stats can be sent. And then, it can send triggers to the training process on when to end the training and when not to*

Step 1: Create a good parser. 

Step 2: Create a good callback and visualization System

Step 3: Add a model

Step 4: After the model is trained, automatically get accuracy and other required metrics. 

Step 5: Know that in AI, the main file is going to be a notebook. So, do accordingly. You will have a main.ipynb file and that is going to do works for you. So, no need to create an argument parser manually, can just create a parser class.
