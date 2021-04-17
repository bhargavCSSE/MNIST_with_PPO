# MNIST digits classification using PPO
Solving MNIST classification problem using PPO.\
The dataset used in this code is from sklearn.  It can be swapped within the MNIST_trainer class in **main.py** with your dataset. \
The PPO code was derived from an excellent video tutorial on https://youtu.be/hlv79rcHws0 \
\
**HOW TO RUN**
1. Create a python environment (python==3.8 recommended) and install pip dependencies using requirements.txt
```
conda create -n mnist_rl python==3.8
conda activate mnist_rl
pip install -r requirements.txt
```
2. Tune PPO parameters (not neccessary) and adjust number of trials and number of episodes in **main.py**.
3. Run the code using following command.
```
python main.py
```
4. Once the code is finished running, open **plotter.ipynb** and run all the cell to generate a plot in plots folder.

\
**RESULT** \
\
The figure below shows accuracies on training and testing dataset after each trial. \
<img src=https://github.com/bhargavCSSE/MNIST_with_PPO/blob/main/plots/MNIST_classification.png> 

The figure below shows accuracy vs iterations plot. \
<img src=https://github.com/bhargavCSSE/MNIST_with_PPO/blob/main/plots/MNIST-score.png> 
