The code that is written is located under the folder "zanyar". Here, there are 3 main Python scripts that are doing the work:
1. main.py: This python script contains the actual IRL and RLHF trainings.
2. train_expert.py: This python script was used to train the expert agent which provides demonstrations to the AIRL algorithm.
3. plotting.py: This python script was used to plot the experiment results that were collected under the folder "zanyar/ppo_cartpole_tensorboard".

In this experiment we are running AIRL alone, and then AIRL complemented by RLHF in order to see whether we can decrease the number of demonstrations that AIRL requires. 
