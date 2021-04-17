import gym
from gym import spaces
import numpy as np
import pandas as pd
from tqdm import tqdm
from ppo_torch import Agent
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class MNIST_trainer(gym.Env):
    def __init__(self):
        super(MNIST_trainer, self).__init__()
        
        # Environment params
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(8*8, ), dtype=np.int8)

        # Environment data
        mnist = load_digits()
        self.images = mnist['images']
        self.labels = mnist['target']
        self.images = self.flatten(self.images)
        self.images, self.images_test, self.labels, self.labels_test = train_test_split(self.images, self.labels, 
                                                                                        test_size=0.2, train_size=0.8, 
                                                                                        shuffle=True)
        self.current_state = 0
        self.end_state = len(self.labels) - 1

        self.current_state_test = 0
        self.end_state_test = len(self.labels_test) - 1

    def step(self, action):
        reward = 0
        done = False
        info = "MNIST-Trainer"
        predicted_label = action
        correct_label = self.labels[self.current_state]

        if predicted_label == correct_label:
            reward = 1
        else:
            reward = 0
        
        if self.current_state == self.end_state:
            done = True

        if done is False:
            self.current_state += 1
            next_state = self.images[self.current_state]
        else:
            next_state = -np.inf

        return next_state, reward, done, info
    
    def reset(self):
        self.current_state = 0
        state = self.images[self.current_state]

        return state

    def render(self, mode='human', close=False):
        pass

    def flatten(self, images):
        flatten_images = []
        for i in range(images.shape[0]):
            image = np.array(images[i])
            flatten_images.append(image.flatten())
        return np.array(flatten_images)


if __name__ == "__main__":

    # Environment settings
    env = MNIST_trainer() 
    
    # Trainer's settings
    load_checkpoint = False
    chkpt_dir = 'tmp/ppo'
    render = False
    n_trials = 5
    n_episodes = 1000

    # PPO params
    N = 20
    batch_size = 10
    n_epochs = 3
    alpha = 0.00003 
    best_score = env.reward_range[0]
    layer_1_dim = 128
    layer_2_dim = 128

    # Final results
    score_book = {}
    actor_loss_book = {}
    critic_loss_book = {}
    total_loss_book = {}

    for trial in range(n_trials):
        print('\nTrial:', trial+1)
        agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                        n_epochs=n_epochs, input_dims=env.observation_space.shape,
                        fc1_dims=layer_1_dim, fc2_dims=layer_2_dim, chkpt_dir=chkpt_dir)
        
        # Initialize storage pointers
        score_history = []
        avg_score_history = []
        loss = []
        actor_loss = []
        critic_loss = []
        total_loss = []

        # Initialize the run
        learn_iters = 0
        avg_score = 0
        n_steps = 0

        if load_checkpoint:
            agent.load_models()

        for i in tqdm(range(n_episodes)):
            observation = env.reset()
            done = False
            score = 0

            while not done:
                if render:
                    env.render(mode='human')
                
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                n_steps += 1
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                
                if not load_checkpoint:
                    if n_steps % N == 0:
                        loss.append(agent.learn())
                        learn_iters += 1

                observation = observation_
            
            if not load_checkpoint:
                avg_loss = np.mean(loss, axis=0)
                actor_loss.append(avg_loss[0])
                critic_loss.append(avg_loss[1])
                total_loss.append(avg_loss[2])

            score = 100*(score/env.labels.shape[0])
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)
            
            if avg_score >= best_score:
                best_score = avg_score
                if not load_checkpoint:
                    # print("Saving Model")
                    agent.save_models()
        
            # print('episode', i, 'score %.2f' % score, 'avg_score %.2f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
        
        # TEST THE MODEL
        test_score = 0
        for i in range(env.images_test.shape[0]):
            action, _, _ = agent.choose_action(env.images_test[i])
            if action == env.labels_test[i]:
                test_score += 1
        test_accuracy = 100*(test_score/env.labels_test.shape[0])
        print("training accuracy: {}".format(score))
        print("testing accuracy: {}".format(test_accuracy))


        score_book[trial] = score_history
        actor_loss_book[trial] = actor_loss
        critic_loss_book[trial] = critic_loss
        total_loss_book[trial] = total_loss

            
    print("\nStoring rewards data...")
    a = pd.DataFrame(score_book)
    a.to_csv('data/PPO-MNIST-rewards-train.csv')
    if not load_checkpoint:
        print("\nStoring losses...")
        b = pd.DataFrame(actor_loss_book)
        b.to_csv('data/PPO-MNIST-actor_loss.csv')
        c = pd.DataFrame(critic_loss_book)
        c.to_csv('data/PPO-MNIST-critic_loss.csv')
        d = pd.DataFrame(total_loss_book)
        d.to_csv('data/PPO-MNIST-total_loss.csv')
    print("Experiment finshed")