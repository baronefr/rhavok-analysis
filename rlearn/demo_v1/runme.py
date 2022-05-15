#!/usr/bin/env python3

###########################################################################
#   Laboratory of Computational Physics // University of Padua, AY 2021/22
#   Group 2202 / Barone Nagaro Ninni Valentini
#
#  Train an actor critic DNN (reinforcement learning) for 
#  Lorenz transient state recovery.
#
#  coder: Barone Francesco, last edit: 15 may
#--------------------------------------------------------------------------
#  adapted from keras.io documentation 
#      @ https://keras.io/examples/rl/actor_critic_cartpole/
#  under Apache License 2.0
#--------------------------------------------------------------------------


try:
    import cayde_env  # importing a custom environment for my server
    cayde_env.tensorflow('CPU')
except: print('no need of custom environment')


import sys
import os
from datetime import datetime

import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gym_lorenz import DynamicalSystem
from sys_lorenz import LorenzSystem


# parameters
gamma = 0.99  # discount factor for past rewards
max_steps_per_episode = 170

eps = np.finfo(np.float32).eps.item()  # smallest number such that 1.0 + eps != 1.0
print('\n\nprecision:', eps)


################################
# setup Keras model
num_inputs = 3
num_actions = 20  # critical
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

################################
# setup the gym environment

system = LorenzSystem(x0_dataset = './data/havok_critic_Lorenz.csv')
env = DynamicalSystem(system, dt=0.01)
env.seed(420)

################################

optimizer = keras.optimizers.Adam(learning_rate=0.005)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

stationary_count = 0

action_smooth_factor = 0.1

print('\n\n------ BEGIN OPTIMIZATION ------')

while True:  # run until solved
    state = env.reset()
    episode_reward = 0
    action_smoothed = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
        
            if episode_count % 20 == 0:
                env.render()

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # predict action probabilities and estimated 
            # future rewards from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # sample action from action probability distribution
            probs = np.copy(np.squeeze(action_probs))
            action = np.random.choice(num_actions, p=probs )
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # apply the (smoothed) sampled action in our environment
            action_smoothed = action_smooth_factor*action + (1-action_smooth_factor)*action_smoothed
            state, reward, done, _ = env.step(action_smoothed)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break
                
        if episode_count % 20 == 0:
            print('p =', np.squeeze(action_probs))

        # update running reward
        prev_running_reward = running_reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # calculate expected value from rewards
        # - at each timestep what was the total reward received after that timestep
        # - rewards in the past are discounted by multiplying them with gamma
        # - these are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # calculate loss values to update network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # clear loss & reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # log details
    episode_count += 1
    if episode_count % 1 == 0:
        template = "[episode {}] running reward: {:.2f}"
        print(template.format(episode_count, running_reward))
    
    # task execution policy
    drew = running_reward - prev_running_reward
    if (drew > 0) and (drew < 20):  
        stationary_count +=1
        if stationary_count == 10: # condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break
    else:
        stationary_count = max(stationary_count-1, 0)
        
        
        

print('p_final =', np.squeeze(action_probs))

dt_string = datetime.now().strftime("%m-%d_%H-%M")
model.save(f'lorenz_{dt_string}.keras')
print('saved model as', f'lorenz_{dt_string}.keras')



print('\n\n------ BEGIN DEMO ------')

try:
    while True:
        state = env.reset()
        episode_reward = 0
        action_smoothed = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):
                env.render()

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])

                probs = np.copy(np.squeeze(action_probs))
                action = np.random.choice(num_actions, p=probs )
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                action_smoothed = action_smooth_factor*action + (1-action_smooth_factor)*action_smoothed
                state, reward, done, _ = env.step(action_smoothed)
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break
except:
    print('\n\nExit from program.')

# eop
os._exit(os.EX_OK)
