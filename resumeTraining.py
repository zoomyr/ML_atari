from tf_agents.environments import suite_gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt
from functools import partial
import os
import tf_agents.environments.wrappers
from functools import partial
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function
import matplotlib.animation as animation
import PIL
from tf_agents.policies.policy_saver import PolicySaver
import tf_agents



def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

env = suite_gym.load("Assault-v0")
img = env.render(mode="rgb_array")

repeating_env = ActionRepeat(env, times=4)


limited_repeating_env = suite_gym.load(
    "Assault-v0",
    gym_env_wrappers=[partial(TimeLimit, max_episode_steps=10000)],
    env_wrappers=[partial(ActionRepeat, times=8)],
)

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "AssaultNoFrameskip-v0"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])
time_step = env.step(1) # FIRE
for _ in range(4):
    time_step = env.step(3) # LEFT

def plot_observation(obs):
    # Since there are only 3 color channels, you cannot display 4 frames
    # with one primary color per frame. So this code computes the delta between
    # the current frame and the mean of the other frames, and it adds this delta
    # to the red and blue channels to get a pink color for the current frame.
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")
    

tf_env = TFPyEnvironment(env)

preprocessing_layer = keras.layers.Lambda(
                          lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)


train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ?
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ?


checkpoint_directory = "lastModelCheckpoint"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer)
checkpoint.restore(tf.train.latest_checkpoint('lastModelCheckpoint/ckpt-600.data-00000-of-00001'))

saved_policy = tf.compat.v2.saved_model.load('savedPolicy')
policy_state = saved_policy.get_initial_state(tf_env.batch_size)

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=1000000)

replay_buffer_observer = replay_buffer.add_batch


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)




collect_driver = DynamicStepDriver(
    tf_env,
    saved_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration


initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(200)],
    num_steps=200) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()


tf.random.set_seed(888) # chosen to show an example of trajectory at the end of an episode

trajectories, buffer_info = replay_buffer.get_next(
    sample_batch_size=2, num_steps=3)
time_steps, action_steps, next_time_steps = to_transition(trajectories)


dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)



def resume_training(n_iterations):
	time_step = None
	saved_policy = tf.compat.v2.saved_model.load('savedPolicy')
	policy_state = saved_policy.get_initial_state(tf_env.batch_size)
	saver = PolicySaver(agent.policy, batch_size=None)
	iterator = iter(dataset)
	for iteration in range(n_iterations):
		time_step, policy_state = collect_driver.run(time_step, policy_state)
		trajectories, buffer_info = next(iterator)
		train_loss = agent.train(trajectories)
		print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
		if iteration % 1000 == 0:
			log_metrics(train_metrics)
			saver.save('savedPolicy')
			checkpoint.save(file_prefix=checkpoint_prefix)

resume_training(10000)