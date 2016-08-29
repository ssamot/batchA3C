#!/usr/bin/env python

import os



os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['OMP_NUM_THREADS'] = '1'

from atari_environment import AtariEnvironment
import numpy as np
import time
import gym
from model_batcha3c import *
from keras import backend as K
import threading
import math




flags = tf.app.flags

flags.DEFINE_string('experiment', 'space_invaders', 'Name of the current experiment')
flags.DEFINE_string('game', 'SpaceInvaders-v0',
                    'Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
flags.DEFINE_integer('num_concurrent', 6, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 10 * 10**7, 'Number of training timesteps.')


flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_string('summary_dir', '/tmp/summaries', 'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoints', 'Directory for storing model checkpoints')
flags.DEFINE_integer('summary_interval', 5,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval.')
flags.DEFINE_integer('checkpoint_interval', 100000,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'global frames (rounded up to statistics interval.')
flags.DEFINE_boolean('show_training', True, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', 'path/to/recent.ckpt', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_string('eval_dir', '/tmp/', 'Directory to store gym evaluation')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')

##  you shouldn't really change these
flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
flags.DEFINE_integer('agent_history_length', 4, 'Use this number of recent screens as the environment state.')

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

initial_learning_rate = log_uniform(1e-4 ,
                                    1e-2,
                                    0.4226)

FLAGS = flags.FLAGS
T = 0
TMAX = FLAGS.tmax
START_TIME = time.time()




def actor_learner_thread(lock, thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    """
    Actor-learner thread implementing batch AC3, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    Based on this: https://github.com/coreylynch/async-rl
    """
    global TMAX, T, START_TIME


    # Unpack graph ops
    s = graph_ops["s"]
    a = graph_ops["a"]
    R = graph_ops["R"]
    lr = graph_ops["learning_rate"]

    actor_update = graph_ops[get_name("update", thread_id)]
    v_p = graph_ops[get_name("v_p", thread_id)]

    copy_from_target = graph_ops[get_name("copy_from_target", thread_id)]


    summary_placeholders, update_ops, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height,
                           agent_history_length=FLAGS.agent_history_length)



    print "Starting thread ", thread_id
    time.sleep(3 * thread_id)
    t = 0
    games_played = 0

    while T < TMAX:

        # Get initial game observation
        s_t = env.get_initial_state()

        ep_reward = 0
        episode_v = 0
        episode_ave_max_p = 0
        ep_t = 0
        # actions = []
        start = time.time()

        s_batch = []
        a_batch = []
        reward_batch = []



        while True:



            readout = v_p.eval(session=session, feed_dict={s: [s_t]})[0]

            readout_v = readout[0]
            readout_p = readout[1:]

            action_index = np.random.choice(num_actions, p=readout_p)

            a_t = np.zeros([num_actions])
            a_t[action_index] = 1

            # Gym executes action in game environment on behalf of actor-learner
            s_t1, r_t, terminal, info = env.step(action_index)

            clipped_r_t = np.clip(r_t, -1, 1)  # try using sigmoid? later
            reward_batch.append(clipped_r_t)

            a_batch.append(a_t)
            s_batch.append(s_t)

            s_t = s_t1
            T = T + 1
            t += 1
            ep_t += 1
            ep_reward += r_t
            episode_v += (readout_v)
            episode_ave_max_p += np.max(readout_p)
            #print values.eval(session=session, feed_dict={s: [s_t]})[0][0]

            # Save model progress
            if T % FLAGS.checkpoint_interval == 0:
                saver.save(session, FLAGS.checkpoint_dir + "/" + FLAGS.experiment + ".ckpt", global_step=T)

            if terminal or (ep_t % 20 == 0):

                R_batch = []

                if(terminal):
                    R_t = 0.0
                else:
                    R_t = v_p.eval(session=session, feed_dict={s: [s_t1]})[0][0]


                # The forward view of TD calculated in reverse
                for reward in reversed(reward_batch):
                    R_t = reward + FLAGS.gamma * R_t
                    R_batch.append(R_t)

                R_batch = list(reversed(R_batch))

                learning_rate = initial_learning_rate * (FLAGS.tmax - T) / FLAGS.tmax
                learning_rate = max(learning_rate, 0)
                #
                session.run(actor_update, feed_dict={lr: learning_rate,
                                                     R : R_batch,
                                                     a : a_batch,
                                                     s : s_batch})

                s_batch = []
                a_batch = []
                reward_batch = []


                session.run(copy_from_target)

            if(terminal):
                games_played += 1
                #stats = [ep_reward, episode_v / float(ep_t)]
                # for i in range(len(stats)):
                #     session.run(update_ops[i], feed_dict={summary_placeholders[i]: float(stats[i])})
                end = time.time()
                print "THREAD:", thread_id, \
                    "/ TIME", T, \
                    "/ TIMESTEP", t, \
                    "/ REWARD", ep_reward,\
                    "/ Q_MAX %.4f" % (episode_v / float(ep_t), ), \
                    "/ P_MAX %.4f" % (episode_ave_max_p / float(ep_t), ), \
                    "/ Total(Hours):" , str((end - START_TIME)/60.0/60.0),\
                    "/ Million Steps/Hour:" , (T/((end - START_TIME)/60.0/60.0))/1000000.0,\
                    "/ Learning Rate: ", str(initial_learning_rate * (FLAGS.tmax - T) / FLAGS.tmax), \
                    "/ Time:" + str(end - start)



                break



def get_name(op_name, thread_id):
    return op_name + "_" + str(thread_id)

def build_graph(num_actions):
    graph_ops = {}

    # Create shared deep q network
    s, target_v_network, target_p_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                                 resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height)

    target_network_params_v = target_v_network.trainable_weights
    target_network_params_p = target_p_network.trainable_weights



    a = tf.placeholder("float", [None, num_actions])
    R = tf.placeholder("float", [None])
    graph_ops["s"] = s
    graph_ops["a"] = a
    graph_ops["R"] = R

    learning_rate = tf.placeholder(tf.float32, shape=[])
    graph_ops["learning_rate"] = learning_rate


    p_values = target_p_network(s)


    graph_ops["target_p"] = p_values


    optimizer_V = tf.train.RMSPropOptimizer(learning_rate, decay = 0.99, epsilon=0.1)
    optimizer_P = tf.train.RMSPropOptimizer(learning_rate, decay = 0.99, epsilon=0.1)

    for thread_id in range(FLAGS.num_concurrent):

        _, v_network, p_network = build_network(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
                                 resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height)

        network_params_v = v_network.trainable_weights
        network_params_p = p_network.trainable_weights

        v_values = v_network(s)
        p_values = p_network(s)


        graph_ops[get_name("v_p", thread_id)] = tf.concat(1,[v_values,p_values])



        # Some of this code has been shamelessly stolen from
        # https://github.com/miyosuda/async_deep_reinforce
        td = tf.sub((R), tf.transpose(v_values))
        td = tf.squeeze(td)

        v_cost =  0.5 * tf.nn.l2_loss(td)
        entropy_beta = 0.01
        log_p = tf.log(tf.clip_by_value(p_values, 1e-20, 1.0))

        entropy = -tf.reduce_sum(p_values * log_p, reduction_indices=1)

        actor_cost = -tf.reduce_sum( tf.reduce_sum( tf.mul( log_p, a ), reduction_indices=1 ) * td + entropy * entropy_beta )


        gvs = optimizer_V.compute_gradients(v_cost, var_list=network_params_v)
        capped_gvs = [(tf.clip_by_norm(gvs[i][0], 40.0), target_network_params_v[i]) for i in range(len(gvs))]
        x_v = optimizer_V.apply_gradients(capped_gvs)

        gvs = optimizer_P.compute_gradients(actor_cost, var_list=network_params_p)
        capped_gvs = [(tf.clip_by_norm(gvs[i][0], 40.0), target_network_params_p[i]) for i in range(len(gvs))]
        x_p = optimizer_P.apply_gradients(capped_gvs)




        update = [x_v, x_p]

        #graph_ops[get_name"actor_cost"] = actor_cost
        graph_ops[get_name("update", thread_id)] = update
        graph_ops[get_name("cost", thread_id)] = v_cost

        copy_from_target = [network_params_v[i].assign(target_network_params_v[i]) for i in range(len(target_network_params_v))] + \
                            [network_params_p[i].assign(target_network_params_p[i]) for i in range(len(target_network_params_p))]
        graph_ops[get_name("copy_from_target", thread_id)] = copy_from_target



    return graph_ops


# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.scalar_summary("Max Q Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.scalar_summary("Epsilon", logged_epsilon)
    logged_T = tf.Variable(0.)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, update_ops, summary_op


def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    # Figure out number of actions from gym env
    env = gym.make(FLAGS.game)
    num_actions = env.action_space.n
    if (FLAGS.game == "Pong-v0" or FLAGS.game == "Breakout-v0"):
        # Gym currently specifies 6 actions for pong
        # and breakout when only 3 are needed. This
        # is a lame workaround.
        num_actions = 3
    return num_actions


def train(session, graph_ops, num_actions, saver):
    # Initialize target network weights


    # Set up game environments (one per thread)
    envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]

    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
    writer = tf.train.SummaryWriter(summary_save_path, session.graph)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    lock = threading.RLock()

    # Start num_concurrent actor-learner training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(lock, thread_id, envs[thread_id], session, graph_ops, num_actions,
                                                    summary_ops, saver)) for thread_id in range(FLAGS.num_concurrent)]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0

    if FLAGS.show_training:
        while True:

            for env in envs:
                env.render()
            now = time.time()
            if now - last_summary_time > FLAGS.summary_interval:
                summary_str = session.run(summary_op)
                writer.add_summary(summary_str, float(T))
                last_summary_time = now
    for t in actor_learner_threads:
        t.join()


def evaluation(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    print "Restored model weights from ", FLAGS.checkpoint_path
    monitor_env = gym.make(FLAGS.game)
    monitor_env.monitor.start(FLAGS.eval_dir + "/" + FLAGS.experiment + "/eval")

    # Unpack graph ops
    target_p = graph_ops["target_p"]
    s = graph_ops["s"]



    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height,
                           agent_history_length=FLAGS.agent_history_length, mode = "test")

    mean = []
    for i_episode in xrange(FLAGS.num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_p = target_p.eval(session=session, feed_dict={s: [s_t]})[0]
            action_index = np.argmax(readout_p)
            #action_index = np.random.choice(len(readout_p), p=readout_p)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        mean.append(ep_reward)
        print "Mean", np.array(mean).mean()
    monitor_env.monitor.close()


def main(_):
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        K.set_session(session)

        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver()

        if FLAGS.testing:
            evaluation(session, graph_ops, saver)
        else:
            train(session, graph_ops, num_actions, saver)


if __name__ == "__main__":
    tf.app.run()
