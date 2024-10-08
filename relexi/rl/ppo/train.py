#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Relexi, a reinforcement learning framework for training
# machine learning models in simulations on high-performance computing systems.
#
# Copyright (c) 2022-2024 Marius Kurz, Andrea Beck
#
# Relexi is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Relexi is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Relexi. If not, see <http://www.gnu.org/licenses/>.


"""Implements main PPO training routine for Relexi."""

import os
import time
import datetime
import random
import copy
import contextlib

import tensorflow as tf

from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import relexi.rl.models
import relexi.rl.tf_helpers
import relexi.env
import relexi.io.readin as rlxin
import relexi.io.output as rlxout


def train( config_file
          ,runtime
          ,parameter_file
          ,executable_path
          ,train_files
          ,train_learning_rate
          ,discount_factor
          ,reward_spectrum_file
          ,reward_scale
          ,reward_kmax
          ,env_name = 'hit'
          ,reward_kmin  = 1
          ,run_name     = None # Output directory will be named accordingly
          ,restart_dir  = None # Directory with checkpoints to restart from
          ,random_seed  = None
          ,log_interval = 1
          ,num_procs_per_environment  = 1
          ,num_parallel_environments  = 1
          ,train_num_epochs           = 5
          ,train_num_iterations       = 1000
          ,train_buffer_capacity      = 1000
          ,entropy_regularization     = 0.0
          ,importance_ratio_clipping  = 0.2
          ,action_std        = 0.02
          ,actor_type        = 'cnn_actor'
          ,critic_type       = 'cnn_critic'
          ,dist_type         = 'normal'
          ,ckpt_interval     = 5
          ,ckpt_num          = 1000
          ,eval_files        = None
          ,eval_num_episodes = 1
          ,eval_interval     = 5
          ,use_XLA     = False
          ,do_profile  = False
          ,env_launcher = 'mpirun'
          ,mpi_launch_mpmd = False
          ,strategy = None
          ,debug = 0
          ,**kwargs
        ):
    """
    Main training routine. Here, the (FLEXI) environment, the art. neural networks, the optimizer,...
    are instantiated and initialized. Then the main training loop iteratively collects trajectories
    and trains the agent on the sampled trajectories.
    """

    # Set output dir to restart dir, if given.
    # TODO: make this more robust
    if restart_dir is not None:
        base_dir = restart_dir

    # Otherwise, use run_name as output dir if given
    elif run_name is not None:
        base_dir = "logs/"+run_name

    # Else, use current time stamp
    else:
        base_dir = datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        base_dir = "logs/"+base_dir

    # Set all output directories of the run accordingly
    train_dir = base_dir+"/train/"
    save_dir  = base_dir+"/models/"
    ckpt_dir  = base_dir+"/ckpt/"

    # Check if all necessary files actually exist
    missing_files = rlxin.files_exist([executable_path,parameter_file,train_files,eval_files,reward_spectrum_file])
    for item in missing_files:
        rlxout.warning("The specified file "+item+" does not exist")

    # Activate XLA for performance
    if use_XLA:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
        tf.config.optimizer.set_jit(True)

    # Load environment
    env_class = relexi.env.load_environment(env_name)

    # Instantiate parallel collection environment
    my_env = tf_py_environment.TFPyEnvironment(
             env_class(runtime
                       ,executable_path
                       ,parameter_file
                       ,tag              = 'train'
                       ,n_envs           = num_parallel_environments
                       ,n_procs          = num_procs_per_environment
                       ,spectra_file     = reward_spectrum_file
                       ,reward_kmin      = reward_kmin
                       ,reward_kmax      = reward_kmax
                       ,reward_scale     = reward_scale
                       ,restart_files    = train_files
                       ,env_launcher     = env_launcher
                       ,mpi_launch_mpmd  = mpi_launch_mpmd
                       ,debug            = debug
                       ))

    # Instantiate serial evaluation environment
    if eval_files is None:
        rlxout.warning('No specific Files for Evaluation specified. Using Training files instead')
        eval_files = train_files

    my_eval_env = tf_py_environment.TFPyEnvironment(
                  env_class(runtime
                            ,executable_path
                            ,parameter_file
                            ,tag              = 'eval'
                            ,n_procs          = num_procs_per_environment
                            ,spectra_file     = reward_spectrum_file
                            ,reward_kmin      = reward_kmin
                            ,reward_kmax      = reward_kmax
                            ,reward_scale     = reward_scale
                            ,restart_files    = eval_files
                            ,random_restart_file = False
                            ,env_launcher     = env_launcher
                            ,debug            = debug
                            ))

    # Get training variables
    optimizer   = tf.keras.optimizers.Adam(learning_rate=train_learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # For distribution strategy, networks and agent have to be initialized within strategy.scope
    if strategy:
        context = strategy.scope()
    else:
        context = contextlib.nullcontext()  # Placeholder that does nothing

    with context:
        # Set TF random seed within strategy to obtain reproducible results
        if random_seed:
            random.seed(random_seed)        # Python seed
            tf.random.set_seed(random_seed)  # TF seed

        # Get required network architecture and instantiate
        actor = relexi.rl.models.from_string(actor_type)
        actor_net = actor(my_env.observation_spec()
                         ,my_env.action_spec()
                         ,action_std=action_std
                         ,dist_type=dist_type
                         ,debug=debug)
        critic = relexi.rl.models.from_string(critic_type)
        value_net = critic( my_env.observation_spec()
                           ,debug=debug)

        # PPO Agent
        tf_agent = ppo_clip_agent.PPOClipAgent(
                my_env.time_step_spec(),
                my_env.action_spec(),
                optimizer,
                actor_net=actor_net,
                value_net=value_net,
                importance_ratio_clipping=importance_ratio_clipping,
                discount_factor=discount_factor,
                entropy_regularization=entropy_regularization,
                normalize_observations=False,
                normalize_rewards=False,
                use_gae=True,
                num_epochs=train_num_epochs,
                train_step_counter=global_step)
        tf_agent.initialize()

    # Get Agent's Policies
    eval_policy    = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    # Instantiate Replay Buffer, which holds the sampled trajectories.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                       data_spec  = tf_agent.collect_data_spec,
                       batch_size = my_env.batch_size,
                       max_length = train_buffer_capacity)

    # Currently sampling in several times not supported
    num_episodes_per_iteration=num_parallel_environments

    # Instantiate driver for data collection
    num_episodes = tf_metrics.NumberOfEpisodes(name='num_episodes')
    env_steps    = tf_metrics.EnvironmentSteps(name='num_env_steps')
    avg_return   = tf_metrics.AverageReturnMetric(  name='avg_return'
                                                   ,buffer_size=num_episodes_per_iteration
                                                   ,batch_size=num_parallel_environments)
    min_return   = tf_metrics.MinReturnMetric(      name='min_return'
                                                   ,buffer_size=num_episodes_per_iteration
                                                   ,batch_size=num_parallel_environments)
    max_return   = tf_metrics.MaxReturnMetric(      name='max_return'
                                                   ,buffer_size=num_episodes_per_iteration
                                                   ,batch_size=num_parallel_environments)
    train_metrics  = [num_episodes
                     ,env_steps
                     ,avg_return
                     ,min_return
                     ,max_return
                     ]

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(my_env
                                                                ,collect_policy
                                                                ,observers = train_metrics + [replay_buffer.add_batch]
                                                                ,num_episodes=num_episodes_per_iteration
                                                                )

    # Instantiate driver for evaluation
    eval_avg_eplen  = tf_metrics.AverageEpisodeLengthMetric(name='eval_avg_episode_length',buffer_size=eval_num_episodes)
    eval_avg_return = tf_metrics.AverageReturnMetric(       name='eval_avg_return'        ,buffer_size=eval_num_episodes)
    eval_metrics    = [eval_avg_eplen,eval_avg_return]
    eval_driver     = dynamic_episode_driver.DynamicEpisodeDriver(my_eval_env
                                                                 ,eval_policy
                                                                 ,observers = eval_metrics
                                                                 ,num_episodes=eval_num_episodes
                                                                 )
    # Write summary to TensorBoard
    summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=1000)

    # Define checkpointer to save policy
    train_checkpointer = common.Checkpointer(
                                 ckpt_dir=ckpt_dir,
                                 max_to_keep=ckpt_num,
                                 agent=tf_agent,
                                 policy=tf_agent.policy,
                                 global_step=global_step)
    train_checkpointer.initialize_or_restore()

    # Main train loop
    rlxout.banner('Starting Training Loop!')
    with summary_writer.as_default():
        if do_profile:
            rlxout.info('Starting profiling....')
            tf.profiler.experimental.start(train_dir)

        start_time = time.time()

        # In case of restart don't start counting at 0
        starting_iteration = int(global_step.numpy()/train_num_epochs) + 1

        # Write parameter files to Tensorboard
        tf.summary.text("training_config"
                       ,rlxin.read_file(config_file,newline='  \n') # TF uses markdown EOL
                       ,step=starting_iteration)
        tf.summary.text("flexi_config"
                       ,rlxin.read_file(parameter_file,newline='  \n') # TF uses markdown EOL
                       ,step=starting_iteration)

        # Run initial evaluation
        mytime = time.time()
        relexi.rl.tf_helpers.collect_trajectories(eval_driver,my_eval_env)
        # Log to console
        rlxout.info(f'Initial eval time: [{time.time()-mytime:5.2f}s]')
        rlxout.info(f'Initial eval return: {eval_avg_return.result().numpy():.6f}', newline=False)
        # Only write to TB and checkpoint if starting from scratch to not overwrite previous data
        if starting_iteration == 1:
            # Write metrics and plot to Tensorboard
            relexi.rl.tf_helpers.write_metrics(eval_metrics,global_step,'MetricsEval')
            if my_eval_env.can_plot:
                tf.summary.image("Environment", my_eval_env.plot(), step=global_step)
            # Save
            rlxout.info('Saving initial checkpoint to: ' + ckpt_dir)
            train_checkpointer.save(global_step)
            rlxout.info('Saving initial model to: ' + save_dir)
            actor_net.model.save(os.path.join(save_dir,f'model_{0:06d}'))

        # Actual train loop
        for i in range(starting_iteration ,train_num_iterations + 1):
            # Collect trajectories, i.e. run simulations
            mytime = time.time()
            relexi.rl.tf_helpers.collect_trajectories(collect_driver,my_env)
            relexi.rl.tf_helpers.write_metrics(train_metrics,global_step,'MetricsTrain')
            collect_time = time.time()-mytime

            # Train agent on collected trajectories
            mytime = time.time()
            if strategy:
                relexi.rl.tf_helpers.train_agent_distributed(tf_agent,replay_buffer,strategy)
            else:
                relexi.rl.tf_helpers.train_agent(tf_agent,replay_buffer)
            train_time = time.time()-mytime

            # Log to console every log_interval iterations
            if (i % log_interval) == 0:
                rlxout.small_banner(f'ITERATION {i}')
                rlxout.info(f'Episodes:    {num_episodes.result().numpy()}',      newline=False)
                rlxout.info(f'Env. Steps:  {env_steps.result().numpy()}',         newline=False)
                rlxout.info(f'Train Steps: {tf_agent.train_step_counter.numpy()}',newline=False)

                total_time=time.time()-start_time
                rlxout.info(f'Sample time: [{collect_time:5.2f}s]')
                rlxout.info(f'Train time:  [{train_time:5.2f}s]',newline=False)
                rlxout.info(f'TOTAL:       [{total_time:5.2f}s]',newline=False)

            # Run eval every eval_interval iterations
            if (i % eval_interval) == 0:
                mytime = time.time()
                relexi.rl.tf_helpers.collect_trajectories(eval_driver,my_eval_env)
                # Log to console
                rlxout.info(f'Eval time: [{time.time()-mytime:5.2f}s]')
                rlxout.info(f'Eval return: {eval_avg_return.result().numpy():.6f}', newline=False)
                # Write metrics and plot to Tensorboard
                relexi.rl.tf_helpers.write_metrics(eval_metrics,global_step,'MetricsEval')
                if my_eval_env.can_plot:
                    tf.summary.image("Environment", my_eval_env.plot(), step=global_step)

            # Checkpoint the policy every ckpt_interval iterations
            if (i % ckpt_interval) == 0:
                rlxout.info('Saving checkpoint to: ' + ckpt_dir)
                train_checkpointer.save(global_step)
                rlxout.info('Saving current model to: ' + save_dir, newline=False)
                actor_net.model.save(os.path.join(save_dir,f'model_{global_step.numpy():06d}'))

            # Flush summary to TensorBoard
            tf.summary.flush()

    if do_profile:
        rlxout.info('End profiling.')
        tf.profiler.experimental.stop()

    # Close all
    del my_env
    del my_eval_env
