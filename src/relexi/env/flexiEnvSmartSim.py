#!/usr/bin/env python3

import io
import csv
import random
import numpy as np
import tensorflow as tf

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from smartsim.settings import MpirunSettings, RunSettings
from smartredis import Client

import relexi.io.output as rlxout

# Try to load matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_FOUND = True
except ImportError:
    MATPLOTLIB_FOUND = False
    rlxout.warning('Could not import Matplotlib. No figures will be created!')


class flexiEnv(py_environment.PyEnvironment):
    """FLEXI environment to be used within TF-Agents.

    This FLEXI environment is implemented as standard python environment,
    which should be compatible with most Reinforcement Learning libraries.
    It leverages the shared FLEXI library to perform simulation steps and
    call the init and finalize routines.

    TODO:
        * Implement base FLEXI environment, from which application-specific
            environments can be subclassed.
    """

    def __init__(self,
                 exp,
                 flexi_path,
                 prm_file,
                 spectra_file,
                 reward_kmin,
                 reward_kmax,
                 reward_scale,
                 n_procs=1,
                 n_envs=1,
                 n_procs_per_node=1,
                 restart_files=None,
                 random_restart_file=True,
                 entry_db="127.0.0.1",
                 port=6780,
                 is_db_cluster=False,
                 debug=0,
                 tag=None,
                 hosts=None,
                 rankfiles=None,
                 mpi_launch_mpmd=False,
                 env_launcher='mpirun'
                 ):
        """Initialize TF and FLEXI specific properties."""
        # Path to FLEXI executable
        self.n_envs = n_envs
        self.n_procs = n_procs
        self.n_procs_per_node = n_procs_per_node
        self.prm_file = prm_file
        self.flexi_path = flexi_path
        self.hosts = hosts
        self.rankfiles = rankfiles

        # Save values for reward function
        self.reward_kmin = reward_kmin
        self.reward_kmax = reward_kmax
        self.reward_scale = reward_scale

        self.e_les = np.zeros((self.n_envs, self.reward_kmax))

        # Sanity Check Launcher
        self.env_launcher = env_launcher
        if ((self.env_launcher == 'local') and (n_procs != 1)):
            rlxout.warning("For env_launcher 'local', only single execution is allowed! Setting 'n_procs=1'!")
            rlxout.warning("To run evironments in parallel with MPI, use env_launcher='mpi'!")
            n_procs = 1

        if self.env_launcher == 'mpirun':
            self.mpi_launch_mpmd = mpi_launch_mpmd
        else:
            self.mpi_launch_mpmd = False

        # Save list of restart files
        self.random_restart_file = random_restart_file
        self.restart_files = restart_files

        # Read target DNS spectra from file
        if spectra_file:
            with open(spectra_file, 'r', encoding='ascii') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                col_e = next(reader).index('E')
                e = []
                for rows in reader:
                    e.append(float(rows[col_e]))
            self.e_dns = e

        # Get experiment handle and port of db
        self.exp = exp
        self.port = port
        # Should be IP address not hostname, since "-.,'" in hostname will cause a crash
        self.entry_db = entry_db
        self.is_db_cluster = is_db_cluster

        # Connect python redis client to an orchestrator database
        self.client = Client(address=f"{self.entry_db}:{str(self.port)}", cluster=self.is_db_cluster)

        # Build tag from tag plus env number
        if tag:
            self.tag = [tag+str(i)+'_' for i in range(self.n_envs)]
        else:
            self.tag = None

        # Startup FLEXI instances inside experiment to get state size
        self.flexi = self._start_flexi(self.exp, self.n_procs, self.n_envs)

        # Get current state from FLEXI environment
        self._state = self._get_current_state()

        # End FLEXI again. Otherwise it will compute the entire run...
        self._end_flexi()

        # Specify action and observation dimensions (neglect first batch dimension)
        self._action_spec = array_spec.ArraySpec(
            shape=(self._state.shape[1],),
            dtype=np.float32, name='action')
        self._observation_spec = array_spec.ArraySpec(
            shape=self._state.shape[1:],
            dtype=np.float32, name='observation')
        self._episode_ended = False

    def __del__(self):
        """Finalize launched FLEXI instances if deleted."""
        self.stop()

    @property
    def batched(self):
        """Override batched property to indicate that environment is batched."""
        return True

    @property
    def batch_size(self):
        """Override batch size property according to chosen batch size."""
        return self.n_envs

    def stop(self):
        """Stops all flexi instances inside launched in this environment."""
        if self.exp:
            self._stop_flexi_instances(self.exp)

    def start(self):
        """Starts all flexi instances with configuration specified in initialization."""
        # Start new FLEXI instance and get initial state
        self.flexi = self._start_flexi(self.exp, self.n_procs, self.n_envs)
        self._state = self._get_current_state()

    def _stop_flexi_instances(self, exp):
        """Stop all FLEXI instances.

        Uses the current SmartSim experiment to loop over all FLEXI instances
        and stop them if they have not finished yet.

        Args:
            exp (smartsim.Experiment): Experiment in which the Orchestrator and
                the FLEXI instances were launched.

        Returns:
            None

        TODO:
            * `exp` should be a property of the class.
        """
        if self.flexi:
            for flexi in self.flexi:
                if not exp.finished(flexi):
                    exp.stop(flexi)

    def _start_flexi(self, exp, n_procs, n_envs):
        """Start FLEXI instances within SmartSim experiment.
            
            Args:
                exp (smartsim.Experiment): Experiment in which the Orchestrator
                    was launched.
                n_procs (int): Number of processors used to run each simulation
                    environment.
                n_envs (int): Number of environments to be launched.

            Returns:
                List of `smartsim` handles for each started FLEXI environment.
        """

        # Build hostlist to specify on which hosts each flexi is started
        # First check: Are there enough free ranks?
        ranks_avail = self.n_procs_per_node*len(self.hosts)
        ranks_needed = n_envs*n_procs
        if ranks_needed > ranks_avail:
            rlxout.warning(f'Only {ranks_avail} ranks are available, but {ranks_needed} would be required '+
                            'to start {n_envs} on {n_procs} each.')

        # Distribute ranks to instances in a round robin fashion
        # TODO: Get ranks directly from hostfile for PBS Orchestrator
        hosts_per_flexi = np.zeros((n_envs, 2), dtype=np.int8)
        n_cores_used = 0
        for i in range(n_envs):
            # 1. Get first node the instance has ranks on
            hosts_per_flexi[i, 0] = n_cores_used // self.n_procs_per_node
            # 2. Increase amount of used cores accordingly
            n_cores_used = n_cores_used + n_procs
            # 3. Get last node the instance has ranks on
            hosts_per_flexi[i,1] = (n_cores_used-1) // self.n_procs_per_node    # last  node

        flexi = []
        # Build list of individual FLEXI instances
        for i in range(n_envs):

            # Select (possibly random drawn) restart file for the run
            if self.random_restart_file:
                restart_file = random.choice(self.restart_files)
            else:
                restart_file = self.restart_files[0]

            args = [self.prm_file, restart_file]
            if self.tag[i]:
                # Tags are given to FLEXI with the Syntax "--tag [value]"
                args.append('--tag')
                args.append(self.tag[i])

            if self.env_launcher == 'mpirun':
                run_args = {
                    "rankfile": self.rankfiles[i],
                    "report-bindings": ""
                }
                run = MpirunSettings(
                    exe=self.flexi_path,
                    exe_args=args,
                    run_args=run_args)
                run.set_tasks(n_procs)

                # Create MPMD Settings and start later in single command
                if self.mpi_launch_mpmd:
                    if i == 0:
                        f_mpmd = run
                    else:
                        f_mpmd.make_mpmd(run)

            else:  # Otherwise do not use launcher
                run = RunSettings(exe=self.flexi_path, exe_args=args)

            # Create and directly start FLEXI instances
            if not self.mpi_launch_mpmd:
                flexi_instance = exp.create_model(self.tag[i]+"flexi", run)
                exp.start(flexi_instance, block=False, summary=False)
                flexi.append(flexi_instance)

        # Create MPMD Model from settings and start
        if self.mpi_launch_mpmd:
            flexi = exp.create_model(self.tag[0]+"flexi", f_mpmd)
            exp.start(flexi, block=False, summary=False)
            flexi = [flexi]

        return flexi

    def _end_flexi(self):
        """Stop FLEXI experiment with SmartSim."""
        for flexi in self.flexi:
            if not self.exp.finished(flexi):
                self.exp.stop(flexi)

    def _reset(self):
        """Resets the FLEXI environment.

        Finalizes FLEXI and restarts it again.

        Returns:
            Start of a trajectory containing the state.

        TODO:
            This is now done outside of the reset function by calling the
            functions "start()" and "stop()" manually. This function is thus
            deprecated.
        """

        # Close FLEXI instance
        # self._end_flexi()
        self._episode_ended = False

        # Start new FLEXI instance and get initial state
        # self.flexi = self._start_flexi(self.exp,self.n_procs,self.n_envs)
        # self._state = self._get_current_state()

        return ts.restart(self._state, batch_size=self.n_envs)

    def _step(self, action):
        """Performs single step in the FLEXI environment.

        The actions of the agents based on the current flow state are written
        to Orchestrator to be read by FLEXI. Then, timestepping is performed
        and the new flow state is obtained together with the quantities to
        computed reward.

        Args:
            action: Array containing current predictions of agent.

        Returns:
            Transition containing (state, reward, discount)
        ."""
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        # Update Prediction
        self._set_prediction(action)

        # Poll New State
        # ATTENTION: HERE THE FLEXI TIMESTEPPING OCCURS!
        self._state = self._get_current_state()

        # Get Reward
        reward = self._get_reward()

        # Determine if simulation finished
        self._episode_ended = self._flexi_ended()

        # Return termination of ended
        if self._episode_ended:
            return ts.termination(self._state, reward)

        # Discount is later multiplied with global discount from user input
        return ts.transition(self._state, reward, discount=np.ones((self.n_envs,)))

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _set_prediction(self, action):
        """Write action for current state to database for FLEXI client.
            
            Args:
                action (np.array): Array containing the agent's action for each
                    environment.

            Returns:
                None
        """
        # Scale actions to [0,0.5]
        # TODO: make this a user parameter
        action_mod = action * 0.5
        for i in range(self.n_envs):
            _ = self.client.put_tensor(self.tag[i]+"actions", action_mod[i,::].astype(np.float64))

    def _get_current_state(self):
        """Get current flow state from the database.
        
        Returns:
            Array containing states of all environments

        ATTENTION: This is the routine the enviroment will idle until the
                   necessary data becomes available
        """
        do_init = True
        key = "state"
        for tag in self.tag:
            self.client.poll_tensor(tag+key, 10, 10000)
            try:
                data = self.client.get_tensor(tag+key)
            except Exception:
                rlxout.warning(f"Did not get state from environment {tag[:-1]}")
            self.client.delete_tensor(tag+key)
            # Account for Fortran/C memory layout and 32bit for TF
            data = np.transpose(data)
            data = np.expand_dims(data, axis=0)
            if do_init:
                state = data
                do_init = False
            else:
                state = np.append(state, data, axis=0)

        return state

    def _flexi_ended(self):
        """Checks if FLEXI instances are still running.

        Returns:
            List containing boolean flag for each environment indicating
            whether it has already finished.
        """
        has_ended = np.empty((self.n_envs))

        key = "step_type"
        for i in range(self.n_envs):
            self.client.poll_tensor(self.tag[i]+key, 10, 1000)
            step_type = self.client.get_tensor(self.tag[i]+key)
            self.client.delete_tensor(self.tag[i]+key)
            if step_type > 0:
                has_ended[i] = False
            else:
                has_ended[i] = True
        return np.all(has_ended)

    def _get_reward(self):
        """Compute the reward for the agent, based on the current flow state.

        Returns:
            Array containing scalar reward for each environment.
        """
        reward = np.zeros((self.n_envs,))
        key = "Ekin"
        for i in range(self.n_envs):
            # Poll Tensor
            self.client.poll_tensor(self.tag[i]+key, 10, 1000)
            data = self.client.get_tensor(self.tag[i]+key)
            self.client.delete_tensor(self.tag[i]+key)
            self.e_les[i, :] = data[0:self.reward_kmax]
            # Compute Reward
            error = self.e_les[i, self.reward_kmin-1:self.reward_kmax] - self.e_dns[self.reward_kmin-1:self.reward_kmax]
            error = error/self.e_dns[self.reward_kmin-1:self.reward_kmax]
            error = np.square(error)
            reward[i] = 2.*np.exp(-1.*np.mean(error)/self.reward_scale)-1.
        return reward

    @property
    def can_plot(self):
        """Ability to generate an image if te current state.

        Returns:
            Boolean indicating ability to plot.
        """
        return MATPLOTLIB_FOUND

    def plot(self, k_max=None):
        """Wrapper routine for plotting energy spectra of environments.
        
        Args:
            k_max (int): Maximum wavenumber up to which plot will be generated.

        Returns:
            TF-compatible buffer with image that can be passed to Tensorboard.
        """
        if not MATPLOTLIB_FOUND:
            raise ImportError("Matplotlib was not found but is necesasry to generate plots for Tensorboard!")

        if k_max:
            k_plot = k_max
        else:
            k_plot = self.reward_kmax
        return self._plot_spectra(self.e_dns[:], self.e_les[0, :], k_max=k_plot)

    def _plot_spectra(self, e_dns, e_les, k_max):
        """Plots current LES and DNS energy spectra.

        Args:
            e_dns (list): List with energy contained in each wavenumber for DNS
            e_les (list): List with energy contained in each wavenumber for LES
            k_max (int): Upper limit of range to plot.

        Returns:
            TF-compatible buffer with image that can be passed to Tensorboard.
        """
        # Plot Spectra
        fig = plt.figure(figsize=(6, 5))
        k = range(1, k_max+1)
        plt.plot(k, e_dns[0:k_max], label='DNS', linestyle='-', linewidth=2)
        plt.plot(k, e_les[0:k_max], label='LES', linestyle='-', linewidth=2)
        plt.xlabel('k', fontsize=21)
        plt.ylabel(r'$E_{kin}$', fontsize=21)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1, k_max)
        plt.ylim(4.e-3, 1.e0)
        plt.legend(fontsize=12)
        plt.grid(which='both')
        plt.tight_layout()

        # Buffer image as png in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Convert PNG buffer to TF image and add batch dimension
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        return image
