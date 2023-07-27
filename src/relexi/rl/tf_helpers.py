import tensorflow as tf
import relexi.io.output as rlxout
import relexi.smartsim.helpers

def write_metrics(metrics,step,category_name):
  """
  Adds all metrics in the list 'metrics' to the tf summary, which will be written to disk.
  'step' gives the corresponding global index and 'category_name' sets a prefix, which
  helps to group metrics in TensorBoard.
  """
  for metric in metrics:
    tf.summary.scalar(category_name+"/"+metric.name,metric.result().numpy(),step=step)


def init_gpus(num_gpus=1,gpu_memory_growth=False):
  """
  Check if tensorflow finds a GPU to run on and employs a distribution strategy
  on multi-GPU systems
  """
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      rlxout.printNotice('Found '+str(len(gpus))+' physical GPU(s) on system.')
      # Set Memory growth
      if gpu_memory_growth:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True) # Allocate only neccessary memory

      # Check if enough GPUs available
      if (num_gpus > len(gpus)):
        rlxout.printWarning('Requested more GPUs than available on the system. Use '+str(len(gpus))+' GPUs instead.')
        num_gpus = len(gpus)

      # Get Distribution Strategy
      if (num_gpus == 1):
        rlxout.printNotice('Running on single GPU. To run on multiple GPUs use commandline argument "-n NUM_GPUS"')
        return None
      else:
        devices = []
        for i in range(num_gpus):
          devices.append("/gpu:"+str(i))

        rlxout.printNotice('Running Mirrored Distribution Strategy on GPUs: '+",".join(devices))
        return tf.distribute.MirroredStrategy(devices=devices,cross_device_ops=tf.distribute.NcclAllReduce())

    except RuntimeError as e:
      rlxout.printWarning(e) # Memory growth must be set before GPUs have been initialized
  else:
    rlxout.printWarning('No GPU found on system, Tensorflow will probably run on CPU')
    return None

@tf.function
def distributed_train_step(agent,experience):
  return agent.train(experience).loss

def train_agent_distributed(agent,replay_buffer,strategy):
  with strategy.scope():
    # Get Dataset
    # TODO: Currently, we simply take all, thus num_steps and sample_batch_size have to simply be very large
    #       Implement this more cleanly
    dataset = replay_buffer.as_dataset(num_steps=50,sample_batch_size=512,single_deterministic_pass=True)
    # Set correct sharding policy to avoid tensorflow warnings
    dataset_options = tf.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset.with_options(dataset_options)
    # Generate distributed dataset from dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    # Get first element (we only have a single one...)
    dist_dataset_iterator = iter(dist_dataset)
    experience,_ = dist_dataset_iterator.get_next()
    # Train distributed
    per_replica_losses = strategy.run(distributed_train_step, args=(agent,experience,))
    #total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

  # Clear replay_buffer
  replay_buffer.clear()
  return


@tf.function
def train_agent(agent,replay_buffer):
  # Get Dataset
  dataset = replay_buffer.gather_all()
  # Train agent
  agent.train(experience=dataset)
  # Clear replay_buffer
  replay_buffer.clear()
  return


#@tf.function
def collect_trajectories(driver,env):
  # Startup FLEXI instances
  env.start()
  # Run FLEXI instances
  driver_run(driver)
  # Stop FLEXI instances
  env.stop()
  # Cleanup OMP files
  relexi.smartsim.helpers.clean_ompi_tmpfiles()
  return

@tf.function
def driver_run(driver):
  final_time_step, policy_state = driver.run()
