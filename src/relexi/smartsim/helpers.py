def generate_rankefile_hawk_ompi(hosts: list, cores_per_node: int, n_par_env: int, ranks_per_env: int, base_path=None):
  """Generate rank file for openmpi process binding
  :param host: list of hosts
  :type host: list[str]
  :param cores_per_node: number of cores per node
  :type cores_per_node: int
  :param n_par_env: number of parallel environments
  :type n_par_env: int
  :param ranks_per_env: number of ranks per environments
  :type ranks_per_env: int
  :param base_path: path to the base directory of the rank files
  :type base_path: str
  """

  # If no base_path given, use CWD
  if base_path:
    rankfile_dir = os.path.join(base_path, "ompi-rankfiles")
  else:
    rankfile_dir = "ompi-rankfiles"

  if os.path.exists(rankfile_dir):
    shutil.rmtree(rankfile_dir)
  os.makedirs(rankfile_dir, exist_ok=True)

  rankfiles = list()
  next_free_slot = 0
  n_cores_used = 0
  for env_idx in range(n_par_env):
    filename = os.path.join(rankfile_dir, f"par_env_{env_idx:05d}")
    rankfiles.append(filename)
    with open(filename, 'w') as rankfile:
      for i in range(ranks_per_env):
        rankfile.write(f"rank {i}={hosts[n_cores_used // cores_per_node]} slot={next_free_slot}\n")
        next_free_slot = next_free_slot + 1
        n_cores_used = n_cores_used + 1

        if next_free_slot > ( cores_per_node - 1 ):
          next_free_slot = 0

    files = os.listdir(rankfile_dir)

  return rankfiles


def parser_flexi_parameters(parameter_file, keyword, new_keyword_value):
  """
  """

  pattern = re.compile(r"(%s)\s*=.*" % keyword, re.IGNORECASE)
  subst = keyword + "=" + new_keyword_value
  parameter_file_in = parameter_file
  pbs_jobID = os.environ['PBS_JOBID']
  parameter_file_out = "parameter_flexi-" + pbs_jobID[0:7] + ".ini"

  with open(parameter_file_out,'w') as new_file:
    with open(parameter_file_in, 'r') as old_file:
      for line in old_file:
        new_file.write(pattern.sub(subst, line))

  return parameter_file_out


def clean_ompi_tmpfiles():
  """
  Cleans up temporary files which are created by openmpi in TMPDIR
  Avoids running out of space in TMPDIR
  If TMPDIR is not found exists with -1 status
  """
  try:
    tmpdir = os.environ['TMPDIR']
  except:
    return -1

  path = os.path.join(tmpdir,'ompi.*')
  path = glob.glob(path)

  for folder in path:
    for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      if os.path.isdir(file_path):
        shutil.rmtree(file_path)



def copy_to_nodes(my_files, base_path, hosts, subfolder=None):
  """
  This routine takes the files given in [my_files] and copies them
  to 'base_path' on the ssh targets 'hosts' via the scp command.
  If the path does not exists, it tries to create it via 'mkdir'.
  A optional 'subfolder' can be given, which will be appended to
  the 'base_path'.
  TODO: Implement a fail-safe, i.e. only overwite filepaths for
        which copying worked.
  """

  # If input not a list, i.e. a single element, transform into list
  if isinstance(my_files, list):
    conv_to_list = False
  else:
    my_files = [my_files]
    conv_to_list = True

  # Append subfolder if given
  if subfolder:
    target = os.path.join(base_path, subfolder)
  else:
    target = base_path

  # Copy to all given hosts
  for host in hosts:
    # Create folder if necessary
    os.system('ssh %s mkdir -p %s' % (host, target))
    # Copy files
    for my_file in my_files:
      os.system('scp -q "%s" "%s:%s"' % (my_file, host, target))

  # Get new path of files
  my_files_new = []
  for my_file in my_files:
    file_name = os.path.split(my_file)[1]
    my_files_new.append(os.path.join(target,file_name))

  # Convert back to single string if input is single string
  if conv_to_list:
    my_files_new = my_files_new[0]

  return my_files_new
