import itertools
import os
import random
import time

def compute_hamming_distance(dict_a, dict_b, exclude_keys = None):
  # compute the hamming distance between two dictionaries
  # (the number of key[values] that are not identical)

  # get the keys that are in both dictionaries:
  keys = set(dict_a.keys()).intersection(set(dict_b.keys()))

  # count the number of keys that are not identical:
  distance = 0
  for k in keys:
      if dict_a[k] != dict_b[k] and (exclude_keys is None or k not in exclude_keys):
          distance += 1

  return distance

def compute_min_hamming_distance(dict_a, list_of_dicts, exclude_keys = None):
  # compute the maximum hamming distance between a dictionary and a list of dictionaries
  # (the maximum number of key[values] that are not identical)

  # count the number of keys that are not identical:
  min_distance = 10000
  for dict_b in list_of_dicts:
      distance = compute_hamming_distance(dict_a, dict_b, exclude_keys = exclude_keys)
      if distance < min_distance:
          min_distance = distance

  return min_distance





def run_lora_experiment(param_grid, 
      n=1000, test = 0, 
      dirname = "grid_search_results", 
      seed = None, 
      hamming_distance_to__skip = 1,  # if the number of keys that are different between two experiments is less than this value, then skip the experiment
      global_experiments_run = None # provide an external list of experiments that have already been run
      ):

  if seed is not None:
      random.seed(seed)
  else:
      random.seed(int(time.time()))

  # Split the parameter grid into fixed and variable arguments
  fixed_args = {k: v for k, v in param_grid.items() if isinstance(v, (int, str, bool, float))}
  variable_args = {k: v for k, v in param_grid.items() if k not in fixed_args}

  # Generate all combinations of variable arguments
  variable_values = list(itertools.product(*[v if not isinstance(v, list) else [v] for v in variable_args.values()]))
  variable_keys = list(variable_args.keys())

  # Generate a long list of grid_values by randomly sampling each argument list
  long_grid_values = []
  for i in range(10000):
      values = {}
      for k in variable_keys:
          if isinstance(param_grid[k], list):
              values[k] = random.choice(param_grid[k])
          else:
              values[k] = param_grid[k]
      long_grid_values.append(values)
  
  # Randomly sample a subset of the long list of grid_values
  grid_values = random.sample(long_grid_values, n)

  # Combine fixed and variable arguments into a single dictionary
  grid_values = [{**fixed_args, **values} for values in grid_values]

  # Define the command to execute your Python job with input arguments
  cmd = 'python lora_diffusion/cli_lora_pti.py'

  # shuffle the grid values ordering:
  random.shuffle(grid_values)

  if global_experiments_run is None:
      experiments_run = []
  else:
      experiments_run = global_experiments_run.copy()

  # Loop over the grid values and execute the Python job with each combination of input arguments
  for i, values in enumerate(grid_values[:n]):
    if len(experiments_run) > 0:
      d = compute_min_hamming_distance(values, experiments_run, exclude_keys = ['output_dir'])
      if d <= hamming_distance_to__skip:
        print(f"Skipping experiment because it is too similar (d = {d} < {hamming_distance_to__skip}) to a previous experiment")
        continue

    experiments_run.append(values.copy())

    # get the datadirectory name:
    data_dir = "_".join(values['instance_data_dir'].split('/')[-2:])

    # generate a short, pseudorandom character id for this run:
    id_str = ''.join(random.choice('0123456789abcdef') for i in range(6))

    values['output_dir'] = f"./exps/{dirname}/{data_dir}_{i:02d}_{id_str}"

    arg_str = ' '.join([f'--{k} {v}' for k, v in values.items()])
    full_cmd = f'{cmd} {arg_str}'
    print('------------------------------------------')
    print(f'Running command: {i+1}/{n}')
    print(full_cmd)

    if 0:
      # pretty print the values dictionary:
      for k, v in values.items():
        print(f'{k}:{" "*(50-len(k))}{v}')

    if not test:
      os.system(full_cmd)

  return experiments_run



"""

export CUDA_VISIBLE_DEVICES=3
cd /home/xander/Projects/cog/lora
nohup python grid_train_lora.py &


export CUDA_VISIBLE_DEVICES=3
cd /home/xander/Projects/cog/lora
python grid_train_lora.py


"""


################################################################################################################################


if 0: # Style
  output_directory = "gordon-lora-768_final"
  training_dirs = [
    "/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/gordon_02/portraits",
    "/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/gordon_02/Paintings",
    "/home/xander/Projects/cog/eden-sd-pipelines/eden/xander/assets/gordon_02/Drawings"
    ]

  param_grid = {
    'pretrained_model_name_or_path': ['runwayml/stable-diffusion-v1-5'],
    'instance_data_dir':             "",

    'train_text_encoder':            True,
    'perform_inversion':             True,
    'learning_rate_ti':              [1e-4],
    'continue_inversion':            True,
    'continue_inversion_lr':         [0.5e-5],
    'learning_rate_unet':            [0.5e-5],
    'learning_rate_text':            [1.5e-5],
    'save_steps':                    300,
    'max_train_steps_ti':            [400], 
    'max_train_steps_tuning':        [3000], 
    'weight_decay_ti':               [0.0005],
    'weight_decay_lora':             [0.0010],
    'lora_rank_unet':                [4],
    'lora_rank_text_encoder':        [8],
    'use_extended_lora':             [True],

    'use_face_segmentation_condition': False,
    'use_mask_captioned_data':       False,
    'placeholder_tokens':            ["\"<s1>\""],
    'clip_ti_decay':                 True,

    'cached_latents':                False,
    'train_batch_size':              4,
    'gradient_accumulation_steps':   1,
    'color_jitter':                  True,
    'scale_lr':                      True,
    'lr_scheduler':                  "linear",
    'lr_warmup_steps':               0,

    'resolution':                    [768],
    'enable_xformers_memory_efficient_attention': True,

  }

  global_experiments_run = []
  print('#####################################################')

  from collections import Counter

  for i in range(10000):

    training_dir = training_dirs[i % len(training_dirs)]
    param_grid['instance_data_dir'] = training_dir

    global_experiments_run = run_lora_experiment(param_grid, n=1, 
            dirname = output_directory, 
            test=0,
            hamming_distance_to__skip=0,
            global_experiments_run = global_experiments_run, 
            seed = int(time.time()))

    # Print some info:
    print("Total n exp run: ", len(global_experiments_run))
    # Loop over all the run experiments and get their training_dir:
    training_dirs_experimented = []
    for exp in global_experiments_run:
      training_dirs_experimented.append(exp['instance_data_dir'])

    # Count the number of times each training_dir appears:
    training_dir_counts = Counter(training_dirs_experimented)
    print(training_dir_counts)




"""

export CUDA_VISIBLE_DEVICES=2
cd /home/xander/Projects/cog/lora
nohup python grid_train_lora.py &


"""


if 1: # Person SLOW
  input_dir  = "/home/xander/Projects/cog/lora/exps/training_imgs/hetty"
  output_directory = "hetty_lora"

  subdir_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, f))]
  training_dirs = [os.path.join(f, "train") for f in subdir_paths]

  print("Training LORA on:")
  for f in training_dirs:
    print(f)

  param_grid = {
    'pretrained_model_name_or_path': ['dreamlike-art/dreamlike-photoreal-2.0'],
    'instance_data_dir':             "",

    'train_text_encoder':            True,
    'perform_inversion':             True,
    'learning_rate_ti':              [2e-4],
    'continue_inversion':            True,
    'continue_inversion_lr':         [1e-5],
    'learning_rate_unet':            [1e-5],
    'learning_rate_text':            [2.5e-5],

    'save_steps':                    200,
    'max_train_steps_ti':            [350], 
    'max_train_steps_tuning':        [800], 
    'weight_decay_ti':               [0.0010],
    'weight_decay_lora':             [0.0015],
    'lora_rank_unet':                [2],
    'lora_rank_text_encoder':        [8],
    'use_extended_lora':             [False],

    'use_face_segmentation_condition': True,
    'use_mask_captioned_data':       False,
    'placeholder_tokens':            ["\"<person1>\""],
    'proxy_token':                   "person",
    'use_template':                  "person",
    'initializer_tokens':            [None],
    'clip_ti_decay':                 True,
    'load_pretrained_inversion_embeddings_path': [None],
    'cached_latents':                False,
    'train_batch_size':              [6],
    'gradient_accumulation_steps':   1,
    'color_jitter':                  True,
    'scale_lr':                      True,
    'lr_scheduler':                  "linear",
    'lr_warmup_steps':               0,
    'resolution':                    [512, 640],
    'enable_xformers_memory_efficient_attention': True,

  }

  global_experiments_run = []
  print('#####################################################')

  from collections import Counter

  for i in range(10000):

    training_dir = training_dirs[i % len(training_dirs)]
    param_grid['instance_data_dir'] = training_dir

    global_experiments_run = run_lora_experiment(param_grid, n=1, 
            dirname = output_directory, 
            test=0,
            hamming_distance_to__skip=0,
            global_experiments_run = global_experiments_run, 
            seed = int(time.time()))

    # Print some info:
    print("Total n exp run: ", len(global_experiments_run))
    # Loop over all the run experiments and get their training_dir:
    training_dirs_experimented = []
    for exp in global_experiments_run:
      training_dirs_experimented.append(exp['instance_data_dir'])

    # Count the number of times each training_dir appears:
    training_dir_counts = Counter(training_dirs_experimented)
    print(training_dir_counts)




