# import logging
# import os
# import pprint
# import threading
# import time
# import timeit
# import traceback
# import typing
# from absl import app
# from absl import flags
# os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

# import torch
# from torch import multiprocessing as mp
# from torch import nn
# from torch.nn import functional as F

# from torchbeast.core import environment
# from torchbeast.core import file_writer
# from torchbeast.core import prof
# from torchbeast.core import vtrace

# # SC stuff
# from pysc2.env import sc2_env
# from SC_Utils.game_utils import IMPALA_ObsProcesser_v2, FullObsProcesser
# from AC_modules.IMPALA import IMPALA_AC_v2
# import absl 
# import sys
# import numpy as np

# # Define flags using absl.flags
# FLAGS = flags.FLAGS

# # Game arguments
# flags.DEFINE_integer('res', 32, 'Screen and minimap resolution')
# flags.DEFINE_string('map_name', 'MoveToBeacon', 'Name of the minigame')
# flags.DEFINE_boolean('select_all_layers', True, 'If True, selects all useful layers of screen and minimap')
# flags.DEFINE_list('screen_names', ['visibility_map', 'player_relative', 'selected', 'unit_density', 'unit_density_aa'],
#                   'List of strings containing screen layers names to use. Overridden by select_all_layers=True')
# flags.DEFINE_list('minimap_names', ['visibility_map', 'camera'],
#                   'List of strings containing minimap layers names to use. Overridden by select_all_layers=True')
# flags.DEFINE_list('action_names', ['no_op', 'move_camera', 'select_point', 'select_rect', 'select_idle_worker',
#                                    'select_army', 'Attack_screen', 'Attack_minimap', 'Build_Barracks_screen',
#                                    'Build_CommandCenter_screen', 'Build_Refinery_screen', 'Build_SupplyDepot_screen',
#                                    'Harvest_Gather_SCV_screen', 'Harvest_Return_SCV_quick', 'HoldPosition_quick',
#                                    'Move_screen', 'Move_minimap', 'Rally_Workers_screen', 'Rally_Workers_minimap',
#                                    'Train_Marine_quick', 'Train_SCV_quick'],
#                   'List of strings containing action names to use.')

# # Agent arguments
# flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')
# flags.DEFINE_string('xpid', None, 'Experiment id (default: None).')

# # Training settings.
# flags.DEFINE_boolean('disable_checkpoint', False, 'Disable saving checkpoint.')
# flags.DEFINE_string('savedir', './logs/torchbeast', 'Root dir where experiment data will be saved.')
# flags.DEFINE_integer('num_actors', 4, 'Number of actors (default: 4).')
# flags.DEFINE_integer('total_steps', 12000, 'Total environment steps to train for.')
# flags.DEFINE_integer('batch_size', 8, 'Learner batch size.')
# flags.DEFINE_integer('unroll_length', 60, 'The unroll length (time dimension).')
# flags.DEFINE_integer('num_buffers', None, 'Number of shared-memory buffers.')
# flags.DEFINE_integer('num_learner_threads', 1, 'Number learner threads.')
# flags.DEFINE_boolean('disable_cuda', False, 'Disable CUDA.')

# # Loss settings.
# flags.DEFINE_float('entropy_cost', 0.0005, 'Entropy cost/multiplier.')
# flags.DEFINE_float('baseline_cost', 0.5, 'Baseline cost/multiplier.')
# flags.DEFINE_float('discounting', 0.99, 'Discounting factor.')
# flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'none'], 'Reward clipping.')

# # Optimizer settings.
# flags.DEFINE_string('optim', 'RMSprop', 'Optimizer. Choose between RMSprop and Adam.')
# flags.DEFINE_float('learning_rate', 0.0007, 'Learning rate.')
# flags.DEFINE_float('alpha', 0.99, 'RMSProp smoothing constant.')
# flags.DEFINE_float('momentum', 0.0, 'RMSProp momentum.')
# flags.DEFINE_float('epsilon', 0.01, 'RMSProp epsilon.')
# flags.DEFINE_float('grad_norm_clipping', 40.0, 'Global gradient norm clip.')

# # New argument for checkpoint path
# flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint to resume training from.')

# logging.basicConfig(
#     format=(
#         "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
#     ),
#     level=0,
# )

# Buffers = typing.Dict[str, typing.List[torch.Tensor]] 

# def init_game(game_params, map_name='MoveToBeacon', step_multiplier=8, **kwargs):

#     race = sc2_env.Race(1) # 1 = terran
#     agent = sc2_env.Agent(race, "Testv0") # NamedTuple [race, agent_name]
#     agent_interface_format = sc2_env.parse_agent_interface_format(**game_params) #AgentInterfaceFormat instance

#     game_params = dict(map_name=map_name, 
#                        players=[agent], # use a list even for single player
#                        game_steps_per_episode = 0,
#                        step_mul = step_multiplier,
#                        agent_interface_format=[agent_interface_format] # use a list even for single player
#                        )  
#     env = sc2_env.SC2Env(**game_params, **kwargs)

#     return env

# def compute_baseline_loss(advantages):
#     return 0.5 * torch.sum(advantages ** 2)

# def compute_policy_gradient_loss(log_prob, advantages):
#     log_prob = log_prob.view_as(advantages)
#     return - torch.sum(log_prob * advantages.detach())


# def init_flags_for_multiprocessing(argv):
#     """
#     This function reinitializes absl.flags in the subprocesses.
#     It ensures that the flags are parsed again inside the child processes.
#     """
#     # Reinitialize absl.flags for the new process (required in multiprocessing)
#     if not FLAGS.is_parsed():
#         FLAGS(argv)  # Parse the flags again inside the subprocess


# def act(
#     actor_index: int,
#     free_queue: mp.SimpleQueue,
#     full_queue: mp.SimpleQueue,
#     model: torch.nn.Module,
#     buffers: Buffers,
#     initial_agent_state_buffers,
# ):
#     try:
#         init_flags_for_multiprocessing(sys.argv)

#         logging.info("Actor %i started.", actor_index)
#         timings = prof.Timings()  # Keep track of how fast things are.

#         seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
#         sc_env = init_game(game_params['env'], FLAGS.map_name, random_seed=seed)
#         obs_processer = IMPALA_ObsProcesser_v2(env=sc_env, action_table=model.action_table, **game_params['obs_processer'])
#         env = environment.Environment_v2(sc_env, obs_processer, seed)
#         # initial rollout starts here
#         env_output = env.initial() 
#         new_res = model.spatial_processing_block.new_res
#         agent_state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
#                                                                             image_size=(new_res,new_res)
#                                                                            )
        
#         with torch.no_grad():
#             agent_output, new_agent_state = model.actor_step(env_output, *agent_state[0]) 

#         agent_state = agent_state[0] # _init_hidden yields [(h,c)], whereas actor step only (h,c)
#         while True:
#             index = free_queue.get()
#             if index is None:
#                 break

#             # Write old rollout end. 
#             for key in env_output:
#                 buffers[key][index][0, ...] = env_output[key]
#             for key in agent_output:
#                 if key not in ['sc_env_action']: # no need to save this key on buffers
#                     buffers[key][index][0, ...] = agent_output[key]
            
#             # lstm state in syncro with the environment / input to the agent 
#             # that's why agent_state = new_agent_state gets executed afterwards
#             initial_agent_state_buffers[index][0][...] = agent_state[0]
#             initial_agent_state_buffers[index][1][...] = agent_state[1]
            
            
#             # Do new rollout.
#             for t in range(FLAGS.unroll_length):
#                 timings.reset()

#                 env_output = env.step(agent_output["sc_env_action"])
                
#                 timings.time("step")
                
#                 # update state
#                 agent_state = new_agent_state 
            
#                 with torch.no_grad():
#                     agent_output, new_agent_state = model.actor_step(env_output, *agent_state)
                
#                 timings.time("model")
                
#                 #env_output = env.step(agent_output["sc_env_action"])

#                 #timings.time("step")

#                 for key in env_output:
#                     buffers[key][index][t+1, ...] = env_output[key] 
#                 for key in agent_output:
#                     if key not in ['sc_env_action']: # no need to save this key on buffers
#                         buffers[key][index][t+1, ...] = agent_output[key] 
#                 # env_output will be like
#                 # s_{0}, ..., s_{T}
#                 # act_mask_{0}, ..., act_mask_{T}
#                 # discount_{0}, ..., discount_{T}
#                 # r_{-1}, ..., r_{T-1}
#                 # agent_output will be like
#                 # a_0, ..., a_T with a_t ~ pi(.|s_t)
#                 # log_pi(a_0|s_0), ..., log_pi(a_T|s_T)
#                 # so the learner can use (s_i, act_mask_i) to predict log_pi_i
#                 timings.time("write")
#             full_queue.put(index)

#         if actor_index == 0:
#             logging.info("Actor %i: %s", actor_index, timings.summary())

#     except KeyboardInterrupt:
#         pass  # Return silently.
#     except Exception as e:
#         logging.error("Exception in worker process %i", actor_index)
#         traceback.print_exc()
#         print()
#         raise e


# def get_batch(
#     free_queue: mp.SimpleQueue,
#     full_queue: mp.SimpleQueue,
#     buffers: Buffers,
#     initial_agent_state_buffers,
#     timings,
#     lock=threading.Lock(),
# ):
#     with lock:
#         timings.time("lock")
#         indices = [full_queue.get() for _ in range(FLAGS.batch_size)]
#         timings.time("dequeue")
#     batch = {
#         key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
#     }
#     initial_agent_state = [torch.stack([initial_agent_state_buffers[m][i][0] for m in indices], axis=0)
#                       for i in range(2)]
#     #print("initial_agent_state[0].shape: ", initial_agent_state[0].shape)
#     timings.time("batch")
#     for m in indices:
#         free_queue.put(m)
#     timings.time("enqueue")
#     batch = {k: t.to(device=FLAGS.device, non_blocking=True) for k, t in batch.items()}
#     initial_agent_state = [t.to(device=FLAGS.device, non_blocking=True) for t in initial_agent_state]
#     timings.time("device")
#     return batch, initial_agent_state

# def learn(
#     actor_model, # single actor model with shared memory? Confirm that?
#     model,
#     batch,
#     initial_agent_state,
#     optimizer,
#     scheduler,
#     lock=threading.Lock(),  # noqa: B008
# ):
#     """Performs a learning (optimization) step."""
#     with lock:
        
#         learner_outputs = model.learner_step(batch, initial_agent_state) 
        
#         # Take final value function slice for bootstrapping.
#         bootstrap_value = learner_outputs["baseline_trg"][-1] # V_learner(s_T)
#         entropy = learner_outputs['entropy']
        
#         # gets [log_prob_{0}, ..., log_prob_{T-1}] and [V_{0},...,V_{T-1}]
#         learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items() if key != 'entropy'}

#         rewards = batch['reward'][1:]
#         if FLAGS.reward_clipping == "abs_one":
#             clipped_rewards = torch.clamp(rewards, -1, 1)
#         elif FLAGS.reward_clipping == "none":
#             clipped_rewards = rewards

#         vtrace_returns = vtrace.from_logits(
#             behavior_action_log_probs=batch['log_prob'][:-1], # actor
#             target_action_log_probs=learner_outputs["log_prob"], # learner
#             not_done=(~batch['done'][1:]).float(),
#             bootstrap=batch['bootstrap'][1:],
#             gamma=FLAGS.discounting,
#             rewards=clipped_rewards,
#             values=learner_outputs["baseline"],
#             values_trg=learner_outputs["baseline_trg"],
#             bootstrap_value=bootstrap_value, # coming from the learner too
#         )

#         pg_loss = compute_policy_gradient_loss(
#             learner_outputs["log_prob"],
#             vtrace_returns.pg_advantages,
#         )
       
#         baseline_loss = FLAGS.baseline_cost * compute_baseline_loss(
#             vtrace_returns.vs - learner_outputs["baseline"]
#         )

#         entropy_loss = FLAGS.entropy_cost * entropy
#         total_loss = pg_loss + baseline_loss + entropy_loss
#         # not every time we get an episode return because the unroll length is shorter than the episode length, 
#         # so not every time batch['done'] contains some True entries
#         episode_returns = batch["episode_return"][batch["done"]] # still to check, might be okay
#         stats = {
#             "episode_returns": tuple(episode_returns.cpu().numpy()),
#             "mean_episode_return": torch.mean(episode_returns).item() if len(episode_returns) > 0 else 0,
#             "total_loss": total_loss.item(),
#             "pg_loss": pg_loss.item(),
#             "baseline_loss": baseline_loss.item(),
#             "entropy_loss": entropy_loss.item(),
#         }

#         optimizer.zero_grad()
#         total_loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_norm_clipping)
#         optimizer.step()
#         if FLAGS.optim == "RMSprop":
#             scheduler.step()
#         actor_model.load_state_dict(model.state_dict())
#         return stats


# def create_buffers(
#     screen_shape,
#     minimap_shape,
#     player_shape, 
#     num_actions, 
#     max_num_spatial_args, 
#     max_num_categorical_args
# ) -> Buffers:
#     """`FLAGS` must contain unroll_length and num_buffers"""
#     T = FLAGS.unroll_length
#     # specs is a dict of dict which containt the keys 'size' and 'dtype'
#     specs = dict(
#         screen_layers=dict(size=(T+1, *screen_shape), dtype=torch.float32), 
#         minimap_layers=dict(size=(T+1, *minimap_shape), dtype=torch.float32),
#         player_state=dict(size=(T+1, player_shape), dtype=torch.float32), 
#         screen_layers_trg=dict(size=(T+1, *screen_shape), dtype=torch.float32), 
#         minimap_layers_trg=dict(size=(T+1, *minimap_shape), dtype=torch.float32),
#         player_state_trg=dict(size=(T+1, player_shape), dtype=torch.float32), 
#         last_action=dict(size=(T+1,), dtype=torch.int64),
#         action_mask=dict(size=(T+1, num_actions), dtype=torch.bool), 
#         reward=dict(size=(T+1,), dtype=torch.float32),
#         done=dict(size=(T+1,), dtype=torch.bool),
#         bootstrap=dict(size=(T+1,), dtype=torch.bool),
#         episode_return=dict(size=(T+1,), dtype=torch.float32),
#         episode_step=dict(size=(T+1,), dtype=torch.int32),
#         log_prob=dict(size=(T+1,), dtype=torch.float32),
#         main_action=dict(size=(T+1,), dtype=torch.int64), 
#         categorical_indexes=dict(size=(T+1, max_num_categorical_args), dtype=torch.int64),
#         spatial_indexes=dict(size=(T+1, max_num_spatial_args), dtype=torch.int64),
#     )
#     buffers: Buffers = {key: [] for key in specs}
#     for _ in range(FLAGS.num_buffers):
#         for key in buffers:
#             buffers[key].append(torch.empty(**specs[key]).share_memory_())
#     return buffers

# def train():  # pylint: disable=too-many-branches, too-many-statements
#     """
#     1. Init actor model and create_buffers()
#     2. Starts 'num_actors' act() functions
#     3. Init learner model and optimizer, loads the former on the GPU
#     4. Launches 'num_learner_threads' threads executing batch_and_learn()
#     5. train finishes when all batch_and_learn threads finish, i.e. when steps >= FLAGS.total_steps
#     """
#     if FLAGS.xpid is None:
#         FLAGS.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
#     # plogger = file_writer.FileWriter(
#     #     xpid=FLAGS.xpid, xp_args=FLAGS.__dict__, rootdir=FLAGS.savedir
#     # )
#     plogger = file_writer.FileWriter(
#     xpid=FLAGS.xpid, xp_args=FLAGS.flag_values_dict(), rootdir=FLAGS.savedir
# )
#     checkpointpath = os.path.expandvars(
#         os.path.expanduser("%s/%s/%s" % (FLAGS.savedir, FLAGS.xpid, "model.tar"))
#     )
#     print("checkpointpath: ", checkpointpath)
#     if FLAGS.num_buffers is None:  # Set sensible default for num_buffers. IMPORTANT!!
#         FLAGS.num_buffers = max(2 * FLAGS.num_actors, FLAGS.batch_size)
#     if FLAGS.num_actors >= FLAGS.num_buffers:
#         raise ValueError("num_buffers should be larger than num_actors")
#     if FLAGS.num_buffers < FLAGS.batch_size:
#         raise ValueError("num_buffers should be larger than batch_size")

#     T = FLAGS.unroll_length
#     B = FLAGS.batch_size

#     FLAGS.device = None
#     if not FLAGS.disable_cuda and torch.cuda.is_available():
#         logging.info("Using CUDA.")
#         FLAGS.device = torch.device("cuda")
#     else:
#         logging.info("Not using CUDA.")
#         FLAGS.device = torch.device("cpu")

#     env = init_game(game_params['env'], FLAGS.map_name) 

#     model = IMPALA_AC_v2(env=env, device='cpu', **game_params['HPs']) 
#     screen_shape = (game_params['HPs']['screen_channels'], *model.screen_res)
#     minimap_shape = (game_params['HPs']['minimap_channels'], *model.screen_res)
#     player_shape = game_params['HPs']['in_player']
#     num_actions = model.action_space
#     buffers = create_buffers(
#                              screen_shape, 
#                              minimap_shape,
#                              player_shape, 
#                              num_actions,
#                              model.max_num_spatial_args, 
#                              model.max_num_categorical_args) 
    
#     model.share_memory() # see if this works out of the box for my A2C

#     # Add initial RNN state.
#     initial_agent_state_buffers = []
#     new_res = model.spatial_processing_block.new_res
#     for _ in range(FLAGS.num_buffers):
#         state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
#                                                                       image_size=(new_res, new_res)
#                                                                   )
        
#         state = state[0] # [(h,c)] -> (h,c)
#         for t in state:
#             t.share_memory_()
#         initial_agent_state_buffers.append(state)
        
#     actor_processes = []
#     ctx = mp.get_context("spawn")
#     free_queue = ctx.SimpleQueue()
#     full_queue = ctx.SimpleQueue()

#     for i in range(FLAGS.num_actors):
#         actor = ctx.Process(
#             target=act,
#             args=(
#                 i,
#                 free_queue,
#                 full_queue,
#                 model, # with share memory
#                 buffers,
#                 initial_agent_state_buffers,
#             ),
#         )
#         actor.start()
#         actor_processes.append(actor)

#     # only model loaded into the GPU ?
#     learner_model = IMPALA_AC_v2(env=env, device='cuda', **game_params['HPs']).to(device=FLAGS.device) 

#     if FLAGS.optim == "Adam":
#         optimizer = torch.optim.Adam(
#             learner_model.parameters(),
#             lr=FLAGS.learning_rate
#         )
#     else:
#         optimizer = torch.optim.RMSprop(
#             learner_model.parameters(),
#             lr=FLAGS.learning_rate,
#             momentum=FLAGS.momentum,
#             eps=FLAGS.epsilon,
#             alpha=FLAGS.alpha,
#         )

#     def lr_lambda(epoch):
#         """
#         Linear schedule from 1 to 0 used only for RMSprop. 
#         To be adjusted multiplying or not by batch size B depending on how the steps are counted.
#         epoch = number of optimizer steps
#         total_steps = optimizer steps * time steps * batch size
#                     or optimizer steps * time steps
#         """
#         return 1 - min(epoch * T, FLAGS.total_steps) / FLAGS.total_steps #epoch * T * B if using B steps

#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#     logger = logging.getLogger("logfile")
#     stat_keys = [
#         "total_loss",
#         "mean_episode_return",
#         "pg_loss",
#         "baseline_loss",
#         "entropy_loss",
#     ]
#     logger.info("# Step\t%s", "\t".join(stat_keys))

#     # Load checkpoint if available
#     if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
#         logging.info(f"Loading checkpoint from {FLAGS.checkpoint_path}")
#         checkpoint = torch.load(FLAGS.checkpoint_path, map_location=FLAGS.device)
#         learner_model.load_state_dict(checkpoint["model_state_dict"])
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#         if FLAGS.optim != "Adam" and "scheduler_state_dict" in checkpoint:
#             scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
#         step = checkpoint.get("step", 0)
#         # Also update the actor model
#         model.load_state_dict(learner_model.state_dict())
#         logging.info(f"Resumed training from checkpoint {FLAGS.checkpoint_path} at step {step}.")
#     else:
#         step = 0

#     stats = {}

#     def batch_and_learn(i, lock=threading.Lock()):
#         """Thread target for the learning process."""
#         nonlocal step, stats
#         timings = prof.Timings()
#         while step < FLAGS.total_steps:
#             timings.reset()
#             batch, agent_state = get_batch(
#                 free_queue,
#                 full_queue,
#                 buffers,
#                 initial_agent_state_buffers,
#                 timings,
#             )
#             stats = learn(
#                 model, learner_model, batch, agent_state, optimizer, scheduler
#             )
#             timings.time("learn")
#             with lock:
#                 to_log = dict(step=step)
#                 to_log.update({k: stats[k] for k in stat_keys})
#                 plogger.log(to_log)
#                 step += T #* B # just count the parallel steps 
#     # end batch_and_learn
    
#         if i == 0:
#             logging.info("Batch and learn: %s", timings.summary())

#     for m in range(FLAGS.num_buffers):
#         free_queue.put(m)

#     threads = []
#     for i in range(FLAGS.num_learner_threads):
#         thread = threading.Thread(
#             target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
#         )
#         thread.start()
#         threads.append(thread)

#     def checkpoint():
#         if FLAGS.disable_checkpoint:
#             return
#         logging.info("Saving checkpoint to %s", checkpointpath)
#         # save_dict = {
#         #     "model_state_dict": model.state_dict(), 
#         #     "optimizer_state_dict": optimizer.state_dict(),
#         #     "flags": vars(FLAGS),
#         #     "step": step,
#         # }
#         save_dict = {
#     "model_state_dict": model.state_dict(), 
#     "optimizer_state_dict": optimizer.state_dict(),
#     "flags": FLAGS.flag_values_dict(),
#     "step": step,
# }
#         if FLAGS.optim != "Adam":
#             save_dict["scheduler_state_dict"] = scheduler.state_dict()
#         torch.save(
#             save_dict,
#             checkpointpath, # only one checkpoint at the time is saved
#         )
#     # end checkpoint
    
#     timer = timeit.default_timer
#     try:
#         last_checkpoint_time = timer()
#         while step < FLAGS.total_steps:
#             start_step = step
#             start_time = timer()
#             time.sleep(5)

#             if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
#                 checkpoint()
#                 last_checkpoint_time = timer()

#             sps = (step - start_step) / (timer() - start_time) # steps per second
#             if stats.get("episode_returns", None):
#                 mean_return = (
#                     "Return per episode: %.1f. " % stats["mean_episode_return"]
#                 )

#             else:
#                 mean_return = ""
#             total_loss = stats.get("total_loss", float("inf"))
#             logging.info(
#                 "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
#                 step,
#                 sps,
#                 total_loss,
#                 mean_return,
#                 pprint.pformat(stats),
#             )
#     except KeyboardInterrupt:
#         return  # Try joining actors then quit.
#     else:
#         for thread in threads:
#             thread.join()
#         logging.info("Learning finished after %d steps.", step)
#     finally:
#         for _ in range(FLAGS.num_actors):
#             free_queue.put(None)
#         for actor in actor_processes:
#             actor.join(timeout=1)

#     checkpoint()
#     plogger.close()

# def test(num_episodes: int = 100):
#     if FLAGS.xpid is None:
#         raise Exception("Specify a experiment id with --xpid. `latest` option not working.")
#     else:
#         checkpointpath = os.path.expandvars(
#             os.path.expanduser("%s/%s/%s" % (FLAGS.savedir, FLAGS.xpid, "model.tar"))
#         )

#     sc_env = init_game(game_params['env'], FLAGS.map_name)
#     model = IMPALA_AC_v2(env=sc_env, device='cpu', **game_params['HPs']) # let's use cpu as default for test
#     obs_processer = IMPALA_ObsProcesser_v2(env=sc_env, action_table=model.action_table, **game_params['obs_processer'])
#     env = environment.Environment_v2(sc_env, obs_processer)
#     model.eval() # disable dropout
#     checkpoint = torch.load(checkpointpath, map_location="cpu")
#     model.load_state_dict(checkpoint["model_state_dict"]) 

#     observation = env.initial() # env.reset
#     returns = []
#     # Init agent LSTM hidden state
#     new_res = model.spatial_processing_block.new_res
#     agent_state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
#                                                                             image_size=(new_res,new_res)
#                                                                            )
#     agent_state = agent_state[0] # _init_hidden yields [(h,c)], whereas actor step only (h,c)
    
#     while len(returns) < num_episodes:
#         with torch.no_grad():
#             agent_outputs, agent_state = model.actor_step(observation, *agent_state) 
#         observation = env.step(agent_outputs["sc_env_action"])
#         if observation["done"].item():
#             returns.append(observation["episode_return"].item())
#             logging.info(
#                 "Episode ended after %d steps. Return: %.1f",
#                 observation["episode_step"].item(),
#                 observation["episode_return"].item(),
#             )
#     env.close()
#     returns = np.array(returns)
#     logging.info(
#         "Average returns over %i episodes: %.2f (std %.2f) ", num_episodes, returns.mean(), returns.std()
#     )
#     print("Saving to file")
#     np.save('%s/%s/test_results'%(FLAGS.savedir, FLAGS.xpid), returns)

# def main(argv):
#     del argv  # Unused.

#     assert FLAGS.optim in ['RMSprop', 'Adam'], \
#         "Expected --optim to be one of [RMSprop, Adam], got "+FLAGS.optim
#     # Environment parameters
#     RESOLUTION = FLAGS.res
#     global game_params  # Make game_params accessible to other functions
#     game_params = {}
#     game_params['env'] = dict(feature_screen=RESOLUTION, feature_minimap=RESOLUTION, action_space="FEATURES") 
#     game_names = ['MoveToBeacon','CollectMineralShards','DefeatRoaches','FindAndDefeatZerglings',
#                   'DefeatZerglingsAndBanelings','CollectMineralsAndGas','BuildMarines']
#     map_name = FLAGS.map_name
#     game_params['map_name'] = map_name
#     if map_name not in game_names:
#         raise Exception("map name "+map_name+" not recognized.")
    
#     # Action and state space params
#     if FLAGS.select_all_layers:
#         obs_proc_params = {'select_all':True}
#     else:
#         obs_proc_params = {'screen_names':FLAGS.screen_names, 'minimap_names':FLAGS.minimap_names}
#     game_params['obs_processer'] = obs_proc_params
#     op = FullObsProcesser(**obs_proc_params)
#     screen_channels, minimap_channels, in_player = op.get_n_channels()

#     HPs = dict(action_names=FLAGS.action_names,
#                screen_channels=screen_channels+1, # counting binary mask tiling
#                minimap_channels=minimap_channels+1, # counting binary mask tiling
#                encoding_channels=32,
#                in_player=in_player
#               )
#     game_params['HPs'] = HPs
    
#     if FLAGS.mode == "train":
#         train()
#     else:
#         test()

# if __name__ == "__main__":
#     app.run(main)


import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing
from absl import app
from absl import flags
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

# SC stuff
from pysc2.env import sc2_env
from SC_Utils.game_utils import IMPALA_ObsProcesser_v2, FullObsProcesser
from AC_modules.IMPALA import IMPALA_AC_v2
import absl 
import sys
import numpy as np

# Define flags using absl.flags
FLAGS = flags.FLAGS

# Game arguments
flags.DEFINE_integer('res', 32, 'Screen and minimap resolution')
flags.DEFINE_string('map_name', 'MoveToBeacon', 'Name of the minigame')
flags.DEFINE_boolean('select_all_layers', True, 'If True, selects all useful layers of screen and minimap')
flags.DEFINE_list('screen_names', ['visibility_map', 'player_relative', 'selected', 'unit_density', 'unit_density_aa'],
                  'List of strings containing screen layers names to use. Overridden by select_all_layers=True')
flags.DEFINE_list('minimap_names', ['visibility_map', 'camera'],
                  'List of strings containing minimap layers names to use. Overridden by select_all_layers=True')
flags.DEFINE_list('action_names', ['no_op', 'move_camera', 'select_point', 'select_rect', 'select_idle_worker',
                                   'select_army', 'Attack_screen', 'Attack_minimap', 'Build_Barracks_screen',
                                   'Build_CommandCenter_screen', 'Build_Refinery_screen', 'Build_SupplyDepot_screen',
                                   'Harvest_Gather_SCV_screen', 'Harvest_Return_SCV_quick', 'HoldPosition_quick',
                                   'Move_screen', 'Move_minimap', 'Rally_Workers_screen', 'Rally_Workers_minimap',
                                   'Train_Marine_quick', 'Train_SCV_quick'],
                  'List of strings containing action names to use.')

# Agent arguments
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')
flags.DEFINE_string('xpid', None, 'Experiment id (default: None).')

# Training settings.
flags.DEFINE_boolean('disable_checkpoint', False, 'Disable saving checkpoint.')
flags.DEFINE_string('savedir', './logs/torchbeast', 'Root dir where experiment data will be saved.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors (default: 4).')
flags.DEFINE_integer('total_steps', 120000, 'Total environment steps to train for.')
flags.DEFINE_integer('batch_size', 8, 'Learner batch size.')
flags.DEFINE_integer('unroll_length', 60, 'The unroll length (time dimension).')
flags.DEFINE_integer('num_buffers', None, 'Number of shared-memory buffers.')
flags.DEFINE_integer('num_learner_threads', 1, 'Number learner threads.')
flags.DEFINE_boolean('disable_cuda', False, 'Disable CUDA.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.0005, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', 0.5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', 0.99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'none'], 'Reward clipping.')

# Optimizer settings.
flags.DEFINE_string('optim', 'RMSprop', 'Optimizer. Choose between RMSprop and Adam.')
flags.DEFINE_float('learning_rate', 0.0007, 'Learning rate.')
flags.DEFINE_float('alpha', 0.99, 'RMSProp smoothing constant.')
flags.DEFINE_float('momentum', 0.0, 'RMSProp momentum.')
flags.DEFINE_float('epsilon', 0.01, 'RMSProp epsilon.')
flags.DEFINE_float('grad_norm_clipping', 40.0, 'Global gradient norm clip.')

# New argument for checkpoint path
flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint to resume training from.')

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]] 

def init_game(game_params, map_name='MoveToBeacon', step_multiplier=8, **kwargs):

    race = sc2_env.Race(1)  # 1 = terran
    agent = sc2_env.Agent(race, "Testv0")  # NamedTuple [race, agent_name]
    agent_interface_format = sc2_env.parse_agent_interface_format(**game_params)  # AgentInterfaceFormat instance

    game_params = dict(map_name=map_name, 
                       players=[agent],  # use a list even for single player
                       game_steps_per_episode=0,
                       step_mul=step_multiplier,
                       agent_interface_format=[agent_interface_format]  # use a list even for single player
                       )  
    env = sc2_env.SC2Env(**game_params, **kwargs)

    return env

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_policy_gradient_loss(log_prob, advantages):
    log_prob = log_prob.view_as(advantages)
    return - torch.sum(log_prob * advantages.detach())

def init_flags_for_multiprocessing(argv):
    """
    This function reinitializes absl.flags in the subprocesses.
    It ensures that the flags are parsed again inside the child processes.
    """
    # Reinitialize absl.flags for the new process (required in multiprocessing)
    if not FLAGS.is_parsed():
        FLAGS(argv)  # Parse the flags again inside the subprocess

def act(
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
    game_params,
):
    try:
        init_flags_for_multiprocessing(sys.argv)

        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        sc_env = init_game(game_params['env'], FLAGS.map_name, random_seed=seed)
        obs_processer = IMPALA_ObsProcesser_v2(env=sc_env, action_table=model.action_table, **game_params['obs_processer'])
        env = environment.Environment_v2(sc_env, obs_processer, seed)
        # initial rollout starts here
        env_output = env.initial() 
        new_res = model.spatial_processing_block.new_res
        agent_state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
                                                                            image_size=(new_res,new_res)
                                                                           )
        
        with torch.no_grad():
            agent_output, new_agent_state = model.actor_step(env_output, *agent_state[0]) 

        agent_state = agent_state[0]  # _init_hidden yields [(h,c)], whereas actor step only (h,c)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end. 
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                if key not in ['sc_env_action']:  # no need to save this key on buffers
                    buffers[key][index][0, ...] = agent_output[key]
            
            # lstm state in syncro with the environment / input to the agent 
            # that's why agent_state = new_agent_state gets executed afterwards
            initial_agent_state_buffers[index][0][...] = agent_state[0]
            initial_agent_state_buffers[index][1][...] = agent_state[1]
            
            
            # Do new rollout.
            for t in range(FLAGS.unroll_length):
                timings.reset()

                env_output = env.step(agent_output["sc_env_action"])
                
                timings.time("step")
                
                # update state
                agent_state = new_agent_state 
            
                with torch.no_grad():
                    agent_output, new_agent_state = model.actor_step(env_output, *agent_state)
                
                timings.time("model")
                
                for key in env_output:
                    buffers[key][index][t+1, ...] = env_output[key] 
                for key in agent_output:
                    if key not in ['sc_env_action']:  # no need to save this key on buffers
                        buffers[key][index][t+1, ...] = agent_output[key] 
                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

def get_batch(
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    device,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(FLAGS.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = [torch.stack([initial_agent_state_buffers[m][i][0] for m in indices], axis=0)
                      for i in range(2)]
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = [t.to(device=device, non_blocking=True) for t in initial_agent_state]
    timings.time("device")
    return batch, initial_agent_state

def learn(
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        
        learner_outputs = model.learner_step(batch, initial_agent_state) 
        
        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline_trg"][-1]  # V_learner(s_T)
        entropy = learner_outputs['entropy']
        
        # gets [log_prob_{0}, ..., log_prob_{T-1}] and [V_{0},...,V_{T-1}]
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items() if key != 'entropy'}

        rewards = batch['reward'][1:]
        if FLAGS.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif FLAGS.reward_clipping == "none":
            clipped_rewards = rewards

        vtrace_returns = vtrace.from_logits(
            behavior_action_log_probs=batch['log_prob'][:-1],  # actor
            target_action_log_probs=learner_outputs["log_prob"],  # learner
            not_done=(~batch['done'][1:]).float(),
            bootstrap=batch['bootstrap'][1:],
            gamma=FLAGS.discounting,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            values_trg=learner_outputs["baseline_trg"],
            bootstrap_value=bootstrap_value,  # coming from the learner too
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["log_prob"],
            vtrace_returns.pg_advantages,
        )
       
        baseline_loss = FLAGS.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )

        entropy_loss = FLAGS.entropy_cost * entropy
        total_loss = pg_loss + baseline_loss + entropy_loss
        # not every time we get an episode return because the unroll length is shorter than the episode length, 
        # so not every time batch['done'] contains some True entries
        episode_returns = batch["episode_return"][batch["done"]]  # still to check, might be okay
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item() if len(episode_returns) > 0 else 0,
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_norm_clipping)
        optimizer.step()
        if FLAGS.optim == "RMSprop":
            scheduler.step()
        actor_model.load_state_dict(model.state_dict())
        return stats

def create_buffers(
    screen_shape,
    minimap_shape,
    player_shape, 
    num_actions, 
    max_num_spatial_args, 
    max_num_categorical_args
) -> Buffers:
    """`FLAGS` must contain unroll_length and num_buffers"""
    T = FLAGS.unroll_length
    # specs is a dict of dict which contain the keys 'size' and 'dtype'
    specs = dict(
        screen_layers=dict(size=(T+1, *screen_shape), dtype=torch.float32), 
        minimap_layers=dict(size=(T+1, *minimap_shape), dtype=torch.float32),
        player_state=dict(size=(T+1, player_shape), dtype=torch.float32), 
        screen_layers_trg=dict(size=(T+1, *screen_shape), dtype=torch.float32), 
        minimap_layers_trg=dict(size=(T+1, *minimap_shape), dtype=torch.float32),
        player_state_trg=dict(size=(T+1, player_shape), dtype=torch.float32), 
        last_action=dict(size=(T+1,), dtype=torch.int64),
        action_mask=dict(size=(T+1, num_actions), dtype=torch.bool), 
        reward=dict(size=(T+1,), dtype=torch.float32),
        done=dict(size=(T+1,), dtype=torch.bool),
        bootstrap=dict(size=(T+1,), dtype=torch.bool),
        episode_return=dict(size=(T+1,), dtype=torch.float32),
        episode_step=dict(size=(T+1,), dtype=torch.int32),
        log_prob=dict(size=(T+1,), dtype=torch.float32),
        main_action=dict(size=(T+1,), dtype=torch.int64), 
        categorical_indexes=dict(size=(T+1, max_num_categorical_args), dtype=torch.int64),
        spatial_indexes=dict(size=(T+1, max_num_spatial_args), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(FLAGS.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def train():  # pylint: disable=too-many-branches, too-many-statements
    """
    1. Init actor model and create_buffers()
    2. Starts 'num_actors' act() functions
    3. Init learner model and optimizer, loads the former on the GPU
    4. Launches 'num_learner_threads' threads executing batch_and_learn()
    5. train finishes when all batch_and_learn threads finish, i.e. when steps >= FLAGS.total_steps
    """
    if FLAGS.xpid is None:
        FLAGS.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=FLAGS.xpid, xp_args=FLAGS.flag_values_dict(), rootdir=FLAGS.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (FLAGS.savedir, FLAGS.xpid, "model.tar"))
    )
    print("checkpointpath: ", checkpointpath)
    if FLAGS.num_buffers is None:  # Set sensible default for num_buffers. IMPORTANT!!
        FLAGS.num_buffers = max(2 * FLAGS.num_actors, FLAGS.batch_size)
    if FLAGS.num_actors >= FLAGS.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if FLAGS.num_buffers < FLAGS.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = FLAGS.unroll_length
    B = FLAGS.batch_size

    # Determine the device
    if not FLAGS.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        device = torch.device("cpu")

    env = init_game(game_params['env'], FLAGS.map_name) 

    model = IMPALA_AC_v2(env=env, device='cpu', **game_params['HPs']) 
    screen_shape = (game_params['HPs']['screen_channels'], *model.screen_res)
    minimap_shape = (game_params['HPs']['minimap_channels'], *model.screen_res)
    player_shape = game_params['HPs']['in_player']
    num_actions = model.action_space
    buffers = create_buffers(
                             screen_shape, 
                             minimap_shape,
                             player_shape, 
                             num_actions,
                             model.max_num_spatial_args, 
                             model.max_num_categorical_args) 
    
    model.share_memory()  # see if this works out of the box for my A2C

    # Add initial RNN state.
    initial_agent_state_buffers = []
    new_res = model.spatial_processing_block.new_res
    for _ in range(FLAGS.num_buffers):
        state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
                                                                      image_size=(new_res, new_res)
                                                                  )
        
        state = state[0]  # [(h,c)] -> (h,c)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)
        
    actor_processes = []
    ctx = mp.get_context("spawn")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(FLAGS.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                i,
                free_queue,
                full_queue,
                model,  # with shared memory
                buffers,
                initial_agent_state_buffers,
                game_params,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    # Initialize learner model on the device
    learner_model = IMPALA_AC_v2(env=env, device='cuda', **game_params['HPs']).to(device=device) 

    if FLAGS.optim == "Adam":
        optimizer = torch.optim.Adam(
            learner_model.parameters(),
            lr=FLAGS.learning_rate
        )
    else:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(),
            lr=FLAGS.learning_rate,
            momentum=FLAGS.momentum,
            eps=FLAGS.epsilon,
            alpha=FLAGS.alpha,
        )

    def lr_lambda(epoch):
        """
        Linear schedule from 1 to 0 used only for RMSprop. 
        To be adjusted multiplying or not by batch size B depending on how the steps are counted.
        epoch = number of optimizer steps
        total_steps = optimizer steps * time steps * batch size
                    or optimizer steps * time steps
        """
        return 1 - min(epoch * T, FLAGS.total_steps) / FLAGS.total_steps  # epoch * T * B if using B steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    # Load checkpoint if available
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        logging.info(f"Loading checkpoint from {FLAGS.checkpoint_path}")
        checkpoint = torch.load(FLAGS.checkpoint_path, map_location=device)
        learner_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if FLAGS.optim != "Adam" and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = checkpoint.get("step", 0)
        # Also update the actor model
        model.load_state_dict(learner_model.state_dict())
        logging.info(f"Resumed training from checkpoint {FLAGS.checkpoint_path} at step {step}.")
    else:
        step = 0

    stats = {}

    def batch_and_learn(i, device, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < FLAGS.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
                device,
            )
            stats = learn(
                model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T  # * B  # just count the parallel steps 
    # end batch_and_learn
    
        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(FLAGS.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(FLAGS.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i, device)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if FLAGS.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        save_dict = {
            "model_state_dict": model.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict(),
            "flags": FLAGS.flag_values_dict(),
            "step": step,
        }
        if FLAGS.optim != "Adam":
            save_dict["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(
            save_dict,
            checkpointpath,  # only one checkpoint at the time is saved
        )
    # end checkpoint
    
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < FLAGS.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)  # steps per second
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )

            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(FLAGS.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()

def test(num_episodes: int = 100):
    if FLAGS.xpid is None:
        raise Exception("Specify a experiment id with --xpid. `latest` option not working.")
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (FLAGS.savedir, FLAGS.xpid, "model.tar"))
        )

    sc_env = init_game(game_params['env'], FLAGS.map_name)
    model = IMPALA_AC_v2(env=sc_env, device='cpu', **game_params['HPs'])  # let's use cpu as default for test
    obs_processer = IMPALA_ObsProcesser_v2(env=sc_env, action_table=model.action_table, **game_params['obs_processer'])
    env = environment.Environment_v2(sc_env, obs_processer)
    model.eval()  # disable dropout
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"]) 

    observation = env.initial()  # env.reset
    returns = []
    # Init agent LSTM hidden state
    new_res = model.spatial_processing_block.new_res
    agent_state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
                                                                            image_size=(new_res,new_res)
                                                                           )
    agent_state = agent_state[0]  # _init_hidden yields [(h,c)], whereas actor step only (h,c)
    
    while len(returns) < num_episodes:
        with torch.no_grad():
            agent_outputs, agent_state = model.actor_step(observation, *agent_state) 
        observation = env.step(agent_outputs["sc_env_action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    returns = np.array(returns)
    logging.info(
        "Average returns over %i episodes: %.2f (std %.2f) ", num_episodes, returns.mean(), returns.std()
    )
    print("Saving to file")
    np.save('%s/%s/test_results' % (FLAGS.savedir, FLAGS.xpid), returns)

def main(argv):
    del argv  # Unused.

    assert FLAGS.optim in ['RMSprop', 'Adam'], \
        "Expected --optim to be one of [RMSprop, Adam], got " + FLAGS.optim
    # Environment parameters
    RESOLUTION = FLAGS.res
    global game_params  # Make game_params accessible to other functions
    game_params = {}
    game_params['env'] = dict(feature_screen=RESOLUTION, feature_minimap=RESOLUTION, action_space="FEATURES") 
    game_names = ['MoveToBeacon','CollectMineralShards','DefeatRoaches','FindAndDefeatZerglings',
                  'DefeatZerglingsAndBanelings','CollectMineralsAndGas','BuildMarines']
    map_name = FLAGS.map_name
    game_params['map_name'] = map_name
    if map_name not in game_names:
        raise Exception("map name " + map_name + " not recognized.")
    
    # Action and state space params
    if FLAGS.select_all_layers:
        obs_proc_params = {'select_all': True}
    else:
        obs_proc_params = {'screen_names': FLAGS.screen_names, 'minimap_names': FLAGS.minimap_names}
    game_params['obs_processer'] = obs_proc_params
    op = FullObsProcesser(**obs_proc_params)
    screen_channels, minimap_channels, in_player = op.get_n_channels()

    HPs = dict(action_names=FLAGS.action_names,
               screen_channels=screen_channels+1,  # counting binary mask tiling
               minimap_channels=minimap_channels+1,  # counting binary mask tiling
               encoding_channels=32,
               in_player=in_player
              )
    game_params['HPs'] = HPs
    
    if FLAGS.mode == "train":
        train()
    else:
        test()

if __name__ == "__main__":
    app.run(main)
