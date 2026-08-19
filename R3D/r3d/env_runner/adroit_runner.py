import wandb
import numpy as np
import torch
import tqdm
from r3d.env import AdroitEnv
from r3d.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from r3d.gym_util.multistep_wrapper import MultiStepWrapper
from r3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from r3d.policy.base_policy import BasePolicy
from r3d.common.pytorch_util import dict_apply
from r3d.env_runner.base_runner import BaseRunner
from r3d.env_runner.frame_adapter_wrapper import environment_action
import r3d.common.logger_util as logger_util
from termcolor import cprint


class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 deterministic_eval_seed=None,
                 render_device_id=0,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(
                        env=AdroitEnv(
                            env_name=task_name,
                            use_point_cloud=True,
                            render_device_id=render_device_id,
                        ),
                        env_name='adroit_'+task_name,
                        use_point_crop=use_point_crop,
                        render_device_id=render_device_id,
                    )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.deterministic_eval_seed = deterministic_eval_seed

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, epoch=None, task_config=None):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_terminal_goal_rates = []
        episode_metrics = []
        


        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            if self.deterministic_eval_seed is None:
                obs = env.reset()
            else:
                episode_seed = int(self.deterministic_eval_seed) + episode_idx
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(episode_seed)
                obs = env.reset(seed=episode_seed)
            policy.reset()

            done = False
            actual_step_count = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    if self.deterministic_eval_seed is not None:
                        # Keep the initial Flow noise identical across NFE
                        # sweeps for the same episode and policy call.
                        policy_seed = (
                            int(self.deterministic_eval_seed)
                            + episode_idx * 1000003
                            + actual_step_count
                        )
                        torch.manual_seed(policy_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(policy_seed)
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input, command=self.task_name)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = environment_action(np_action_dict).squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                actual_step_count += 1

            episode_infos = env.get_infos()
            goal_counts = np.asarray(
                episode_infos.get('n_goal_achieved', []), dtype=np.int64)
            if goal_counts.size == 0:
                raise RuntimeError(
                    "Adroit evaluation requires per-step n_goal_achieved info")
            num_goal_achieved = int(np.sum(goal_counts))
            success_threshold = 20 if self.task_name == 'pen' else 25
            episode_success = float(num_goal_achieved > success_threshold)
            terminal_goal_rate = float(np.mean(info['goal_achieved']))

            all_success_rates.append(episode_success)
            all_goal_achieved.append(num_goal_achieved)
            all_terminal_goal_rates.append(terminal_goal_rate)
            episode_metrics.append({
                'episode': episode_idx,
                'success': bool(episode_success),
                'n_goal_achieved': num_goal_achieved,
                'success_threshold': success_threshold,
                'wrapped_env_steps': len(goal_counts),
                'policy_steps': actual_step_count,
                'terminal_goal_rate': terminal_goal_rate,
            })


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['mean_terminal_goal_rate'] = np.mean(all_terminal_goal_rates)
        log_data['episode_metrics'] = episode_metrics

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env

        return log_data
