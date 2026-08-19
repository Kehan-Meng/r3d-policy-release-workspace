import wandb
import numpy as np
import torch
import collections
import tqdm
from r3d.env import MetaWorldEnv
from r3d.gym_util.multistep_wrapper import MultiStepWrapper
from r3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from r3d.policy.base_policy import BasePolicy
from r3d.common.pytorch_util import dict_apply
from r3d.env_runner.base_runner import BaseRunner
from r3d.env_runner.frame_adapter_wrapper import environment_action
import r3d.common.logger_util as logger_util
from termcolor import cprint

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, epoch=None, task_config=None, save_video=False):
        if isinstance(epoch, bool) and task_config is None:
            save_video = epoch
            epoch = None
        device = policy.device

        all_traj_rewards = []
        all_success_rates = []
        episode_details = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            decision_count = 0
            target_pos = None
            initial_object_pos = None
            final_object_target_distance = None
            min_object_target_distance = float("inf")
            ever_entered_goal_but_far = False
            if self.task_name == "soccer":
                metaworld_env = env.env.env
                target_pos = np.asarray(metaworld_env._target_pos, dtype=np.float32)
                initial_object_pos = np.asarray(
                    metaworld_env._get_pos_objects(), dtype=np.float32
                )
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input, command=self.task_name)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = environment_action(np_action_dict).squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
                decision_count += 1
                if self.task_name == "soccer":
                    object_pos = np.asarray(
                        metaworld_env._get_pos_objects(), dtype=np.float32
                    )
                    final_object_target_distance = float(
                        np.linalg.norm(object_pos - target_pos)
                    )
                    min_object_target_distance = min(
                        min_object_target_distance,
                        final_object_target_distance,
                    )
                    crossed_goal_line = object_pos[1] > target_pos[1] - 0.1
                    inside_goal_width = abs(object_pos[0] - target_pos[0]) <= 0.10
                    ever_entered_goal_but_far |= bool(
                        crossed_goal_line
                        and inside_goal_width
                        and final_object_target_distance > 0.07
                    )

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            detail = {
                "episode": episode_idx,
                "success": bool(is_success),
                "reward": float(np.asarray(traj_reward).sum()),
                "decision_count": decision_count,
            }
            if self.task_name == "soccer":
                detail.update({
                    "target_pos": target_pos.tolist(),
                    "initial_object_pos": initial_object_pos.tolist(),
                    "final_object_target_distance_m": final_object_target_distance,
                    "min_object_target_distance_m": min_object_target_distance,
                    "ever_entered_goal_but_far": ever_entered_goal_but_far,
                })
            episode_details.append(detail)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        log_data['episode_count'] = len(episode_details)
        log_data['episode_details'] = episode_details
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data


class MetaworldRunnerFast(MetaworldRunner):
    """Fast MetaWorld eval: no video recording, no RGB rendering.

    Uses MetaWorldEnvFast which skips get_rgb() on every step/reset and
    returns a dummy render() frame.  Simply wrap the fast env directly —
    no SimpleVideoRecordingWrapper overhead.
    """

    def __init__(self, **kwargs):
        # Bypass MetaworldRunner.__init__ (too coupled to SimpleVideoRecordingWrapper).
        # Call BaseRunner directly, then replicate the minimal setup.
        output_dir = kwargs.pop("output_dir")
        super(MetaworldRunner, self).__init__(output_dir)

        from r3d.env.metaworld.metaworld_wrapper import MetaWorldEnvFast
        from r3d.gym_util.multistep_wrapper import MultiStepWrapper

        self.task_name = kwargs.get("task_name")
        device = kwargs.get("device", "cuda:0")
        use_point_crop = kwargs.get("use_point_crop", True)
        num_points = kwargs.get("num_points", 512)
        n_obs_steps = kwargs.get("n_obs_steps", 8)
        n_action_steps = kwargs.get("n_action_steps", 8)
        max_steps = kwargs.get("max_steps", 1000)

        def env_fn(task_name):
            return MultiStepWrapper(
                MetaWorldEnvFast(
                    task_name=task_name,
                    device=device,
                    use_point_crop=use_point_crop,
                    num_points=num_points,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )

        self.eval_episodes = kwargs.get("eval_episodes", 20)
        self.env = env_fn(self.task_name)
        self.fps = kwargs.get("fps", 10)
        self.crf = kwargs.get("crf", 22)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = kwargs.get("tqdm_interval_sec", 5.0)

        import r3d.common.logger_util as logger_util
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
