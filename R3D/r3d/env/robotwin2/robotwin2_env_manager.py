"""
RoboTwin 2.0 Environment Manager
Directly adapted from the original eval_3dpolicy.py with path corrections
"""

import sys
import os
import subprocess
from pathlib import Path

# The simulator and assets stay in a pinned external checkout. This package
# contains only the R3D integration layer.
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
release_root = Path(current_file_path).resolve().parents[4]
runtime_root = Path(
    os.environ.get("ROBOTWIN2_ROOT", release_root / "third_party" / "robotwin2")
).expanduser().resolve()

if not runtime_root.is_dir():
    raise RuntimeError(
        f"RoboTwin2 runtime not found at {runtime_root}. Run "
        "environment/install_benchmarks.sh robotwin2 or set ROBOTWIN2_ROOT."
    )
if str(runtime_root) not in sys.path:
    sys.path.insert(0, str(runtime_root))

# Add envs directory to path
envs_dir = os.path.join(runtime_root, 'envs')
if envs_dir not in sys.path:
    sys.path.insert(0, envs_dir)

# Add description utils to path
description_utils_dir = os.path.join(runtime_root, 'description', 'utils')
if description_utils_dir not in sys.path:
    sys.path.insert(0, description_utils_dir)

from _GLOBAL_CONFIGS import CONFIGS_PATH
from utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

import numpy as np
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import json
import torch
import time

from r3d.env.robotwin2.ee_execution import (
    EE_EXECUTION_CONTRACT_VERSION,
    execute_ee_action_chunk,
)


class RoboTwin2EnvManager:
    """
    RoboTwin 2.0 Environment Manager

    This is the original Env class from eval_3dpolicy.py with minimal modifications
    to work within the R3D directory structure.
    """

    def __init__(self):
        self.profile_eval = False
        self.lean_observation = False
        self.defer_intermediate_render = False
        self._eval_profile = {}
        self.last_controller_execution = None
        self.last_action_step_count = 0

    def configure_eval_observation(
            self, profile=False, lean=False, defer_intermediate_render=False,
            rt_samples_per_pixel=None, camera_shader=None):
        self.profile_eval = bool(profile)
        self.lean_observation = bool(lean)
        self.defer_intermediate_render = bool(defer_intermediate_render)
        self.rt_samples_per_pixel = rt_samples_per_pixel
        self.camera_shader = camera_shader
        self._eval_profile = {
            "take_action_calls": 0,
            "action_execution_calls": 0,
            "observation_calls_from_action": 0,
        }
        if self.lean_observation and hasattr(self, "args"):
            # Apply before setup_demo so unused cameras are never constructed.
            self.args["camera"]["collect_wrist_camera"] = False
            self.args["camera"]["eval_head_only"] = True
            self.args["data_type"]["rgb"] = False
        if rt_samples_per_pixel is not None and hasattr(self, "args"):
            if int(rt_samples_per_pixel) < 1:
                raise ValueError("rt_samples_per_pixel must be positive")
            self.args["rt_samples_per_pixel"] = int(rt_samples_per_pixel)
        if camera_shader is not None and hasattr(self, "args"):
            self.args["camera_shader"] = camera_shader
        if hasattr(self, "task"):
            self.task.profile_observation = self.profile_eval
            self.task.reset_obs_profile()
            self.apply_eval_observation_flags()

    def apply_eval_observation_flags(self):
        """Reapply flags after setup_demo recreates task cameras and data_type."""
        if not hasattr(self, "task"):
            return
        self.task.profile_observation = self.profile_eval
        self.task.defer_intermediate_render = self.defer_intermediate_render
        if self.lean_observation:
            if hasattr(self.task, "data_type"):
                self.task.data_type["rgb"] = False
            if hasattr(self.task, "cameras"):
                self.task.cameras.collect_wrist_camera = False
                self.task.cameras.capture_head_only = True

    def _profile_add(self, key, value):
        if self.profile_eval:
            self._eval_profile[key] = self._eval_profile.get(key, 0.0) + value

    def get_eval_profile(self):
        result = dict(self._eval_profile)
        if hasattr(self, "task") and hasattr(self.task, "get_obs_profile"):
            result["observation_stages"] = self.task.get_obs_profile()
        result["lean_observation"] = self.lean_observation
        result["defer_intermediate_render"] = self.defer_intermediate_render
        result["rt_samples_per_pixel"] = getattr(self, "rt_samples_per_pixel", None) or 32
        result["camera_shader"] = getattr(self, "camera_shader", None) or "rt"
        if hasattr(self, "task") and hasattr(self.task, "cameras"):
            cameras = self.task.cameras
            result["captured_cameras"] = (
                ["head_camera"] if getattr(cameras, "capture_head_only", False)
                else (
                    (["left_camera", "right_camera"] if cameras.collect_wrist_camera else [])
                    + list(cameras.static_camera_name)
                )
            )
        return result

    def get_last_controller_execution(self):
        return self.last_controller_execution

    def get_last_action_step_count(self):
        return int(self.last_action_step_count)

    @staticmethod
    def class_decorator(task_name):
        """Create task environment instance"""
        envs_module = importlib.import_module(f"envs.{task_name}")
        try:
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
        except:
            raise SystemExit(f"No Task: {task_name}")
        return env_instance

    @staticmethod
    def get_camera_config(camera_type):
        """Get camera configuration"""
        camera_config_path = os.path.join(runtime_root, "task_config", "_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        assert camera_type in args, f"camera {camera_type} is not defined"
        return args[camera_type]

    @staticmethod
    def get_embodiment_config(robot_file):
        """Get embodiment configuration"""
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args

    @staticmethod
    def get_embodiment_file(embodiment_types, embodiment_type):
        """Get embodiment file path"""
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    def dual_arm(self):
        """Check if using dual arm"""
        return self.task.get_dual_arm()

    def Create_env(self, task_name, head_camera_type, seed, task_num, instruction_type, task_config):
        """
        Create and initialize the environment

        Returns:
            tuple: (seed_list, id_list, episode_info_list_total)
        """
        task_config_file_path = os.path.join(runtime_root, 'task_config', f'{task_config}.yml')
        self.time_str = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

        with open(task_config_file_path, 'r', encoding='utf-8') as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.args['task_name'] = task_name
        self.args["task_config"] = task_config
        self.args["ckpt_setting"] = None

        self.embodiment_type = self.args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            self._embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
            self._camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = self.args["camera"]["head_camera_type"]
        self.args["head_camera_h"] = self._camera_config[head_camera_type]["h"]
        self.args["head_camera_w"] = self._camera_config[head_camera_type]["w"]

        if len(self.embodiment_type) == 1:
            self.args["left_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["right_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["dual_arm_embodied"] = True
        elif len(self.embodiment_type) == 3:
            self.args["left_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[0])
            self.args["right_robot_file"] = self.get_embodiment_file(self._embodiment_types, self.embodiment_type[1])
            self.args["embodiment_dis"] = self.embodiment_type[2]
            self.args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        if len(self.embodiment_type) == 1:
            self.embodiment_name = str(self.embodiment_type[0])
        else:
            self.embodiment_name = str(self.embodiment_type[0]) + "+" + str(self.embodiment_type[1])

        self.args["left_embodiment_config"] = self.get_embodiment_config(self.args["left_robot_file"])
        self.args["right_embodiment_config"] = self.get_embodiment_config(self.args["right_robot_file"])

        print("============= Config =============\n")
        print("\033[95mMessy Table:\033[0m " + str(self.args["domain_randomization"]["cluttered_table"]))
        print("\033[95mRandom Background:\033[0m " + str(self.args["domain_randomization"]["random_background"]))
        if self.args["domain_randomization"]["random_background"]:
            print(" - Clean Background Rate: " + str(self.args["domain_randomization"]["clean_background_rate"]))
        print("\033[95mRandom Light:\033[0m " + str(self.args["domain_randomization"]["random_light"]))
        if self.args["domain_randomization"]["random_light"]:
            print(" - Crazy Random Light Rate: " + str(self.args["domain_randomization"]["crazy_random_light_rate"]))
        print("\033[95mRandom Table Height:\033[0m " + str(self.args["domain_randomization"]["random_table_height"]))
        print("\033[95mRandom Head Camera Distance:\033[0m " + str(self.args["domain_randomization"]["random_head_camera_dis"]))

        print("\033[94mHead Camera Config:\033[0m " + str(self.args["camera"]["head_camera_type"]) + f", " +
              str(self.args["camera"]["collect_head_camera"]))
        print("\033[94mWrist Camera Config:\033[0m " + str(self.args["camera"]["wrist_camera_type"]) + f", " +
              str(self.args["camera"]["collect_wrist_camera"]))
        print("\033[94mEmbodiment Config:\033[0m " + self.embodiment_name)
        print("\n==================================")

        self.task = self.class_decorator(self.args["task_name"])
        self.st_seed = 100000 * (1 + seed)
        self.task_num = task_num
        self.clear_cache_freq = self.args['clear_cache_freq']
        self.args["eval_mode"] = True
        self.instruction_type = instruction_type

        return self.find_seed(task_num)

    def Init_task_env(
            self,
            seed,
            id,
            episode_info_list,
            run_dir,
            epoch,
            task_config,
            save_video=True,
            ):
        """Initialize task environment for a specific episode"""
        self.env_state = 0  # 0:running 1:success 2:fail
        self.step = 0
        self.last_controller_execution = None
        self.last_action_step_count = 0
        self.succ_seed = seed

        self.task.setup_demo(now_ep_num=id, seed=seed, is_test=True, **self.args)

        results = generate_episode_descriptions(self.args["task_name"], episode_info_list, 1)
        instruction = np.random.choice(results[0][self.instruction_type])
        self.task.set_instruction(instruction=instruction)

        self.eval_video_log = save_video
        self.video_size = str(self.args['head_camera_w']) + 'x' + str(self.args['head_camera_h'])
        self.save_dir = str(epoch) + '-' + run_dir + '-' + task_config

        if self.eval_video_log:
            # Save to output directory relative to current location
            self.save_dir = Path(current_directory) / 'eval_video' / self.save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.success_path = os.path.join(self.save_dir, f'success_{seed}.mp4')
            self.fail_path = os.path.join(self.save_dir, f'fail_{seed}.mp4')
            log_file = open(f'{self.save_dir}/{self.time_str}_ffmpeg_log.txt', 'w')
            self.file_path = os.path.join(self.save_dir, f'{seed}.mp4')

            self.ffmpeg = subprocess.Popen([
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pixel_format', 'rgb24',
                '-video_size', self.video_size,
                '-framerate', '4',
                '-i', '-',
                '-pix_fmt', 'yuv420p',
                '-vcodec', 'libx264',
                '-preset', 'veryfast',
                '-tune', 'zerolatency',
                '-g', '15',
                '-threads', '0',
                f'{self.save_dir}/{seed}.mp4'
            ], stdin=subprocess.PIPE, stdout=log_file, stderr=log_file)

        return instruction

    def save_seed(self, seedlist, episode_info_list=None, st_seed=None):
        """Save valid seeds to file"""
        if st_seed is None:
            st_seed = self.st_seed
        st_seed_key = str(st_seed) + self.args['task_config']

        save_path = os.path.join(current_directory, 'seeds_list')
        file_path = os.path.join(save_path, f'{self.args["task_name"]}.json')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        new_seeds = set(seedlist)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if st_seed_key in data:
                    existing_seeds = set(data[st_seed_key]["seeds"])
                    all_seeds = sorted(list(existing_seeds.union(new_seeds)))
                    data[st_seed_key]["seeds"] = all_seeds

                    if episode_info_list:
                        if "episode_info" not in data[st_seed_key]:
                            data[st_seed_key]["episode_info"] = {}

                        for i, seed in enumerate(seedlist):
                            data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]
                else:
                    data[st_seed_key] = {"seeds": sorted(list(new_seeds))}
                    if episode_info_list:
                        data[st_seed_key]["episode_info"] = {}
                        for i, seed in enumerate(seedlist):
                            data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]
            except (json.JSONDecodeError, FileNotFoundError):
                data = {
                    st_seed_key: {
                        "seeds": sorted(list(new_seeds)),
                        "episode_info": {}
                    }
                }
                if episode_info_list:
                    for i, seed in enumerate(seedlist):
                        data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]
        else:
            data = {
                st_seed_key: {
                    "seeds": sorted(list(new_seeds)),
                    "episode_info": {}
                }
            }
            if episode_info_list:
                for i, seed in enumerate(seedlist):
                    data[st_seed_key]["episode_info"][str(seed)] = episode_info_list[i]

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def find_seed(self, task_num):
        """Find or load valid seeds"""
        save_path = os.path.join(current_directory, 'seeds_list')
        file_path = os.path.join(save_path, f'{self.args["task_name"]}.json')
        st_seed_key = str(self.st_seed) + self.args['task_config']
        existing_seeds = []
        existing_episode_info = {}

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if st_seed_key in data and "seeds" in data[st_seed_key]:
                    existing_seeds = data[st_seed_key]["seeds"]
                    if "episode_info" in data[st_seed_key]:
                        existing_episode_info = data[st_seed_key]["episode_info"]
            except (json.JSONDecodeError, FileNotFoundError):
                existing_seeds = []

        valid_seeds = existing_seeds

        if len(valid_seeds) >= task_num:
            selected_seeds = valid_seeds[:task_num]
            id_list = list(range(task_num))

            episode_info_list = []
            for seed in selected_seeds:
                seed_str = str(seed)
                if seed_str in existing_episode_info:
                    episode_info_list.append(existing_episode_info[seed_str])
                else:
                    episode_info_list.append([])

            print(f"Found {len(selected_seeds)} valid seeds in group {st_seed_key}")
            return selected_seeds, id_list, episode_info_list

        print(f"Insufficient seeds in group {st_seed_key} ({len(valid_seeds)}/{task_num}), starting to find new seeds...")

        needed_seeds = task_num - len(valid_seeds)
        start_seed = self.st_seed
        if valid_seeds:
            start_seed = max(valid_seeds) + 1

        new_seeds, new_ids, new_episode_info_list = self.Check_seed(needed_seeds, start_seed)

        final_seeds = valid_seeds + new_seeds
        final_seeds = final_seeds[:task_num]
        final_ids = list(range(len(final_seeds)))

        existing_episode_info_list = [existing_episode_info[str(s)] for s in valid_seeds if str(s) in existing_episode_info]
        final_episode_info_list = existing_episode_info_list + new_episode_info_list
        final_episode_info_list = final_episode_info_list[:task_num]

        self.save_seed(new_seeds, new_episode_info_list)

        print(f"Total found {len(final_seeds)} valid seeds in group {st_seed_key}")
        return final_seeds, final_ids, final_episode_info_list

    def Check_seed(self, test_num, start_seed):
        """Check and validate seeds"""
        expert_check = True
        print("Task name: ", self.args["task_name"])
        suc_seed_list = []
        now_id_list = []
        succ_tnt = 0
        now_seed = start_seed
        now_id = 0
        self.task.cus = 0
        self.task.test_num = 0
        episode_info_list_total = []

        while succ_tnt < test_num:
            render_freq = self.args['render_freq']
            self.args['render_freq'] = 0

            if expert_check:
                try:
                    self.task.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **self.args)
                    episode_info = self.task.play_once()
                    self.task.close()

                except UnStableError as e:
                    print(" -------------")
                    print("Error: ", e)
                    print(" -------------")
                    self.task.close_env()
                    now_seed += 1
                    self.args["render_freq"] = render_freq
                    continue
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    print(' -------------')
                    print('Error: ', stack_trace)
                    print(' -------------')
                    self.task.close()
                    now_seed += 1
                    self.args['render_freq'] = render_freq
                    print('error occurs !')
                    continue

            if (not expert_check) or (self.task.plan_success and self.task.check_success()):
                suc_seed_list.append(now_seed)
                now_id_list.append(now_id)
                now_id += 1
                succ_tnt += 1
                now_seed += 1
                episode_info_list = [episode_info["info"]]
                episode_info_list_total.append(episode_info_list)
            else:
                now_seed += 1
                self.args["render_freq"] = render_freq
                continue

            self.args['render_freq'] = render_freq

        return suc_seed_list, now_id_list, episode_info_list_total

    def Detect_env_state(self):
        """Detect current environment state"""
        if self.step >= self.task.step_lim:
            self.env_state = 2  # fail
        if self.task.eval_success:
            self.env_state = 1  # success

    def Take_action(
        self,
        actions,
        obs_history,
        n_obs_steps,
        action_types='qpos',
        use_ee_space=False,
        physics_step_budget=None,
    ):
        """
        Execute action chunk in the environment

        Args:
            actions: Action chunk to execute
            obs_history: Observation history deque
            n_obs_steps: Number of observation steps
            action_types: 'qpos' or 'ee'
            use_ee_space: Whether using end effector space

        Returns:
            tuple: (status, obs_history)
                status: "success", "fail", or "run"
                obs_history: Updated observation history
        """
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError(f"actions must be a non-empty [K,D] array, got {actions.shape}")

        first_stage = len(actions) - (n_obs_steps - 1)
        if first_stage <= 0:
            raise ValueError(
                f"Action chunk length {len(actions)} is too short for "
                f"n_obs_steps={n_obs_steps}"
            )
        first_actions = actions[:first_stage]
        second_actions = actions[first_stage:]
        take_action_start = time.perf_counter()
        is_ee_action = action_types in ('ee', 'delta_ee')
        controller_stages = []
        attempted_action_count = 0
        controller_ok = True

        def execute_actions(action_batch):
            nonlocal attempted_action_count, controller_ok
            stage_start = time.perf_counter()
            if is_ee_action:
                stage_result = execute_ee_action_chunk(
                    self.task,
                    action_batch,
                    action_type=action_types,
                    stop_on_failure=True,
                    physics_step_budget=physics_step_budget,
                )
                controller_stages.append(stage_result)
                attempted_action_count += stage_result['attempted_target_count']
                controller_ok = controller_ok and stage_result['controller_success']
            else:
                self.task.take_action(action_batch, action_type=action_types)
                attempted_action_count += len(action_batch)
            self._profile_add("action_execution_sec", time.perf_counter() - stage_start)
            self._profile_add("action_execution_calls", 1)

        def append_observation():
            stage_start = time.perf_counter()
            observation = self.get_observation()
            self._profile_add("observation_wall_sec", time.perf_counter() - stage_start)
            self._profile_add("observation_calls_from_action", 1)

            if use_ee_space:
                agent_pos_vector = np.concatenate([
                    observation['endpose']['left_endpose'],
                    [observation['endpose']['left_gripper']],
                    observation['endpose']['right_endpose'],
                    [observation['endpose']['right_gripper']],
                ])
            else:
                agent_pos_vector = observation['joint_action']['vector']
            obs_history.append({
                'point_cloud': torch.from_numpy(observation['pointcloud']),
                'agent_pos': torch.from_numpy(agent_pos_vector),
            })
            return observation

        execute_actions(first_actions)
        observation = append_observation()

        if controller_ok and not self.task.eval_success:
            for action in second_actions:
                execute_actions(action[None, :])
                observation = append_observation()
                if not controller_ok or self.task.eval_success:
                    break

        if is_ee_action:
            target_results = []
            target_offset = 0
            for stage_result in controller_stages:
                for target_result in stage_result['target_results']:
                    item = dict(target_result)
                    item['target_index'] = int(item['target_index'] + target_offset)
                    target_results.append(item)
                target_offset += stage_result['requested_target_count']
            self.last_controller_execution = {
                'contract_version': (
                    controller_stages[0]['contract_version']
                    if controller_stages else EE_EXECUTION_CONTRACT_VERSION
                ),
                'action_type': action_types,
                'physics_step_budget_per_target': physics_step_budget,
                'requested_target_count': int(len(actions)),
                'attempted_target_count': int(attempted_action_count),
                'fully_planned_target_count': int(sum(
                    int(item['controller_success']) for item in target_results
                )),
                'reached_target_count': int(sum(
                    int(item.get('target_reached', False))
                    for item in target_results
                )),
                'completed_all_targets': bool(attempted_action_count == len(actions)),
                'completed_all_trajectories': bool(
                    attempted_action_count == len(actions)
                    and all(
                        item.get('target_reached', False)
                        for item in target_results
                    )
                ),
                'controller_success': bool(controller_ok),
                'stopped_on_controller_failure': bool(not controller_ok),
                'stopped_for_environment_success': bool(
                    self.task.eval_success and attempted_action_count < len(actions)
                ),
                'executed_physics_steps': int(sum(
                    stage.get('executed_physics_steps', 0)
                    for stage in controller_stages
                )),
                'target_results': target_results,
            }
        else:
            self.last_controller_execution = None

        # Save a video frame only when this runner requested videos.
        if self.eval_video_log:
            self.ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
        self.last_action_step_count = int(attempted_action_count)
        self.step += attempted_action_count
        self._profile_add("take_action_total_sec", time.perf_counter() - take_action_start)
        self._profile_add("take_action_calls", 1)
        self.Detect_env_state()

        if self.env_state == 1:
            print('Task Success!')
            self.Close_env(success=True)
            return "success", obs_history
        elif is_ee_action and not controller_ok:
            print('EE Controller Failed!')
            self.Close_env(success=False)
            return "fail", obs_history
        elif self.env_state == 2:
            print('Task Failed!')
            self.Close_env(success=False)
            return "fail", obs_history
        else:
            return "run", obs_history

    def Close_env(self, success=False):
        """Close environment and rename video based on success"""
        if self.eval_video_log:
            observation = self.get_observation()
            self.ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
        self.task.close_env(clear_cache=((self.succ_seed + 1) % self.clear_cache_freq == 0))

        if self.eval_video_log:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            del self.ffmpeg
            if success:
                os.rename(self.file_path, self.success_path)
            else:
                os.rename(self.file_path, self.fail_path)

        if self.task.render_freq:
            self.task.viewer.close()

        print('Env Closed!')
        self.task._take_picture()

    def get_observation(self):
        """Get current observation from environment"""
        return self.task.get_obs()
