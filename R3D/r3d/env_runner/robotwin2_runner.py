"""
RoboTwin 2.0 Environment Runner for R3D.

Supports both legacy single-task evaluation and multi-task checkpoints that are
conditioned with a task_onehot observation.
"""

import numpy as np
import torch
import tqdm
import os
import random
import time
import hashlib
from pathlib import Path
from datetime import datetime
from termcolor import cprint
from collections import deque

from r3d.policy.base_policy import BasePolicy
from r3d.common.pytorch_util import dict_apply
from r3d.env_runner.base_runner import BaseRunner
import r3d.common.logger_util as logger_util


def _as_plain_dict(item):
    if hasattr(item, "items"):
        return dict(item)
    return item


class RoboTwin2Runner(BaseRunner):
    """
    RoboTwin 2.0 Environment Runner.

    This runner directly uses the original RoboTwin 2.0 Env interface,
    supporting action chunk prediction and execution without MultiStepWrapper.
    """

    _ROBOTWIN2_DIR = str(Path(__file__).parent.parent / 'env' / 'robotwin2')

    def __init__(
        self,
        output_dir: str,
        task_name: str = "beat_block_hammer",
        task_entries=None,
        eval_task_name=None,
        seed: int = 1,
        eval_episodes: int = 20,
        max_steps: int = 1000,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        task_config: str = "demo_clean",
        instruction_type: str = "unseen",
        action_space_type: str = "joint",
        head_camera_type: str = "D435",
        save_video: bool = True,
        tqdm_interval_sec: float = 5.0,
        episode_start: int = 0,
        deterministic_eval_seed=None,
        lean_observation: bool = False,
        profile_eval: bool = False,
        defer_intermediate_render: bool = False,
        rt_samples_per_pixel: int = None,
        camera_shader: str = None,
        **kwargs
    ):
        super().__init__(output_dir)

        self.task_name = task_name
        self.task_entries = None
        if task_entries is not None:
            self.task_entries = [_as_plain_dict(item) for item in task_entries]
        self.eval_task_name = eval_task_name
        self.seed = seed
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.task_config = task_config
        self.instruction_type = instruction_type
        self.action_space_type = action_space_type
        self.head_camera_type = head_camera_type
        self.save_video = save_video
        self.tqdm_interval_sec = tqdm_interval_sec
        self.episode_start = int(episode_start)
        self.deterministic_eval_seed = deterministic_eval_seed
        self.lean_observation = bool(lean_observation)
        self.profile_eval = bool(profile_eval)
        self.defer_intermediate_render = bool(defer_intermediate_render)
        self.rt_samples_per_pixel = rt_samples_per_pixel
        self.camera_shader = camera_shader
        if self.episode_start < 0:
            raise ValueError("episode_start must be non-negative")
        if self.lean_observation and self.save_video:
            raise ValueError("lean_observation requires save_video=False")

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        _orig_cwd = os.getcwd()
        os.chdir(self._ROBOTWIN2_DIR)
        from r3d.env.robotwin2 import RoboTwin2EnvManager
        self.env_manager = RoboTwin2EnvManager()
        os.chdir(_orig_cwd)

        if self.task_entries is None:
            cprint(f"[RoboTwin2Runner] Initialized for task: {task_name}", "cyan")
        else:
            cprint(f"[RoboTwin2Runner] Initialized for {len(self.task_entries)} task entries", "cyan")
            if self.eval_task_name is not None:
                cprint(f"[RoboTwin2Runner] Single-task override: {self.eval_task_name}", "cyan")
        cprint(f"[RoboTwin2Runner] Action space: {action_space_type}", "cyan")
        cprint(f"[RoboTwin2Runner] Eval episodes: {eval_episodes}", "cyan")
        cprint(f"[RoboTwin2Runner] n_obs_steps: {n_obs_steps}, n_action_steps: {n_action_steps}", "cyan")

    def _task_onehot(self, task_idx, num_tasks, device=None):
        task_onehot = torch.zeros(num_tasks, dtype=torch.float32, device=device)
        task_onehot[task_idx] = 1.0
        return task_onehot

    def _add_task_onehot(self, obs, task_idx=None, num_tasks=None):
        if task_idx is None or num_tasks is None:
            return obs
        obs = obs.copy()
        obs['task_onehot'] = self._task_onehot(task_idx, num_tasks)
        return obs

    def _ensure_history_task_onehot(self, obs_history, task_idx=None, num_tasks=None):
        if task_idx is None or num_tasks is None:
            return obs_history
        fixed_history = deque(maxlen=self.n_obs_steps)
        for obs in obs_history:
            fixed_history.append(self._add_task_onehot(obs, task_idx, num_tasks))
        return fixed_history

    def _agent_pos_from_observation(self, observation):
        if self.action_space_type == 'ee':
            left_endpose = observation['endpose']['left_endpose']
            right_endpose = observation['endpose']['right_endpose']
            left_gripper = observation['endpose']['left_gripper']
            right_gripper = observation['endpose']['right_gripper']
            return np.concatenate([
                left_endpose,
                [left_gripper],
                right_endpose,
                [right_gripper],
            ])
        return observation['joint_action']['vector']

    def _episode_task_metadata(self, task_name):
        """Return lightweight task metadata needed for per-condition metrics."""
        task = getattr(self.env_manager, "task", None)
        if task_name == "place_shoe":
            shoe = getattr(task, "shoe", None)
            if shoe is None:
                return {}
            initial_x = float(shoe.get_pose().p[0])
            return {
                "initial_shoe_x": initial_x,
                "arm_routing": "left" if initial_x < 0 else "right",
            }

        if task_name == "move_playingcard_away":
            playingcards = getattr(task, "playingcards", None)
            if playingcards is None:
                return {}

            initial_x = float(playingcards.get_pose().p[0])
            return {
                "initial_playingcard_x": initial_x,
                "playingcard_side": "right" if initial_x > 0 else "left",
            }

        return {}

    def _init_move_card_phase_tracker(self, task_name):
        if task_name != "move_playingcard_away":
            return None

        task = getattr(self.env_manager, "task", None)
        playingcards = getattr(task, "playingcards", None)
        if task is None or playingcards is None:
            return None

        initial_position = np.array(playingcards.get_pose().p, dtype=np.float64, copy=True)
        expected_side = "right" if initial_position[0] > 0 else "left"
        return {
            "expected_side": expected_side,
            "initial_position": initial_position,
            "first_closed_side": None,
            "min_left_tcp_distance": float("inf"),
            "min_right_tcp_distance": float("inf"),
            "saw_card_gripper_contact": False,
            "transport_while_expected_closed": False,
            "max_correct_direction_displacement": 0.0,
            "released_after_correct_move": False,
        }

    def _update_move_card_phase_tracker(self, tracker):
        """Track observable MoveCard behavior without changing environment state."""
        if tracker is None:
            return

        task = self.env_manager.task
        robot = task.robot
        card_position = np.asarray(task.playingcards.get_pose().p, dtype=np.float64)
        left_tcp = np.asarray(robot.get_left_tcp_pose()[:3], dtype=np.float64)
        right_tcp = np.asarray(robot.get_right_tcp_pose()[:3], dtype=np.float64)
        tracker["min_left_tcp_distance"] = min(
            tracker["min_left_tcp_distance"], float(np.linalg.norm(left_tcp - card_position))
        )
        tracker["min_right_tcp_distance"] = min(
            tracker["min_right_tcp_distance"], float(np.linalg.norm(right_tcp - card_position))
        )

        left_closed = robot.is_left_gripper_close()
        right_closed = robot.is_right_gripper_close()
        if tracker["first_closed_side"] is None:
            if left_closed and not right_closed:
                tracker["first_closed_side"] = "left"
            elif right_closed and not left_closed:
                tracker["first_closed_side"] = "right"
            elif left_closed and right_closed:
                tracker["first_closed_side"] = "both"

        contact_positions = task.get_gripper_actor_contact_position("081_playingcards")
        tracker["saw_card_gripper_contact"] |= len(contact_positions) > 0

        direction = 1.0 if tracker["expected_side"] == "right" else -1.0
        directed_displacement = direction * (card_position[0] - tracker["initial_position"][0])
        tracker["max_correct_direction_displacement"] = max(
            tracker["max_correct_direction_displacement"], float(directed_displacement)
        )

        expected_closed = right_closed if tracker["expected_side"] == "right" else left_closed
        if (
            tracker["saw_card_gripper_contact"]
            and expected_closed
            and directed_displacement > 0.02
        ):
            tracker["transport_while_expected_closed"] = True

        if (
            directed_displacement > 0.05
            and robot.is_left_gripper_open()
            and robot.is_right_gripper_open()
        ):
            tracker["released_after_correct_move"] = True

    @staticmethod
    def _finalize_move_card_phase_tracker(tracker):
        if tracker is None:
            return {}

        expected_side = tracker["expected_side"]
        selected_side = tracker["first_closed_side"]
        correct_hand_choice = selected_side == expected_side
        lateral_direction_correct = tracker["max_correct_direction_displacement"] > 0.05
        release_success = tracker["released_after_correct_move"]

        if not correct_hand_choice:
            failure_stage = "wrong_hand_or_no_close"
        elif not tracker["saw_card_gripper_contact"]:
            failure_stage = "contact_failure"
        elif not tracker["transport_while_expected_closed"]:
            failure_stage = "grasp_or_transport_failure"
        elif not lateral_direction_correct:
            failure_stage = "lateral_direction_failure"
        elif not release_success:
            failure_stage = "release_failure"
        else:
            failure_stage = "completed_behavior_chain"

        return {
            "expected_hand": expected_side,
            "selected_hand": selected_side or "none",
            "correct_hand_choice": correct_hand_choice,
            "card_gripper_contact": tracker["saw_card_gripper_contact"],
            "grasp_or_transport_success": tracker["transport_while_expected_closed"],
            "lateral_direction_correct": lateral_direction_correct,
            "release_success": release_success,
            "max_correct_direction_displacement": tracker["max_correct_direction_displacement"],
            "failure_stage": failure_stage,
        }

    def _run_single_task(self, policy: BasePolicy, epoch, task_name, task_config, task_idx=None, num_tasks=None):
        _orig_cwd = os.getcwd()
        os.chdir(self._ROBOTWIN2_DIR)
        try:
            device = policy.device
            result = self.env_manager.Create_env(
                task_name=task_name,
                head_camera_type=self.head_camera_type,
                seed=self.seed,
                task_num=self.episode_start + self.eval_episodes,
                instruction_type=self.instruction_type,
                task_config=task_config,
            )

            if not result:
                cprint("Failed to get valid seeds", "red")
                return {f"{task_name}/{task_config}: success_rate": 0.0, "test_mean_score": 0.0}

            seed_list, id_list, episode_info_list_total = result
            episode_end = self.episode_start + self.eval_episodes
            seed_list = seed_list[self.episode_start:episode_end]
            id_list = id_list[self.episode_start:episode_end]
            episode_info_list_total = episode_info_list_total[self.episode_start:episode_end]
            if len(seed_list) != self.eval_episodes:
                raise RuntimeError(
                    f"Requested episode shard [{self.episode_start}, {episode_end}), "
                    f"but only received {len(seed_list)} episodes"
                )
            self.env_manager.configure_eval_observation(
                profile=self.profile_eval,
                lean=self.lean_observation,
                defer_intermediate_render=self.defer_intermediate_render,
                rt_samples_per_pixel=self.rt_samples_per_pixel,
                camera_shader=self.camera_shader,
            )
            cprint(f"Found {len(seed_list)} valid task seeds: {seed_list}", "green")

            all_success = []
            all_episode_rewards = []
            episode_details = []
            episode_errors = []
            profile_episode_init_sec = 0.0
            observation_probes = []
            action_probes = []
            run_dir = os.path.basename(self.output_dir)

            for i, (episode_seed, task_id, episode_info_list) in enumerate(
                tqdm.tqdm(
                    zip(seed_list, id_list, episode_info_list_total),
                    total=len(seed_list),
                    desc=f"Eval RoboTwin2 {task_name} ({self.action_space_type})",
                    leave=False,
                    mininterval=self.tqdm_interval_sec,
                )
            ):
                try:
                    global_episode = self.episode_start + i
                    if self.deterministic_eval_seed is not None:
                        per_episode_seed = int(self.deterministic_eval_seed) + global_episode
                        random.seed(per_episode_seed)
                        np.random.seed(per_episode_seed)
                        torch.manual_seed(per_episode_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(per_episode_seed)
                    init_start = time.perf_counter()
                    self.env_manager.Init_task_env(
                        episode_seed,
                        task_id,
                        episode_info_list,
                        run_dir,
                        epoch,
                        task_config,
                        save_video=self.save_video,
                    )
                    profile_episode_init_sec += time.perf_counter() - init_start
                    self.env_manager.apply_eval_observation_flags()
                    episode_metadata = self._episode_task_metadata(task_name)
                    phase_tracker = self._init_move_card_phase_tracker(task_name)
                    policy.reset()

                    done = False
                    first_action_recorded = False
                    episode_reward = 0
                    episode_length = 0
                    controller_failure_count = 0
                    last_controller_failure = None
                    obs_history = deque(maxlen=self.n_obs_steps)

                    observation = self.env_manager.get_observation()
                    if self.profile_eval:
                        pointcloud = np.ascontiguousarray(observation['pointcloud'])
                        observation_probes.append({
                            'episode': global_episode,
                            'pointcloud_shape': list(pointcloud.shape),
                            'pointcloud_mean': float(pointcloud.mean()),
                            'pointcloud_std': float(pointcloud.std()),
                            'pointcloud_sha256': hashlib.sha256(pointcloud.tobytes()).hexdigest(),
                            'xyz_sha256': hashlib.sha256(np.ascontiguousarray(pointcloud[:, :3]).tobytes()).hexdigest(),
                            'rgb_sha256': hashlib.sha256(np.ascontiguousarray(pointcloud[:, 3:]).tobytes()).hexdigest(),
                        })
                    agent_pos_vector = self._agent_pos_from_observation(observation)
                    current_obs = {
                        'point_cloud': torch.from_numpy(observation['pointcloud']),
                        'agent_pos': torch.from_numpy(agent_pos_vector),
                    }
                    current_obs = self._add_task_onehot(current_obs, task_idx, num_tasks)

                    for _ in range(self.n_obs_steps):
                        obs_history.append(current_obs.copy())

                    while not done and episode_length < self.max_steps:
                        obs_history = self._ensure_history_task_onehot(obs_history, task_idx, num_tasks)
                        obs_dict = {
                            key: torch.stack([o[key] for o in obs_history], dim=0)
                            for key in obs_history[0].keys()
                        }

                        obs_dict_input = dict_apply(
                            obs_dict,
                            lambda x: x.unsqueeze(0).to(device=device),
                        )

                        with torch.no_grad():
                            action_dict = policy.predict_action(obs_dict_input, command=task_name)

                        action_chunk = action_dict.get(
                            'action_env', action_dict['action']
                        ).squeeze(0).detach().cpu().numpy()
                        if self.profile_eval and not first_action_recorded:
                            action_array = np.ascontiguousarray(action_chunk)
                            action_probes.append({
                                'episode': global_episode,
                                'action_shape': list(action_array.shape),
                                'action_sha256': hashlib.sha256(action_array.tobytes()).hexdigest(),
                            })
                            first_action_recorded = True
                        action_type = 'ee' if self.action_space_type == 'ee' else 'qpos'
                        use_ee_space = self.action_space_type == 'ee'

                        status, obs_history = self.env_manager.Take_action(
                            action_chunk,
                            obs_history,
                            self.n_obs_steps,
                            action_types=action_type,
                            use_ee_space=use_ee_space,
                        )
                        obs_history = self._ensure_history_task_onehot(obs_history, task_idx, num_tasks)
                        self._update_move_card_phase_tracker(phase_tracker)

                        episode_length += self.env_manager.get_last_action_step_count()
                        controller_execution = self.env_manager.get_last_controller_execution()
                        if (
                            controller_execution is not None
                            and not controller_execution['controller_success']
                        ):
                            controller_failure_count += 1
                            last_controller_failure = controller_execution

                        if status == "success":
                            done = True
                            episode_reward = 1.0
                            success = True
                        elif status == "fail":
                            done = True
                            episode_reward = 0.0
                            success = False
                        else:
                            done = False
                            success = False

                    all_success.append(success)
                    all_episode_rewards.append(episode_reward)
                    phase_metadata = self._finalize_move_card_phase_tracker(phase_tracker)
                    controller_metadata = {}
                    if self.action_space_type == 'ee':
                        controller_metadata = {
                            'controller_failure_count': controller_failure_count,
                            'last_controller_failure': last_controller_failure,
                        }
                    episode_details.append({
                        'episode': global_episode,
                        'success': success,
                        'reward': episode_reward,
                        'length': episode_length,
                        'seed': episode_seed,
                        **episode_metadata,
                        **phase_metadata,
                        **controller_metadata,
                    })

                    status_color = 'green' if success else 'red'
                    cprint(
                        f"{task_name}/{task_config}: "
                        f"Episode {i + 1}/{len(seed_list)}: "
                        f"{'SUCCESS' if success else 'FAIL'} "
                        f"(reward: {episode_reward:.2f}, steps: {episode_length})",
                        status_color,
                    )

                except Exception as e:
                    cprint(f"Episode {i} (seed {episode_seed}) failed with error: {e}", 'red')
                    import traceback
                    traceback.print_exc()
                    episode_errors.append(str(e))
                    all_success.append(False)
                    all_episode_rewards.append(0)
                    episode_details.append({
                        'episode': self.episode_start + i,
                        'success': False,
                        'reward': 0,
                        'length': 0,
                        'seed': episode_seed,
                        'error': str(e),
                    })

            success_rate = float(np.mean(all_success)) if len(all_success) > 0 else 0.0
            mean_reward = float(np.mean(all_episode_rewards)) if len(all_episode_rewards) > 0 else 0.0
            self.logger_util_test.record(success_rate)
            self.logger_util_test10.record(success_rate)

            log_prefix = f"{task_name}/{task_config}"
            log_data = {
                f"{log_prefix}: success_rate": success_rate,
                f"{log_prefix}: mean_reward": mean_reward,
                "test_mean_score": success_rate,
                "episode_start": self.episode_start,
                "episode_count": len(episode_details),
                "episode_details": episode_details,
            }
            if self.profile_eval:
                eval_profile = self.env_manager.get_eval_profile()
                eval_profile["episode_init_sec"] = profile_episode_init_sec
                eval_profile["observation_probes"] = observation_probes
                eval_profile["action_probes"] = action_probes
                log_data["eval_profile"] = eval_profile
            if episode_errors:
                log_data[f"{log_prefix}: runner_error_count"] = len(episode_errors)
                log_data[f"{log_prefix}: first_runner_error"] = episode_errors[0]
            if task_name == "move_playingcard_away":
                metric_names = (
                    "correct_hand_choice",
                    "card_gripper_contact",
                    "grasp_or_transport_success",
                    "lateral_direction_correct",
                    "release_success",
                )
                for metric_name in metric_names:
                    metric_values = [
                        detail[metric_name] for detail in episode_details
                        if metric_name in detail
                    ]
                    if metric_values:
                        log_data[f"{log_prefix}: {metric_name}_rate"] = float(np.mean(metric_values))
                for side in ("left", "right"):
                    side_details = [
                        detail for detail in episode_details
                        if detail.get("playingcard_side") == side
                    ]
                    if side_details:
                        side_success_rate = float(np.mean([
                            detail["success"] for detail in side_details
                        ]))
                        log_data[f"{log_prefix}: {side}_episodes"] = len(side_details)
                        log_data[f"{log_prefix}: {side}_success_rate"] = side_success_rate
                        for metric_name in metric_names:
                            metric_values = [
                                detail[metric_name] for detail in side_details
                                if metric_name in detail
                            ]
                            if metric_values:
                                log_data[
                                    f"{log_prefix}: {side}_{metric_name}_rate"
                                ] = float(np.mean(metric_values))
                        for failure_stage in (
                            "wrong_hand_or_no_close",
                            "contact_failure",
                            "grasp_or_transport_failure",
                            "lateral_direction_failure",
                            "release_failure",
                        ):
                            log_data[
                                f"{log_prefix}: {side}_{failure_stage}_count"
                            ] = sum(
                                detail.get("failure_stage") == failure_stage
                                for detail in side_details
                            )

            if task_name == "place_shoe":
                for side in ("left", "right"):
                    side_details = [
                        detail for detail in episode_details
                        if detail.get("arm_routing") == side
                    ]
                    if side_details:
                        log_data[f"{log_prefix}: {side}_episodes"] = len(side_details)
                        log_data[f"{log_prefix}: {side}_success_rate"] = float(
                            np.mean([detail["success"] for detail in side_details])
                        )

            cprint("\n" + "="*60, "cyan")
            cprint(f"RoboTwin 2.0 Evaluation Summary - {task_name}", "cyan")
            cprint(f"Task Config - {task_config}", "cyan")
            cprint("="*60, "cyan")
            cprint(f"Success Rate: {success_rate:.2%} ({np.sum(all_success)}/{len(all_success)})", "yellow")
            cprint(f"Mean Reward: {mean_reward:.3f}", "yellow")
            cprint(f"Action Space: {self.action_space_type}", "yellow")
            cprint(f"Instruction Type: {self.instruction_type}", "yellow")
            cprint("="*60 + "\n", "cyan")

            return log_data
        finally:
            os.chdir(_orig_cwd)

    def run(self, policy: BasePolicy, epoch, task_config):
        if self.task_entries is None:
            resolved_task_config = task_config or self.task_config
            return self._run_single_task(
                policy=policy,
                epoch=epoch,
                task_name=self.task_name,
                task_config=resolved_task_config,
            )

        task_names = list(dict.fromkeys(
            str(entry['task_name']) for entry in self.task_entries
        ))
        task_name_to_idx = {
            task_name: idx for idx, task_name in enumerate(task_names)
        }
        selected_entries = []
        for idx, entry in enumerate(self.task_entries):
            entry = _as_plain_dict(entry)
            entry_task_name = entry['task_name']
            if self.eval_task_name is not None and entry_task_name != self.eval_task_name:
                continue
            selected_entries.append((task_name_to_idx[str(entry_task_name)], entry))

        if len(selected_entries) == 0:
            raise ValueError(f"No RoboTwin eval tasks match eval_task_name={self.eval_task_name}")

        log_data = {}
        success_rates = []
        num_tasks = len(task_names)
        for task_idx, entry in selected_entries:
            entry_task_name = entry['task_name']
            entry_task_config = entry.get('setting') or entry.get('task_config') or task_config or self.task_config
            task_log = self._run_single_task(
                policy=policy,
                epoch=epoch,
                task_name=entry_task_name,
                task_config=entry_task_config,
                task_idx=task_idx,
                num_tasks=num_tasks,
            )
            log_data.update(task_log)
            success_rates.append(task_log.get('test_mean_score', 0.0))

        mean_success = float(np.mean(success_rates)) if len(success_rates) > 0 else 0.0
        log_data['multi_task/mean_success_rate'] = mean_success
        log_data['test_mean_score'] = mean_success
        return log_data

    def _save_results(self, episode_details, success_rate, mean_reward):
        results_dir = os.path.join(self.output_dir, 'evaluation_results')
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            results_dir,
            f'{self.task_name}_{self.action_space_type}_{timestamp}.txt',
        )

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("RoboTwin 2.0 Evaluation Results\n")
            f.write("="*60 + "\n")
            f.write(f"Task: {self.task_name}\n")
            f.write(f"Action Space: {self.action_space_type}\n")
            f.write(f"Instruction Type: {self.instruction_type}\n")
            f.write(f"Task Config: {self.task_config}\n")
            f.write(f"Eval Episodes: {self.eval_episodes}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"n_obs_steps: {self.n_obs_steps}\n")
            f.write(f"n_action_steps: {self.n_action_steps}\n")
            f.write("\n")
            f.write(f"Overall Success Rate: {success_rate:.2%}\n")
            f.write(f"Mean Reward: {mean_reward:.3f}\n")
            f.write("\n")
            f.write("Episode Details:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Episode':<10} {'Seed':<10} {'Success':<10} {'Reward':<10} {'Length':<10}\n")
            f.write("-"*60 + "\n")

            for detail in episode_details:
                f.write(
                    f"{detail['episode']:<10} "
                    f"{detail.get('seed', 'N/A'):<10} "
                    f"{detail['success']!s:<10} "
                    f"{detail['reward']:<10.2f} "
                    f"{detail['length']:<10}\n"
                )

        cprint(f"Detailed results saved to: {results_file}", "green")
