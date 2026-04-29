#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juanjosealcaraz

"""

import numpy as np

DEBUG = True


def _normalize_reset(ret):
    if isinstance(ret, tuple) and len(ret) >= 1:
        return ret[0]
    return ret


def _normalize_step(ret):
    if isinstance(ret, tuple):
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated) or bool(truncated)
            return obs, float(reward), done, (info if isinstance(info, dict) else {})
        if len(ret) == 4:
            obs, reward, done, info = ret
            return obs, float(reward), bool(done), (info if isinstance(info, dict) else {})
    obs, reward = ret[0], ret[1]
    return obs, float(reward), False, {}


def _format_env_action(system, action, n_slices):
    action = np.asarray(action, dtype=np.int16).reshape(-1)
    if action.size != int(n_slices):
        return action

    node_b = getattr(system, "node_b", None)
    if node_b is None:
        env = getattr(system, "env", None)
        node_b = getattr(env, "node_b", None) if env is not None else None

    expected = getattr(node_b, "n_slices_l1", None)
    if expected is None:
        return action
    if int(expected) != int(n_slices):
        return action

    # Current NodeB expects [cross_step_allocations, current_step_allocations].
    return np.concatenate([np.zeros_like(action), action], axis=0)


def _get_node_b(system):
    node_b = getattr(system, "node_b", None)
    cur = system
    depth = 0
    while node_b is None and hasattr(cur, "env") and depth < 10:
        cur = cur.env
        node_b = getattr(cur, "node_b", None)
        depth += 1
    return node_b


def _resource_usage_from_env_action(system, env_action, n_slices):
    a_arr = np.asarray(env_action, dtype=np.int64).ravel()
    if a_arr.size >= 2 * int(n_slices):
        now_use = int(a_arr[int(n_slices): 2 * int(n_slices)].sum())
    else:
        now_use = int(a_arr.sum())

    cross_use = 0
    node_b = _get_node_b(system)
    if node_b is not None:
        try:
            leases = getattr(node_b, "slice_leases", [])
            cross_use = sum(
                int(lease.get("prb", 0))
                for lease in leases
                if int(lease.get("remain", 0)) > 0
            )
        except Exception:
            cross_use = 0
    return now_use + cross_use

class Learner:
    '''
    Auxiliary class with elements and variables for the KBRL agent
    '''
    def __init__(self, algorithm, indexes, initial_action, security_factor):
        self.algorithm = algorithm
        self.indexes = indexes
        self.initial_action = initial_action
        self.security_factor = security_factor
        self.step = 1

class KBRL_Control:
    '''
    KBRL_Control: Kernel Model-Based RL for online learning and control.
    Its objective is to assign spectrum resources (RBs) among several ran slices
    '''
    def __init__(self, learners, n_prbs, alfa = 0.05, accuracy_range = [0.99, 0.999]):
        self.learners = learners # there must be one learner instance per slice
        self.accuracy_range = accuracy_range
        self.n_slices = len(learners)
        self.n_prbs = n_prbs
        self.alfa = alfa
        self.adjusted = 0
        self.action = np.array([h.initial_action for h in learners], dtype = np.int16)
        self.security_factors = np.array([h.security_factor for h in learners], dtype = np.int16)        
        self.margins = np.array([0]*self.n_slices, dtype = np.int16)
        intial_value = (self.accuracy_range[0] + self.accuracy_range[1])/2
        self.accuracies = np.full((self.n_slices, self.n_prbs), intial_value, dtype = float)

    def select_action(self, state):
        action = np.zeros((self.n_slices), dtype = np.int16)
        adjusted = 0
        for i, h in enumerate(self.learners):
            algorithm = h.algorithm
            _i_ = h.indexes
            l1_state = state[_i_]
            min_prbs = 0
            max_prbs = self.n_prbs

            # we check the prediction for assignment "offset" prbs below the action
            offset = self.security_factors[i]
            margin = 0
            for l1_prbs in range(max(min_prbs - offset,0), max_prbs+1):
                x = np.append(l1_state, l1_prbs/self.n_prbs)
                prediction = algorithm.predict(x)
                if prediction == 1:
                    a = min(self.n_prbs, l1_prbs + offset)
                    margin = a - l1_prbs
                    l1_prbs = a
                    break
            action[i] = l1_prbs
            self.margins[i] = margin

        assigned_prbs = action.sum()
        if assigned_prbs > self.n_prbs: # not enough resources
            adjusted = 1
            action, diff = self.adjust_action(action, assigned_prbs, self.n_prbs)
            self.margins = self.margins - diff
        
        self.action = action

        return action, adjusted

    def adjust_action(self, action, assigned_prbs, n_prbs):
        relative_p = action / assigned_prbs
        new_action = np.array([np.floor(n_prbs * p) for p in relative_p], dtype=np.int16)
        return new_action, action - new_action

    def update_control(self, state, action, reward):
        hits = np.zeros((self.n_slices), dtype = np.int16)

        for i, h in enumerate(self.learners):
            algorithm = h.algorithm
            _i_ = h.indexes
            l1_state = state[_i_]
            l1_action = action[i]
            x = np.append(l1_state, l1_action/self.n_prbs)
            y_pred = algorithm.predict(x)
            y = reward[i]
            hit = y == y_pred
            margin = max(0, self.margins[i])
            if y_pred == 1:
                if hit == 0: # with the same or less margin we would have made the same mistake
                    self.accuracies[i,0:margin+1] = (1 - self.alfa) * self.accuracies[i,0:margin+1]
                else: # with the same or more margin we would have succeeded as well
                    self.accuracies[i,margin:] = (1 - self.alfa) * self.accuracies[i,margin:] + self.alfa
            if not self.adjusted: # if the action was not adjusted then update security_factor
                self.security_factors[i] = np.argmax(self.accuracies[i,:] > self.accuracy_range[0])

            hits[i] = hit
            # sample augmentation
            if y == 1: # fulfilled
                for a in range(l1_action, self.n_prbs + 1): # same or more prbs would obtain the same y
                    new_x = np.append(l1_state, a/self.n_prbs)
                    y_pred = algorithm.predict(new_x)
                    algorithm.update(new_x, y)
            else: # not fulfilled (y = -1)
                for a in range(0, l1_action + 1): #  same or fewer prbs would obtain the same y
                    new_x = np.append(l1_state, a/self.n_prbs)
                    y_pred = algorithm.predict(new_x)
                    algorithm.update(new_x, y)

        return hits

    def _build_output(
        self,
        reward_history,
        resources_history,
        allocated_prbs_history,
        hits_history,
        adjusted_actions,
        SLA_history,
        SLA_labels_sum_history,
        violation_history,
        end_idx=None,
        extra_payload=None,
    ):
        if end_idx is None:
            end_idx = len(reward_history)
        payload = {
            'reward': reward_history[:end_idx],
            'resources': resources_history[:end_idx],
            'allocated_prbs': allocated_prbs_history[:end_idx],
            'hits': hits_history[:, :end_idx],
            'adjusted': adjusted_actions[:end_idx],
            'SLA': SLA_history[:end_idx],
            'SLA_labels_sum': SLA_labels_sum_history[:end_idx],
            'violation': violation_history[:end_idx],
        }
        if extra_payload:
            payload.update(extra_payload)
        return payload

    def run(
        self,
        system,
        steps,
        learning_time=-1,
        reset_env=True,
        save_every=0,
        save_path=None,
        save_extras=None,
    ):
        action = self.action
        initial_env_action = _format_env_action(system, action, self.n_slices)

        SLA_history = np.zeros((steps), dtype = np.int16)
        SLA_labels_sum_history = np.zeros((steps), dtype=np.int16)
        reward_history = np.zeros((steps), dtype = np.float32)
        violation_history = np.zeros((steps), dtype = np.int16)
        adjusted_actions = np.zeros((steps), dtype = np.int16)
        resources_history = np.zeros((steps), dtype = np.int16)
        allocated_prbs_history = np.zeros((steps, len(initial_env_action)), dtype=np.int16)
        hits_history = np.zeros((len(action),steps), dtype = np.int16)

        state = _normalize_reset(system.reset()) if reset_env else getattr(system, "obs", None)
        if state is None:
            state = _normalize_reset(system.reset())

        for i in range(steps):
            executed_action = np.array(action, copy=True)
            env_action = _format_env_action(system, executed_action, self.n_slices)
            new_state, reward, done, info = _normalize_step(system.step(env_action))
            SLA_labels = info['SLA_labels']
            if learning_time < 0 or i < learning_time:
                hits = self.update_control(state, executed_action, SLA_labels)
            else:
                hits = np.zeros((self.n_slices), dtype=np.int16)
            action, self.adjusted = self.select_action(new_state)
            state = new_state

            applied_env_action = np.asarray(info.get("delayed_applied_action", env_action), dtype=np.int16).ravel()
            SLA_labels_sum_history[i] = int(np.asarray(SLA_labels).sum())
            SLA_history[i] = int(info['total_violations'] <= 0)
            reward_history[i] = reward
            violation_history[i] = info['total_violations']
            resources_history[i] = _resource_usage_from_env_action(system, applied_env_action, self.n_slices)
            allocated_prbs_history[i] = applied_env_action
            adjusted_actions[i] = self.adjusted
            hits_history[:,i] = hits

            if save_path and save_every and ((i + 1) % int(save_every) == 0):
                payload = self._build_output(
                    reward_history,
                    resources_history,
                    allocated_prbs_history,
                    hits_history,
                    adjusted_actions,
                    SLA_history,
                    SLA_labels_sum_history,
                    violation_history,
                    end_idx=i + 1,
                    extra_payload=save_extras,
                )
                np.savez(save_path, **payload)

            if done:
                state = _normalize_reset(system.reset())

        print('mean resources = {}'.format(resources_history.mean()))
        print('total violations = {}'.format(violation_history.sum()))
        print('mean adjusted = {}'.format(adjusted_actions.mean()))
        print('mean accuracy = {}'.format(hits_history.mean(axis=1)))

        output = self._build_output(
            reward_history,
            resources_history,
            allocated_prbs_history,
            hits_history,
            adjusted_actions,
            SLA_history,
            SLA_labels_sum_history,
            violation_history,
            extra_payload=save_extras,
        )

        if save_path:
            np.savez(save_path, **output)

        return output
