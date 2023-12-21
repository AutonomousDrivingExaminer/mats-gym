from __future__ import annotations

from gymnasium.utils.env_checker import data_equivalence


def seed_action_spaces(env):
    if hasattr(env, "agents"):
        for i, agent in enumerate(env.agents):
            env.action_space(agent).seed(42 + i)


def seed_observation_spaces(env):
    if hasattr(env, "agents"):
        for i, agent in enumerate(env.agents):
            env.observation_space(agent).seed(42 + i)
            

def seed_test(env_fn):
    """Check that two parallel environments execute the same way."""

    def run_episode(env):
        env.reset(seed=42)

        # seed action spaces to ensure sampled actions are the same
        seed_action_spaces(env)

        # seed observation spaces to ensure first observation is the same
        seed_observation_spaces(env)
        iter = 0

        obs, info = env.reset(seed=42)
        episode = [
            (obs, info)
        ]

        seed_action_spaces(env)
        done = False
        while not done: 
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            episode.append((obs, rewards, terminations, truncations, infos))
            done = all(terminations.values()) or all(truncations.values())
        env.close()
        return episode
    
    env = env_fn()
    traj_1 = run_episode(env)
    env = env_fn()
    traj_2 = run_episode(env)

    assert len(traj_1) == len(traj_2), "Incorrect episode length"
    start_1, start_2 = traj_1[0], traj_2[0]
    #assert data_equivalence(start_1, start_2), "Incorrect start obs or info"
    for step_1, step_2 in zip(traj_1[1:], traj_2[1:]):
        obs1, rewards1, terminations1, truncations1, infos1 = step_1
        obs2, rewards2, terminations2, truncations2, infos2 = step_2
        assert data_equivalence(obs1, obs2), "Incorrect observations"
        assert data_equivalence(rewards1, rewards2), "Incorrect values for rewards"
        assert data_equivalence(terminations1, terminations2), "Incorrect terminations."
        assert data_equivalence(truncations1, truncations2), "Incorrect truncations"
        #assert data_equivalence(infos1, infos2), "Incorrect infos"

