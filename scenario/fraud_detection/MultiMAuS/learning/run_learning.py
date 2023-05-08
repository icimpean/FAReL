import numpy as np
import os
from datetime import datetime
from pytz import timezone
import matplotlib.pyplot as plt

from scenario.fraud_detection.MultiMAuS.dqn import DQNAgent
from scenario.fraud_detection.MultiMAuS.learning.agent_qlean import QLearnAgent
from scenario.fraud_detection.MultiMAuS.learning.agent_bandit import BanditAgent
from scenario.fraud_detection.env import TransactionModelMDP, NUM_FRAUD_FEATURES
from scenario.fraud_detection.MultiMAuS.learning.environment import Environment
from scenario.fraud_detection.MultiMAuS.simulator import parameters
from scenario.fraud_detection.MultiMAuS.simulator.transaction_model import TransactionModel
from scenario.fraud_detection.MultiMAuS.experiments import rewards

if __name__ == '__main__':

    auths = [
             # (Environment(BanditAgent(do_reward_shaping=True)), 'Bandit (reward shaping)'),
             # (RandomAuthenticator(), 'Random'),

             # (OracleAuthenticator(), 'Oracle'),

             # (HeuristicAuthenticator(50), 'Heuristic'),
             # (NeverSecondAuthenticator(), 'NeverSecond'),
             # (AlwaysSecondAuthenticator(), 'AlwaysSecond'),

             (Environment(QLearnAgent('zero', 0.01, 0.1, 0.1, False)), 'Q-Learn'),
             (Environment(QLearnAgent('zero', 0.01, 0.1, 0.1, True)), 'Q-Learn with reward shaping'),

             (Environment(BanditAgent()), 'Bandit'),
             (Environment(BanditAgent(do_reward_shaping=True)), 'Bandit with reward shaping'),
        (False, "DQN"),
        (True, "DQN with reward shaping"),
        (False, "DQN, amount only"),
        #
    ]

    authenticator = None
    auth_name = ''

    for k in range(len(auths)):

        all_monetary_rewards = []

        if auth_name != 'Q-Learning (from scratch)':
            authenticator, auth_name = auths[k]
        else:  # if we just did Q-Learning, run it again with the pre-trained one
            auth_name = 'Q-Learning (pre-trained)'

        seed = 0

        print("-----")
        print(auth_name)
        print("-----")

        sum_monetary_rewards = None

        for i in range(0, 5):  # 20
            seed_i = seed + i

            # the parameters for the simulation
            params = parameters.get_default_parameters()
            params['seed'] = seed_i
            params['init_satisfaction'] = 0.9
            params['stay_prob'] = [0.9, 0.6]
            params['num_customers'] = 100
            params['num_fraudsters'] = 10
            # params['end_date'] = datetime(2016, 12, 31).replace(tzinfo=timezone('US/Pacific'))
            params['end_date'] = datetime(2016, 7, 31).replace(tzinfo=timezone('US/Pacific'))

            # path = f'results/{auth_name}_{seed}_{int(params["init_satisfaction"] * 10)}_' \
            #        f'{params["num_customers"]}_{params["num_fraudsters"]}_' \
            #        f'{params["end_date"].year}'
            dir_path = f'results/{auth_name}/'
            os.makedirs(dir_path, exist_ok=True)
            path = f'{dir_path}/{auth_name}_{seed_i}_{int(params["init_satisfaction"] * 10)}_' \
                   f'{params["num_customers"]}_{params["num_fraudsters"]}_' \
                   f'{params["end_date"].year}'

            if os.path.exists(path+'.npy'):
                monetary_rewards = np.load(path+'.npy')
            else:

                if "DQN" in auth_name:
                    # get the model for transactions
                    dqn_agent = DQNAgent(input_shape=NUM_FRAUD_FEATURES, h1=64, actions=[0, 1], learning_rate=0.001)
                    authenticator = TransactionModelMDP(dqn_agent, do_reward_shaping=auths[k][0])
                    model = TransactionModel(params, authenticator=authenticator)
                    authenticator.transaction_model = model

                else:
                    # get the model for transactions
                    model = TransactionModel(params, authenticator=authenticator)

                # run
                while not model.terminated:
                    model.step()

                agent_vars = model.log_collector.get_agent_vars_dataframe()
                agent_vars.index = agent_vars.index.droplevel(1)
                monetary_rewards = rewards.monetary_reward_per_timestep(agent_vars)

                np.save(path, monetary_rewards)

            all_monetary_rewards.append(monetary_rewards)

            if sum_monetary_rewards is None:
                sum_monetary_rewards = monetary_rewards
            else:
                sum_monetary_rewards += monetary_rewards

            # seed += 1

        sum_monetary_rewards /= (i+1)
        if k == 0:
            color = 'r'
        elif k == 1:
            color = 'r--'
        elif k == 2:
            color = 'b'
        elif k == 3:
            color = 'b--'
        # plt.plot(range(len(monetary_rewards)), np.cumsum(monetary_rewards), color, label=auth_name)
        # plt.plot(range(len(monetary_rewards)), np.cumsum(monetary_rewards), label=auth_name)

        cumulative = np.cumsum(all_monetary_rewards, axis=1)
        means = np.mean(cumulative, axis=0)
        vars = np.std(cumulative, axis=0)
        # print(cumulative.shape)
        # print(means.shape)
        # print(vars.shape)
        # exit()
        p = plt.plot(range(len(monetary_rewards)), means, label=auth_name, color="black" if k == len(auths) - 1 else None)
        plt.fill_between(range(len(monetary_rewards)), means - vars, means + vars, alpha=0.1, color=p[0].get_color())

    plt.xlabel('time step')
    plt.ylabel('monetary reward (cumulative)')
    plt.legend()
    plt.tight_layout()
    plt.show()
