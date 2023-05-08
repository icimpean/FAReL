from enum import Enum, auto

import numpy as np

from scenario import CombinedState, Feature
from scenario.fraud_detection.MultiMAuS.simulator.customers import BaseCustomer
from scenario.fraud_detection.MultiMAuS.simulator.transaction_model import TransactionModel


class FraudActions(Enum):
    ignore = 0
    authenticate = 1


class FraudFeature(Feature):
    satisfaction = 0
    fraud_percentage = auto()
    month = auto()
    day = auto()
    weekday = auto()
    hour = auto()
    card_id = auto()
    merchant_id = auto()
    country = auto()
    continent = auto()
    amount = auto()
    currency = auto()


NUM_FRAUD_FEATURES = len(FraudFeature)
context_features = [FraudFeature.satisfaction, FraudFeature.fraud_percentage]
individual_features = [f for f in FraudFeature if f not in context_features]


class TransactionModelMDP(object):
    def __init__(self, transaction_model, do_reward_shaping=False):
        self.transaction_model = transaction_model
        self.do_reward_shaping = do_reward_shaping
        self._params = self.transaction_model.parameters
        self.input_shape = NUM_FRAUD_FEATURES
        self.actions = [a for a in FraudActions]

        # Don't know (full) next state beforehand with this model => update once next transaction is seen
        self.previous_state = None
        self.customer = None
        self.action = None
        self.reward = None
        self.done = None
        self.info = None
        self.state = None
        self.t = 0

        # Replicate transaction model and schedule
        self.scheduler = self.transaction_model.schedule
        self.transaction_model.pre_step()
        self._buffer = self.scheduler.agent_buffer(shuffled=True)

        #
        self.nominal_features = [FraudFeature.card_id, FraudFeature.merchant_id, FraudFeature.country,
                                 FraudFeature.continent, FraudFeature.currency]
        self.numerical_features = [#FraudFeatures.satisfaction, FraudFeatures.fraud_percentage, ==> fraud company features, not individual for fairness notion
                                   FraudFeature.month, FraudFeature.day, FraudFeature.weekday, FraudFeature.hour, FraudFeature.amount]

    @staticmethod
    def full_state(customer: BaseCustomer, transaction_model: TransactionModel):
        # global_date = customer.model.curr_global_date.replace(tzinfo=None)
        local_date = customer.local_datetime.replace(tzinfo=None)

        month = local_date.month
        day = local_date.day
        weekday = local_date.weekday()
        hour = local_date.hour

        country = transaction_model.countries[customer.country]
        continent = transaction_model.continents[transaction_model.continents_countries[customer.country]]
        currency = transaction_model.currencies[customer.currency]

        ft = transaction_model.fraudulent_transactions
        gt = transaction_model.genuine_transactions
        st = ft + gt
        fraud_percentage = 0 if st == 0 else ft / st

        # print(customer.card_id, customer.curr_merchant.unique_id)

        state = np.array([
            # Company satisfaction
            sum((customer.satisfaction for customer in transaction_model.customers)) / len(transaction_model.customers),
            # transaction_model.revenue,  # TODO: redefine
            fraud_percentage,
            # transaction_model.lost_customers,

            # Customer/Transaction
            month,
            day,
            weekday,
            hour,
            #
            customer.card_id,
            customer.curr_merchant.unique_id,
            country,
            continent,
            customer.curr_amount,
            currency,
        ])
        # state = np.array([get_state(customer)])
        state = CombinedState.from_array(state, context_features=context_features,
                                         individual_features=individual_features)

        return state

    def _get_customer(self):
        transaction_attempted = False

        # Wait for a transaction request
        while not transaction_attempted:
            # Get the next customer in line to act
            try:
                customer = next(self._buffer)
            # Empty customer buffer => reset all customers, based on mesa's RandomActivation.step() method
            # and run the post step method of the transaction model as well as the preprocessing for the next buffer
            except StopIteration:
                self.scheduler.steps += 1
                self.scheduler.time += 1
                self.transaction_model.post_step()
                #
                self.transaction_model.pre_step()
                self._buffer = self.scheduler.agent_buffer(shuffled=True)
                customer = next(self._buffer)

            # Let the customer make a transaction
            # (model may have to authorise transaction with the authorise_transaction method below)
            transaction_attempted = customer.step_rl()

        # noinspection PyUnboundLocalVariable
        return customer

    def reset(self):
        self.transaction_model.pre_step()
        self._buffer = self.scheduler.agent_buffer(shuffled=True)
        self.customer = self._get_customer()
        self.state = self.full_state(self.customer, self.transaction_model)
        self.previous_state = None
        return self.state

    def step(self, action):
        self.authorise_transaction(self.customer, action)
        next_customer = self._get_customer()
        self.customer = next_customer
        self.state = self.full_state(self.customer, self.transaction_model)
        return self.state, self.reward, self.done, self.info

    def authorise_transaction(self, customer, action):
        # get the current state we will show to the agent
        # state = full_state(customer, self.transaction_model)
        # self.state = state

        # # Update previous timestep
        # if self.previous_state is not None:
        #     # Store the experience in memory and train the agent
        #     experience = Experience(self.previous_state, self.action, self.reward, self.done, self.state, None)
        #     self.agent.store_experience(experience)
        #     agent_loss = self.agent.train(self.t - 1)

        # call the step function of the model
        # action = self.agent.select_action(self.state, self.t, episode=0)

        # ask the user for authentication
        auth_result = 1
        if action:
            auth_result = customer.give_authentication()

        # calculate the reward
        reward = 0
        if auth_result is not None:
            reward += customer.fraudster * (-customer.curr_amount)
            reward += (1 - customer.fraudster) * (0.003 * customer.curr_amount + 0.01)

        # FROM MultiMAuS: do some reward shaping: reward success after one authentication
        # if self.do_reward_shaping:
        #     if reward > 0:  # transaction was successful
        #         if action == 0:
        #             reward += 0.2
        if self.do_reward_shaping:  # TODO
            if reward > 0:
                reward = 1
            else:
                reward = -1  # negative initial reward means lost amount ==> fraud

        if customer.fraudster:
            self.transaction_model.fraudulent_transactions += 1
        else:
            self.transaction_model.genuine_transactions += 1
        self.transaction_model.revenue += reward

        # update agent
        self.reward = reward
        self.previous_state = self.state
        self.action = action
        self.done = self.transaction_model.terminated
        self.info = {"true_action": FraudActions.authenticate.value if customer.fraudster
                     else FraudActions.ignore.value, "fraudster": customer.fraudster}
        self.t += 1

    def similarity_metric(self, state1, state2, distance="HMOM", alpha=1.0, exp=True):
        n_state1 = self.normalise_state(state1)
        n_state2 = self.normalise_state(state2)

        num1 = np.array([n_state1[f.value] for f in self.numerical_features])
        nom1 = np.array([n_state1[f.value] for f in self.nominal_features])
        num2 = np.array([n_state2[f.value] for f in self.numerical_features])
        nom2 = np.array([n_state2[f.value] for f in self.nominal_features])

        # Heterogeneous Euclidean-Overlap Metric (HEOM)
        if distance == 'HEOM':
            d = np.sum(np.abs(num1 - num2)) + np.sum(nom1 != nom2)
        # Heterogeneous Manhattan-Overlap Metric (HMOM)
        elif distance == 'HMOM':
            d = np.sum((num1 - num2) ** 2) + np.sum(nom1 != nom2)
        else:
            raise ValueError(f"Expected distance: HEOM or HMOM, got: {distance}")

        return np.exp(-alpha * d) if exp else d

    def _get_max_norm(self, parameter):
        return max(1, self._params[parameter].shape[0] - 1)

    def normalise_state(self, state):
        """Normalise based on MultiMAuS transaction model and its parameters:

        ``The transaction amounts range from about 0.5 to 7,800 Euro (after converting everything to the same
        currency). Purchases are made with credit cards from 126 countries (19 for fraudulent transactions) in
        5 (3) different currencies. There are a total of 7 merchants...``
        """
        sat, fraud_per, month, day, weekday, hour, card_id, merchant_id, country, continent, amount, currency = state.to_array()
        norm_array = np.array([
            sat,
            fraud_per,
            month / self._get_max_norm("frac_month"),
            day / self._get_max_norm("frac_monthday"),
            weekday / self._get_max_norm("frac_weekday"),
            hour / self._get_max_norm("frac_hour"),
            card_id / 10000,
            merchant_id / max(1, len(self.transaction_model.merchants) - 1),
            country / self._get_max_norm("country_frac"),
            continent / max(1, len(self.transaction_model.continents) - 1),
            amount / 8000,
            currency / max(1, len(self.transaction_model.currencies) - 1),
        ])
        return norm_array

    @staticmethod
    def get_individual(state):
        individual = {f: state[f] for f in individual_features}
        return individual
