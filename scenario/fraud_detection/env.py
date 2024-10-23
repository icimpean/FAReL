from enum import Enum, auto
from typing import List, Union

import numpy as np

from scenario import CombinedState, Feature, Scenario
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


ALL_FRAUD_FEATURES = [f for f in FraudFeature]
NUM_FRAUD_FEATURES = len(FraudFeature)
context_features = [FraudFeature.satisfaction, FraudFeature.fraud_percentage]
individual_features = [f for f in FraudFeature if f not in context_features]


class TransactionModelMDP(Scenario):
    def __init__(self, transaction_model, do_reward_shaping=False, num_transactions=None,
                 reward_biases=[], exclude_from_distance=()):
        # Super call
        features = [feature for feature in FraudFeature]
        nominal_features = [FraudFeature.card_id, FraudFeature.merchant_id, FraudFeature.country,
                            FraudFeature.continent, FraudFeature.currency]
        numerical_features = [  # FraudFeatures.satisfaction, FraudFeatures.fraud_percentage,
            # ==> fraud company features, not individual for fairness notion
            FraudFeature.month, FraudFeature.day, FraudFeature.weekday, FraudFeature.hour, FraudFeature.amount]
        super(TransactionModelMDP, self).__init__(features=features, nominal_features=nominal_features,
                                                  numerical_features=numerical_features,
                                                  seed=transaction_model.parameters["seed"],
                                                  exclude_from_distance=exclude_from_distance)
        #
        self.transaction_model = transaction_model
        self.do_reward_shaping = do_reward_shaping
        self.num_transactions = num_transactions
        self.reward_biases = reward_biases
        self._params = self.transaction_model.parameters
        self.input_shape = NUM_FRAUD_FEATURES
        self.actions = [a for a in FraudActions]
        self.maxima = {
            f: self._get_max_norm(f)
            for f in ("frac_month", "frac_monthday", "frac_weekday", "frac_hour", "country_frac")
        }
        self.maxima["card_id"] = 10000
        self.maxima["merchant_id"] = max(1, len(self.transaction_model.merchants) - 1)
        self.maxima["continent"] = max(1, len(self.transaction_model.continents) - 1)
        self.maxima["amount"] = 8000
        self.maxima["currency"] = max(1, len(self.transaction_model.currencies) - 1)

        # Don't know (full) next state beforehand with this model => update once next transaction is seen
        self.previous_state = None
        self.customer = None
        self.action = None
        self.reward = None
        self.done = None
        self.info = None
        self.state = None
        self.t = 0
        self.n_transactions = 0

        # Replicate transaction model and schedule
        self.scheduler = self.transaction_model.schedule
        self.transaction_model.pre_step()
        self._buffer = self.scheduler.agent_buffer(shuffled=True)

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
        # current date
        self.transaction_model.curr_global_date = self.transaction_model.parameters['start_date']
        self.transaction_model.next_customer_id = 0
        self.transaction_model.next_fraudster_id = 0
        self.transaction_model.next_card_id = 0
        self.transaction_model.merchants = self.transaction_model.initialise_merchants()
        self.transaction_model.customers = self.transaction_model.initialise_customers()
        self.transaction_model.fraudsters = self.transaction_model.initialise_fraudsters()
        # set termination status
        self.n_transactions = 0
        self.transaction_model.terminated = False
        #
        self.transaction_model.revenue = 0
        self.transaction_model.genuine_transactions = 0
        self.transaction_model.fraudulent_transactions = 0
        self.transaction_model.lost_customers = 0
        self.transaction_model.pre_step()
        self._buffer = self.scheduler.agent_buffer(shuffled=True)
        self.customer = self._get_customer()
        self.state = self.full_state(self.customer, self.transaction_model)
        self.previous_state = None
        # Initialise features
        self.init_features(self.state)
        return self.state

    def step(self, action):
        self.authorise_transaction(self.customer, action)
        self.generate_sample()
        return self.state, self.reward, self.done, self.info

    def generate_sample(self):
        next_customer = self._get_customer()
        self.customer = next_customer
        self.state = self.full_state(self.customer, self.transaction_model)
        return self.state

    def calc_goodness(self, sample: CombinedState):
        return self.customer.fraudster

    def calculate_rewards(self, sample: CombinedState, goodness):
        reward_ignore = 1
        reward_authenticate = -1

        if self.customer.fraudster:
            reward_ignore *= -1
            reward_authenticate *= -1

        rewards = {FraudActions.ignore: reward_ignore, FraudActions.authenticate: reward_authenticate}
        return rewards

    def authorise_transaction(self, customer, action):
        # ask the user for authentication
        auth_result = 1
        if action:
            auth_result = customer.give_authentication()

        # calculate the reward
        reward = 0
        if auth_result is not None:
            reward += customer.fraudster * (-customer.curr_amount)
            reward += (1 - customer.fraudster) * (0.003 * customer.curr_amount + 0.01)

        # Do some reward shaping: reward success after one authentication
        if self.do_reward_shaping:
            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1  # negative initial reward means lost amount ==> fraud
            # else: reward = 0

        # Add bias
        for bias in self.reward_biases:
            self.reward += bias.get_bias(self.state)

        if customer.fraudster:
            self.transaction_model.fraudulent_transactions += 1
        else:
            self.transaction_model.genuine_transactions += 1
        self.transaction_model.revenue += reward
        self.n_transactions += 1

        # update agent
        self.reward = reward
        self.previous_state = self.state
        self.action = action
        self.done = self.transaction_model.terminated
        if not self.done and self.num_transactions is not None:
            if self.n_transactions >= self.num_transactions:
                self.done = True

        self.info = {"true_action": FraudActions.authenticate.value if customer.fraudster
                     else FraudActions.ignore.value, "fraudster": customer.fraudster}
        self.t += 1

    def _normalise_features(self, state: Union[CombinedState, np.ndarray], features: List[FraudFeature] = None,
                            indices=None):
        if isinstance(state, CombinedState):
            if features is None:
                new_values = self.normalise_state(state)
            else:
                new_values = self.normalise_state(state, individual_only=True)
        else:
            new_values = state
        # Already transformed into array, return requested indices
        if indices:
            new_values = new_values[indices]
        return new_values

    def _get_max_norm(self, parameter):
        return max(1, self._params[parameter].shape[0] - 1)

    def normalise_state(self, state, individual_only=False):
        """Normalise based on MultiMAuS transaction model and its parameters:

        ``The transaction amounts range from about 0.5 to 7,800 Euro (after converting everything to the same
        currency). Purchases are made with credit cards from 126 countries (19 for fraudulent transactions) in
        5 (3) different currencies. There are a total of 7 merchants...``
        """
        sat, fraud_per, month, day, weekday, hour, card_id, merchant_id, country, continent, amount, currency = state.to_array()
        if individual_only:
            norm_array = np.array([
                month / self.maxima["frac_month"],
                day / self.maxima["frac_monthday"],
                weekday / self.maxima["frac_weekday"],
                hour / self.maxima["frac_hour"],
                card_id / self.maxima["card_id"],
                merchant_id / self.maxima["merchant_id"],
                country / self.maxima["country_frac"],
                continent / self.maxima["continent"],
                amount / self.maxima["amount"],
                currency / self.maxima["currency"],
            ])
        else:
            norm_array = np.array([
                sat,
                fraud_per,
                month / self.maxima["frac_month"],
                day / self.maxima["frac_monthday"],
                weekday / self.maxima["frac_weekday"],
                hour / self.maxima["frac_hour"],
                card_id / self.maxima["card_id"],
                merchant_id / self.maxima["merchant_id"],
                country / self.maxima["country_frac"],
                continent / self.maxima["continent"],
                amount / self.maxima["amount"],
                currency / self.maxima["currency"],
            ])
        return norm_array

    def get_all_entities_in_state(self, state: CombinedState, action, true_action, score, reward):
        return [(state, action, true_action, score, reward)]
