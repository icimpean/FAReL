from mesa.datacollection import DataCollector
from collections import defaultdict
import pandas as pd


class LogCollector(DataCollector):
    """ 
    Inherits from the DataCollector from the mesa framework,
    and overwrites some functions for our simulator
    """

    def __init__(self, model_reporters=None, agent_reporters=None, tables=None):
        self.agent_vars = {}  # Needs to be before super call, or it's not registered for the remainder of the init
        super(LogCollector, self).__init__(model_reporters, agent_reporters, tables)

    def _new_agent_reporter(self, name, reporter):
        super(LogCollector, self)._new_agent_reporter(name, reporter)
        self.agent_vars[name] = []

    def collect(self, model):
        """ collect only logs from agents that make a transation"""
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(model))

        if self.agent_reporters:
            for var, reporter in self.agent_reporters.items():
                agent_records = [(agent.unique_id, reporter(agent)) for agent in model.schedule.agents if agent.active]
                self.agent_vars[var].append(agent_records)

    def get_agent_vars_dataframe(self):
        """ Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        This function was modified from the original implementation in mesa
        to return None if there are no entries at all

        (the df.index.names = ["Step", "AgentID"] line crashes with "ValueError:
        Length of new names must be 1, got 2" if there are no entries in original
        mesa implementation)

        """
        data = defaultdict(dict)
        found_entries = False

        for var, records in self.agent_vars.items():
            for step, entries in enumerate(records):
                for entry in entries:
                    agent_id = entry[0]
                    val = entry[1]
                    data[(step, agent_id)][var] = val
                    found_entries = True

        if not found_entries:
            return None

        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.names = ["Step", "AgentID"]
        return df
