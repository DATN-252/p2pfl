#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#

"""Data feedback command."""

import json
from typing import Any, Callable
from p2pfl.communication.commands.command import Command
from p2pfl.management.logger import logger

class DataFeedbackCommand(Command):
    """Command to receive new labeled data during runtime (online learning)."""

    def __init__(self, push_data_fn: Callable[[dict[str, Any]], None]) -> None:
        """
        Initialize the command.

        Args:
            push_data_fn: Function to push data to the node.
        """
        super().__init__()
        self.push_data_fn = push_data_fn

    @staticmethod
    def get_name() -> str:
        """Get the command name."""
        return "data_feedback"

    def execute(self, source: str, round: int, *args, **kwargs) -> None:
        """
        Execute the command.

        Args:
            source: The source of the command.
            round: The round of the command (unused for this command).
            *args: The data in JSON string format.
            **kwargs: The command keyword arguments.
        """
        if len(args) < 1:
            logger.error("DataFeedbackCommand", f"Received empty data feedback from {source}")
            return

        try:
            data_json = args[0]
            data = json.loads(data_json)
            self.push_data_fn(data)
        except Exception as e:
            logger.error("DataFeedbackCommand", f"Error parsing data feedback from {source}: {e}")
