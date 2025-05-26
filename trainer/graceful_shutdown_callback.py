import dataclasses
import json
import signal
import sys

from kubernetes import config, client
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class GracefulShutdownCallback(TrainerCallback):
    """
    Callback that catches SIGTERM and stops the training instead of terminating the program.
    """

    def __init__(self):
        self.shutting_down = False

        # Store the previous signal handler
        self.old_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        if self.shutting_down:
            if self.old_handler and callable(self.old_handler):
                print('Second signal received, propagating it')
                self.old_handler(sig, frame)
            else:
                sys.exit('Second signal received, forcefully exiting')
        print('Attempting graceful shutdown')
        self.shutting_down = True

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step.
        """

        if not self.shutting_down:
            return

        control.should_save = True
        control.should_training_stop = True 