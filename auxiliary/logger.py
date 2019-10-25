# MIT License
#
# Copyright (c) 2017 Tinghui Zhou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import progressbar  # This is progressbar2, so you need to install "pip install progresbar2"
import sys
from blessings import Terminal


class TermWriter:
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        with self.t.location(*self.location):
            sys.stdout.write("\033[K")
            print(string)

    def flush(self):
        return


class TermLogger:
    def __init__(self, n_epochs, train_size, valid_size, use_flag=False):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.use_flag = use_flag

        if self.use_flag:
            self.t = Terminal()
            s = 10
            e = 1   # epoch bar position
            tr = 3  # train bar position
            ts = 6  # valid bar position
            h = self.t.height

            for i in range(10):
                print('')

            # Epoch bar
            self.epoch_bar = progressbar.ProgressBar(
                max_value=n_epochs,
                fd=TermWriter(self.t, (0, h - s + e))
            )

            # Train writer & bar
            self.train_writer = TermWriter(self.t, (0, h - s + tr))
            self.train_bar = progressbar.ProgressBar(
                max_value=self.train_size,
                fd=TermWriter(self.t, (0, h - s + tr + 1))
            )

            # Valid writer & bar
            self.valid_writer = TermWriter(self.t, (0, h - s + ts))
            self.valid_bar = progressbar.ProgressBar(
                max_value=self.valid_size,
                fd=TermWriter(self.t, (0, h - s + ts + 1))
            )
        else:
            print("=> TermLogger is not effective!!!")

    def start_bar(self, bar_type="epoch"):
        if self.use_flag:
            if bar_type == "epoch":
                self.epoch_bar.start()
            elif bar_type == "train":
                self.train_bar.start()
            elif bar_type == "eval":
                self.valid_bar.start()

    def update_bar(self, current_val, bar_type="epoch"):
        if self.use_flag:
            if bar_type == "epoch":
                self.epoch_bar.update(current_val)
            elif bar_type == "train":
                self.train_bar.update(current_val)
            elif bar_type == "eval":
                self.valid_bar.update(current_val)

    def finish_bar(self, bar_type="epoch"):
        if self.use_flag:
            if bar_type == "epoch":
                self.epoch_bar.finish()
            elif bar_type == "train":
                self.train_bar.finish()
            elif bar_type == "eval":
                self.valid_bar.finish()

    def write_log(self, log_message, bar_type="train"):
        if self.use_flag:
            if bar_type == "train":
                self.train_writer.write(log_message)
            elif bar_type == "eval":
                self.valid_writer.write(log_message)
