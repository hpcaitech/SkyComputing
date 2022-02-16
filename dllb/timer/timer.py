#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import os.path as osp
import time


class DistributedTimer:
    def __init__(self, root="/tmp"):
        self.file_path = osp.join(root, "dist_timer.txt")

    def clean_prev_file(self):
        if osp.exists(self.file_path):
            os.remove(self.file_path)

    def add_timestamp(self):
        with open(self.file_path, "a") as f:
            new_line = "timestamp: {}\n".format(time.time())
            f.write(new_line)

    def get_prev_interval(self):
        with open(self.file_path, "r") as f:
            all_lines = f.readlines()
            start_time = float(all_lines[-2].split(":")[-1])
            end_time = float(all_lines[-1].split(":")[-1])

            return end_time - start_time
