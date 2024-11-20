import re
import time
import subprocess
import threading
import importlib
from enum import Enum, auto
from numpy import argmin, argmax
from copy import deepcopy

from accelerate.logging import get_logger
from accelerate.tracking import is_wandb_available, on_main_process
from accelerate.tracking import WandBTracker as AceWandBTracker
from typing import Any, Dict, List, Optional, Union

from swanlab.integration.accelerate import SwanLabTracker as AceSwanLabTracker

class NPUType(Enum):
    A = auto()
    B = auto()
    C = auto()

logger = get_logger(__name__)


def _is_package_available(pkg_name, metadata_name=None):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            # Some libraries have different names in the metadata
            _ = importlib.metadata.metadata(pkg_name if metadata_name is None else metadata_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

def is_swanlab_available():
    return _is_package_available("swanlab")

class NPUMonitor:

    main_process_only = True

    @on_main_process
    def __init__(self, run_mode='delay_time', delay_time_interval=1, delay_time_max_num=5):

        npu_type = NPUType.B
        start_idx = 0
        npu_smi_info = self.npu_smi_info()
        # 遍历每一行并提取 Power 和 AICore 列数据
        for idx, line in enumerate(npu_smi_info.splitlines()):
            if line.startswith('|'):
                parts = line.split('|')
                if '910' in parts[1]:
                    if '910A' in parts[1]:
                        npu_type = NPUType.A
                    elif '910B' in parts[1]:
                        npu_type = NPUType.B
                    elif '910C' in parts[1]:
                        npu_type = NPUType.C
                    start_idx = idx
                    break

        self.npu_type = npu_type
        self.start_idx = start_idx
        self.run_mode = run_mode
        self.delay_time_interval = max(delay_time_interval, 1)
        self.delay_time_max_num = min(delay_time_max_num, 10)
        self.__powers_history = None
        self.__aicores_history = None
        if run_mode == 'real_time':
            logger.info("NPU monitor run in real time mode")
        elif run_mode == 'delay_time':
            self.__powers_history = []
            self.__aicores_history = []
            self.delay_get_npu_infos()
            logger.info("NPU monitor run in delay time mode")
        else:
            raise ValueError("run_mode should be 'real_time' or 'delay_time'")
        
    @property
    def powers_history(self):
        return self.__powers_history
    
    @property
    def aicores_history(self):
        return self.__aicores_history

    @staticmethod
    def npu_smi_info():
        # 运行 npu-smi info 命令
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run npu-smi info: {e}")
            return ""
    
    def get_valid_infos(self):
        npu_smi_info = self.npu_smi_info()
        infos = npu_smi_info.splitlines()
        infos = infos[self.start_idx:self.start_idx + 24]
        return infos

    def get_power(self, return_type='avg'):
        infos = self.get_valid_infos()
        powers = []
        if self.npu_type != NPUType.A:
            data = []
            for i in range(0, len(infos)):
                if len(infos[i].split('|')) > 4:
                    data += [infos[i].split('|')]
            for i in range(0, len(data), 2):
                power = re.search(r'\d+\.\d+', data[i][3])
                power = float(power.group()) if power is not None else 0.0
                powers.append(power)
        else:
            raise NotImplementedError("NPU type not supported")

        if return_type == 'avg':
            avg_power = sum(powers) / (len(powers) + 1e-9)
            return avg_power  
        return powers
    
    def get_aicores(self, return_type='avg'):
        infos = self.get_valid_infos()
        aicores = []
        if self.npu_type != NPUType.A:
            data = []
            for i in range(0, len(infos)):
                if len(infos[i].split('|')) > 4:
                    data += [infos[i].split('|')]
            for i in range(0, len(data), 2):
                aicore = re.search(r'\d+', data[i+1][3])
                aicore = int(aicore.group()) if aicore is not None else 0
                aicores.append(aicore)
        else:
            raise NotImplementedError("NPU type not supported")

        if return_type == 'avg':
            avg_aicore = sum(aicores) / (len(aicores) + 1e-9)
            return avg_aicore
        return aicores
    
    def get_powers_and_aicores(self, return_type='avg'):
        infos = self.get_valid_infos()
        if self.npu_type != NPUType.A:
            data = []
            for i in range(0, len(infos)):
                if len(infos[i].split('|')) > 4:
                    data += [infos[i].split('|')]
            powers = []
            aicores = []
            for i in range(0, len(data), 2):
                power = re.search(r'\d+\.\d+', data[i][3])
                power = float(power.group()) if power is not None else 0.0
                aicore = re.search(r'\d+', data[i+1][3])
                aicore = int(aicore.group()) if aicore is not None else 0
                powers.append(power)
                aicores.append(aicore)
        else:
            raise NotImplementedError("NPU type not supported")
        
        if return_type == 'avg':
            avg_power = sum(powers) / (len(powers) + 1e-9)
            avg_aicore = sum(aicores) / (len(aicores) + 1e-9)
            return dict(power=avg_power, aicore=avg_aicore)
        return dict(power=powers, aicore=aicores)

    def delay_get_npu_infos(self):

        def store():
            while True:
                powers_and_aicores = self.get_powers_and_aicores()
                powers = powers_and_aicores['power']
                aicores = powers_and_aicores['aicore']
                self.__powers_history.append(powers)
                self.__aicores_history.append(aicores)
                if len(self.__powers_history) > self.delay_time_max_num:
                    self.__powers_history.pop(0)
                    self.__aicores_history.pop(0)
                time.sleep(self.delay_time_interval)
                    
        log_thread = threading.Thread(target=store, daemon=True)
        log_thread.start()
        

    def get_npu_infos(self, return_type='avg'):
        if return_type == 'avg':
            if self.run_mode == 'delay_time':
                powers_history_copy = self.__powers_history[:]
                aicores_history_copy = self.__aicores_history[:]
                max_value_idx = argmax(powers_history_copy)
                powers, aicores = powers_history_copy[max_value_idx], aicores_history_copy[max_value_idx]
                return dict(power=powers, aicore=aicores)
        return self.get_powers_and_aicores(return_type=return_type)


class WandBTracker(AceWandBTracker):

    name = "wandb"
    requires_logging_directory = False
    main_process_only = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        npu_moniter_run_mode = kwargs.get('npu_moniter_run_mode', 'delay_time')
        delay_time_interval = kwargs.get('delay_time_interval', 1)
        delay_time_max_num = kwargs.get('delay_time_max_num', 5)
        self.npu_monitor = NPUMonitor(
            run_mode=npu_moniter_run_mode,
            delay_time_interval=delay_time_interval,
            delay_time_max_num=delay_time_max_num
        )
   
    @on_main_process
    def log_npu_infos(self, step: Optional[int] = None, **kwargs):
        npu_infos = self.npu_monitor.get_npu_infos()
        self.log(npu_infos, step=step, **kwargs)
        
class SwanLabTracker(AceSwanLabTracker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        npu_moniter_run_mode = kwargs.get('npu_moniter_run_mode', 'delay_time')
        delay_time_interval = kwargs.get('delay_time_interval', 1)
        delay_time_max_num = kwargs.get('delay_time_max_num', 5)
        self.npu_monitor = NPUMonitor(
            run_mode=npu_moniter_run_mode,
            delay_time_interval=delay_time_interval,
            delay_time_max_num=delay_time_max_num
        )
   
    @on_main_process
    def log_npu_infos(self, step: Optional[int] = None, **kwargs):
        npu_infos = self.npu_monitor.get_npu_infos()
        self.log(npu_infos, step=step, **kwargs)