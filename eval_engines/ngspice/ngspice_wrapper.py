import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import subprocess
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import IPython
debug = False

class NgSpiceWrapper(object):

    BASE_TMP_DIR = os.path.join(os.getcwd(), "temp_sim_data")

    def __init__(self, num_process, yaml_path, path, root_dir=None):
        if root_dir == None:
            self.root_dir = NgSpiceWrapper.BASE_TMP_DIR
        else:
            self.root_dir = root_dir

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.UnsafeLoader)
        design_netlist = yaml_data['dsn_netlist']
        design_netlist = path+'/'+design_netlist
 
        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(self.root_dir, "designs_" + self.base_design_name)

        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
        raw_file.close()

    def get_design_name(self, state):
        fname = self.base_design_name
        for value in state.values():
            fname += "_" + str(value)
        return fname

    def create_design(self, state, new_fname):
        design_folder = os.path.join(self.gen_dir, new_fname)+str(random.randint(0,10000))
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):
            if '.include' in line:
                regex = re.compile("\.include\s*\"(.*?)\"")
                found = regex.search(line)
                if found:
                    # current_fpath = os.path.realpath(__file__)
                    # parent_path = os.path.abspath(os.path.join(current_fpath, os.pardir))
                    # parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))
                    # path_to_model = os.path.join(parent_path, 'spice_models/45nm_bulk.mod')
                    # lines[line_num] = lines[line_num].replace(found.group(1), path_to_model)
                    pass # do not change the model path
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            if 'wrdata' in line:
                regex = re.compile("wrdata\s*(\w+\.\w+)\s*")
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1))
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement)

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath

    # [2026-11-13 超时保护] 单次仿真硬上限。正常仿真 0.5~2s；实测有 ngspice
    # 进程以 0 CPU 挂起 23 分钟导致整个训练迭代静默卡死（同步采样等齐 4 个
    # worker，一个挂起全员停摆）。超时即杀进程、返回 info=1（视为仿真失败）。
    NGSPICE_TIMEOUT_SEC = 60

    def simulate(self, fpath):
        info = 0 # this means no error occurred
        command = "ngspice -b %s" % fpath
        if debug:
            print(command)
            print(fpath)
        # [2026-11-14 无窗口化·v3] 不再经 cmd 外壳（shell=True 每次仿真都弹控制台窗口）；
        # 并彻底脱离用户桌面会话：DETACHED_PROCESS（不继承控制台，不再触发焦点抢占/
        # 输入法抖动）+ stdin 断开 + SW_HIDE 兜底。getattr 兜底保证跨平台语法安全。
        if os.name == "nt":
            creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        else:
            creationflags = 0
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        proc = subprocess.Popen(
            ["ngspice", "-b", fpath],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        try:
            exit_code = proc.wait(timeout=self.NGSPICE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print("[ngspice_wrapper] WARNING: ngspice 超时(%ds)被强杀: %s" %
                  (self.NGSPICE_TIMEOUT_SEC, fpath))
            return 1
        if (exit_code % 256):
           # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info


    def create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        if dsn_name == None:
            dsn_name = self.get_design_name(state)
        else:
            dsn_name = str(dsn_name)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state, dsn_name)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)
        return state, specs, info


    def run(self, states, design_names=None, verbose=False):
        """

        :param states:
        :param design_names: if None default design name will be used, otherwise the given design name will be used
        :param verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result
