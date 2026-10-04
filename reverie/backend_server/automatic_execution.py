#!/usr/bin/env python3

import gc
import os
import signal
import sys
import time
import shutil
import reverie
import json
import seed_history
import argparse
import webbrowser
import subprocess
from typing import Tuple
from pathlib import Path
from datetime import datetime
from multiprocessing import Process
from openai_cost_logger import OpenAICostLoggerViz


def parse_args() -> Tuple[str, str, int, bool]:
    """Parse bash arguments

    Returns:
        Tuple[str, str, int, bool]:
            - name of the forked simulation
            - the name of the new simulation
            - total steps to run (step = 10sec in the simulation)
            - open the simulator UI
    """
    parser = argparse.ArgumentParser(description='Reverie Server')
    parser.add_argument(
    '--origin',
    type=str,
    default="base_the_ville_isabella_maria_klaus",
    help='The name of the forked simulation'
    )
    parser.add_argument(
    '--target',
    type=str,
    default="test-simulation",
    help='The name of the new simulation'
    )
    parser.add_argument(
        '--steps',
        type=str,
        default="8640", # 24 hours
        help=('The step number to stop AT, counted from the start of the '
              'simulated day -- NOT how many steps to run. Forking a '
              'checkpoint that already sits at step 2360 and asking for '
              '--steps 30 runs nothing at all. 1 step = 10 simulated seconds, '
              'so the default 8640 is a full 24 h day.')
    )
    parser.add_argument(
        '--ui',
        type=str,
        default="True",
        help='Open the simulator UI'
    )
    parser.add_argument(
        '--browser_path',
        type=str,
        default="/usr/bin/google-chrome %s",
        help='Browser path, default is /usr/bin/google-chrome %s'
    )
    parser.add_argument(
        '--port',
        type=str,
        default="8000",
        help='Port number for the frontend server'
    )
    origin = parser.parse_args().origin
    target = parser.parse_args().target
    steps = parser.parse_args().steps
    ui = parser.parse_args().ui
    ui = True if ui.lower() == "true" else False
    browser_path = parser.parse_args().browser_path
    port = parser.parse_args().port
    
    return origin, target, steps, ui, browser_path, port


def get_starting_step(exp_name: str) -> int:
    """Get the starting step of the experiment. 

    Returns:
        exp_name (str): The name of the experiment
    """
    current_step = 0
    experiments_directory = "../../environment/frontend_server/storage"
    full_path = Path(experiments_directory, exp_name, "movement")  
    if full_path.exists():
        files = os.listdir(full_path)
        steps = [int(os.path.splitext(filename)[0]) for filename in files]
        current_step = max(steps)
    return current_step


def _find_chrome_binary() -> str:
    """Return a Chrome/Chromium executable for headless mode (Linux or macOS)."""
    candidates = [
        "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for c in candidates:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    raise FileNotFoundError("No Chrome/Chromium binary found for headless mode; install Google Chrome or run with --ui True")


def start_web_tab(ui: bool, browser_path: str, port: str) -> int:
    """Open a new tab in the browser with the simulator home page
    
    Args:
        ui (bool): Open the simulator UI.
        browser_path (str): The path of the browser.
        port (str): The port number of the frontend server.
        
    Returns:
        int: The process id of the web tab (only for headless chrome).
    """
    url = f"http://localhost:{port}/simulator_home"
    print("(Auto-Exec): Opening the simulator home page", flush=True)
    time.sleep(5)
    pid = None
    try:
        if ui:
            try:
                webbrowser.get(browser_path).open(url, new=2)
            except webbrowser.Error:
                # browser_path not registered on this OS (e.g. macOS): use the system default browser.
                webbrowser.open(url, new=2)
        else:
            command = [
                _find_chrome_binary(),
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--single-process",
                "--no-zygote",
                url
            ]

            # Own session, so the whole browser (Chrome spawns helpers even
            # with --single-process) can be killed as a group below. Five
            # orphaned headless Chromes were found alive on 22 Sep 2026.
            process = subprocess.Popen(command, start_new_session=True)
            pid = process.pid
            print(f"(Auto-Exec): Web tab process started with pid {pid}", flush=True)
        return pid
    except Exception as e:
        print(e, flush=True)


def get_new_checkpoint(step: int, tot_steps: int, checkpoint_freq: int) -> int:
    """Get the current checkpoint from the checkpoint file

    Args:
        step (int): The current step.
        tot_steps (int): The total steps to run.
        checkpoint_freq (int): The frequency of the checkpoints.

    Returns:
        int: The new checkpoint.
    """
    new_checkpoint = step + checkpoint_freq
    if new_checkpoint > tot_steps:
        new_checkpoint = tot_steps
    return new_checkpoint


def save_checkpoint(rs, idx: int, th: Process) -> Tuple[str, int, int]:
    """Save the checkpoint and return the data to start the new one.

    Args:
        rs (ReverieServer): The reverie server object.
        idx (int): The index of the checkpoint.
        th (Process): The process of the web tab.
    
    Returns:
        Tuple[str, int, int]: The name of the new experiment, the starting step and the index of the checkpoint.
    """
    target = rs.sim_code
    rs.open_server(input_command="fin")
    print(f"(Auto-Exec): Checkpoint saved: {target}", flush=True)    
    return target, get_starting_step(target), idx+1
    

if __name__ == '__main__':
    checkpoint_freq = 200 # 1 step = 10 sec
    log_path = "cost-logs" # where the simulations' prints are stored
    idx = 0
    origin, target, tot_steps, ui, browser_path, port = parse_args()
    current_step = get_starting_step(origin)
    exp_name = target
    start_time = datetime.now()
    tot_steps = int(tot_steps)
    curr_checkpoint = get_new_checkpoint(current_step, tot_steps, checkpoint_freq)

    # A failing step fails again at the same step: the prompts run at
    # temperature 0, so the retry is bit-identical. Without a limit the runner
    # re-forks the whole simulation forever -- 13 identical retries and 346 MB
    # of copies on 22 Sep 2026, with no way to tell from the log.
    max_stalls = int(os.environ.get("AUTO_EXEC_MAX_STALLS", 2))
    stalls = 0

    try:
        with open("../../openai_config.json") as f:
            seed_file = seed_history.configured(json.load(f))
    except Exception:
        seed_file = None

    print("(Auto-Exec): STARTING THE EXPERIMENT", flush=True)
    print(f"(Auto-Exec): Origin: {origin}", flush=True)
    print(f"(Auto-Exec): Target: {target}", flush=True)
    print(f"(Auto-Exec): Total steps: {tot_steps}", flush=True)
    print(f"(Auto-Exec): Checkpoint Freq: {checkpoint_freq}", flush=True)
    print(f"(Auto-Exec): Origin is at step {current_step}; running to "
          f"{tot_steps} ({tot_steps - current_step} steps)", flush=True)
    if seed_file:
        print(f"(Auto-Exec): Seed memories (D13): {seed_file}"
              + ("" if current_step == 0 else
                 " -- NOT loaded: the origin is past step 0"), flush=True)

    # --steps is an absolute end step, not a count. Forking a checkpoint that
    # is already past it used to skip the loop and exit 0 in 7 ms, which reads
    # exactly like a finished run -- on 22 Sep 2026 that cost a "completed"
    # baseline that had run nothing. Say so and fail.
    if current_step >= tot_steps:
        print(f"(Auto-Exec): NOTHING TO RUN: origin '{origin}' is already at "
              f"step {current_step}, and --steps is the step to stop AT, not "
              f"how many to run. For N more steps, pass "
              f"--steps {current_step + 100} for 100, etc.", flush=True)
        sys.exit(2)

    while current_step < tot_steps:
        step_before_stage = current_step
        try:
            steps_to_run = curr_checkpoint - current_step
            target = f"{exp_name}-s-{idx}-{current_step}-{curr_checkpoint}"
            print(f"(Auto-Exec): STAGE {idx}", flush=True)
            print(f"(Auto-Exec): Running experiment '{exp_name}' from step '{current_step}' to '{curr_checkpoint}'", flush=True)
            rs = reverie.ReverieServer(origin, target)
            # D13 (016-IP): the paper's seed memories, loaded once, in the
            # first stage of a run that starts at step 0, before the first
            # step -- the command the paper's server takes by hand, which
            # this runner never typed. Absent from the config, nothing runs.
            if idx == 0 and current_step == 0 and seed_file:
                seed_history.load(rs, seed_file)
            th, pid = None, None 
            # Headless chrome doesn't need a thread since it create a dedicated thread by itself
            if ui:
                th = Process(target=start_web_tab, args=(ui, browser_path, port))
                th.start()
            else:
                pid = start_web_tab(ui, browser_path, port)
            rs.open_server(input_command=f"run {steps_to_run}")
        except KeyboardInterrupt:
            print("(Auto-Exec): KeyboardInterrupt: Stopping the experiment.", flush=True)
            sys.exit(0)
        except Exception as e:
            print(e, flush=True)
            step = e.args[1]
            if step != 0:
                origin, current_step, idx = save_checkpoint(rs, idx, th)
            else:
                shutil.rmtree(f"../../environment/frontend_server/storage/{target}") # Remove the experiment folder if no steps were run
            print(f"(Auto-Exec): Error at step {current_step}", flush=True)
            print(f"(Auto-Exec): Exception {e.args[0]}", flush=True)
        else:
            origin, current_step, idx = save_checkpoint(rs, idx, th)
            curr_checkpoint = get_new_checkpoint(current_step, tot_steps, checkpoint_freq)
        finally:
            time.sleep(10) # Wait for the server to finish and then kill the process
            if th and th.is_alive():
                th.kill()
                th.join()
                th.close()
                gc.collect()
        
            if pid:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except OSError:
                    pass  # already gone
                print(f"(Auto-Exec): Killed web tab process group with pid {pid}", flush=True)
                pid = None

        if current_step > step_before_stage:
            stalls = 0
        else:
            stalls += 1
            print(f"(Auto-Exec): STAGE MADE NO PROGRESS (still at step "
                  f"{current_step}), {stalls} of {max_stalls} allowed", flush=True)
            if stalls >= max_stalls:
                print(f"(Auto-Exec): GIVING UP at step {current_step} after "
                      f"{stalls} stages with no progress. The error above "
                      f"repeats every time; fix it and resume with "
                      f"--origin {origin}", flush=True)
                sys.exit(1)

    print(f"(Auto-Exec): EXPERIMENT FINISHED: {exp_name}")
    OpenAICostLoggerViz.print_experiment_cost(experiment=exp_name, path=log_path)
    OpenAICostLoggerViz.print_total_cost(path=log_path)
    print(f"(Auto-Exec): Execution time: {datetime.now() - start_time}")
    sys.exit(0)