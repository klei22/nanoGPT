import json, logging, threading, time, uuid, tempfile, math
import yaml
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from fabric import Connection  

class JobStatus:
    PENDING = "PENDING"; RUNNING = "RUNNING"; COMPLETED = "COMPLETED"; FAILED = "FAILED"; TIMEOUT = "TIMEOUT"; CANCELLED = "CANCELLED"

# New: metadata for each launched remote job
@dataclass
class RemoteJob:
    id: str
    host: str
    remote_dir: str
    yaml_path: str
    log_path: str
    pid_path: str
    pid: Optional[int] = None
    status: str = JobStatus.PENDING
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # Heartbeat/watchdog
    lease_path: Optional[str] = None
    heartbeat_thread: Optional[threading.Thread] = None
    heartbeat_stop: Optional[threading.Event] = None
    heartbeat_interval: int = 120

class RemoteTrainer:
    def __init__(self, hosts: List[str], user: str, key_filename: Optional[str]=None):
        self.hosts=hosts
        self.user=user
        self.key_filename=key_filename
        
        self.num_hosts = len(hosts)
        # New: list of jobs we launched
        self.jobs: List[RemoteJob] = []
        self.watchdog_timeout: int = 300


    def check_connectivity(self) -> bool:
        all_ok = True
        for i, host in enumerate(self.hosts):
            try:
                conn = Connection(host=host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
                conn.open()
                conn.close()
                logging.info(f"Connectivity OK: host_{i} ({host})")
            except Exception as e:
                logging.error(f"\033[31mConnection to host_{i} ({host}) failed: {e}\033[0m")
                all_ok = False

        print(f"\033[32mConnectivity check passed Number of hosts: {self.num_hosts}\033[0m")
        return all_ok

    def _start_heartbeat(self, job: RemoteJob) -> None:
        """Start a background thread that updates the remote lease file periodically."""
        stop_event = threading.Event()
        job.heartbeat_stop = stop_event

        def beater():
            while not stop_event.is_set():
                try:
                    conn = Connection(host=job.host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
                    try:
                        conn.open()
                        if job.lease_path:
                            conn.run(f"touch {job.lease_path}", hide=True, warn=True)
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                except Exception:
                    # ignore transient failures; watchdog will handle prolonged loss
                    pass
                stop_event.wait(job.heartbeat_interval)
        t = threading.Thread(target=beater, name=f"heartbeat-{job.id}", daemon=True)
        job.heartbeat_thread = t
        t.start()
        
    def perform_git_pull(self, remote_work_dir: str) -> bool:
        overall_ok = True
        for i, host in enumerate(self.hosts):
            try:
                conn = Connection(host=host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
                try:
                    conn.open()
                    cmd = f"cd {remote_work_dir} && git pull"
                    r = conn.run(cmd, hide=True, warn=True)
                    if r.ok:
                        logging.info(f"\033[32mGit pull succeeded on host_{i} ({host})\033[0m")
                    else:
                        logging.error(f"\033[31mGit pull failed on host_{i} ({host}): {r.stderr}\033[0m")
                        overall_ok = False
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
            except Exception as e:
                logging.error(f"\033[31mConnection to host_{i} ({host}) failed during git pull: {e}\033[0m")
                overall_ok = False
        return overall_ok
        
    def clear_all_jobs(self) -> bool:
        """Kill all GPU-using processes detected by nvidia-smi on each host.

        This attempts a best-effort query for GPU process PIDs using
        `nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits` and
        falls back if the option isn't supported. Each discovered PID is
        killed with SIGKILL. Returns True if all hosts succeeded (or had no
        GPU pids), False if any host encountered errors.
        """
        overall_ok = True
        for i, host in enumerate(self.hosts):
            try:
                # Use context manager to ensure proper cleanup/close even on exceptions
                with Connection(host=host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {}) as conn:
                    conn.open()
                    # Robust remote snippet: try query-compute-apps first, then try a looser CSV parse
                    cmd = """
set -o pipefail
if command -v nvidia-smi >/dev/null 2>&1; then
    # try modern query for compute-apps pid (noheader,nounits); fall back to older format if needed
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    if [ -z "$PIDS" ]; then
        # fallback: try query without nounits/noheader or parse the process table
        PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    fi
    if [ -z "$PIDS" ]; then
        # final fallback: try to parse the nvidia-smi textual table for digits in the processes section
        PIDS=$(nvidia-smi 2>/dev/null | awk '/Processes:/{p=1;next} p && /[0-9]+/ { for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i }' | sort -u || true)
    fi
    if [ -n "$PIDS" ]; then
        KILLED=()
        for pid in $PIDS; do
            # sanity: ensure pid is numeric
            case $pid in [0-9]*) true ;; *) continue ;; esac
            kill -9 $pid >/dev/null 2>&1 || true
            KILLED+=("$pid")
        done
        if [ ${#KILLED[@]} -gt 0 ]; then
            printf 'KILLED:%s\n' "$(printf '%s,' "${KILLED[@]}" | sed 's/,$//')"
            exit 0
        else
            printf 'NO_KILLED\n'
            exit 0
        fi
    else
        printf 'NO_GPU_PIDS\n'
        exit 0
    fi
else
    printf 'NO_NVIDIA_SMI\n'
    exit 0
fi
"""
                    r = conn.run(cmd, hide=True, warn=True)
                    out = (r.stdout or "").strip()
                    if r.ok:
                        if out.startswith("KILLED:"):
                            killed_list = out.replace("KILLED:", "").strip()
                            logging.info(f"\033[32mKilled GPU PIDs on host_{i} ({host}): {killed_list}\033[0m")
                        elif out.startswith("NO_GPU_PIDS") or out.startswith("NO_KILLED"):
                            logging.info(f"\033[33mNo GPU PIDs found on host_{i} ({host}).\033[0m")
                        elif out.startswith("NO_NVIDIA_SMI"):
                            logging.warning(f"\033[33mnvidia-smi not found on host_{i} ({host}); nothing to kill.\033[0m")
                        else:
                            # If output is unexpected but command succeeded, log it
                            logging.info(f"\033[32mclear_all_jobs output on host_{i} ({host}): {out}\033[0m")
                    else:
                        logging.error(f"\033[31mKilling GPU jobs failed on host_{i} ({host}): {r.stderr}\033[0m")
                        overall_ok = False
            except Exception as e:
                logging.error(f"\033[31mConnection to host_{i} ({host}) failed during GPU-kill: {e}\033[0m")
                overall_ok = False
        return overall_ok

    def submit_job(self, path_to_yaml: str, remote_work_dir: str, dir_name: str, conda_env: str = "reallmforge", max_iters: int = 10000) -> bool:
        # Load the aggregated YAML (expects a top-level list of configs)
        yaml_path = Path(path_to_yaml)
        if yaml is None:
            raise RuntimeError("PyYAML is required to split YAML configs but is not available.")
        with yaml_path.open("r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of configs in {yaml_path}, got {type(data)}")

        n_hosts = max(1, self.num_hosts)
        # Round-robin split: host_k gets indices k, k+n, k+2n, ...
        splits: List[List[dict]] = [[] for _ in range(n_hosts)]
        for idx, cfg in enumerate(data):
            splits[idx % n_hosts].append(cfg)

        # Prepare per-host YAML files in a temp directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="yaml_slices_"))
        local_slice_files: List[Optional[Path]] = [None] * n_hosts
        base = yaml_path.stem
        for i in range(n_hosts):
            if len(splits[i]) == 0:
                local_slice_files[i] = None
                continue
            slice_path = tmp_dir / f"{base}.host{i}.yaml"
            # Force inline (flow-style) lists for specific keys
            class FlowList(list):
                pass

            def represent_flow_sequence(dumper, data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

            # Register representer for SafeDumper
            yaml.add_representer(FlowList, represent_flow_sequence, Dumper=yaml.SafeDumper)

            # Wrap target lists to render in flow style
            def to_flow_lists(configs: List[dict]) -> List[dict]:
                keys = ("n_head_layerlist", "mlp_size_layerlist")
                out = []
                for cfg in configs:
                    cfg2 = dict(cfg)
                    for k in keys:
                        v = cfg2.get(k)
                        if isinstance(v, list):
                            cfg2[k] = FlowList(v)
                    out.append(cfg2)
                return out

            with slice_path.open("w") as f:
                yaml.safe_dump(to_flow_lists(splits[i]), f, sort_keys=False, width=4096)
            local_slice_files[i] = slice_path
            logging.info(f"Prepared slice for host_{i}: {slice_path} ({len(splits[i])} configs)")

        # Upload each slice to its corresponding host
        overall_ok = True
        # Encode run_id with timestamp for better tracking
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        short_uuid = uuid.uuid4().hex[:4]
        run_id = f"{timestamp}_{short_uuid}"
        new_jobs: List[RemoteJob] = []
        for i, host in enumerate(self.hosts):
            if local_slice_files[i] is None:
                logging.warning(f"\033[33mNo configs assigned to host_{i} ({host}); skipping upload.\033[0m")
                continue

            conn = Connection(host=host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
            try:
                conn.open()
                # Create run directory under remote_work_dir (no per-host subfolder)
                remote_run_dir = f"{remote_work_dir}/nsga_exps/{dir_name}/{base}-{run_id}-host{i}"
                conn.run(f"mkdir -p {remote_run_dir}", hide=True)
                remote_yaml_path = f"{remote_run_dir}/{local_slice_files[i].name}"
                remote_log_path = f"{remote_run_dir}/run.log"
                remote_pid_path = f"{remote_run_dir}/run.pid"
                lease_path = f"{remote_run_dir}/lease"

                conn.put(str(local_slice_files[i]), remote=remote_yaml_path)
                logging.info(f"Uploaded {local_slice_files[i].name} to host_{i}:{remote_yaml_path}")

                # kick off remote job; robust conda detection: prefer `conda run -n base`, else activate via hook or conda.sh; log diagnostics
                cmd = (
                    f"cd {remote_work_dir} && "
                    f"setsid bash -lc '\n"
                    f"{{\n"
                    f"echo \"[launcher] $(date) starting on $(hostname)\";\n"
                    f"echo \"[launcher] PATH: $PATH\";\n"
                    f"CONDA_BIN=$(command -v conda || true);\n"
                    f"echo \"[launcher] which conda: $CONDA_BIN\";\n"
                    f"if [ -n \"$CONDA_BIN\" ] && conda run -n {conda_env} python -V >/dev/null 2>&1; then\n"
                    f"  echo \"[launcher] using conda run -n {conda_env}\";\n"
                    f"  conda run -n {conda_env} python -u optimization_and_search/run_from_yaml.py --yaml {remote_yaml_path} --output_dir {remote_run_dir} --prefix {base} --override_args max_iters={max_iters}; ec=$?;\n"
                    f"else\n"
                    f"  if [ -n \"$CONDA_BIN\" ]; then eval \"$(conda shell.bash hook)\" >/dev/null 2>&1 || true; fi;\n"
                    f"  if command -v conda >/dev/null 2>&1; then\n"
                    f"    conda activate {conda_env} || echo \"[ERROR] conda activate {conda_env} failed\";\n"
                    f"  else\n"
                    f"    CONDA_SH=${{CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}};\n"
                    f"    [ -f \"$CONDA_SH\" ] || CONDA_SH=\"$HOME/anaconda3/etc/profile.d/conda.sh\";\n"
                    f"    if [ -f \"$CONDA_SH\" ]; then . \"$CONDA_SH\"; conda activate {conda_env} || echo \"[ERROR] conda activate {conda_env} failed (from conda.sh)\"; else echo \"[WARN] conda.sh not found at $CONDA_SH\"; fi;\n"
                    f"  fi;\n"
                    f"  echo \"[launcher] conda: $(conda --version 2>/dev/null || echo not-found)\";\n"
                    f"  echo \"[launcher] which python: $(which python 2>/dev/null || echo not-found)\";\n"
                    f"  python -V || true;\n"
                    f"  python -u optimization_and_search/run_from_yaml.py --yaml {remote_yaml_path} --output_dir {remote_run_dir} --prefix {base} --override_args max_iters={max_iters}; ec=$?;\n"
                    f"fi;\n"
                    f"echo $ec > {remote_run_dir}/exit_code\n"
                    f"}} >> {remote_log_path} 2>&1 < /dev/null &\n"
                    f"echo $! > {remote_pid_path}\n"
                    f"' </dev/null >/dev/null 2>&1 &"
                )
                conn.run(cmd, hide=True)

                # read PID back
                pid_out = conn.run(f"cat {remote_pid_path}", hide=True, warn=True)
                pid: Optional[int] = None
                if pid_out.ok:
                    s = pid_out.stdout.strip()
                    if s.isdigit():
                        pid = int(s)

                job = RemoteJob(
                    id=f"{base}-{run_id}-host{i}",
                    host=host,
                    remote_dir=remote_run_dir,
                    yaml_path=remote_yaml_path,
                    log_path=remote_log_path,
                    pid_path=remote_pid_path,
                    pid=pid,
                    status=JobStatus.RUNNING if pid else JobStatus.PENDING,
                    lease_path=lease_path,
                )
                self.jobs.append(job)
                # Defer starting heartbeats until after all hosts are launched
                new_jobs.append(job)
            
                logging.info(f"\033[32mLaunched job on host_{i} ({host}), PID={pid}, log: {remote_log_path}\033[0m")

            except Exception as e:
                logging.error(f"\033[31mFailed to upload to host_{i} ({host}): {e}\033[0m")
                overall_ok = False
                # kill any previously started jobs for this submit call
                for started in new_jobs:
                    try:
                        c2 = Connection(host=started.host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
                        try:
                            c2.open()
                            if started.pid is not None:
                                c2.run(f"kill -9 {started.pid}", hide=True, warn=True)
                        finally:
                            try:
                                c2.close()
                            except Exception:
                                pass
                        started.status = JobStatus.CANCELLED
                        started.finished_at = time.time()
                    except Exception as ke:
                        logging.warning(f"\033[33mRollback kill failed for {started.id}@{started.host}: {ke}\033[0m")
                # stop processing further hosts on failure
                break
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        # Start heartbeats for all newly launched jobs (deferred until after launches) only if overall_ok
        if overall_ok:
            for j in new_jobs:
                self._start_heartbeat(j)

        # Cleanup local temp files
        try:
            for p in local_slice_files:
                if p and p.exists():
                    p.unlink(missing_ok=True)
            # Attempt to remove the temp dir; ignore if not empty
            tmp_dir.rmdir()
        except Exception:
            pass

        return overall_ok

    # New: poll job statuses by checking remote PID liveness
    def poll_jobs(self) -> List[RemoteJob]:
        for job in self.jobs:
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT):
                continue
            conn = Connection(host=job.host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
            try:
                conn.open()
                # Ensure we have a PID
                if job.pid is None:
                    r = conn.run(f"test -f {job.pid_path} && cat {job.pid_path}", hide=True, warn=True)
                    if r.ok:
                        s = r.stdout.strip()
                        if s.isdigit():
                            job.pid = int(s)
                alive = False
                if job.pid is not None:
                    r = conn.run(f"kill -0 {job.pid}", hide=True, warn=True)
                    alive = r.ok
                if alive:
                    job.status = JobStatus.RUNNING
                else:
                    # Process exited: check exit_code file
                    ec_path = f"{job.remote_dir}/exit_code"
                    r = conn.run(f"test -f {ec_path} && cat {ec_path}", hide=True, warn=True)
                    exit_code: Optional[int] = None
                    if r.ok:
                        s = r.stdout.strip()
                        if s.isdigit():
                            exit_code = int(s)
                    if exit_code == 0:
                        job.status = JobStatus.COMPLETED
                        logging.info(f"\033[32mJob {job.id}@{job.host} completed successfully.\033[0m")
                    else:
                        job.status = JobStatus.FAILED
                        logging.error(f"\033[31mJob {job.id}@{job.host} failed.\033[0m")
                    job.finished_at = time.time()
                    # stop heartbeat
                    if job.heartbeat_stop:
                        job.heartbeat_stop.set()
                    if job.heartbeat_thread:
                        try:
                            job.heartbeat_thread.join(timeout=2.0)
                        except Exception:
                            pass
            except Exception as e:
                logging.warning(f"\033[33mPoll failed for {job.id}@{job.host}: {e}\033[0m")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        return self.jobs

    def check_hosts_status(self) -> None:
        for i, host in enumerate(self.hosts):
            # check connectivity: if no connection, kick out the host from the list
            try:
                conn = Connection(host=host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
                conn.open()
                conn.close()
                logging.info(f"Connectivity OK: host_{i} ({host})")
            except Exception as e:
                logging.error(f"\033[31mConnection to host_{i} ({host}) failed: {e}\033[0m")
                self.hosts.remove(host)
                self.num_hosts = len(self.hosts)
                logging.warning(f"\033[33mHost_{i} ({host}) removed from host list. Remaining hosts: {self.num_hosts}\033[0m") 

    def wait_for_all(self, poll_interval: float = 10.0, timeout: Optional[float] = None, verbose: bool = False) -> bool:
        start = time.time()
        while True:
            states = [j.status for j in self.poll_jobs()]
            if all(s in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT) for s in states):
                logging.info(f"\033[32mAll jobs completed: {states}\033[0m")
                return True
            if timeout and (time.time() - start) > timeout:
                for j in self.jobs:
                    if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT):
                        j.status = JobStatus.TIMEOUT
                        logging.warning(f"\033[33mJob {j.id}@{j.host} timed out.\033[0m")
                return False
            if verbose:
                status_counts = {s: states.count(s) for s in set(states)}
                logging.info("-" * 40)
                logging.info(f"Elapsed time: {time.time() - start:.1f}s")
                logging.info(f"Job statuses: {status_counts}")
            time.sleep(poll_interval)

    def kill_all(self) -> List[RemoteJob]:
        for job in self.jobs:
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT):
                continue
            conn = Connection(host=job.host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
            try:
                conn.open()
                if job.pid is not None:
                    conn.run(f"kill -9 {job.pid}", hide=True, warn=True)
                job.status = JobStatus.CANCELLED
                job.finished_at = time.time()
            except Exception as e:
                logging.warning(f"\033[33mKill failed for {job.id}@{job.host}: {e}\033[0m")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            # stop heartbeat for cancelled jobs
            if job.heartbeat_stop:
                job.heartbeat_stop.set()
            if job.heartbeat_thread:
                try:
                    job.heartbeat_thread.join(timeout=2.0)
                except Exception:
                    pass
        return self.jobs

    def fetch_results(self, gen: int, local_dir: str = "train") -> str:
        local_base = Path(local_dir)
        local_base.mkdir(parents=True, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        agg_yaml_path = local_base / f"{time_stamp}_gen{gen}.yaml"
        gen_csv_path = local_base / f"{time_stamp}_gen{gen}.csv"
        if not gen_csv_path.exists():
            gen_csv_path.write_text("#idx, best_val_loss\n")

        summary_csv_path = local_base / "best_val_loss.csv"
        # ensure CSV header once
        if not summary_csv_path.exists():
            summary_csv_path.write_text("#gen, job_id,formatted_name, best_val_loss\n")
        logs_local_dir = local_base / "logs"
        logs_local_dir.mkdir(parents=True, exist_ok=True)
        for job in self.jobs:
            if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT):
                logging.warning(f"\033[33mSkipping fetch for incomplete job {job.id}@{job.host} (status: {job.status})\033[0m") 
                continue
            conn = Connection(host=job.host, user=self.user, connect_kwargs={"key_filename": self.key_filename} if self.key_filename else {})
            try:
                conn.open()
                # Tar the run directory remotely for robust transfer
                # parent = "/".join(job.remote_dir.rstrip("/").split("/")[:-1])
                # leaf = job.remote_dir.rstrip("/").split("/")[-1]
                # tar_path = f"{parent}/{job.id}.tar.gz"
                # conn.run(f"cd {parent} && tar -czf {job.id}.tar.gz {leaf}", hide=True)
                # conn.get(tar_path, local=str(local_base / f"{job.id}.tar.gz"))
                # logging.info(f"Fetched {job.id} to {local_base}/{job.id}.tar.gz")
              
                # peel the job directory for logs
                parts = job.remote_dir.rstrip("/").split("/")
                remote_work_dir = "/".join(parts[:-1])
                remote_logs_path = f"{remote_work_dir}/{job.id}.yaml"
                
                # Always fetch the job's run.log first for diagnostics
                try:
                    remote_log_path = job.log_path
                    local_log_copy = logs_local_dir / f"{job.id}.log"
                    lr = conn.run(f"test -f {remote_log_path}", hide=True, warn=True)
                    if lr.ok:
                        conn.get(remote_log_path, local=str(local_log_copy))
                except Exception:
                    pass

                job_idx_set = set()
                recorded_idx = set()
                # Fetch the configuration slice to know which indices were expected on this host
                try:
                    local_slice_path = logs_local_dir / f"{job.id}.slice.yaml"
                    conn.get(job.yaml_path, local=str(local_slice_path))
                    with local_slice_path.open("r") as slice_file:
                        slice_docs = yaml.safe_load(slice_file) or []
                    if not isinstance(slice_docs, list):
                        slice_docs = []
                    for cfg in slice_docs:
                        if isinstance(cfg, dict) and "idx" in cfg:
                            try:
                                job_idx_set.add(int(cfg["idx"]))
                            except (TypeError, ValueError):
                                continue
                except Exception:
                    logging.warning(f"\033[33mUnable to fetch config slice for {job.id}@{job.host}\033[0m")

                r = conn.run(f"test -f {remote_logs_path}", hide=True, warn=True)
                if r.ok:
                    # Download the YAML results file locally
                    local_yaml_path = logs_local_dir / f"{job.id}.yaml"
                    conn.get(remote_logs_path, local=str(local_yaml_path))
                    # Parse and append to aggregate + CSV
                    try:
                        with local_yaml_path.open("r") as f:
                            docs = list(yaml.safe_load_all(f))
                        if not docs:
                            logging.warning(f"\033[33mEmpty YAML in results from {job.id}@{job.host}\033[0m")
                        for doc in docs:
                            if not isinstance(doc, dict):
                                continue
                            config = doc.get("config") or {}
                            if "idx" not in config:
                                continue
                            idx = config["idx"]
                            try:
                                idx_value = int(idx)
                            except (TypeError, ValueError):
                                logging.warning(f"\033[33mSkipping result without valid idx from {job.id}@{job.host}: {idx}\033[0m")
                                continue
                            raw_bvl = doc.get("best_val_loss")
                            try:
                                bvl = float(raw_bvl)
                            except (TypeError, ValueError):
                                bvl = float("inf")
                            if not math.isfinite(bvl):
                                bvl = float("inf")
                            doc["best_val_loss"] = bvl
                            name = doc.get("formatted_name", "")
                            # Append to aggregate.yaml with sanitized loss
                            with agg_yaml_path.open("a") as fa:
                                yaml.safe_dump(doc, fa, explicit_start=True, sort_keys=False, width=4096)
                            bvl_str = "inf" if math.isinf(bvl) else f"{bvl}"
                            with summary_csv_path.open("a") as fc:
                                fc.write(f"{gen},{job.id},{name},{bvl_str}\n")
                            with gen_csv_path.open("a") as fc:
                                fc.write(f"{idx_value},{bvl_str}\n")
                            recorded_idx.add(idx_value)
                        logging.info(f"\033[32mFetched results from {job.id}@{job.host}\033[0m")
                    except Exception as ye:
                        logging.error(f"\033[31mYAML parse error for results from {job.id}@{job.host}: {ye}\033[0m")
                else:
                    logging.warning(f"\033[33mNo results YAML found at {remote_logs_path} for {job.id}@{job.host}\033[0m")

                # For any configs that failed before producing results, emit a fallback record with inf loss
                missing_idx = sorted(job_idx_set - recorded_idx)
                if missing_idx:
                    for idx_value in missing_idx:
                        name = f"{job.id}-idx{idx_value}"
                        fallback_doc = {
                            "formatted_name": name,
                            "config": {"idx": idx_value},
                            "best_val_loss": float("inf"),
                            "status": job.status,
                        }
                        with agg_yaml_path.open("a") as fa:
                            yaml.safe_dump(fallback_doc, fa, explicit_start=True, sort_keys=False, width=4096)
                        with summary_csv_path.open("a") as fc:
                            fc.write(f"{gen},{job.id},{name},inf\n")
                        with gen_csv_path.open("a") as fc:
                            fc.write(f"{idx_value},inf\n")
                    logging.warning(f"\033[33mRecorded fallback inf losses for {len(missing_idx)} configs from {job.id}@{job.host}\033[0m")
            except Exception as e:
                logging.error(f"Fetch failed for {job.id}@{job.host}: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        # reorder gen_csv by idx
        try:
            lines = gen_csv_path.read_text().strip().split("\n")
            header = lines[0]
            entries = []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                idx_str, bvl_str = parts
                try:
                    idx = int(idx_str)
                    bvl = float(bvl_str)
                    entries.append((idx, bvl))
                except ValueError:
                    continue
            entries.sort(key=lambda x: x[0])
            with gen_csv_path.open("w") as fc:
                fc.write(header + "\n")
                for idx, bvl in entries:
                    fc.write(f"{idx},{bvl}\n")
        except Exception as re:
            logging.error(f"\033[31mFailed to reorder gen CSV {gen_csv_path}: {re}\033[0m")
        return str(gen_csv_path)
