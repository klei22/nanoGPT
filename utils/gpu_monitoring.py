import os
import subprocess


def _run_nvidia_smi_query(query: str) -> list[str]:
    result = subprocess.run(
        [
            'nvidia-smi',
            f'--query-{query}',
            '--format=csv,nounits,noheader',
        ],
        stdout=subprocess.PIPE,
        encoding='utf-8',
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

def get_gpu_memory_info(info_type='used'):
    """
    Returns specific GPU memory information based on the info_type parameter.
    Valid options for info_type: 'used', 'free', 'total'
    """
    # Run nvidia-smi command to query memory info
    lines = _run_nvidia_smi_query('gpu=memory.total,memory.used,memory.free')

    for line in lines:
        total, used, free = line.split(', ')
        if info_type == 'used':
            return int(used)  # Return used memory in MiB
        elif info_type == 'free':
            return int(free)  # Return free memory in MiB
        elif info_type == 'total':
            return int(total)  # Return total memory in MiB
        else:
            raise ValueError("Invalid info_type. Choose 'used', 'free', or 'total'.")


def get_process_gpu_memory_bytes(device_index: int, pid: int | None = None) -> int:
    """
    Returns GPU memory usage in bytes for a single PID on a specific CUDA device.

    Uses pynvml when available and falls back to nvidia-smi query-compute-apps.
    Returns 0 if the process is not present in running compute contexts.
    """
    pid = os.getpid() if pid is None else pid

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
        except AttributeError:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            if process.pid == pid:
                return int(process.usedGpuMemory)
        return 0
    except Exception:
        pass

    gpu_uuid_lines = _run_nvidia_smi_query('gpu=index,uuid')
    device_uuid = None
    for line in gpu_uuid_lines:
        idx_str, uuid = [value.strip() for value in line.split(',', maxsplit=1)]
        if int(idx_str) == int(device_index):
            device_uuid = uuid
            break

    if device_uuid is None:
        return 0

    process_lines = _run_nvidia_smi_query('compute-apps=gpu_uuid,pid,used_gpu_memory')
    for line in process_lines:
        gpu_uuid, proc_pid, used_mib = [value.strip() for value in line.split(',', maxsplit=2)]
        if gpu_uuid == device_uuid and int(proc_pid) == pid:
            return int(float(used_mib) * 1024 * 1024)
    return 0
