from remote_trainer import RemoteTrainer  
from fabric import Connection
import logging
from run_exp import load_hosts_from_file
import argparse


# Configure logging to only show INFO:root messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

def main():
    parser = argparse.ArgumentParser(description="Stop/clear all remote training jobs across hosts.")
    parser.add_argument(
        "--hosts",
        type=str,
        default="../host_configs/hosts_8instances.yaml",
        help="Path to YAML file listing remote hosts",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="xinting",
        help="SSH username",
    )
    parser.add_argument(
        "--key",
        dest="key_filename",
        type=str,
        default="/home/xinting/.ssh/id_rsa",
        help="Path to SSH private key file",
    )

    args = parser.parse_args()

    hosts = load_hosts_from_file(args.hosts)
    user = args.user
    key_filename = args.key_filename

    trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
    trainer.check_connectivity()
    # Report available disk space on each host
    print("Disk availability per host (df -h /):")
    for i, host in enumerate(hosts):
        try:
            with Connection(host=host, user=user, connect_kwargs={"key_filename": key_filename} if key_filename else {}) as conn:
                conn.open()
                # Use df with explicit columns for stable parsing
                r = conn.run("df -h --output=target,avail,pcent / | tail -n 1", hide=True, warn=True)
                if r.ok:
                    mount, avail, used_pct = r.stdout.strip().split()
                    print(f"  host_{i} ({host}): mount={mount}, avail={avail}, used={used_pct}")
                else:
                    print(f"  host_{i} ({host}): failed to read disk space ({r.stderr.strip()})")

                # Run maintenance commands sequentially; reboot will end the session
                print(f"  host_{i} ({host}): running apt update")
                conn.sudo("apt update", hide=True, warn=True)

                print(f"  host_{i} ({host}): running apt full-upgrade")
                conn.sudo("apt full-upgrade -y", hide=True, warn=True)

                print(f"  host_{i} ({host}): running apt auto-remove --purge")
                conn.sudo("apt auto-remove --purge -y", hide=True, warn=True)

                print(f"  host_{i} ({host}): rebooting")
                conn.sudo("reboot", hide=True, warn=True)
        except Exception as e:
            print(f"  host_{i} ({host}): error: {e}")
    


if __name__ == "__main__":
    main()



