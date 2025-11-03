"""Discovery of .cfg files locally and remotely over SSH key auth."""

from __future__ import annotations

import os
import pathlib
from typing import List, Optional


def discover_cfg_files(cfg_root: Optional[str]) -> List[str]:
    if not cfg_root:
        return []
    root = pathlib.Path(cfg_root)
    return [str(p) for p in root.rglob("*.cfg")]


def _load_pkey_from_file(path: str, passphrase: Optional[str]):
    import paramiko
    exc = None
    for KeyCls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
        try:
            return KeyCls.from_private_key_file(path, password=passphrase or None)
        except Exception as e:
            exc = e
            continue
    raise exc or RuntimeError("Unsupported private key format")


def discover_cfg_files_remote_key(
    host: str, user: str, remote_path: str,
    port: int = 22, key_path: Optional[str] = None, passphrase: Optional[str] = None,
    known_hosts: Optional[str] = None
) -> List[str]:
    import paramiko
    files: List[str] = []
    client = paramiko.SSHClient()
    if known_hosts and os.path.exists(known_hosts):
        client.load_host_keys(known_hosts)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    else:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    pkey = None
    agent = None
    try:
        agent = paramiko.Agent()
        if not agent.get_keys():
            agent = None
    except Exception:
        agent = None
    if agent is None and key_path and os.path.exists(key_path):
        pkey = _load_pkey_from_file(key_path, passphrase)

    try:
        client.connect(hostname=host, port=port, username=user,
                       pkey=pkey, allow_agent=bool(agent is not None),
                       look_for_keys=False, timeout=15)
        cmd = f"ls {remote_path}/*.cfg 2>/dev/null"
        stdin, stdout, stderr = client.exec_command(cmd, timeout=20)
        output = stdout.read().decode(errors="ignore")
        errout = stderr.read().decode(errors="ignore")
        if errout and not output:
            print(f"[WARN] SSH ls stderr: {errout.strip()}")

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            base = os.path.basename(line)
            if base.lower().endswith(".cfg"):
                files.append(base)
    except Exception as e:
        print(f"[WARN] SSH key auth failed or ls failed: {e}")
    finally:
        try:
            client.close()
        except Exception:
            pass

    return sorted(files)


