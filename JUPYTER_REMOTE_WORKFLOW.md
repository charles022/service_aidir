# Remote Jupyter via SSH Tunnel

## Objective
Run Jupyter Notebook/Lab on a remote server over SSH, and use a browser on the laptop as if Jupyter were local.

## Why this setup
- Jupyter stays bound to `127.0.0.1` on the server (not publicly exposed).
- Access happens through encrypted SSH port forwarding.
- Daily use is fast: start server-side Jupyter, open SSH tunnel, browse locally.

## Network model
- Server runs Jupyter on: `127.0.0.1:8888`
- Laptop SSH tunnel maps: `localhost:8888 -> server 127.0.0.1:8888`
- Browser opens: `http://127.0.0.1:8888`

## Config files involved
- Server Jupyter config: `~/.jupyter/jupyter_server_config.py`
- Laptop SSH config: `~/.ssh/config`

## Automation script
This repo includes:
- `setup_jupyter_ssh.sh`

It creates/updates both config files idempotently and keeps timestamped backups.

## Script usage
```bash
bash setup_jupyter_ssh.sh <host_alias> <server_host_or_ip> <server_user> <identity_file>
```

Example:
```bash
bash setup_jupyter_ssh.sh myserver 203.0.113.10 chuck ~/.ssh/id_ed25519
```

## Daily workflow
1. SSH to server and start Jupyter:
```bash
ssh myserver
source ~/.venvs/jupyter/bin/activate
jupyter lab
```

2. On laptop, open tunnel (if not already open):
```bash
ssh -Nf myserver
```

3. Open browser on laptop:
```text
http://127.0.0.1:8888
```

4. Authenticate with Jupyter token shown in server logs.

## Recommended reliability
Use `tmux` on the server so Jupyter survives SSH disconnects:
```bash
tmux new -s jupyter
# start jupyter lab inside tmux
```

## Security notes
- Keep `c.ServerApp.ip = "127.0.0.1"`.
- Do not expose Jupyter directly on `0.0.0.0` unless intentionally protected.
- Use SSH keys and avoid password SSH when possible.
