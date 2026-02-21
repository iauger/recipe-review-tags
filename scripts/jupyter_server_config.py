# scripts/jupyter_server_config.py
c = get_config()  # type: ignore[name-defined]

c.ServerApp.ip = "127.0.0.1"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False

# Disable token links
c.IdentityProvider.token = ""

# Require password (you already created it via `jupyter server password`)
c.PasswordIdentityProvider.enabled = True