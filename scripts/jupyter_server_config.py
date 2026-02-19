c = get_config()  # type: ignore[name-defined]

c.ServerApp.ip = "127.0.0.1"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False

# Disable tokens (belt + suspenders)
c.ServerApp.token = ""
c.IdentityProvider.token = ""

# Ensure password auth is allowed (password set via `jupyter server password`)
# If you haven't set it yet, run: `jupyter server password`