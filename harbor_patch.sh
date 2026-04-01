#!/bin/bash
# Harbor patches for autophd_gym
# Usage: uv venv && uv sync && bash harbor_patch.sh
set -euo pipefail

VENV_SITE=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
echo "Site-packages: $VENV_SITE"

# ── Patch 1: terminus-2 max duration 60 → 1200 seconds ──
python3 -c "
from pathlib import Path
p = Path('$VENV_SITE/harbor/agents/terminus_2/terminus_2.py')
s = p.read_text()
old = 'duration_sec=min(parsed_cmd.duration, 60),'
new = 'duration_sec=min(parsed_cmd.duration, 1200),'
if old in s:
    p.write_text(s.replace(old, new, 1))
    print('patched terminus_2 max duration: 60 → 1200')
elif new in s:
    print('terminus_2 max duration already patched')
else:
    raise RuntimeError('terminus_2 duration pattern not found and not already patched')
"

# ── Patch 2: docker environment GPU support ──
python3 -c "
from pathlib import Path
p = Path('$VENV_SITE/harbor/environments/docker/docker.py')
s = p.read_text()
old = '    def supports_gpus(self) -> bool:\n        return False'
new = '    def supports_gpus(self) -> bool: \n        return True'
if old in s:
    p.write_text(s.replace(old, new, 1))
    print('patched docker GPU support: False → True')
elif 'return True' in s.split('supports_gpus')[1][:50]:
    print('docker GPU support already patched')
else:
    raise RuntimeError('docker GPU pattern not found and not already patched')
"

# ── Patch 3: GPU device selection via HARBOR_GPU_DEVICES env var ──
python3 << 'PYEOF'
from pathlib import Path
import sysconfig, textwrap

venv_site = sysconfig.get_path("purelib")
p = Path(f"{venv_site}/harbor/environments/docker/docker.py")
s = p.read_text()

marker = "# PATCH: GPU device selection"
if marker in s:
    print("GPU device selection already patched")
else:
    old_method = "    @property\n    def _docker_compose_paths(self) -> list[Path]:"
    if old_method not in s:
        print("WARNING: could not find _docker_compose_paths property")
    else:
        helper = textwrap.dedent('''\
    # PATCH: GPU device selection
    def _gpu_override_compose_path(self) -> "Path | None":
        """Generate a temp compose override to pin specific GPU devices."""
        gpu_devices = os.environ.get("HARBOR_GPU_DEVICES")
        if not gpu_devices:
            return None
        import tempfile
        content = f"""services:
  main:
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: "{gpu_devices}"
    deploy:
      resources:
        reservations:
          devices: !reset []
"""
        tmp = Path(tempfile.mktemp(suffix="-gpu-override.yaml", prefix="harbor-"))
        tmp.write_text(content)
        return tmp

''')
        s = s.replace(old_method, helper + "    " + old_method.lstrip())

        # Append GPU override as last compose file
        old_return = "        return paths"
        new_return = "        # PATCH: GPU device override\n        _gpu_ov = self._gpu_override_compose_path()\n        if _gpu_ov:\n            paths.append(_gpu_ov)\n\n        return paths"
        idx = s.rfind(old_return)
        if idx != -1:
            s = s[:idx] + new_return + s[idx + len(old_return):]
            p.write_text(s)
            print("patched GPU device selection: use HARBOR_GPU_DEVICES env var")
        else:
            print("WARNING: could not find return paths in _docker_compose_paths")
PYEOF

echo ""
echo "All patches applied. Usage:"
echo "  HARBOR_GPU_DEVICES=3 harbor run -p tasks/...     # use GPU 3"
echo "  HARBOR_GPU_DEVICES=0,1 harbor run -p tasks/...   # use GPU 0 and 1"
echo "  harbor run -p tasks/...                           # use all GPUs (default)"
echo ""
echo "setup complete"
