#!/usr/bin/env bash
# RawCull installer for Ubuntu / Debian.  Run from the project folder:  ./install.sh
set -euo pipefail
cd "$(dirname "$0")"
APP_DIR="$(pwd)"

echo "==> System packages (needs sudo)"
sudo apt-get update
# exiftool: reads the camera's AF point + capture time.  opencv-data: face detection cascades.
# libxcb-cursor0: required by Qt 6.5+ on X11.
sudo apt-get install -y python3 python3-venv python3-pip libimage-exiftool-perl opencv-data \
    libxcb-cursor0 libegl1 libxkbcommon-x11-0

echo "==> Python environment"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> Launcher"
cat > run.sh <<EOF
#!/usr/bin/env bash
cd "$APP_DIR"
exec .venv/bin/python -m rawcull "\$@"
EOF
chmod +x run.sh

mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/rawcull.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=RawCull
Comment=Find the best shots in a folder of RAW photos
Exec=$APP_DIR/run.sh
Terminal=false
Categories=Graphics;Photography;
EOF

echo
echo "Done.  Start it with:  ./run.sh            (or open 'RawCull' from the app menu)"
echo "Open a folder directly:  ./run.sh ~/Pictures/shoot"
echo "Command-line mode:       ./run.sh --cli ~/Pictures/shoot --write-xmp"
