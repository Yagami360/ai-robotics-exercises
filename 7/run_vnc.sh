#!/bin/bash
set -eu

export DISPLAY=:1

# if [ ! -f ~/.vnc/xstartup ]; then
#     mkdir -p ~/.vnc
#     cat > ~/.vnc/xstartup << 'EOF'
#     #!/bin/bash
#     unset SESSION_MANAGER
#     unset DBUS_SESSION_BUS_ADDRESS
#     export XKL_XMODMAP_DISABLE=1

#     # マウスとキーボードの設定
#     xsetroot -solid grey
#     xrdb $HOME/.Xresources 2>/dev/null || true

#     # VNC設定
#     vncconfig -iconic &

#     # デスクトップ環境を起動
#     if command -v startxfce4 >/dev/null 2>&1; then
#         exec startxfce4
#     else
#         exec xterm -geometry 80x24+10+10 -ls -title "$VNCDESKTOP Desktop"
#     fi
#     EOF

#     chmod +x ~/.vnc/xstartup
# fi

set +e
vncserver -kill :1
set -e

vncserver :1 -geometry 1024x768 -depth 24 -localhost no -SecurityTypes VncAuth -SendCutText=0 -AcceptCutText=0 -AcceptPointerEvents=1 -AcceptKeyEvents=1
