#!/usr/bin/env bash
# Nautilus Pod bootstrap.  Start GPU recovery before non-essential setup because
# opportunistic allocations can be preempted within a few minutes.
set +e
export DEBIAN_FRONTEND=noninteractive
PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID2ByjSDwjcOftF2GedGDDRPq5qe0XBv40WPP1f3HnVJ downeyflyfan@downeyflyfan-OMEN-35L-Gaming-Desktop-GT16-0xxx"

echo "[init] immediately launching guarded MFVideo recovery"
/bin/bash /init/nautilus_auto_resume_train.sh \
  >> /root/downeyflyfan/MFVideo/nautilus-auto-resume.log 2>&1 &

echo "[init] apt install system pkgs (zsh, sshd, cli tools)"
apt-get update -qq
apt-get install -y -qq zsh openssh-server sudo git curl unzip file ca-certificates \
  ripgrep fd-find direnv neovim >/dev/null 2>&1
apt-get install -y -qq eza >/dev/null 2>&1 || true

mkdir -p /run/sshd /root/.local/bin /root/.config

echo "[init] starship"
if [ ! -x /root/.local/bin/starship ]; then
  curl -fsSL https://starship.rs/install.sh | sh -s -- -y -b /root/.local/bin >/dev/null 2>&1
fi

echo "[init] yazi"
if [ ! -x /root/.local/bin/yazi ]; then
  curl -fsSL -o /tmp/yazi.zip https://github.com/sxyazi/yazi/releases/latest/download/yazi-x86_64-unknown-linux-gnu.zip
  ( cd /tmp && unzip -oq yazi.zip && cp yazi-*-linux-gnu/yazi yazi-*-linux-gnu/ya /root/.local/bin/ )
fi

echo "[init] zinit"
if [ ! -d /root/.local/share/zinit/zinit.git ]; then
  git clone -q https://github.com/zdharma-continuum/zinit /root/.local/share/zinit/zinit.git
fi

echo "[init] .zshrc"
if [ ! -f /root/.zshrc ]; then
cat > /root/.zshrc <<'ZRC'
# --- Nautilus zsh init (managed) ---
export PATH="$HOME/.local/bin:$PATH"
export EDITOR=vim
ZINIT_HOME="$HOME/.local/share/zinit/zinit.git"
if [[ -f "$ZINIT_HOME/zinit.zsh" ]]; then
  source "$ZINIT_HOME/zinit.zsh"
  zinit light zsh-users/zsh-autosuggestions
  zinit light zsh-users/zsh-completions
  zinit light zsh-users/zsh-syntax-highlighting
fi
autoload -Uz compinit && compinit -u
command -v starship >/dev/null && eval "$(starship init zsh)"
function yy() {
  local tmp="$(mktemp -t yazi-cwd.XXXXXX)"
  yazi "$@" --cwd-file="$tmp"
  local cwd="$(cat -- "$tmp")"
  [[ -n "$cwd" && "$cwd" != "$PWD" ]] && cd -- "$cwd"
  rm -f -- "$tmp"
}
# --- end managed ---
ZRC
fi

echo "[init] cutlass (python pkg + source headers on PVC)"
python -c "import cutlass" >/dev/null 2>&1 || pip install -q nvidia-cutlass >/dev/null 2>&1
[ -d /root/cutlass ] || git clone -q --depth 1 https://github.com/NVIDIA/cutlass /root/cutlass

echo "[init] .zshenv (PATH for all shell modes)"
[ -f /root/.zshenv ] || echo 'export PATH="$HOME/.local/bin:$PATH"' > /root/.zshenv

echo "[init] ssh authorized_keys"
mkdir -p /root/.ssh && chmod 700 /root/.ssh
grep -qF "$PUBKEY" /root/.ssh/authorized_keys 2>/dev/null || echo "$PUBKEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

echo "[init] non-root user downeyflyfan (uid 1001, passwordless sudo)"
groupadd -g 1001 downeyflyfan 2>/dev/null
useradd -u 1001 -g 1001 -M -d /root/downeyflyfan -s /usr/bin/zsh downeyflyfan 2>/dev/null
mkdir -p /root/downeyflyfan && chown 1001:1001 /root/downeyflyfan
echo "downeyflyfan ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/downeyflyfan
chmod 440 /etc/sudoers.d/downeyflyfan
chmod 711 /root
mkdir -p /root/downeyflyfan/.ssh && chmod 700 /root/downeyflyfan/.ssh
grep -qF "$PUBKEY" /root/downeyflyfan/.ssh/authorized_keys 2>/dev/null || echo "$PUBKEY" >> /root/downeyflyfan/.ssh/authorized_keys
chmod 600 /root/downeyflyfan/.ssh/authorized_keys
chown -R 1001:1001 /root/downeyflyfan/.ssh

echo "[init] sshd config + start"
sed -i 's/#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
chsh -s /usr/bin/zsh root >/dev/null 2>&1
/usr/sbin/sshd

echo "[init] keep-alive (no jupyter; Pod stays up on sshd)"
exec sleep infinity
