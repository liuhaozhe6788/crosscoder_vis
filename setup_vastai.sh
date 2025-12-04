vim ~/.bashrc
source ~/.bashrc

# Utility functions for colored output
print_info() {
    echo -e "\033[0;36m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

# Conda environment setup with Python 3.12
CONDA_ENV_NAME="gpu-server-env"

if ! conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    print_info "Creating conda environment '$CONDA_ENV_NAME' with Python 3.12..."
    conda create -y -n "$CONDA_ENV_NAME" python=3.12
    print_info "Conda environment '$CONDA_ENV_NAME' created."
else
    print_info "Conda environment '$CONDA_ENV_NAME' already exists."
fi

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

if [ -f "requirements.txt" ]; then
    print_info "Installing Python dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_info "Python dependencies installed."
else
    print_warning "requirements.txt not found. Skipping Python dependencies installation."
fi

if [ -z "$(git config --global user.name)" ]; then
    print_warning "Git user.name not configured"
    git config --global user.name "haozliu"
fi

if [ -z "$(git config --global user.email)" ]; then
    print_warning "Git user.email not configured"
    git config --global user.email "haozliu@ethz.ch"
fi

# GitHub SSH Key Setup
print_info "Setting up GitHub SSH authentication..."

SSH_DIR="$HOME/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [ ! -f "$SSH_DIR/id_ed25519" ] && [ ! -f "$SSH_DIR/id_rsa" ]; then
    print_info "Generating SSH key for GiittHub..."
    ssh-keygen -t ed25519 -C "liuhaozhe2000@gmail.com" -f "$SSH_DIR/id_ed25519" -N ""
    
    print_info "SSH public key generated. Add this to your GitHub account:"
    echo "=========================================="
    cat "$SSH_DIR/id_ed25519.pub"
    echo "=========================================="
    print_info "Visit: https://github.com/settings/keys"
    print_info "Click 'New SSH key' and paste the key above"
    
    read -p "Press Enter after adding the key to GitHub..."
else
    print_info "SSH key already exists"
    if [ -f "$SSH_DIR/id_ed25519.pub" ]; then
        print_info "Your SSH public key:"
        cat "$SSH_DIR/id_ed25519.pub"
    fi
fi