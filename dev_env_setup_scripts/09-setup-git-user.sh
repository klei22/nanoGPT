#!/bin/bash
# 09-setup-git-user.sh
# Configures global Git settings for the user.

set -e

# --- 1. Set the Default Editor ---
# Change "nvim" to your preferred editor, e.g., "vim", "nano", or "code --wait".
EDITOR="nvim"

echo "🚀 Setting default Git editor to '$EDITOR'..."
git config --global core.editor "$EDITOR"
echo "✅ Editor set."
echo ""


# --- 2. Prompt for and Set User Name ---
echo "👤 Please enter your full name for Git commits:"
read -p "Full Name: " GIT_USER_NAME

echo "🚀 Setting global git user.name to '$GIT_USER_NAME'..."
git config --global user.name "$GIT_USER_NAME"
echo "✅ User name set."
echo ""


# --- 3. Prompt for and Set User Email ---
echo "✉️ Please enter your email address for Git commits:"
read -p "Email Address: " GIT_USER_EMAIL

echo "🚀 Setting global git user.email to '$GIT_USER_EMAIL'..."
git config --global user.email "$GIT_USER_EMAIL"
echo "✅ User email set."
echo ""


# --- Final Message ---
echo "🎉 Git configuration complete!"
git config --list | grep "user."
