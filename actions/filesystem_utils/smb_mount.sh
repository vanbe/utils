#!/bin/bash

# SMB Mount Script
# Usage: ./smb_mount.sh <share>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <share>"
    exit 1
fi

SHARE=$1

# Read credentials and host from .env
if [ -f "../actions/.env" ]; then
    USER=$(grep '^NAS_USER=' ../actions/.env | cut -d'=' -f2)
    PASS=$(grep '^NAS_PASS=' ../actions/.env | cut -d'=' -f2)
    HOST=$(grep '^NAS_HOST=' ../actions/.env | cut -d'=' -f2)
fi
if [ -z "$HOST" ]; then
    echo "NAS_HOST not set in .env"
    exit 1
fi

SHARE_PATH="//$HOST/$SHARE"
MOUNT_POINT="/mnt/nas/$SHARE"

echo "Mounting NAS share: $SHARE_PATH to $MOUNT_POINT"

# Ping the host
echo "Pinging host $HOST..."
if ! ping -c 1 -W 4 "$HOST" >/dev/null 2>&1; then
    echo "Ping failed: host not reachable"
    exit 1
fi
echo "Ping successful."

# Construct options
if [ -n "$USER" ] && [ -n "$PASS" ]; then
    OPTIONS="username=$USER,password=$PASS,uid=$(id -u),gid=$(id -g),file_mode=0664,dir_mode=0775"
else
    OPTIONS="guest,uid=$(id -u),gid=$(id -g),file_mode=0664,dir_mode=0775"
fi

# Create mount directory
echo "Creating mount directory..."
if ! sudo mkdir -p "$MOUNT_POINT"; then
    echo "Failed to create mount directory"
    exit 1
fi
echo "Directory created."

# Mount the share
echo "Mounting share..."
if sudo mount.cifs "$SHARE_PATH" "$MOUNT_POINT" -o "$OPTIONS"; then
    echo "NAS share mounted successfully"
else
    echo "Failed to mount NAS share"
    exit 1
fi