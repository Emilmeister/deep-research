#!/bin/sh
# shellcheck shell=dash
set -u

# Check if it's a valid file
check_file() {
    local target="$1"

    if [ ! -f "$target" ]; then
        cat <<EOF
!!!
!!! ERROR
!!! "$target" is not a valid file, exiting...
!!!
EOF
        exit 127
    fi
}

# Check if it's a valid directory
check_directory() {
    local target="$1"

    if [ ! -d "$target" ]; then
        cat <<EOF
!!!
!!! ERROR
!!! "$target" is not a valid directory, exiting...
!!!
EOF
        exit 127
    fi
}

setup_ownership() {
    local target="$1"
    local type="$2"

    case "$type" in
    file | directory) ;;
    *)
        cat <<EOF
!!!
!!! ERROR
!!! "$type" is not a valid type, exiting...
!!!
EOF
        exit 1
        ;;
    esac

    target_ownership=$(stat -c %U:%G "$target")

    if [ "$target_ownership" != "searxng:searxng" ]; then
        if [ "${FORCE_OWNERSHIP:-true}" = true ] && [ "$(id -u)" -eq 0 ]; then
            chown -R searxng:searxng "$target"
        else
            cat <<EOF
!!!
!!! WARNING
!!! "$target" $type is not owned by "searxng:searxng"
!!! This may cause issues when running SearXNG
!!!
!!! Expected "searxng:searxng"
!!! Got "$target_ownership"
!!!
EOF
        fi
    fi
}

# Handle volume mounts
volume_handler() {
    local target="$1"

    check_directory "$target"
    setup_ownership "$target" "directory"
}

# Handle configuration file updates
config_handler() {
    local target="$1"
    local template="$2"
    local new_template_target="$target.new"

    # Create/Update the configuration file
    if [ -f "$target" ]; then
        setup_ownership "$target" "file"

        if [ "$template" -nt "$target" ]; then
            cp -pfT "$template" "$new_template_target"

            cat <<EOF
...
... INFORMATION
... Update available for "$target"
... It is recommended to update the configuration file to ensure proper functionality
...
... New version placed at "$new_template_target"
... Please review and merge changes
...
EOF
        fi
    else
        cat <<EOF
...
... INFORMATION
... "$target" does not exist, creating from template...
...
EOF
        cp -pfT "$template" "$target"

        sed -i "s/ultrasecretkey/$(head -c 24 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9')/g" "$target"
    fi

    check_file "$target"
}

cat <<EOF
SearXNG $SEARXNG_VERSION
EOF

# Check for volume mounts
volume_handler "$CONFIG_PATH"
volume_handler "$DATA_PATH"

# Check for files
config_handler "$SEARXNG_SETTINGS_PATH" "/usr/local/searxng/searx/settings.yml"

# Создаем директорию для логов, если её нет
mkdir -p /var/log/searxng

# Запускаем python mcp_search.py в фоновом режиме с nohup
echo "Starting python mcp.py in background with nohup..."
nohup /usr/local/searxng/venv/bin/python /u01/mcp/mcp_search.py > /var/log/searxng/mcp.log 2>&1 &
MCP_PID=$!
echo "python mcp.py started with PID: $MCP_PID, logs: /var/log/searxng/mcp.log"

# Небольшая пауза, чтобы убедиться что процесс запустился
sleep 20

# Проверяем, что процесс mcp_search.py действительно запустился
if ! kill -0 $MCP_PID 2>/dev/null; then
    echo "ERROR: Failed to start python mcp.py"
    exit 1
fi

# Запускаем основную программу SearXNG
echo "Starting SearXNG..."
exec /usr/local/searxng/venv/bin/granian searx.webapp:app