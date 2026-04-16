#!/bin/bash
# Daily arXiv workflow: fetch, summarize, and extract knowledge graph
# Usage: ./daily.sh [channel] [category]
#
# Environment variables (from .env):
#   ARXIV_CATEGORY
#   ARXIV_SUMMARIZE_MODEL
#   ARXIV_SUMMARIZE_MODEL_PROVIDER
#   ARXIV_SUMMARIZE_OUTPUT
#   ARXIV_EXTRACTOR_OUTPUT
#   ARXIV_LOG_DIR   # 日志文件目录，默认为 $XDG_DATA_HOME/arxiv-daily/logs 或 ~/.local/share/arxiv-daily/logs
#   ARXIV_LOG_FILE  # 日志文件路径（优先级高于 ARXIV_LOG_DIR），默认为 $ARXIV_LOG_DIR/arxiv_daily.log

set -euo pipefail

# Logging function
# Determine user data directory
if [ -n "${ARXIV_LOG_DIR:-}" ]; then
    LOG_DIR="$ARXIV_LOG_DIR"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
    LOG_DIR="$XDG_DATA_HOME/arxiv-daily/logs"
else
    LOG_DIR="$HOME/.local/share/arxiv-daily/logs"
fi
mkdir -p "$LOG_DIR"
LOG_FILE="${ARXIV_LOG_FILE:-${LOG_DIR}/arxiv_daily.log}"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

CHANNEL="${1:-astro-ph}"
CATEGORY="${2:-${ARXIV_CATEGORY:-}}"

# Build category argument
CATEGORY_ARG=""
if [ -n "$CATEGORY" ]; then
    CATEGORY_ARG="--category $CATEGORY"
fi

log "Starting daily arXiv workflow for channel: $CHANNEL"

# Fetch articles and extract arxiv IDs
log "Fetching latest articles..."
ARXIV_IDS=$(uv run arXiv --log-level ERROR new --channel "$CHANNEL" $CATEGORY_ARG 2>/dev/null | grep -oE '[0-9]{4}\.[0-9]{4,5}' | sort -u)

if [ -z "$ARXIV_IDS" ]; then
    log "No articles found."
    exit 0
fi

log "Found $(echo "$ARXIV_IDS" | wc -w) articles."

# Process each article
for ARXIV_ID in $ARXIV_IDS; do
    log "Processing article: $ARXIV_ID"
    
    if uv run arXiv --log-level ERROR summarize "$ARXIV_ID" >/dev/null 2>&1; then
        log "Successfully summarized $ARXIV_ID"
    else
        log "Failed to summarize $ARXIV_ID"
    fi
    
    if uv run arXiv --log-level ERROR extractor "$ARXIV_ID" >/dev/null 2>&1; then
        log "Successfully extracted knowledge graph for $ARXIV_ID"
    else
        log "Failed to extract knowledge graph for $ARXIV_ID"
    fi
done

log "Daily workflow completed."
