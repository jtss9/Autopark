#!/usr/bin/env bash
# Claude Code status line: folder, git branch, model name, context progress bar, effort level

input=$(cat)

model=$(echo "$input" | jq -r '.model.display_name // "Claude"')
used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
effort=$(echo "$input" | jq -r '.effortLevel // empty')
cwd=$(echo "$input" | jq -r '.workspace.current_dir // .cwd // empty')

# Folder: basename of cwd
if [ -n "$cwd" ]; then
  folder=$(basename "$cwd")
else
  folder=$(basename "$(pwd)")
  cwd=$(pwd)
fi

# Git branch (skip optional locks, suppress errors)
git_branch=$(git -C "$cwd" --no-optional-locks symbolic-ref --short HEAD 2>/dev/null)

# Build effort label
if [ -n "$effort" ]; then
  effort_label="effort: ${effort}"
  effort_len=${#effort_label}
else
  effort_label=""
  effort_len=0
fi

reset="\033[0m"
cyan="\033[36m"
magenta="\033[35m"
bold="\033[1m"

# Folder + branch prefix
if [ -n "$git_branch" ]; then
  prefix_visible="${folder} (${git_branch})  "
  prefix="${cyan}${bold}${folder}${reset} ${magenta}(${git_branch})${reset}  "
else
  prefix_visible="${folder}  "
  prefix="${cyan}${bold}${folder}${reset}  "
fi

# Build context bar
if [ -n "$used" ]; then
  pct=$(printf '%.0f' "$used")

  filled=$(( pct / 10 ))
  empty=$(( 10 - filled ))
  bar=""
  for i in $(seq 1 $filled); do bar="${bar}█"; done
  for i in $(seq 1 $empty);  do bar="${bar}░"; done

  if [ "$pct" -ge 80 ]; then
    ctx_color="\033[31m"
  elif [ "$pct" -ge 50 ]; then
    ctx_color="\033[33m"
  else
    ctx_color="\033[32m"
  fi

  ctx_visible="ctx: ${bar} ${pct}%"
  ctx_part="ctx: ${ctx_color}${bar} ${pct}%${reset}"
else
  ctx_visible=""
  ctx_part=""
fi

# Assemble left side
if [ -n "$ctx_visible" ]; then
  left_visible="${prefix_visible}${model}  ${ctx_visible}"
  left="${prefix}${model}  ${ctx_part}"
else
  left_visible="${prefix_visible}${model}"
  left="${prefix}${model}"
fi
left_len=${#left_visible}

# Get terminal width (default 80)
cols=$(tput cols 2>/dev/null || echo 80)

if [ -n "$effort_label" ]; then
  case "$effort" in
    low)      eff_color="\033[32m" ;;
    medium)   eff_color="\033[33m" ;;
    high|max) eff_color="\033[31m" ;;
    *)        eff_color="\033[0m"  ;;
  esac

  pad=$(( cols - left_len - effort_len ))
  [ "$pad" -lt 1 ] && pad=1
  spaces=$(printf '%*s' "$pad" '')

  printf "%b%s%b%s%b" "$left" "" "" "${spaces}${eff_color}${effort_label}${reset}" ""
else
  printf "%b" "$left"
fi
