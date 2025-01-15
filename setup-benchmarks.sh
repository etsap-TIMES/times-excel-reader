#!/bin/bash

# A helper script to setup or update the repositories containing benchmark models under `benchmarks/`

set -eo pipefail

# Commit SHA for each repository:
REF_TIMES_model="b488fb07f0899ee8b7e710c230b1a9414fa06f7d"
REF_demos_xlsx="34a2a5c044cc0bbea1357de50db2f5f02d575181"
REF_demos_dd="2848a8a8e2fdcf0cdf7f83eefbdd563b0bb74e86"
REF_tim_xlsx="e820d8002adc6b1526a3bffcc439219b28d0eed5"
REF_tim_gams="cfe2628dbb5974b99c8a5664a9358849324e31ac"
REF_TIMES_NZ="4170d720e1c5cb0e31537a3168188169209ceb4d"

# If no GitHub token is provided, try to clone using SSH
if [ -z "$GH_PAT_DEMOS_XLSX" ]; then
    echo "Warning: no GitHub token provided, will try to clone private repos using SSH"
    use_SSH=1
fi

# Move to the directory containing this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

mkdir -p benchmarks

# Function to check out a repository at a specified commit
checkout_repo() {
    local repo=$1
    local dest_dir=$2
    local commit=$3
    local private=$4

    if [ -d "$dest_dir" ]; then
        echo "Directory $dest_dir already exists. Checking if on correct commit."
        pushd "$dest_dir" > /dev/null
        git fetch --depth=1 origin "$commit"
    else
        echo "Directory $dest_dir does not exist. Cloning repository."
        if [ -n "$private" ]; then
            if [ -n "$use_SSH" ]; then
                repo_url="git@github.com:${repo}.git"
            else
                repo_url="https://$GH_PAT_DEMOS_XLSX@github.com/${repo}/"
            fi
        else
            repo_url="https://github.com/${repo}/"
        fi
        git clone --filter=blob:none "$repo_url" "$dest_dir"
        pushd "$dest_dir" > /dev/null
    fi
    git checkout "$commit" || exit 1
    popd > /dev/null
    echo "$dest_dir: successfully checked out $repo at $commit"
}

# Array of repositories to check out, in the form repo|dest_dir|commit|private
repositories=(
    "etsap-TIMES/TIMES_model|TIMES_model|$REF_TIMES_model"
    "olejandro/demos-dd|benchmarks/dd|$REF_demos_dd"
    "olejandro/demos-xlsx|benchmarks/xlsx|$REF_demos_xlsx|true"
    "esma-cgep/tim|benchmarks/xlsx/TIM|$REF_tim_xlsx"
    "esma-cgep/tim-gams|benchmarks/dd/TIM|$REF_tim_gams"
    "olejandro/TIMES-NZ-Model-Files|benchmarks/TIMES-NZ|$REF_TIMES_NZ"
)

# Setup / update the repositories
for repo_info in "${repositories[@]}"; do
    IFS='|' read -r repo dest_dir commit private <<< "$repo_info"
    checkout_repo "$repo" "$dest_dir" "$commit" "$private"
done

# Create symlinks for TIMES-NZ since xlsx & dd files are in same repo
ln -s "$SCRIPT_DIR/benchmarks/TIMES-NZ/TIMES-NZ" "$SCRIPT_DIR/benchmarks/xlsx/TIMES-NZ"
ln -s "$SCRIPT_DIR/benchmarks/TIMES-NZ/TIMES-NZ-GAMS/times_scenarios/kea-v2_1_3" "$SCRIPT_DIR/benchmarks/dd/TIMES-NZ-KEA"
ln -s "$SCRIPT_DIR/benchmarks/TIMES-NZ/TIMES-NZ-GAMS/times_scenarios/tui-v2_1_3" "$SCRIPT_DIR/benchmarks/dd/TIMES-NZ-TUI"

echo "All benchmark repositories are set up and up to date :)"
