#!/usr/bin/env bash
# Linux counterpart of build-release.ps1.
# Builds the xai-grok-pager release binary and packages it (with LICENSE and
# THIRD-PARTY-NOTICES) as grok-linux-<arch>.tar.gz under the output directory.
#
# Usage: ./build-release.sh [output-directory]
#   default output directory: <repo>/target/package

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-$repo/target/package}"
binary="$repo/target/release/xai-grok-pager"

case "$(uname -m)" in
    x86_64|amd64)  arch='x64' ;;
    aarch64|arm64) arch='arm64' ;;
    *)             arch="$(uname -m)" ;;
esac

package_dir="$output_dir/grok-linux-$arch"
archive="$package_dir.tar.gz"

# The Windows script routes protoc through a wrapper exe when present; on
# Linux a plain protoc from PATH is sufficient.
if [[ -z "${PROTOC:-}" ]]; then
    if ! command -v protoc >/dev/null 2>&1; then
        echo "error: protoc not found in PATH (or set the PROTOC env var)" >&2
        exit 1
    fi
    export PROTOC="$(command -v protoc)"
fi

export CARGO_HTTP_CHECK_REVOKE='false'
if [[ -z "${CARGO_BUILD_JOBS:-}" ]]; then
    cpus="$(nproc 2>/dev/null || echo 1)"
    export CARGO_BUILD_JOBS="$(( cpus < 12 ? cpus : 12 ))"
fi

cd "$repo"

# The /DEBUG:NONE link-arg from the .ps1 is a Windows link.exe workaround
# (LNK1318); a plain release build is fine on Linux.
cargo build -p xai-grok-pager-bin --release --bin xai-grok-pager

rm -rf "$package_dir"
rm -f "$archive"
mkdir -p "$package_dir"

cp "$binary" "$package_dir/grok"
cp "$repo/LICENSE" "$package_dir/"
cp "$repo/THIRD-PARTY-NOTICES" "$package_dir/"

# Match Compress-Archive behavior: files at the archive root, no wrapping dir.
tar -czf "$archive" -C "$package_dir" grok LICENSE THIRD-PARTY-NOTICES
rm -rf "$package_dir"

echo "Release package: $archive"
