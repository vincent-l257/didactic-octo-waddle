#!/usr/bin/env bash
# Engine code outside src/platform and include/basalt/platform must not use raw OS calls or platform macros.
# tests/platform tests the backends themselves and is allowed like src/platform. tools/platform_boundary.allow lists the files that still do, with their hit counts. A count may only go down:
# a higher count fails, and a lower one fails too until the allowlist is lowered to match.
set -euo pipefail
cd "$(dirname "$0")/.."

allow=tools/platform_boundary.allow
pattern='_WIN32|_MSC_VER|__linux__|__unix__|windows\.h|sys/mman\.h|unistd\.h|sys/stat\.h|\bmmap\(|\bmunmap\(|madvise|posix_fadvise|sched_setaffinity|pthread_|VirtualAlloc|CreateFileW?\b|__builtin_(ctz|ia32|cpu_supports)|__attribute__\(\(target|\bdlopen\(|\bwaitpid\(|"/proc/|"/sys/'

declare -A allowed
while read -r count path; do
  [[ -z "${count}" || "${count}" == \#* ]] && continue
  allowed["${path}"]="${count}"
done < "${allow}"

status=0
declare -A seen
while IFS=: read -r path count; do
  seen["${path}"]=1
  have="${allowed[${path}]:-0}"
  if (( count > have )); then
    echo "platform boundary: ${path} has ${count} raw OS/platform hits, allowed ${have}." >&2
    echo "  Move the call behind src/platform (see CONTRIBUTING.md)." >&2
    status=1
  elif (( count < have )); then
    echo "platform boundary: ${path} is down to ${count} hits (allowlist says ${have}); lower it in ${allow}." >&2
    status=1
  fi
done < <(git grep -cE "${pattern}" -- src include tools tests \
           ':!src/platform' ':!include/basalt/platform' ':!tests/platform' ':!src/kernels/cpu/ggml' ':!*stb_image.h' \
           ':!*.py' ':!tools/check_platform_boundary.sh' ':!tools/platform_boundary.allow' || true)

for path in "${!allowed[@]}"; do
  if [[ -z "${seen[${path}]:-}" ]]; then
    echo "platform boundary: ${path} no longer has hits; remove it from ${allow}." >&2
    status=1
  fi
done

(( status == 0 )) && echo "platform boundary: ok"
exit "${status}"
