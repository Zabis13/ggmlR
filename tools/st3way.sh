#!/usr/bin/env bash
# Same as fn3way.sh, for struct/class/enum/union definitions.
#   tools/st3way.sh OUTDIR BASE_DIR UP_DIR OURS_DIR
exec env KIND=struct "$(dirname "$0")/fn3way.sh" "$@"
