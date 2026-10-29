#!/usr/bin/env bash
# Three-way index of top-level definitions: base (what ggmlR was synced from),
# up (new upstream), ours (ggmlR). Classifies each definition by body hash.
#
#   tools/fn3way.sh OUTDIR BASE_DIR UP_DIR OURS_DIR   # full index
#   tools/fn3way.sh OUTDIR                            # reindex ours only (dirs from OUTDIR/dirs)
#
# Env: KIND=fn|struct (default fn), PAT="glob ..." (default "ggml-vulkan*.cpp ggml-vulkan*.h")
#
# Output: OUTDIR/{base,up,ours}.idx  — key name file start end hash
#         OUTDIR/classes.tsv         — key class ours_loc up_loc base_loc
# Key = name; for a name defined more than once on one side, name|normalized-signature.
#
# Classes: SAME, TAKE_UP (ours == base != up), OURS_ONLY_CHANGED (up == base != ours),
#   SAME_CHANGE (up == ours != base), MERGE (all differ), NEW_UP, OURS_ONLY,
#   BOTH_NEW, SAME_NEW, REMOVED_OURS (in base+up), REMOVED_UP (in base+ours), REMOVED_BOTH.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=${1:?OUTDIR}
mkdir -p "$out"

if [ $# -ge 4 ]; then
    printf '%s\n%s\n%s\n%s\n%s\n' "$2" "$3" "$4" "${KIND:-fn}" "${PAT:-ggml-vulkan*.cpp ggml-vulkan*.h}" > "$out/dirs"
    sides="base up ours"
else
    [ -f "$out/dirs" ] || { echo "no $out/dirs: run the full form first" >&2; exit 1; }
    sides="ours"
fi
mapfile -t cfg < "$out/dirs"
declare -A dir=([base]=${cfg[0]} [up]=${cfg[1]} [ours]=${cfg[2]})
kind=${cfg[3]}
pat=${cfg[4]}

index_side() {
    local side=$1 d=${dir[$1]} files=() f bodies
    for f in $pat; do
        for g in "$d"/$f; do [ -f "$g" ] && files+=("$g"); done
    done
    [ ${#files[@]} -gt 0 ] || { echo "$side: no files matching '$pat' in $d" >&2; exit 1; }
    bodies=$(mktemp -d)
    awk -v KIND="$kind" -v BODYDIR="$bodies" -f "$here/fnidx.awk" "${files[@]}" > "$bodies/raw"
    ( cd "$bodies" && find . -maxdepth 1 -name '[0-9]*' -printf '%f\n' | xargs -r md5sum ) \
        | awk '{print $2 "\t" $1}' > "$bodies/hash"
    # join hash, then build key (name, or name|sig when the name repeats on this side)
    awk -F'\t' -v OFS='\t' '
        FNR == NR { h[$1] = $2; next }
        { n[++m] = $0; cnt[$1]++ }
        END {
            for (i = 1; i <= m; i++) {
                split(n[i], a, "\t")
                key = (cnt[a[1]] > 1) ? a[1] "|" a[5] : a[1]
                print key, a[1], a[2], a[3], a[4], h[a[6]]
            }
        }' "$bodies/hash" "$bodies/raw" > "$out/$side.idx"
    rm -rf "$bodies"
    echo "$side: $(wc -l < "$out/$side.idx") definitions in ${#files[@]} files"
}

for s in $sides; do index_side "$s"; done

awk -F'\t' -v OFS='\t' '
    FILENAME ~ /base\.idx$/ { b[$1] = $6; bl[$1] = $3 ":" $4 "-" $5; k[$1]; next }
    FILENAME ~ /up\.idx$/   { u[$1] = $6; ul[$1] = $3 ":" $4 "-" $5; k[$1]; next }
    FILENAME ~ /ours\.idx$/ { o[$1] = $6; ol[$1] = $3 ":" $4 "-" $5; k[$1]; next }
    END {
        for (x in k) {
            ib = (x in b); iu = (x in u); io = (x in o)
            if (ib && iu && io) {
                if (b[x] == u[x] && u[x] == o[x]) c = "SAME"
                else if (b[x] == o[x]) c = "TAKE_UP"
                else if (b[x] == u[x]) c = "OURS_ONLY_CHANGED"
                else if (u[x] == o[x]) c = "SAME_CHANGE"
                else c = "MERGE"
            } else if (ib && iu) c = "REMOVED_OURS"
            else if (ib && io)   c = "REMOVED_UP"
            else if (iu && io)   c = (u[x] == o[x]) ? "SAME_NEW" : "BOTH_NEW"
            else if (iu)         c = "NEW_UP"
            else if (io)         c = "OURS_ONLY"
            else                 c = "REMOVED_BOTH"
            print x, c, (io ? ol[x] : "-"), (iu ? ul[x] : "-"), (ib ? bl[x] : "-")
        }
    }' "$out/base.idx" "$out/up.idx" "$out/ours.idx" | sort -t$'\t' -k2,2 -k1,1 > "$out/classes.tsv"

cut -f2 "$out/classes.tsv" | sort | uniq -c | sort -rn
