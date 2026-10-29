#!/usr/bin/env bash
# Replace one definition in ours with the upstream body (take) or with a
# three-way merge of ours/base/up bodies (merge; conflicts left as markers).
#
#   tools/fnapply.sh OUTDIR KEY take|merge
#
# OUTDIR is an index made by fn3way.sh / st3way.sh. Refuses when the key is
# missing on a needed side, when ours no longer matches the index (STALE),
# or when the new body would be empty. Reindexes ours afterwards.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=${1:?OUTDIR}; key=${2:?KEY}; mode=${3:?take|merge}
case $mode in take|merge) ;; *) echo "mode must be take or merge" >&2; exit 1 ;; esac

row() { awk -F'\t' -v k="$key" '$1 == k { print; n++ } END { exit n == 1 ? 0 : 1 }' "$out/$1.idx"; }

o=$(row ours) || { echo "REFUSE: '$key' not found exactly once in ours" >&2; exit 1; }
u=$(row up)   || { echo "REFUSE: '$key' not found exactly once in up" >&2; exit 1; }
if [ "$mode" = merge ]; then
    b=$(row base) || { echo "REFUSE: '$key' not found exactly once in base (merge needs it)" >&2; exit 1; }
fi

IFS=$'\t' read -r _ _ of os oe oh <<< "$o"
IFS=$'\t' read -r _ _ uf us ue _  <<< "$u"

cur=$(sed -n "${os},${oe}p" "$of" | sed 's/[ \t\r]*$//' | md5sum | cut -d' ' -f1)
[ "$cur" = "$oh" ] || { echo "REFUSE: STALE — $of:$os-$oe changed since indexing; rerun tools/fn3way.sh $out" >&2; exit 1; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
sed -n "${us},${ue}p" "$uf" > "$tmp/up"
sed -n "${os},${oe}p" "$of" > "$tmp/new"
conf=0
sigmsg=""
# signature = lines up to the first one ending in "{"; prints the line count, 0 if none
siglen() { awk '/\{[ \t\r]*$/ { print NR; f = 1; exit } END { if (!f) print 0 }' "$1"; }
# compare modulo static/inline and default arguments
signorm() { head -n "$2" "$1" | tr '\n' ' ' | sed -E 's/\b(static|inline) //g; s/[ \t]*=[ \t]*([^,()]|\([^()]*\))*//g; s/[ \t\r]+/ /g; s/ ?\{ ?$//; s/^ //; s/ $//'; }
if [ "$mode" = take ]; then
    # upstream moved some functions into separate .cpp files: there they lose
    # "static" and keep default arguments in a header. ggmlR is one TU, all static.
    ko=$(siglen "$tmp/new"); ku=$(siglen "$tmp/up")
    if [ "$ko" -gt 0 ] && [ "$ku" -gt 0 ] && [ "$(signorm "$tmp/new" "$ko")" = "$(signorm "$tmp/up" "$ku")" ]; then
        { head -n "$ko" "$tmp/new"; tail -n +$((ku + 1)) "$tmp/up"; } > "$tmp/take"
        cmp -s <(head -n "$ko" "$tmp/new") <(head -n "$ku" "$tmp/up") || sigmsg=", sig: kept ours"
    else
        cp "$tmp/up" "$tmp/take"
        if grep -m1 -v '^template' "$tmp/new" | grep -q '^static ' && ! grep -m1 -v '^template' "$tmp/up" | grep -q '^static '; then
            awk '!d && !/^template/ { $0 = "static " $0; d = 1 } { print }' "$tmp/up" > "$tmp/take"
        fi
        sigmsg=", SIG CHANGED — review (default arguments are not carried over)"
    fi
    mv "$tmp/take" "$tmp/new"
else
    IFS=$'\t' read -r _ _ bf bs be _ <<< "$b"
    sed -n "${bs},${be}p" "$bf" > "$tmp/base"
    set +e
    git merge-file -L ours -L base -L up "$tmp/new" "$tmp/base" "$tmp/up"
    conf=$?
    set -e
    [ $conf -ge 0 ] && [ $conf -lt 128 ] || { echo "REFUSE: git merge-file failed ($conf)" >&2; exit 1; }
fi
[ -s "$tmp/new" ] || { echo "REFUSE: new body is empty — nothing written" >&2; exit 1; }

total=$(wc -l < "$of")
{ head -n $((os - 1)) "$of"; cat "$tmp/new"; tail -n +$((oe + 1)) "$of"; } > "$tmp/file"
want=$(( total - (oe - os + 1) + $(wc -l < "$tmp/new") ))
got=$(wc -l < "$tmp/file")
[ "$got" -eq "$want" ] || { echo "REFUSE: line count mismatch ($got != $want) — nothing written" >&2; exit 1; }
cat "$tmp/file" > "$of"

echo "$mode $key -> $of:$os (was $((oe - os + 1)) lines, now $(wc -l < "$tmp/new"))$( [ "$mode" = merge ] && echo ", conflicts: $conf")$sigmsg"
"$here/fn3way.sh" "$out" > /dev/null
