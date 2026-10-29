# Index top-level definitions of a C/C++ file.
#   KIND=fn     : functions (definition line at column 0, body up to the first "}" at column 0)
#   KIND=struct : struct/class/enum/union definitions (same body rule)
# A "template <...>" line directly above the definition is included in the range.
# Output (TSV): name  file  start  end  sig  bodyfile
# Each body is written, trailing whitespace stripped, to BODYDIR/<n> for hashing.

function trim(s) { gsub(/^[ \t]+|[ \t\r]+$/, "", s); return s }

function normsig(s) {
    gsub(/[ \t\r]+/, " ", s)
    sub(/^template ?<[^>]*> ?/, "", s)
    gsub(/(^| )(static|inline) /, " ", s)
    sub(/ ?\{.*$/, "", s)
    return trim(s)
}

function fn_name(s,    pre) {
    pre = s
    sub(/\(.*$/, "", pre)
    pre = trim(pre)
    if (match(pre, /[A-Za-z_~][A-Za-z0-9_:~]*$/)) return substr(pre, RSTART, RLENGTH)
    return ""
}

function struct_name(s,    t) {
    t = s
    sub(/^typedef[ \t]+/, "", t)
    sub(/^(struct|class|enum|union)[ \t]+/, "", t)
    sub(/^class[ \t]+/, "", t)
    gsub(/(alignas|__attribute__)[ \t]*\([^)]*\)+[ \t]*/, "", t)
    if (match(t, /^[A-Za-z_][A-Za-z0-9_]*/)) return substr(t, RSTART, RLENGTH)
    return ""
}

function begin_cand(    ) {
    cstart = FNR; csig = $0; nbody = 0; body[++nbody] = $0
    if (prev ~ /^template[ \t]*</) { cstart = FNR - 1; csig = prev " " $0; body[1] = prev; body[++nbody] = $0 }
    cname = (KIND == "fn") ? fn_name($0) : struct_name($0)
    state = 1
}

function emit(endline,    i, bf, n) {
    if (cname == "" && KIND == "struct") {
        # anonymous typedef struct { ... } name;
        n = $0; sub(/^}[ \t]*/, "", n); sub(/[ \t]*;.*$/, "", n); cname = n
    }
    if (cname == "") { state = 0; return }
    ++cnt
    bf = BODYDIR "/" cnt
    for (i = 1; i <= nbody; i++) { s = body[i]; sub(/[ \t\r]+$/, "", s); print s > bf }
    close(bf)
    printf "%s\t%s\t%d\t%d\t%s\t%s\n", cname, FILENAME, cstart, endline, normsig(csig), cnt
    state = 0
}

# "enum ggml_status f(" — a function returning a tagged type, not a type definition
BEGIN { RETTYPE = "^(struct|class|enum|union)[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t*&]+[A-Za-z_][A-Za-z0-9_:]*[ \t]*\\(" }

FNR == 1 { state = 0; prev = "" }

{
    line = $0
    if (state == 0) {
        if (KIND == "fn") {
            if (line ~ /^[A-Za-z_]/ && line ~ /\(/ && line !~ /;[ \t\r]*(\/\/.*)?$/ &&
                (line !~ /^(struct|class|enum|union)[ \t]/ || line ~ RETTYPE) &&
                line !~ /^(typedef|namespace|using|extern|template[ \t]*<[^>]*>[ \t\r]*$)/) {
                begin_cand()
            }
        } else {
            if (line ~ /^(typedef[ \t]+)?(struct|class|enum|union)[ \t]/ && line !~ /;[ \t\r]*(\/\/.*)?$/ && line !~ RETTYPE) {
                begin_cand()
            }
        }
        if (state == 1) {
            if (line ~ /\{/) {
                state = 2
                o = gsub(/\{/, "{", line); c = gsub(/\}/, "}", line)
                if (o == c && line ~ /\}[ \t;\r]*(\/\/.*)?$/) emit(FNR)
            }
        }
    } else if (state == 1) {
        csig = csig " " line; body[++nbody] = line
        if (line ~ /\{/) state = 2
        else if (line ~ /;[ \t\r]*$/ || FNR - cstart > 30) state = 0
        if (state == 2 && line ~ /^\}/) emit(FNR)
    } else if (state == 2) {
        body[++nbody] = line
        if (line ~ /^\}/) emit(FNR)
    }
    prev = line
}
