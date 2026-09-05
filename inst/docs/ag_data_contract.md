# Contract: how `ag_tensor` values are accessed

Status: **draft, not yet implemented.** Written before the residency work
(TODO "Автограф: резидентность .ag_run_op", variant A) so that the rules exist
before code depends on them.

## Why this document exists

The previous contract was one comment on `R/autograd.R:71`:

```r
e$data <- data       # numeric matrix — always kept for backward
```

That line was not merely documentation — four sites relied on it as a mutable
storage invariant, and a runtime trace (`GGMLR_AG_TRACE_DATA=1`, 4827 tests,
56 656 reads across 107 sites) is what found them. None of them would fail
loudly if the invariant quietly stopped holding. That is the failure mode this
document exists to prevent, so the new rules are written down first and the
dangerous cases get their own named API rather than a comment.

## What changes

Today `$data` is always an R matrix and always authoritative. Once the tape is
GPU-resident, a tensor's value may live only in a backend buffer, with `$data`
absent until something asks for it. "Read the field" and "get the value" stop
being the same operation.

## The three access paths

### 1. `.ag_data(t)` — read

Returns the value as a plain R matrix, materialising from the device if that is
where it lives. Read-only: callers must not assume the result is connected to
the tensor, and writing to it changes nothing.

This is the default and covers the overwhelming majority of reads — every
compute site on the tape (`ag_matmul`, `ag_add`, losses, activations) is of this
kind, ~50 000 of the 56 656 traced reads.

### 2. `.ag_data_mut(t)` — read for modification

Returns a materialised, writable copy **and obliges the caller to write it back
through `.ag_data_set()`**. Nothing is observed until that call.

For read-modify-write cycles, where code today does `t$data <- f(t$data)`.
Distinct from `.ag_data()` so that the special case is visible in the code
rather than resting on a comment.

### 3. `.ag_data_set(t, value)` — write

Installs a new value, invalidating or refreshing device residency as needed.
The only supported way to change a tensor's value.

## Rules

1. **Never read `$data` directly.** Use `.ag_data()`. A direct read gets `NULL`
   for a resident tensor, and `NULL` entering R matrix arithmetic surfaces far
   from its cause.

2. **Never assign to `$data` directly.** Use `.ag_data_set()`. A direct write
   leaves the device copy stale and the two disagree silently.

3. **The value handle is never an arithmetic operand.** Whatever represents a
   resident value must have no `Ops` group methods, no `-`, `*`, `%*%`. Bare
   `externalptr` already errors in arithmetic (verified: "non-numeric argument
   to binary operator"), but that safety comes from it having no methods, not
   from the language. Should the handle become an S3 class, defining arithmetic
   on it would turn `p$data <- p$data - lr * g` from a loud error into a wrong
   answer. So: no arithmetic methods on the handle, ever.

4. **`$data` may be `NULL`.** Absence means "not materialised", never "empty".
   Emptiness is a 0-row matrix.

5. **A pointer is only valid with its generation.** `$ptr` and `$ctx_gen` travel
   together; `.ag_data()` refuses a pointer from an older generation rather than
   reading freed memory. Clearing one clears the other.

## Sites that must use the mutable path

Found by trace, not by reading the source — each does read-modify-write on
`$data` and would break silently under a read-only contract:

| site | what it does |
|---|---|
| `autograd.R:1279-1299` (`ag_gradcheck`) | perturbs one element, computes, restores |
| `autograd.R:789-800`, `856` (SGD, Adam) | `p$data <- p$data - lr * g` |
| `ag_training.R:431`, `443` (`dp_train`, `.sync_weights`) | copies weights between replicas |
| `ag_layers.R:520-524` (`ag_embedding`) | reads the weight table; comment already notes it must survive gradcheck substitution |

`ag_gradcheck` is the sharpest: it writes a perturbed value and expects the next
forward to see it. If forward reads a device buffer while the write lands in an
R matrix, the check silently validates the unperturbed network — a green test
that proves nothing. Its existing tests must give identical numbers before and
after the move.

## Order of work

1. This contract, and `autograd.R:71` updated to match. ← *current step*
2. `.ag_data()` read path + lazy materialisation. The bulk.
3. `.ag_data_mut()` / `.ag_data_set()` and the four sites above, each with a test.
4. ~14 mechanical replacements of direct `$data` reads in `ag_layers.R` (cold:
   414 reads and below).

Re-run the tracer after each step. A site whose count jumps has had a
materialisation introduced into a hot path — the round trip that residency is
meant to remove, reappearing.
