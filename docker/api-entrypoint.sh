#!/bin/sh
# Entrypoint for the API image.
#
# When MEEPLEMATE_SOPS_ENV holds a sops-encrypted dotenv file, decrypt it and
# run the command with the decrypted values (the usual MM_* variables) in its
# environment. Otherwise run the command unchanged, so plain MM_* variables
# keep working.
#
# If the variable is set, decryption must succeed: sops exits non-zero before
# the command starts, and there is deliberately no fallback to plain
# variables. Set-but-empty counts as set.
set -eu

if [ "$#" -eq 0 ]; then
    echo "api-entrypoint: no command given" >&2
    exit 64
fi

if [ -z "${MEEPLEMATE_SOPS_ENV+set}" ]; then
    exec "$@"
fi

# sops picks the input format from the extension. The file holds ciphertext
# only; plaintext exists only in the command's environment.
sops_file="$(mktemp --suffix=.env)"
printf '%s' "$MEEPLEMATE_SOPS_ENV" > "$sops_file"
unset MEEPLEMATE_SOPS_ENV

# Single-quote an argument for /bin/sh.
quote() {
    rest=$1
    quoted=
    while :; do
        case $rest in
            *\'*)
                quoted="$quoted${rest%%\'*}'\\''"
                rest=${rest#*\'}
                ;;
            *)
                break
                ;;
        esac
    done
    printf "'%s%s'" "$quoted" "$rest"
}

# sops runs its command as one string through `/bin/sh -c`, so the arguments
# have to be quoted back into a single string. sops has already read the file
# by then, so that shell removes it before exec-ing the command, which leaves
# the command (e.g. uvicorn) receiving signals directly.
command="rm -f -- $(quote "$sops_file"); exec"
for arg in "$@"; do
    command="$command $(quote "$arg")"
done

exec sops exec-env --same-process "$sops_file" "$command"
