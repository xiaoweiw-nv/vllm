#!/usr/bin/env bash
set -euo pipefail

pin="${FLASHINFER_SM12X_PIN:-61503db7ee6442e393ed119662f4d066d5098524}"
manifest="${FLASHINFER_SM12X_MANIFEST:-/tmp/flashinfer-sm12x-manifest.txt}"

if [[ ! -f "${manifest}" ]]; then
    echo "missing FlashInfer SM12x manifest: ${manifest}" >&2
    exit 1
fi

flashinfer_dst="${FLASHINFER_DST:-}"
if [[ -z "${flashinfer_dst}" ]]; then
    for candidate in \
        /opt/venv/lib/python*/site-packages/flashinfer \
        /usr/local/lib/python*/site-packages/flashinfer \
        /usr/local/lib/python*/dist-packages/flashinfer; do
        if [[ -d "${candidate}" ]]; then
            flashinfer_dst="${candidate}"
            break
        fi
    done
fi

if [[ -z "${flashinfer_dst}" || ! -d "${flashinfer_dst}" ]]; then
    echo "could not find installed flashinfer package" >&2
    exit 1
fi

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

archive="${workdir}/flashinfer.tar.gz"
curl -fsSL \
    "https://github.com/flashinfer-ai/flashinfer/archive/${pin}.tar.gz" \
    -o "${archive}"
tar -xzf "${archive}" -C "${workdir}"

src="${workdir}/flashinfer-${pin}/flashinfer"
sm12x_src="${src}/fused_moe/cute_dsl/blackwell_sm12x"
sm12x_dst="${flashinfer_dst}/fused_moe/cute_dsl/blackwell_sm12x"

if [[ ! -d "${sm12x_src}" ]]; then
    echo "missing pinned FlashInfer SM12x source: ${sm12x_src}" >&2
    exit 1
fi

while IFS= read -r relpath; do
    [[ -n "${relpath}" ]] || continue
    [[ "${relpath}" != /* ]] || {
        echo "manifest path must be relative: ${relpath}" >&2
        exit 1
    }
    [[ "${relpath}" != *..* ]] || {
        echo "manifest path must not contain '..': ${relpath}" >&2
        exit 1
    }
    install -D -m 0644 "${sm12x_src}/${relpath}" "${sm12x_dst}/${relpath}"
done < "${manifest}"

install -D -m 0644 "${src}/tllm_enums.py" "${flashinfer_dst}/tllm_enums.py"

find "${sm12x_dst}" -name '*.pyc' -delete
find "${sm12x_dst}" -type d -name __pycache__ -empty -delete
if [[ -d "${flashinfer_dst}/__pycache__" ]]; then
    find "${flashinfer_dst}/__pycache__" -name 'tllm_enums*.pyc' -delete
fi
