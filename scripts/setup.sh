#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
mode=${1:-all}
case "$mode" in
  all|control|learner|sampler) ;;
  -h|--help) printf 'Usage: setup.sh [all|control|learner|sampler]\n'; exit 0 ;;
  *) printf 'Usage: setup.sh [all|control|learner|sampler]\n' >&2; exit 2 ;;
esac
command -v uv >/dev/null
mkdir -p .venvs
exec 7>.venvs/setup.lock
flock -n 7
uv python install 3.12.14
for role in control learner sampler; do
  if [[ "$mode" == all || "$mode" == "$role" ]]; then
    [[ -x ".venvs/$role/bin/python" ]] || uv venv --python 3.12.14 ".venvs/$role"
    uv pip sync --python ".venvs/$role/bin/python" --require-hashes --only-binary :all: "requirements/$role.lock"
    uv pip check --python ".venvs/$role/bin/python"
  fi
done
if [[ ! -f .venvs/openenv/envs/harbor_env/harness.py ]]; then
  stage=$(mktemp -d .venvs/openenv-stage.XXXXXX)
  trap 'rm -rf -- "$stage"' EXIT
  curl --fail --silent --show-error --location --max-time 120 \
    https://codeload.github.com/openenv/OpenEnv/tar.gz/refs/tags/v0.15.0 -o "$stage/source.tgz"
  printf '%s  %s\n'     9ff2b3d2b714f12924f233e76c06ddc579253729bca3d721dc627c7146b87e81 "$stage/source.tgz" | sha256sum -c -
  mkdir "$stage/tree"
  tar -xzf "$stage/source.tgz" -C "$stage/tree" --strip-components=1 --wildcards '*/envs/harbor_env/*'
  mv "$stage/tree" .venvs/openenv
fi
