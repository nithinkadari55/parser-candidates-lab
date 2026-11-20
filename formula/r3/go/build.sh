#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VERSION=v0.4.1
tmp=$(mktemp -d)
(cd "$tmp" && go mod init dl >/dev/null 2>&1 && go mod download -json "github.com/go-pdfkit/latex@$VERSION") > "$tmp/dl.json"
src=$(sed -n 's/^[[:space:]]*"Dir": "\(.*\)",\{0,1\}$/\1/p' "$tmp/dl.json")
sum=$(sed -n 's/^[[:space:]]*"Sum": "\(.*\)",\{0,1\}$/\1/p' "$tmp/dl.json")
echo "go-pdfkit/latex $VERSION $sum from $src"
rm -rf third_party/latex "$tmp"
mkdir -p third_party
cp -r "$src" third_party/latex
chmod -R u+w third_party/latex
cp _bridge/harness_export.go third_party/latex/harness_export.go
if [ ! -f go.sum ]; then
  go mod tidy
fi
go vet .
go build -trimpath -o bin/gopdfkit-region .
