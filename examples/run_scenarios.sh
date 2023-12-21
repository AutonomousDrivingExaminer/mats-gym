#!/bin/bash
scenarios=${1:-$(ls *.py)}

cd ..
for f in $scenarios; do
    echo "Running $f"
    PYTHONPATH=. python examples/${f}
done