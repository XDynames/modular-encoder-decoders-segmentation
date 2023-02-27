echo "Running isort..."
isort .

echo "Running black..."
black . -l 79

echo "Running flake8..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi