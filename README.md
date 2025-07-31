Reproduction of Hinton's 1986 experiment on family tree prediction from the paper "Learning Distributed Representations of Concepts"

## Installation (Optional)

If you haven't installed `uv`, install it from [here](https://docs.astral.sh/uv/#installation).

## Main Result

After installion of `uv`, simply run

`uv run main.py`

The required packages will be installed for you and you'll get:

```
Average test accuracy: 0.755
Total perfect accuracies percentage: 0.3
```

in the end. Feel free to change anything (model architecture / optimizer / loss, etc) and let me know if you can beat this number. :)

## Visualization

Run

`uv run main.py -s -v`

to save the model weights and generate visualizations of first layer weights similar to the paper.
