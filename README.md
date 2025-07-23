Reproduction of Hinton's 1986 experiment on family tree prediction from the paper "Learning Distributed Representations of Concepts"

## Installation (Optional)

If you haven't installed uv, install it from [here](https://docs.astral.sh/uv/#installation).

## Main Result

Run

`uv run main.py`

You'll get:

```
Average test accuracy: 0.67
Total perfect accuracies percentage: 0.2
```

If everything goes well.

## Visualization

Run

`uv run main.py -s -v`

to save the model weights and generate visualizations of first layer weights similar to the paper.