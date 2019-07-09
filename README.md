# Generate To Adapt
PyTorch implementation of "Generate To Adapt: Aligning Domains using Generative Adversarial Networks".

## Datasets
Download the dataset from [here](http://www.cs.umd.edu/~yogesh/datasets/digits.zip) and extract it.
This folder contains the dataset in the same format as needed by the code.

## Training
```
python train.py --data-root <path to dataset> --output-root <path to output results>
```
Current checkpoint and the best-performing model will be stored in the output directory provided.

## Testing
```
python test.py --data-root <path to dataset>
```
