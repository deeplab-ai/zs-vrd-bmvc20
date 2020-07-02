# From Saturation to Zero-Shot Visual Relationship Detection Using Local Context
Code for our BMVC 2020 submission.

## Requirements
Tested with Python 3.5 and 3.6.
```
python3 -r requirements.txt
```

## Setup
1. Clone the repository
```
git clone https://github.com/vrdmaniac/zs-vrd.git
cd zs-vrd
```
2. Setup data
```
python3 main_prerequisites.py
```

## Model zoo
The full list of tested models is included in common/models.
There are two basic model categories:
* (Scene Graph) Generators: models that use standard linear classifiers.
* (Scene Graph) Projectors: models that construct linear classifiers using language.

## Train/test a model
```
python3 main_research.py --model=MODEL --test_dataset=TEST_DATASET
```
See main_research.py for other input arguments.

Example:
```
python3 main.py --model=visual_spat_projector
```
will train a visual-spatial net with a local-context-aware classifier and test it on VRD.

Other useful flags:
* compute_accuracy: compute accuracy instead of precision
* use_weighted_ce: weight cross-entropy terms
* test_dataset: specify a different dataset for testing (only valid for projectors)

## Still pending
This code is part of a wider project. Specifically, the code for synonym mining is messy. We will upload a clean version in the next few days.