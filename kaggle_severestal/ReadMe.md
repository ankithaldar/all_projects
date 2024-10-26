# [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection)

WIP 1st Place solution to the [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) hosted on Kaggle.

## Context
> In this competition, you’ll help engineers improve the algorithm by localizing and classifying surface defects on a steel sheet.

## Evaluation
> This competition is evaluated on the mean Dice coefficient. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:
>
> [$ \frac{2 * \left| X \cap Y \right|}{ \left| X \right| + \left| Y \right| } $](https://www.kaggle.com/code/yerramvarun/understanding-dice-coefficient)
>
> ![Dice Coefficient](https://miro.medium.com/v2/resize:fit:429/1*yUd5ckecHjWZf6hGrdlwzA.png)
> where X is the predicted set of pixels and Y is the ground truth. The Dice coefficient is defined to be 1 when both X and Y are empty. The leaderboard score is the mean of the Dice coefficients for each <ImageId, ClassId> pair in the test set.

## Solution
> Our solution is a two step pipeline :
> - First remove images with no faults with a classifier
> - Segment the remaining images

### Classification
- __Models__
  - efficientnet-b1 [batch size: 32]
  - resnet34 [batch size: 32]
- __Ensemble__
  - 3 * efficientnet-b1 + 1 * resnet34
- __Optimizer__
  - Stochastic gradient descent
- __Train Augmentation__
  - Random Crop [224x1568]
  - Horizontal Flip
  - Vertical Flip
  - RandomBrightnessContrast (from albumentations)
  - A customized defect blackout
- __Test Augmentation__
  - None
  - Horizontal Flip
  - Vertical Flip
- __Threshold__
  - 0.6, 0.6, 0.6, 0.6
