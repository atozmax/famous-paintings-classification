# Report

This project trains a classifier that names the painter of an image. There are 25 painters and 6,516 labeled paintings. A held-out course test set has 300 images. The model submitted for that test set is a fine-tuned ResNet-152, saved as `models/resnet152_progressive.pth`. It labels **274 / 300** test images correctly (**91.3%**).

## Data

The archive is the course file `famous_paintings.zip` (Google Drive id `1-0d315aj7Ai8NNqat65XDvaOcHDcHiUD`, about 580 MB). Training files are named `Painter_Name_<number>.jpg`. Locally they are stored in `data/0` through `data/24`. The folder number is the class index, and it matches alphabetical order of the painter name.

Test files are `test_data/test_0.jpg` through `test_data/test_299.jpg`. They have no painter name in the filename.

The class sizes are uneven. Vincent van Gogh has 877 paintings and Edgar Degas has 702. Frida Kahlo has 120 and Kazimir Malevich has 126. The largest class is about seven times the smallest.

## First models

`code.ipynb` loads ImageNet ResNet-18, ResNet-34, and ResNet-152. Torchvision has no ResNet-150, so ResNet-152 is the deep model. Only the final linear layer is replaced (25 outputs). The rest of each network starts from ImageNet weights.

There is no augmentation. Each image is resized to 224×224 and normalized with ImageNet statistics.

The split is `random_split`, 80% train and 20% validation, seed 42. That split is **not stratified**. A painter’s validation share ranges from 15.4% (Leonardo da Vinci) to 24.4% (Amedeo Modigliani) instead of a flat 20%.

Training uses Adam at `1e-4`, at most 50 epochs, batch size 32. If validation accuracy does not improve for 10 epochs, the learning rate is multiplied by 0.1. Training stops after 20 epochs without an improvement. The kept weights from this stage are `models/resnet18.pth`, `models/resnet34.pth`, and `models/resnet152.pth`.

A follow-up in the same notebook reloads ResNet-152 and unfreezes `layer4` together with the classifier, then a further pass also unfreezes `layer3`. The checkpoint kept from that retraining is `models/resnet152_retrained.pth`.

## Progressive fine-tune

`code2.ipynb` starts from `resnet152_retrained.pth` and trains from the classifier backward:

1. `fc`
2. `layer4`
3. `layer3`
4. `layer2`
5. `layer1`
6. stem (`conv1` and `bn1`)

Each stage lasts at most 50 epochs and starts at learning rate `1e-3`. If validation accuracy does not improve for 7 epochs, the learning rate is multiplied by 0.1. The stage stops after 14 epochs without an improvement. A stage that never beats its starting accuracy reloads the previous best weights, so a bad stage cannot overwrite a better model.

Training images use only small geometric changes: resize to 248, rotation of ±8°, scale between 0.94 and 1.06, then a crop to 224. There is no color jitter and no other color change. Validation uses resize 248 and a center crop.

The classifier stage reached a best validation accuracy of **0.882** and stopped at epoch 33. Unfreezing `layer4` did not beat 0.882. The `layer3` stage dropped validation accuracy to about 0.72 and the run was interrupted. The file `models/resnet152_progressive.pth` is therefore the best classifier-stage checkpoint, at 0.882 validation accuracy on the original non-stratified split.

## What the errors look like

`code3.ipynb` evaluates that checkpoint on the same validation split.

Validation accuracy is **0.882** (1,303 images). Per-painter accuracy is high for several small classes: Sandro Botticelli is perfect on its validation images, Kazimir Malevich is 0.933, and Frida Kahlo is 0.842. Vincent van Gogh is 0.962 and Edgar Degas is 0.901.

The painters under 0.80 are mid-sized and are confused with stylistically close neighbors:

| painter | validation accuracy |
|---|---:|
| Leonardo da Vinci | 0.727 |
| Marc Chagall | 0.732 |
| Salvador Dali | 0.750 |
| Mikhail Vrubel | 0.767 |
| Titian | 0.778 |
| Henri Matisse | 0.784 |

Of 154 validation mistakes, **99** have predicted probability at least 0.80 (mean confidence 0.953) and **55** are below 0.80 (mean confidence 0.607). The confident mistakes put almost no probability on the true painter. Typical swaps are Picasso with van Gogh, Degas with van Gogh, Titian with Rubens, and Chagall with Picasso.

Because the weak classes are not the small classes, class weighting was not used for the submitted model. Inverse-frequency weights would down-weight van Gogh and Degas, who are already accurate, and would not separate paintings that the network is already sure about.

A later cell in `code3.ipynb` sets up a stratified retrain (100 epochs, learning rate `1e-3`, reduce-on-plateau patience 10, early stop patience 30). That retrain was not run on this machine.

`colab.ipynb` is the same idea prepared for a GPU runtime: stratified 80/20 split, the same geometric augmentation, and full fine-tunes of ResNet-18, ResNet-34, and ResNet-152. Each run allows 200 epochs, multiplies the learning rate by 0.5 after 10 epochs without a validation macro-F1 gain, and stops after 45 such epochs. Every parameter is trainable. After each model, the best validation-F1 weights and the best validation-accuracy weights are written to Google Drive (`MyDrive/famous_paintings_weights`), six files in total. Those Colab runs are separate from the checkpoint measured below.

## Test set

`artist.ipynb` loads `models/resnet152_progressive.pth` and predicts `test_data/`. The submission table has two columns: `file` (for example `test_0`, with no extension) and `artist` (for example `Leonardo_da_Vinci`). The class index is the numeric folder id, not the string-sorted order `'0', '1', '10', '11', ...` that `image_dataset_from_directory` would produce.

Compared with `true_labels.csv`, **274 of 300** predictions match. Accuracy is **0.913**. The 26 misses are saved in `wrong_test_predictions.png`, with the true painter and the predicted painter under each image. Several of those misses are plausible style collisions: Durer’s rhinoceros drawing predicted as Malevich, Bosch predicted as Bruegel, and van Gogh predicted as Vrubel.

`submission.csv` and `result.zip` (the csv plus `artist.ipynb`) are the files produced for the course submission.

## Inference service

`app/` serves this same checkpoint with FastAPI. `POST /predict` accepts an image and returns the painter id, the painter name, and the top class probabilities. Preprocessing matches validation: resize 248, center crop 224, ImageNet normalization, no color change. The network stays in evaluation mode.
