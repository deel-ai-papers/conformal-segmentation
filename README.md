# COSE: Conformal Segmentation


Repository with the code of our [paper](https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/html/Mossina_Conformal_Semantic_Image_Segmentation_Post-hoc_Quantification_of_Predictive_Uncertainty_CVPRW_2024_paper.html):
> L. Mossina, J. Dalmau and L. Andéol (2024). _Conformal Semantic Image Segmentation: Post-hoc Quantification of Predictive Uncertainty_. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2024, pp. 3574-3584 


## Idea
We apply Conformal Prediction to semantic image segmentation with multiple classes. Our contribution includes:
- Novel application of [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg) ([arXiv](https://arxiv.org/abs/2208.02814)), by Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2022);
- Novel visualization of conformal sets via heatmaps;
- Tests on multiple datasets: Cityscapes (automotive), ADE20K (daily scenes), LoveDA (aerial imaging).

An example of conformalized segmentation on the Cityscapes dataset:

<img src="notebooks/paper/figures/city_seg_varisco.png" style="max-width:500px;width:100%">


### How it works
The user must provide a calibration dataset of $n$ *labeled images* not used during training, from the same distribution as the test set and representative of the inputs given to the model when deployed.

1. Choose a *conformal loss* $L(\lambda) = \ell(\cdot, \cdot)$ that corresponds to your *notion of error* (see examples in [Section 4.3](https://arxiv.org/html/2405.05145v1#S4.SS3))
2. Select a risk level $\alpha \in (0,1)$: smaller values lead to more conservative prediction sets.
3. Compute the optimal parameter $\hat{\lambda} := \inf \\{ \lambda \in [0,1] : \frac{n}{n+1} \hat{R}_{n}(\lambda) + \frac{1}{n+1} \leq \alpha \\}$.
    - This is a simple optimization problem because the empirical risk $\hat{R}\_{n}(\lambda) = \frac{1}{n}\sum_{i=1}^{n}L_i(\lambda)$ is monotonic in $\lambda$.
    - $\hat{\lambda}$ is the smallest threshold such that the risk is controlled at the level $\alpha$ specified by the user (see [Theorem 4.1](https://arxiv.org/html/2405.05145v1#S4)).
    

The parameter $\lambda$ acts as a **threshold** on the underlying softmax (see [Eq. 1](https://arxiv.org/html/2405.05145v1#S1.E1) and [Sec. 5](https://arxiv.org/html/2405.05145v1#S5)): for each pixel, we include all classes with softmax score is above $1 - \lambda$. This results in a multilabeled segmentation mask.
In the heatmap, we count the number of classes included after thresholding. The color skews toward red as more classes are included.


Visually, the value of the threshold ${\lambda}$ and the resulting heatmaps are connected like this:


<img src="doc/figures/thresholding_cityscapes.gif" style="max-width:500px;width:100%">




## Get started
This repository relies on the libraries of the [OpenMMLab codebase](https://platform.openmmlab.com/modelzoo/) (via [`mmseg`](https://mmsegmentation.readthedocs.io/en/latest/) & `mmengine`) to handle the pretrained ML models and the datasets, and `pytorch` for all other things ML.

For the moment, you must either choose some existing models and datasets from `mmsegmentation` or adapt your code to this library.
We plan on releasing a more general version that works with basic pytorch and dataloaders, with minimal requirements (softmax output).

### 1. Make a virtual environment and install our repo
The following steps should ensure that the library and experiments run correctly:

1. Make a virtual environment named `.venv`, as specified in the [`Makefile`](Makefile)
    ```
    $ make venv
    $ make cose_path
    ```

2. Write the project's environmental variables to a file named `.env`, which should not be commited. For example:
    ```
    $ DATASET_NAME='Cityscapes'
    $ cd path/to/DATASET_NAME
    $ echo COSE_DATA_DATASET_NAME=$PWD >> ~/projects/vision/cose/.env
    ```
Repeat these steps for every dataset: "Cityscapes", "ADE20K", "LoveDA". 


### 2. Alternative installation
If the `make` commands above do not work, try to reproduce the following steps:

1. Create a virtual environment with the [`venv` package](https://docs.python.org/3/library/venv.html) from the Python Standard Library:
   ```
   $ python3.9 -m venv .venv
   ```
2. You must ensure that your GPU/CUDA, `pytorch` and `mmsegmentation` (and their dependencies) libraries are compatible. It can require a process of trial and error, uninstalling and reinstalling different version of the same package (e.g. `mmcv` below). For our our machines, this worked:
   ```
   $ .venv/bin/python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   $ .venv/bin/python -m pip install -r requirements.txt
   $ .venv/bin/python -m pip uninstall mmcv
   $ .venv/bin/mim install mmcv
   ```
3. To install locally the `cose` packages (it allows to execute `cose` after modifying it, with a reload of the notebook):
   ```
   $ .venv/bin/python -m pip install --editable 
   ```


### 3. Software architecture: pytorch, mmsegmentation, etc.
For Conformal segmentation, we just assume that the **logits of an inference** are available from the prediction.
This could require you to modify the NN code to explicitly return them instead of the simple segmentation mask obtained via an argmax.

In this repo, we use [`mmsegmentation`](https://github.com/open-mmlab/mmsegmentation) and other packages of the [**OpenMMLab**](https://openmmlab.com) projects.
We use their models' specifications (pytorch), their pre-trained weights and their dataset wrappers.
Their repos are being actively developed and are vast: we only use a small set of tools, 
ignoring most of the pre-baked ones for training or running inferences.

If you use other models/dataset (e.g. via torch-hub), you will need to adapt your code to their idiosynchrasies (should be straightforward).

In the future, we would like to make a a version that does not depend on `mmseg`.
In the meantime, write an **issue** if you have problems.

### 4. Demo notebooks

**TODO**: Write some clean and simple [`notebooks`](notebooks/) to demo the approach.

In the meantime, have a look at the [`experiments`](experiments) directory:
- [`crc_calibration.py`](https://github.com/deel-ai-papers/conformal-segmentation/blob/main/experiments/crc_calibration.py) does the conformalization
- [`crc_test_metrics.py`](https://github.com/deel-ai-papers/conformal-segmentation/blob/main/experiments/crc_test_metrics.py) evaluates the conformalized model (i.e. using the estimated $\hat{\lambda}$) on test data


### 5. Interactive web applications
We wrote two simple applications (see [`src/app`](src/app)) using the [Gradio](https://www.gradio.app/guides/quickstart) library by HuggingFace.
To run it, you must download the datasets and models we used in our experiments: [scripts/downloaders/download_mods_weights.ipynb](scripts/downloaders/download_mods_weights.ipynb).

1. [Thresholding app](src/app/app_threshold.py): observe how the value of the parameter $\lambda \in [0,1]$ influences the heatmap. This is the value we estimate with the CRC conformal algorithm.
2. [Conformal heatmap](src/app/app_vision.py): run inferences with pre-conformalized models



## Run the experiments
See the [README.md](experiments/README.md) in the [`experiments`](experiments/) directory.


## Citation
```
@InProceedings{Mossina_2024_conformal_segmentation,
    author    = {Mossina, Luca and Dalmau, Joseba and And\'eol, L\'eo},
    title     = {Conformal Semantic Image Segmentation: Post-hoc Quantification of Predictive Uncertainty},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3574-3584}
}

```

## Acknowledgments
We would like to thank our colleagues of the [DEEL](https://www.deel.ai/about-us)  Project (DEpendable Explainable Learning), for their invaluable feedback.

We work on uncertainty, explainability, OOD detection and other topics in trustworthy and certifiable AI.

Have a look at our open-source projects and publications:
- repos: https://github.com/deel-ai
- publications: https://www.deel.ai/publications

