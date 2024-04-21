# Experiments: crc on multiple datasets

As an example, here is how to run one of our experiments:
1. Conformalize the predictor: compute optimal $\hat{\lambda}$ on calibration data $D_{\text{calib}}$.
    - Multiple runs: for better empirical evaluation, you can repeat with multiple random seeds (see config files)
2. Evaluate conformalization on $D_{\text{test}}$: we expect empirical risk to be close to nominal risk $\alpha$
    - For each random split in conformalization (point (1)), we test on the remaining data.

For details, see directly the python scripts and the configuration files called within the `.sh` scripts below.

### (1) Conformalize the predictor
Example: Cityscapes, conformalize miscoverage loss with CRC on the second gpu (`cuda:1`)

You will need to specify the gpu to run the experiment on.
- If only 1 gpu: `cuda:0`
- if 2 or more gpus: `cuda:0` or `cuda:1` etc.
- if NO gpus: `cpu`

```
$ cd /path/to/cose
$ source .venv/bin/activate
(.venv) $ sh experiments/run_crc_miscov_loss_cityscapes.sh cuda:1
```

Post-process the output of conformalization:
```
(.venv) $ python ./cose/experiments/postprocess_expes.py --input-dir experiments/outputs/Cityscapes/miscoverage_loss/
```

Or directly post-processing all datasets and losses in `experiments/outputs`:
```
sh experiments/process_all_opt_lambdas.sh
```

### (2) Evaluation on test split
Example: Cityscapes.

```
$ cd /path/to/output
$ source .venv/bin/activate
(.venv) $ sh experiments/[...].sh