# Example Code for Bimanual Manipulation

This example demonstrates the work presented in our paper _A System for Imitation Learning of Contact-Rich Bimanual Manipulation Policies_

In order to run the code, please create a new conda environment from the provided _environment.yaml_ file
```
conda env create -f environment.yml
```
and install the _irl\_control_ package as described in the main README.md of the main package.

After creating the new environment, please activate it with 
```
conda activate bimanual
```

Collecting data can be accomplished with the following command, collecting a single demonstration in the _data/BIP_ directory:
```
python collect_data.py
```

Training and subsequently evaluating the BIP, run the following command:
```
python run_bip.py
```

## Citation

```
@article{stepputtis2022bimanual,
  author    = {Stepputtis, Simon and Bandari, Maryam and Schaal, Stefan and Ben Amor, Heni},
  title     = {A System for Imitation Learning of Contact-Rich Bimanual Manipulation Policies},
  journal   = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2022},
}
```
