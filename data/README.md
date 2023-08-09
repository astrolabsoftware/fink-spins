## Data

### ZTF

In order to generate the dataset, you need to run:

```python
python fink_spins/generate_data.py --verbose -version 2023.08
```

This will download the SSOFT for all flavors (SHG1G2, HG1G2, and HG), and build the master table.

### ZTF x ATLAS

(to be done)

### BFT from SsODNet

get_bft.sh download the SsODNet.bft parquet file


## CHANGELOG

|When| What|
|----|-----|
|2023/07/04 | Initial dataset from fink-tutorials@spins |
|2023/08/09 | Dataset from the SSOFT |
