DETAILED DESCRIPTIONS OF DATA FILES
====================================

Here are brief descriptions of the data and metadata.
For easy-loading, it is recommended to use pandas, e.g., 

import pandas as pd
df = pd.read_csv(filename, sep="\t").

===DATA DESCRIPTION===

The provided dataset splits (train and test) are in the form of TSV (tab-separated values). 
Altogether, they contain 14,000 records by 849 users on 518 items. 

The train/test are split 80%/20% based on the time of the interactions,
in particular, the test set has the latest 20% of the interactions.

The first row contains the header names (item_id, user_id, rating, and timestamp).
Each row after the first row represents a rating 'rating' to the item 'item_id' by the user 'user_id' at time 'timestamp'.

Note that these splits have been subject through a series of data filtering, including but not limited to: filtering based on item metadata and random sampling from the files available in https://github.com/RUCAIBox/RecSysDatasets/tree/master.

===METADATA DESCRIPTION===
The provided metadata file is in the form of TSV. 
It contains item id, title, and description for all the items in the dataset.
It is a filtered version of the metadata for the Musical_Instruments datasset available at: https://amazon-reviews-2023.github.io/index.html 

The first row contains the header names (item id, title, and description).