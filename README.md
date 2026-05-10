### Setup
In order to run, you must first ensure that the two zip files, containing the
two necessary datasets for the post nodes and comment nodes are unzipped into the "meirl_datasets" folder, so after unzipping that folder, within the 
CS4230_Reddit_meirl_dashboard folder containing the dashbaord file and the readme, ensure that the meirl_datasets folder contains:
* the-reddit-irl-dataset-posts.csv
* the-reddit-irl-dataset-comments.csv

After this, make sure that all dependencies, located inside the requirements.txt folder, are downloaded, however everything we used for this dashboard was used in class, so no insnae installations should be necessary.

### Running
Once in the terminal for VSCode (or preferred IDE) and in the directory containing dashboard_run.py, you should able to run teh porrgam via the command:
`python -m streamlit run dashboard_run.py`

This will then open up the dashbaord that you can use to navigate our research and findings.

#### NOTE
Because our dataset is so large, it may take quite a bit of time to load up for the first time (as it has to sort through 7 million comments to match it to our proportionally sampled posts,) so we appreciate patience when working with our dashboard.