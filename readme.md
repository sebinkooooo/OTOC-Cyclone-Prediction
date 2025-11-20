read the paper/paper.tex publication to understand the mathematical and engineering framework

create a virtual environment and install depdencies from requirements.txt
setup by downloading ERA5 paper in nc format from: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download 


use era5_preprocessor.py which reads the downlaoded nc format and preforms necessary calculations to obtian top-8 PCA dimensions.

This informaiton is saved to era5_processed.json

To visualize input data of cyclone, run plot_inputdata.py or access plot/ERA5. 

add quantinuum qnexus username password in env and set QNEXUS_USE_PASSWORD=True. Also specify your qnexus project name (needed to run quantum hrdware and emulators on their SDK)

QNEXUS_USERNAME=
QNEXUS_PASSWORD=
QNEXUS_USE_PASSWORD=True
QNEXUS_PROJECT=

run quantum_pipeline.py, currently this is run on a H1 emulator (H1-Emulator) and takes approx 15minutes on a m4 pro CPU. You can specify how many timestpe snapshots.

This saves results of OTOC into otoc_results.json

run plot functions to plot all data

run comparison.py to compare this method against traditional emthods and access reuslts in comaprison.json

run nextstep_comapriosn and access json to see that we only are able to predict chaos and not actual cyclone movements (implied - not a redflag)

run plot functions to plot these comapriosns and access images in /plot folder
