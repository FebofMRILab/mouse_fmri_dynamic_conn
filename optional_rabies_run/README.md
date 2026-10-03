### BASH code to run the RABIES workflow on BIDS data (mice)

Code for running rabies workflow on data in https://www.biorxiv.org/content/10.64898/2026.04.26.720907v4. 
BIDS converted data following the correct structure. The rabies directory is preferred in bids_converted/derivatives (check code for correct directory calls). Parameters should be checked and adjusted according to specific studies. 

### Atlas location
This depends on study. We use an adaptation of the CIVM atlas by:

Johnson GA, Badea A, Brandenburg J, Cofer G, Fubara B, Liu S, Nissanov J (2010) Waxholm space: an image-based reference for coordinating mouse brain research. Neuroimage 53:365-372.

### RABIES 
Used in a container environment using singularity calls in a network environment. 

Desrosiers-Gregoire G, Devenyi GA, Grandjean J, Chakravarty MM (2024) A standardized image processing and data quality platform for rodent fMRI. Nat Commun 15:6708.

