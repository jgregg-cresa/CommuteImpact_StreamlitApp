About The Project
This repository allows the user to calculate an Origin-Destination Matrix using the Google API

The relevant files which you will need to use are the following

ZIP_Code_Population_Weighted_Centroids.csv
Contains the US HUD created opoulation weighted zipcodes. These are to be used for cases where the employees only have zipcodes and no lat, long, or other address fields.

PublicTransit_GoogleScript_V5.ipynb
Contains the functions and necessary code blocks to calculate transit, driving, and mixed method commutes

Dashboard_Transformation_Arc.ipynb
Contains the functions and necessary code blocks to transform the results of the PublicTransit script to a dashboard for Arc Online.

alt text

Dashboard_Transformation_Tableau.ipynb
Contains the functions and necessary code blocks to transform the results of the PublicTransit script to the indesign standard commute impact chart.

alt text

Getting Started
In order to access and run the OD Matrix Code, you first must create a codespace for yourself. If you already have a codespace, you can skip to Step 3.

alt text

Click on the Green Code Button next to the "Add File" Button.

Click on Create Codespace on main aka the main branch where the most up-to-date code resides.

Once opened, go to the Scripts folder in the left panel and select PublicTransit_GoogleScript_V5.ipynb


alt text

Prerequisites
In order to run the code you will need to do the following:

Copy over your Destinations and Origins files to the Inputs folder. You can drag and drop the files into it if you like. Make sure they are labeled appropriately as "Origins" and "Destinations".

Install the following packages (Uncomment the first cell of code in the script). You will be prompted to install the recommended Python extension and Jupyter Extensions. Click "Yes" and then select the Python 3.10 environment.

  !pip install googlemaps
  !sudo apt-get install libkrb5-dev
  !pip install folium
Running the Analysis
After getting the Python and Jupyter Notebook extensions installed, proceed with examining the cell labeled "Read in the Inputs and choose your transit method."

Choose your transit method and the name of your output file
Proceed with running the cells one by one using Shift + Enter
After Running
Proceed to choosing between the other two scripts in the folder and opening said files in the same codespace. Make sure that the outputs folder only contains the results of the commute impact analysis.