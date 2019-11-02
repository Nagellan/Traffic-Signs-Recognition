# Traffic-Signs-Recognition
Assignment for Introduction to Machine Learning subject at Innopolis University, 5th semester

## Task description
Task description is available [here](Task-Description.md).

## Report
Work report is available [here](Report.pdf).

## How to run
* Download all necessary files and folders described in Task-Description.md
* Reorganize downloaded folders and files in a way described below, in project folder tree section
* Go to the project root folder
* Run ``main.py`` file located in root directory of the project (considered that you have python3 installed)

**Linux:**
```
cd Traffic-Signs-Recognition
python3 main.py
```

**Windows:**
```
cd Traffic-Signs-Recognition
python main.py
```

## Project folder tree
```

root
├───GTSRB
│   ├───Final_Test
│   │   └───Images
│   |       ├───00000.ppm 
|   |       |   ...
│   |       ├───12628.ppm
│   |       └───GT-final_test.csv
|   |
│   └───Final_Training
│       └───Images    
│           ├───00000
|           |   ├───00000_00000.ppm 
|           |   |   ...
|           |   ├───00006_00029.ppm
|           |   └───GT-00000.csv
|           |   ...
│           └───00042
|               ├───00000_00000.ppm 
|               |   ...
|               ├───00006_00029.ppm
|               └───GT-00042.csv
|
├───Report.pdf
├───Task-Description.md
├───README.md
└───main.py
```

**Attention** to the file `GTSRB/Final_Test/Images/GT-final_test.csv`! This file is taken from `GTSRB_Final_Test_GT.zip` which you have to download by task description. For program to work, it must be moved to `GTSRB/Final_Test/Images/` replacing the old `.csv` file located there.
