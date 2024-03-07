## End To End ML project practice 1

## Create Virtual Environment
- conda create -p venv
## Activate Virtual Environment
- conda activate venv/

## create a git repository and add files
- 1. Initialise git:   git init
- 2. add file :      git add README.md
- 3. check git status:    git status
- 4. create main branch : git branch -M main
- 5. push file to repository:   git push -u origin  main
    
- 6. create .gitignore file in repository and pull that file in our project:   git pull origin main
## Create requirements.txt file to store required libraries and packages install requirements.txt
- pip install -r requirements.txt

## Create a template.py file that contains python code which creates all directories and files we need for project.

## create a src folder in which two other directories are present as follows
   - src
      - components:
         - data_ingestion.py : to fetch data and store in 'train.csv' and 'test.csv' file.
         - data_transformation.py : data cleaning and feature engineering process
         - model_trainer.py 
         - model_monitoring.py 
      - pipelines:
         - trainig_pipeline.py : to train machine learning model
         - prediction_pipeline.py : to prediction by using trained machine learning model.
      - exception.py : Custome Exception for exception handling
      - logger.py : to store execution information in log files.
      - utils.py : code for read data from databases
         
## 