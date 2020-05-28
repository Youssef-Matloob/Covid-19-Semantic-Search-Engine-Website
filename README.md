# Covid-19 Semantic Search Engine Website

This Project is part of Society Impact Project - Electronics and Electrical Communication Department - Faculty of Engineering - Cairo University - Dr. Hassan Mostafa

## Developed a website like a search engine for the medical community to help them in their fight against COVID-19 Under Supervision of Dr. Hassan Mostafa

This Repo contains a website for a COVID-19 Search Engine that can be used by Medical community to search for topics in covid-19 published papers, it's based on LDA and trained on +40K papers

![CORD19](https://wlni.com/wp-content/uploads/2020/03/CORONAVIRUS-2.jpg)

### Dataset - COVID-19 Open Research Dataset (CORD-19):

COVID-19 Open Research Dataset (CORD-19) is a free resource of scholarly articles, aggregated by a coalition of leading research groups, about COVID-19 and the coronavirus family of viruses. The dataset can be found on [Semantic Scholar](https://pages.semanticscholar.org/coronavirus-research) and there is a research challenge on [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).

This project builds an index over the CORD-19 dataset to assist with analysis and data discovery. A series of tasks were explored to identify relevant articles and help find answers to key scientific questions on a number of COVID-19 research topics.

### Tasks
The following files show the top query results for each task provided in the CORD-19 Research Challenge using this model. A highlights section is also shown for each task, which highlights the most relevant sentences from the query results.

- [What is known about transmission, incubation, and environmental stability?](https://www.kaggle.com/davidmezzetti/cord-19-transmission-incubation-environment)
- [What do we know about COVID-19 risk factors?](https://www.kaggle.com/davidmezzetti/cord-19-risk-factors)
- [What do we know about virus genetics, origin, and evolution?](https://www.kaggle.com/davidmezzetti/cord-19-virus-genetics-origin-and-evolution)
- [What do we know about vaccines and therapeutics?](https://www.kaggle.com/davidmezzetti/cord-19-vaccines-and-therapeutics)
- [What do we know about non-pharmaceutical interventions?](https://www.kaggle.com/davidmezzetti/cord-19-non-pharmaceutical-interventions)
- [What has been published about medical care?](https://www.kaggle.com/davidmezzetti/cord-19-medical-care)
- [What do we know about diagnostics and surveillance?](https://www.kaggle.com/davidmezzetti/cord-19-diagnostics-and-surveillance)
- [What has been published about information sharing and inter-sectoral collaboration?](https://www.kaggle.com/davidmezzetti/cord-19-sharing-and-collaboration)
- [What has been published about ethical and social science considerations?](https://www.kaggle.com/davidmezzetti/cord-19-ethical-and-social-science-considerations)


### Installation
You can use Git to clone the repository from GitHub and install it.

Python 3.5+ is supported

### Building a model
Download all the files in the Download CORD-19 section on [Semantic Scholar](https://pages.semanticscholar.org/coronavirus-research). Go the directory with the files
and run the following commands.

    cd <download_path>

For each tar.gz file run the following
    mkdir <file> && tar -C <file> -xvzf <file.tar.gz>

Once completed, there should be a file name metadata.csv and subdirectories for each data subset with all json articles.

To build the model locally:

    # run loader.py to prepare the dataset
    python -m loader.py 

    # Build model files 
    python -m model.py

The model will be stored in the same directory

### Tech Overview

The model is a built on LDA and using CountVectorizer


### Please find attached the video with a demo for the website working at this [link](https://www.youtube.com/watch?v=kx_NjmCFVeo)

#### Under Supervision of:
* Dr Hassan Mostafa

#### Submitted by:

* Abdallah Ahmed 
* Abdelrahman Ahmed
* Mohamed Sabry
* Mohamed Abd Elhalim
* Youssef Mostafa
