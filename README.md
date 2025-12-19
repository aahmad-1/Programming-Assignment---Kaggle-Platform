im a computaitonal enginneering student at LUT uni and im taking a calss called Computational Science and Artificial Intelligence working life. i attached the latex template and references.bib of the assignment in the github repo as well.

assignment premise:
Programming Assignment: Kaggle Platform
This year, as a practical programming assignment, we will explore the __Kaggle__ data science competition platform and online community. Kaggle focuses on machine learning, artificial intelligence, and related topics. It offers access to code, models, methods, and datasets. You can even participate in data science competitions.
Practical objectives 
* Familiarise yourself with the Kaggle platform.
* Explore datasets, models, code, and other resources.
* Choose one specific example from Kaggle to study in detail, such as:
* 
* A pretrained model of interest (e.g., LLMs, text segmentation, image-based models, or any other topic you prefer).
* Participation in a competition or modelling task.
* Get familiar with different frameworks, such as Keras, PyTorch, and TensorFlow.
* Study various datasets, such as greenhouse datasets, car price prediction, MedMNIST models, etc.
After selecting your topic, implement your solutions, simulations, or visualisations. These may take different forms depending on your chosen topic. Ask the lecturers for guidance, and discuss your choices with your fellow students.
Use the provided programming task template to report your findings. Follow the suggested structure unless your specific topic requires adaptation. A certain degree of improvisation and innovation is welcome; however, maintain an academic tone throughout your report.

------------------------------
About Dataset
Context:
The data were obtained in a survey of students math and portuguese language courses in secondary school. It contains a lot of interesting social, gender and study information about students. You can use it for some EDA or try to predict students final grade.
Content:
Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:
1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2. sex - student's sex (binary: 'F' - female or 'M' - male)
3. age - student's age (numeric: from 15 to 22)
4. address - student's home address type (binary: 'U' - urban or 'R' - rural)
5. famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
6. Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
7. Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
8. Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
9. Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10. Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
11. reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12. guardian - student's guardian (nominal: 'mother', 'father' or 'other')
13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15. failures - number of past class failures (numeric: n if 1<=n<3, else 4)
16. schoolsup - extra educational support (binary: yes or no)
17. famsup - family educational support (binary: yes or no)
18. paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19. activities - extra-curricular activities (binary: yes or no)
20. nursery - attended nursery school (binary: yes or no)
21. higher - wants to take higher education (binary: yes or no)
22. internet - Internet access at home (binary: yes or no)
23. romantic - with a romantic relationship (binary: yes or no)
24. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25. freetime - free time after school (numeric: from 1 - very low to 5 - very high)
26. goout - going out with friends (numeric: from 1 - very low to 5 - very high)
27. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29. health - current health status (numeric: from 1 - very bad to 5 - very good)
30. absences - number of school absences (numeric: from 0 to 93)
These grades are related with the course subject, Math or Portuguese:
1. G1 - first period grade (numeric: from 0 to 20)
2. G2 - second period grade (numeric: from 0 to 20)
3. G3 - final grade (numeric: from 0 to 20, output target)
Additional note: there are several (382) students that belong to both datasets . These students can be identified by searching for identical attributes that characterize each student, as shown in the annexed R file.
Source Information
P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
Fabio Pagnotta, Hossain Mohammad Amran. Email:fabio.pagnotta@studenti.unicam.it, mohammadamra.hossain '@' studenti.unicam.it University Of Camerino
https://archive.ics.uci.edu/ml/datasets/STUDENT+ALCOHOL+CONSUMPTION

---------------------------------
folder structure:

```markdown
Programming Assignment - Kaggle Platform
├─ notebooks
│  ├─ linear_regression_test1.ipynb
│  ├─ linear_regression_test2.ipynb
│  ├─ random_forest_test1.ipynb
│  ├─ random_forest_test2.ipynb
│  └─ student_data_analysis.ipynb
├─ outputs
│  ├─ factor_importance.npy
│  ├─ metrics
│  │  └─ results.txt
│  ├─ plots
│  │  ├─ lr_without_g1g2.png
│  │  ├─ lr_with_g1g2.png
│  │  ├─ rf_without_g1g2.png
│  │  ├─ rf_without_g1g2_importance.png
│  │  ├─ rf_with_g1g2.png
│  │  └─ rf_with_g1g2_importance.png
│  └─ predictions.npy
├─ README.md
├─ references.bib
├─ src
│  ├─ data_prep.py
│  ├─ train.py
│  ├─ visualizations.py
│  └─ __pycache__
│     └─ data_prep.cpython-313.pyc
├─ student_data
│  └─ student-mat.csv
├─ Surname_Firstname_CompSci_Special_Assignment_2025.pdf
└─ Surname_Firstname_CompSci_Special_Assignment_2025.tex
```
```
Programming Assignment - Kaggle Platform
├─ notebooks
│  ├─ linear_regression_test1.ipynb
│  ├─ linear_regression_test2.ipynb
│  ├─ random_forest_test1.ipynb
│  ├─ random_forest_test2.ipynb
│  └─ student_data_analysis.ipynb
├─ outputs
│  ├─ factor_importance.npy
│  ├─ metrics
│  │  └─ results.txt
│  ├─ plots
│  │  ├─ lr_without_g1g2.png
│  │  ├─ lr_with_g1g2.png
│  │  ├─ rf_without_g1g2.png
│  │  ├─ rf_without_g1g2_importance.png
│  │  ├─ rf_with_g1g2.png
│  │  └─ rf_with_g1g2_importance.png
│  └─ predictions.npy
├─ README.md
├─ references.bib
├─ src
│  ├─ data_prep.py
│  ├─ train.py
│  ├─ visualizations.py
│  └─ __pycache__
│     └─ data_prep.cpython-313.pyc
├─ student_data
│  └─ student-mat.csv
├─ Surname_Firstname_CompSci_Special_Assignment_2025.pdf
└─ Surname_Firstname_CompSci_Special_Assignment_2025.tex

```