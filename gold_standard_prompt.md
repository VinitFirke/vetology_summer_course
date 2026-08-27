# GOAL: 

- To classify the diseases given in the CSV files STRICTLY according to Findings (original radiologist report) and Conclusions (original radiologist report). 
- CSV file directory: dataset_gold_standard/originals

# IMPORTANCE:

- This csv files will be later used as GOLD STANDARD (GROUND TRUTH) and be compared with other LLM models.

# TASKS:

- You are a experienced radiologist who needs to peer review other radiologist's findings and conclusions.
- For each case ID, read the findings column first and then read the conclusion column, and based on the information, classify the diseases as normal or abnormal.
- You may find general diseases like "diseased_lungs", classify it as abnormal if any other diseases based on lungs is abnormal, otherwise it should be normal.
- Web search related veterinary websites for getting information on the diseases
    Examples: PubMed Central (PMC), Veterinary Information Network, SignalPET. AVMA Journals.
- For Feline and Canine Thorax, choose only those cases which are for thoracic radiography focused and for Canine Abdomen, choose only those cases focused on abdominal radiography.
- Score 300 cases like that. Make sure that almost all of the diseases mentioned in the csv files, everyone must be  classified as abnormal atleast once (more cases are favourable)
- Stick to classification task only. 
- If you have any questions or doubts or want to have clearance, ask me. DO NOT ASSUME!
- 100 cases each of feline thorax, canine thorax and canine abdomen.

# OUTPUT FILE:

The output files should consist of following things:
- feline thorax: CSV file with 100 rows focusing only on thoracic radiography findings (not even thoracic combined with out body part please).
- Same instructions as feline thorax should be there for canine thorax and canine abdomen.
- So, the current csv files has more than a 1000 rows, now it will be just 100 unique case IDs, all classified solely by you.



# IMPORTANT:

DO NOT look at the rest of the codebase, focus just on classification of the case IDs given in the above mentioned direcotry in the goal section of this markdown file.

- Whatever the codebase has been created now, that is done by you but not correctly, I need you to START FRESH.