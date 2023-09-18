# DstGPT

## Introduction
Denmark has some of the worlds most comprehensive national statistics about all aspects of the danish society. All these statistics are neatly organised, well documented and freely available from a [public API](https://www.dst.dk/da/Statistik/brug-statistikken/muligheder-i-statistikbanken/api). This creates an opportunity to create an analyst that can quickly give comprehensive answers to most question about Denmark: Which municipalities are running the biggest deficits? How are the birth rates evolving? Or what are the fastest growing sports?

Whatever the question, if it can be measured then it likely is and the answer will be readily available from one of the 2304 tables available from the Denmark Statistiscs public API.

## TODO List

- [x] Create initial querying framework that gets the correct table with correct filters applied.
- [ ] Create inital visualisation framework
- [ ] Create inital analysis framework
- [ ] Create inital command framework. Once you have an initial table and graphs displayed make it easy to work with. It could commands like: Add this group to barchat. Or display timeseries plot of these variables instead. Or maybe add figure about this query to the dashboard etc.


## Initial querying framework - notes
There are 2304 different tables available from Denmark Statistics public API. All of these have short highlevel descriptions of the table content. The tables furthermore has descriptions of all variables and each variable that segments the data has a descriptive list of unique values it can take. 

This is used to create an initial querying framework in the following way: All table descriptions are embedded. When writing a question/query the query is then embedded and a the 5 tables with descriptions closest in embedding space to the query is selected. These 5 tables are then filtered down to one table using are rerank stage (in practice I just use coheres rerank api and likewise cohere is used for all embeddings).
When a table has been selected then a GPT message is created in order for GPT to select the right table specifications. For all variables with more than 20 unique values GPT guesses the correct values and these are then used in an embedding search on all unique values that variable can take.

A quick note on architecture is that since their are quite a few tables, 2304, and each table variable almost always has fewer than 300 unique values then I do not use a database. Instead I create a folder for each table and store the table descriptions and a dictionary with the variable embeddings. When creating the table specifications then I just read the pickle files. Since I never look at more than one table at a time and each dictionary with variable embeddings i fairly small then read time is fast and the memory footprint small. Also since the unique values a variable can take mostly stays constant for the foreseable future then their is no need for CRUD functionality. All files are created once and then read, when needed during program execution.

The files with table descriptions and embeddings can be downloaded from the following [dropbox link](https://www.dropbox.com/scl/fi/vsk3ykzzxofiyfuavqy8r/dstgpt_en_table_info_embs.zip?rlkey=ejkytokt98f2au34rg86reu7x&dl=0)