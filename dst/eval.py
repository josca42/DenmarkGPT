### Create framework for evaluating llm setup

# get GPT to generate questions
import pickle
import pandas as pd
from dst.data import config, DST_SUBJECTS_INDEX_0_1, EVAL_DIR
from dst import llm
from datetime import datetime
import json
from jinja2 import Template
from dst.data import TABLE_INFO_DA_DIR, TABLE_INFO_EN_DIR, EVAL_DIR
from dst.query import (
    find_table_candidates,
    load_and_process_table_info,
    determine_query_type,
)
from dst.db import crud, models
import ast
import time


def test_find_table(test_name):
    eval_run_dir = EVAL_DIR / "find_table" / test_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    fp_df_test_run = eval_run_dir / "df_test_run.csv"

    if fp_df_test_run.exists():
        df_test_run = pd.read_csv(fp_df_test_run, sep=";")
    else:
        df_test_run = test_run_find_table(TEST_QA_QUESTIONS, lang="en", verbose=True)
        df_test_run.to_csv(fp_df_test_run, index=False, sep=";")

    eval_results = eval_find_table_test_run(df_test_run)

    with open(eval_run_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)


def test_run_find_table(questions, lang, verbose=False):
    results = []
    prev_table_descr, prev_api_request = "", ""
    for i, question in enumerate(questions):
        if verbose:
            print(f"Question {i}: {question}")

        start_time = time.time()
        setting_info = {
            "prompt": question,
            "query_type": -1,
            "prev_table_descr": prev_table_descr,
            "prev_api_request": prev_api_request,
            "lang": lang,
        }
        action_type, table_descr = determine_query_type(question, setting_info)
        match_action_time = time.time() - start_time
        if verbose:
            print(
                f"Action type: {action_type}, Table description: {table_descr}, Time: {match_action_time}"
            )

        start_time = time.time()
        if action_type == 1:
            table, table_candidates = find_table_candidates(
                table_descr=table_descr, lang=lang, k=10, query=question, rerank=True
            )
        else:
            continue

        find_table_time = time.time() - start_time
        if verbose:
            print(
                f"Table id: {table['id']}, Table descr: {table['description']}, Time: {find_table_time}"
            )

        results.append(
            dict(
                question=question,
                action_type=action_type,
                action_table_descr=table_descr,
                table_id=table["id"],
                table_descr=table["description"],
                table_ids_options=table_candidates,
                lang=lang,
                match_action_time=match_action_time,
                find_table_time=find_table_time,
            )
        )
    return pd.DataFrame(results)


def eval_find_table_test_run(df_test_run):
    def agg_list_of_results(results):
        combined_results = dict()
        for result in results:
            for key, value in result.items():
                if key not in combined_results:
                    combined_results[key] = value
                else:
                    combined_results[key] += value
        return combined_results

    results = []
    for i in range(0, len(df_test_run), 30):
        chunk = df_test_run.iloc[i : i + 30]
        sys_msg = dict(role="system", content=EVAL_FIND_TABLE_QA_SYS_MSG)
        user_msg = dict(
            role="user",
            content=EVAL_FIND_TABLE_QA_USER_MSG.render(
                qa_run=chunk[["question", "table_descr"]].to_dict(orient="records")
            ),
        )
        response_txt = llm.gpt(
            messages=[sys_msg, user_msg], model="gpt-4", temperature=0
        )
        result = response_txt.split("Result: ")[1]
        result = ast.literal_eval(result)
        results.append(result)

    results = agg_list_of_results(results)
    return results


def create_test_questions():
    msgs = [
        dict(
            role="user",
            content=CREATE_EVAL_QUESTIONS.render(
                subjects=json.dumps(DST_SUBJECTS_INDEX_0_1, ensure_ascii=False)
            ),
        )
    ]
    response_txt = llm.gpt(messages=msgs, model="gpt-4", temperature=1)
    questions = [
        question.split(". ", 1)[-1].strip()
        for question in response_txt.split("\n")
        if question
    ]
    return questions


###   Prompts   ###
CREATE_EVAL_QUESTIONS = Template(
    """Your are Test GPT. Your task is to test a system that answers questions using data about the following highlevel areas:
{{ subjects }}

Create 100 test questions that could be asked about the above subjects"""
)

EVAL_FIND_TABLE_QA_SYS_MSG = """You are quality assurance GPT. Your task is to do quality assurance on question answer framework. More specifically you evaluate a proposed table is likely to contain data that can be used to answer a question.

You get data on questions and the proposed tables on the following form:
QA run: [{"question": The question asked, "table_descr": description of the proposed table}, ...]

Before answering go through each question-table pair and write your conclusion of wether the table could been used to answer the question.

Finally write up a summary of the results as a formatted array of strings that can be used in JSON.parse(). Use the structure:
Result: {"success": Number of questions where the table could be used, "failure soft": Number of questions where the table is partially suited or lack comprehensiveness", "failure soft questions": list of each question that is failure soft, "failure strict": Number of questions where the table is not suited to provide the required information, "failure strict questions": list of each question that is failure strict
}

Do not write anything after the Result json."""

EVAL_FIND_TABLE_QA_USER_MSG = Template("""QA run: {{ qa_run }}""")


###   Test questions   ###
TEST_QA_QUESTIONS = [
    "What has the inflation been over time",
    "How has the public expenses to health care evolved over time",
    "How has unemployment evolved over time",
    "How has unemployment evolved",
    "Which municipalities have the highest expenses",
    "Which cities have the biggest cost",
    "Which municipalities have the biggest deficits",
    "WHich municipalities have the highest number of traffic accidents",
    "Which regions have the highest number of traffic accidents",
    "How has the number of traffic accidents evolved over time",
    "what was the most popular name in 2022?",
    "which city has the most accidents?",
    "What names are most popular",
    "What has the inflation been recently",
    "which city has the most accidents",
    "number of unemployed in 2021?",
    "Middle age in denmark?",
    "How is the quality of life in denmark?",
    "How was life quality changed over time?",
    "How has employment rate changed over time?",
    "How is the age distribution",
    "How many people live in the city of Odense",
    "The average salary",
    "Citizens of copenhagen",
    "Danish population in total",
    "how many have internet access in denmark?",
    "how many are studying in university?",
    "What is the total number of households?",
    "How many children per family are on average?",
    "What is the migration rate in Denmark?",
    "What are the housing conditions?",
    "What is the health status of Danish citizens?",
    "What is the state of democracy in Denmark?",
    "What are the most common names?",
    "What is the employment status of the Danish population?",
    "How many people are employed?",
    "What is the unemployment rate?",
    "What is the rate of absenteeism and labor disputes?",
    "What is the average income and salary?",
    "What is the average wealth and how has it changed over time?",
    "What is the state of public finance in Denmark?",
    "What is the balance of payments and foreign trade in Denmark?",
    "What is the inflation rate?",
    "What is the average consumption?",
    "What is the state of real estate?",
    "What are the common digital payments in Denmark?",
    "How many people receive public support?",
    "Which cities have the highest crime rates?",
    "What is the status of living conditions in different municipalities?",
    "What is the general education level and how has it changed over time?",
    "What cities have the highest education level",
    "What is the state of full-time education?",
    "What is the state of research, development and innovation in Denmark?",
    "How many international companies are there in Denmark?",
    "What is the average life expectancy?",
    "What is the fertility rate. Which regions have the highest fertility rate?",
    "What is the average income of a working woman in Denmark?",
    "What is the average household electricity consumption",
    "What is the ratio of doctors to patients in Denmark",
    "How many peoples receive social support",
    "What is the main type of transportation used",
]
