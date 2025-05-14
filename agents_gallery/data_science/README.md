# Data Science with Multiple Agents

## Overview

This project demonstrates a multi-agent system designed for sophisticated data analysis. It integrates several specialized agents to handle different aspects of the data pipeline, from data retrieval to advanced analytics and machine learning. The system is built to interact with BigQuery, perform complex data manipulations, generate data visualizations and execute machine learning tasks using BigQuery ML (BQML). The agent can generate text response as well as visuals, including plots and graphs for data analysis and exploration.


## Agent Details
The key features of the Data Science Multi-Agent include:

| Feature | Description |
| --- | --- |
| **Interaction Type:** | Conversational |
| **Complexity:**  | Advanced |
| **Agent Type:**  | Multi Agent |
| **Components:**  | Tools, AgentTools, Session Memory, RAG |
| **Vertical:**  | All (Applicable across industries needing advanced data analysis) |


### Architecture
![Data Science Architecture](data-science-architecture.png)


### Key Features

*   **Multi-Agent Architecture:** Utilizes a top-level agent that orchestrates sub-agents, each specialized in a specific task.
*   **Database Interaction (NL2SQL):** Employs a Database Agent to interact with BigQuery using natural language queries, translating them into SQL.
*   **Data Science Analysis (NL2Py):** Includes a Data Science Agent that performs data analysis and visualization using Python, based on natural language instructions.
*   **Machine Learning (BQML):** Features a BQML Agent that leverages BigQuery ML for training and evaluating machine learning models.
*   **Code Interpreter Integration:** Supports the use of a Code Interpreter extension in Vertex AI for executing Python code, enabling complex data analysis and manipulation.
*   **ADK Web GUI:** Offers a user-friendly GUI interface for interacting with the agents.
*   **Testability:** Includes a comprehensive test suite for ensuring the reliability of the agents.



## Setup and Installation

### Prerequisites

*   **Google Cloud Account:** You need a Google Cloud account with BigQuery enabled.
*   **Python 3.9+:** Ensure you have Python 3.9 or a later version installed.


### Project Setup

1.  **Set up Environment Variables:**
    - Create a .env that contains the required variables. Included is a file named .env.copy that you can use as a template. 
  
        ```bash
        cp .env.copy .env
        ```

2.  **BigQuery Setup:**
    These steps will load the sample data provided in this repository to BigQuery.
    For our sample use case, we are working on the Forecasting Sticker Sales data from Kaggle:

    _Walter Reade and Elizabeth Park. Forecasting Sticker Sales. https://kaggle.com/competitions/playground-series-s5e1, 2025. Kaggle._

    *   First, set the BigQuery project ID in the `.env` file. This can be the same GCP Project you use for `GOOGLE_CLOUD_PROJECT`,
        but you can use other BigQuery projects as well, as long as you have access permissions to that project.
        If you have an existing BigQuery table you wish to connect, specify the `BQ_DATASET_ID` in the `.env` file as well.
        Make sure you leave `BQ_DATASET_ID='forecasting_sticker_sales'` if you wish to use the sample data.

        Alternatively, you can set the variables from your terminal:

        ```bash
        export BQ_PROJECT_ID='YOUR-BQ-PROJECT-ID'
        export BQ_DATASET_ID='YOUR-DATASET-ID' # leave as 'forecasting_sticker_sales' if using sample data
        ```

        You can skip the upload steps if you are using your own data. We recommend not adding any production critical datasets to this sample agent.
        If you wish to use the sample data, continue with the next step.

    *   You will find the datasets inside 'data_science/utils/data/'.
        Make sure you are still in the working directory (`agents_gallery/data-science`). To load the test and train tables into BigQuery, run the following commands:
        ```bash
        uv run utils/create_bq_table.py
        ```


3.  **BQML Setup:**
    The BQML Agent uses the Vertex AI RAG Engine to query the full BigQuery ML Reference Guide.

    Before running the setup, ensure your project ID is added in .env file: `"GOOGLE_CLOUD_PROJECT"`.
    Leave the corpus name empty in the .env file: `BQML_RAG_CORPUS_NAME = ''`. The corpus name will be added automatically once it's created.

    To set up the RAG Corpus for your project, run the methods `create_RAG_corpus()` and `ingest_files()` in
    `data_science/utils/reference_guide_RAG.py` by running the below command from the working directory:

    ```bash
    python utils/reference_guide_RAG.py
    ```


4.  **Other Environment Variables:**

    *   `NL2SQL_METHOD`: (Optional) Either `BASELINE` or `CHASE`. Sets the method for SQL Generation. Baseline uses Gemini off-the-shelf, whereas CHASE uses [CHASE-SQL](https://arxiv.org/abs/2410.01943)
    *   `CODE_INTERPRETER_EXTENSION_NAME`: (Optional) The full resource name of
        a pre-existing Code Interpreter extension in Vertex AI. 
        * If you wish to create a new Code Interpreter extension, you can execute `utils/create_code_executor_extension.py`  which will both create a new extension for you and set the environment variable in the `.env` file.


### Example Agent Interaction

Here's a quick example of how a user might interact with the Data Science Multi-Agent System:

> ´**User:** Hi, What data do you have access to?

> **Agent:**  I have access to two tables: `train` and `test`. Both tables contain sticker sales data with the following columns: `id`, `date`, `country`, `store`, `product`, and `num_sold`.

> **User:** I need more details on the train table. What countries exist? How many stores are there?

> **Agent:** Result: The train table contains data from Canada, Finland, Italy, Kenya, Norway, and Singapore. Each of these countries has 3 distinct stores.

>Explanation: The database agent executed a SQL query against the train table to find the distinct countries and the number of distinct stores within each country. The query grouped the results by country and counted the distinct store values.

> **User:** Please generate a plot with total sales per country.

> **Agent:**  Result: A bar plot has been generated showing the total sales per country. The x-axis represents the country, and the y-axis represents the total sales.

> Explanation:
> 1. The database agent was first called to retrieve the total sales per country from the train table.
> 2. The data science agent was then called to generate a bar plot visualizing the results. The plot displays each country on the x-axis and its corresponding total sales on the y-axis.

> **User:** What kinds of forecasting models can I train in BQML?

> **Agent:** You can train several types of forecasting models in BQML, including ARIMA, Exponential Smoothing, and Temporal Fusion Transformer (TFT) models. Would you like to proceed with training a forecasting model?

> **User:** Yes

> **Agent:** Okay, before proceeding, let's confirm the project and dataset ID for BQML. We'll be using the `PROJECT_ID` project and the `forecasting_sticker_sales` dataset. Does that sound right?

> **User:** Yes

> **Agent:** The BQML supports `ARIMA_PLUS`, `ARIMA_PLUS_XREG`.

>To proceed, I need to know which model you'd like to use. I also need to know the following:

>*   The name of the column that provides the time points (e.g., 'date').
>*   The name of the column that contains the data to forecast (e.g., 'num_sold').
>*   Do you want to fit and forecast multiple time series using a single query? If so, what are the ID columns? (e.g., `country`, `store`, `product`)


## Running Evaluations


Evaluation tests assess the overall performance and capabilities of the agent in a holistic manner.

**Run Evaluation Tests:**

    ```bash
    uv run pytest eval
    ```


- This command executes all test files within the `eval/` directory.
- `uv run` ensures that pytest runs within the project's virtual environment.



## Running Tests

Tests assess the overall executability of the agents.

**Test Categories:**

*   **Integration Tests:** These tests verify that the agents can interact correctly with each other and with external services like BigQuery. They ensure that the root agent can delegate tasks to the appropriate sub-agents and that the sub-agents can perform their intended tasks.
*   **Sub-Agent Functionality Tests:** These tests focus on the specific capabilities of each sub-agent (e.g., Database Agent, BQML Agent). They ensure that each sub-agent can perform its intended tasks, such as executing SQL queries or training BQML models.
*   **Environment Query Tests:** These tests verify that the agent can handle queries that are based on the environment.

**Run Tests:**

    ```bash
    uv run pytest tests
    ```

- This command executes all test files within the `tests/` directory.
- `uv run` ensures that pytest runs within the project's virtual environment.



## Deployment on Vertex AI Agent Engine

* Use either the `webui_manager.py` or `interactive_manager.py` to walk you through the deployment to Agent Engine.


## Optimizing and Adjustment Tips

*   **Prompt Engineering:** Refine the prompts for `root_agent`, `bqml_agent`, `db_agent`
    and `ds_agent` to improve accuracy and guide the agents more effectively.
    Experiment with different phrasing and levels of detail.
*   **Extension:** Extend the multi-agent system with your own AgentTools or sub_agents.
    You can do so by adding additional tools and sub_agents to the root agent inside
    `agents/data-science/data_science/agent.py`.
*   **Partial imports:** If you only need certain capabilities inside the multi-agent system,
    e.g. just the data agent, you can import the data_agent as an AgentTool into your own root agent.
*   **Model Selection:** Try different language models for both the top-level
    agent and the sub-agents to find the best performance for your data and
    queries.


## Troubleshooting

*   If you face `500 Internal Server Errors` when running the agent, simply re-run your last command.
    That should fix the issue.
*   If you encounter issues with the code interpreter, review the logs to
    understand the errors. Make sure you're using base-64 encoding for
    files/images if interacting directly with a code interpreter extension
    instead of through the agent's helper functions.
*   If you see errors in the SQL generated, try the following:
    - including clear descriptions in your tables and columns help boost performance
    - if your database is large, try setting up a RAG pipeline for schema linking by storing your table schema details in a vector store