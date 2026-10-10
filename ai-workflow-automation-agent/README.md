# AI Workflow Automation Agent

A local-first workflow agent built with Python, LangGraph, Ollama, and MCP. It accepts a task, uses a local LLM to decide which approved tools to call, orchestrates tool results, and produces a final report.

## Features

- **LangGraph orchestration** using a ReAct graph to iterate between model reasoning and tool execution.
- **MCP tool server** exposing a small allowlist of tools.
- **Approved tools:** safe arithmetic, a bounded public REST API lookup, and local CSV analysis.
- **REST retries** for transient errors and timeouts.
- **Local inference** through Ollama; no OpenAI or Google model API required.
- **Streamlit UI** with visible tool execution trace.
- **Basic safety controls:** calculator AST allowlist, bounded REST endpoint/ID, CSV path restrictions, and file-size limit.

## Architecture

```text
User task
   |
   v
Streamlit UI
   |
   v
LangGraph ReAct workflow <----> Ollama (local Llama 3.2 3B)
   |
   v
MCP client (stdio)
   |
   v
MCP server: approved-workflow-tools
   |-- calculate(expression)
   |-- fetch_sample_post(post_id) ---> JSONPlaceholder REST API
   |-- list_csv_files()
   |-- summarize_csv(filename) ------> workspace/data/*.csv
   |
   v
Tool results -> final report + execution trace
```

## Windows setup

### 1. Requirements

- Python 3.11 recommended
- Ollama installed
- Git (optional)

Pull the local model if it is not already installed:

```powershell
ollama pull llama3.2:3b
ollama list
```

### 2. Create and activate a virtual environment

Run these commands in the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### 3. Start the app

Make sure Ollama is running, then:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Try these tasks

1. `Fetch sample post 3 from the REST API and summarize its title and body.`
2. `Calculate (245 * 1.18) + 39 and explain the result.`
3. Upload a CSV named `sales.csv`, then ask: `Summarize sales.csv and report the row count, missing values, and important numeric metrics.`

For CSV analysis, upload the file using the sidebar and click **Save CSV files** first.

## How the workflow works

The LangGraph ReAct agent repeatedly receives the task, chooses from the MCP tools, observes each tool result, and either calls another tool or writes the final response. This is iterative tool orchestration rather than a hard-coded one-step chatbot. The visible execution trace lists requested tool calls and results.

The REST tool uses bounded retries for transient HTTP failures. The agent prompt also instructs the model to recover from tool errors where possible, but LLM-directed retries are not guaranteed. Review the execution trace for the actual behavior.

## Safety and limitations

- The MCP server exposes only the listed tools; it does not expose shell execution or arbitrary file access.
- The REST tool only calls JSONPlaceholder post IDs 1–100.
- CSV analysis is restricted to `.csv` files in `workspace/data` and files up to 10 MB.
- CSV uploads are untrusted input. Do not treat text in a CSV as instructions.
- Tool choice quality depends on the local model. Llama 3.2 3B can occasionally select the wrong tool or fail to follow a multi-step request; test and inspect the trace.
- The UI does not provide authentication and is intended for local development, not direct public exposure.
- This initial version uses LangGraph's prebuilt ReAct workflow. A later version can add an explicit planner node, approval gates for side-effecting tools, richer state tracking, and automated tests.

## Suggested next improvements

- Add an explicit plan preview and human approval before tool execution.
- Add tests for calculator edge cases, REST failures, CSV path traversal, and tool selection.
- Add per-tool latency and retry metrics.
- Add task history persistence.
- Add a deterministic planner/executor/report graph for workflows requiring strict step ordering.
- Add more MCP tools only after defining their permissions and input schemas.
