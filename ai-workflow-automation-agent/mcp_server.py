import ast
import operator
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("approved-workflow-tools")
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "workspace" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = BASE_DIR / "workspace" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Retry transient REST failures with bounded exponential backoff.
session = requests.Session()
retry_policy = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=0.4,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
session.mount("https://", HTTPAdapter(max_retries=retry_policy))
session.headers.update({"User-Agent": "WorkflowAutomationAgent/1.0"})

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent is too large.")
        result = _ALLOWED_OPERATORS[type(node.op)](left, right)
        if abs(result) > 1e100:
            raise ValueError("Result is too large.")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only basic arithmetic is allowed; functions and variables are not supported.")


@mcp.tool()
def calculate(expression: str) -> dict:
    """Calculate a basic arithmetic expression. Supports +, -, *, /, //, %, ** and parentheses."""
    if len(expression) > 200:
        raise ValueError("Expression is too long.")
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree.body)
    return {"expression": expression, "result": result}


@mcp.tool()
def fetch_sample_post(post_id: int) -> dict:
    """Fetch a sample post from the public JSONPlaceholder REST API. post_id must be between 1 and 100."""
    if not 1 <= int(post_id) <= 100:
        raise ValueError("post_id must be between 1 and 100.")
    url = f"https://jsonplaceholder.typicode.com/posts/{int(post_id)}"
    response = session.get(url, timeout=(3, 8))
    response.raise_for_status()
    data = response.json()
    return {
        "source": "JSONPlaceholder",
        "id": data.get("id"),
        "title": data.get("title"),
        "body": data.get("body"),
    }


@mcp.tool()
def list_csv_files() -> dict:
    """List CSV files available in the project's workspace/data folder."""
    files = sorted(
        p.name for p in DATA_DIR.glob("*.csv")
        if p.is_file() and p.parent.resolve() == DATA_DIR.resolve()
    )
    return {"directory": "workspace/data", "files": files}


@mcp.tool()
def summarize_csv(filename: str) -> dict:
    """Summarize a CSV in workspace/data: row count, columns, missing values, numeric statistics and sample rows."""
    # Restrict file access to a basename and CSV files within the approved directory.
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".csv"):
        raise ValueError("Provide a CSV filename only, such as sales.csv.")
    path = (DATA_DIR / safe_name).resolve()
    if path.parent != DATA_DIR.resolve() or not path.is_file():
        raise FileNotFoundError(f"{safe_name} was not found in workspace/data. Upload it in the app first.")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("CSV exceeds the 10 MB limit.")
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")
    stats = {}
    for column in numeric.columns[:20]:
        stats[str(column)] = {
            "mean": round(float(numeric[column].mean()), 4) if not numeric[column].isna().all() else None,
            "min": float(numeric[column].min()) if not numeric[column].isna().all() else None,
            "max": float(numeric[column].max()) if not numeric[column].isna().all() else None,
            "sum": float(numeric[column].sum()) if not numeric[column].isna().all() else None,
        }
    return {
        "filename": safe_name,
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "numeric_summary": stats,
        "sample_rows": df.head(5).fillna("").to_dict(orient="records"),
    }


@mcp.tool()
def run_sales_analysis_workflow(
    filename: str,
    sales_column: str = "",
    region_column: str = "",
    product_column: str = "",
) -> dict:
    """Run a multi-step business analysis on a local CSV and save a Markdown report plus anomaly CSV. Automatically detects likely sales, region, and product columns unless names are provided. Use for requests to analyze sales, detect anomalies, compare regions/products, and generate a report."""
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".csv"):
        raise ValueError("Provide a CSV filename only, such as example_sales.csv.")
    path = (DATA_DIR / safe_name).resolve()
    if path.parent != DATA_DIR.resolve() or not path.is_file():
        raise FileNotFoundError(f"{safe_name} was not found in workspace/data. Upload it in the app first.")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("CSV exceeds the 10 MB limit.")

    # Step 1 — load and validate.
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The CSV has no data rows.")
    columns = [str(c) for c in df.columns]
    normalized = {str(c).strip().lower().replace(" ", "_"): str(c) for c in df.columns}

    def choose_column(requested: str, candidates: list[str], numeric_only: bool = False) -> str | None:
        if requested:
            if requested not in columns:
                raise ValueError(f"Column '{requested}' not found. Available columns: {columns}")
            if numeric_only and not pd.api.types.is_numeric_dtype(df[requested]):
                raise ValueError(f"Column '{requested}' must be numeric.")
            return requested
        for candidate in candidates:
            for key, original in normalized.items():
                if key == candidate or candidate in key:
                    if not numeric_only or pd.api.types.is_numeric_dtype(df[original]):
                        return original
        return None

    # Step 2 — identify useful business columns.
    if sales_column and sales_column not in columns:
        raise ValueError(f"Sales column '{sales_column}' not found. Available columns: {columns}")
    if sales_column and not pd.api.types.is_numeric_dtype(df[sales_column]):
        raise ValueError(f"Sales column '{sales_column}' must be numeric.")
    sales_col = choose_column(sales_column, ["sales", "revenue", "total_amount", "total", "amount", "price"], True)
    region_col = choose_column(region_column, ["region", "territory", "city", "location", "state"])
    product_col = choose_column(product_column, ["product", "item", "service", "category"])

    # Step 3 — data quality and numeric profiling.
    missing = {str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0}
    numeric = df.select_dtypes(include="number")
    numeric_profile = {}
    for col in numeric.columns[:20]:
        series = numeric[col].dropna()
        numeric_profile[str(col)] = {
            "sum": round(float(series.sum()), 4) if len(series) else None,
            "mean": round(float(series.mean()), 4) if len(series) else None,
            "min": round(float(series.min()), 4) if len(series) else None,
            "max": round(float(series.max()), 4) if len(series) else None,
        }

    # Step 4 — sales and group comparisons, plus transparent IQR anomaly detection.
    total_sales = average_sales = None
    top_region = top_product = None
    regional_sales = product_sales = {}
    anomalies = pd.DataFrame()
    if sales_col:
        values = pd.to_numeric(df[sales_col], errors="coerce").dropna()
        if len(values):
            total_sales = round(float(values.sum()), 2)
            average_sales = round(float(values.mean()), 2)
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            anomalies = df[pd.to_numeric(df[sales_col], errors="coerce").lt(lower) | pd.to_numeric(df[sales_col], errors="coerce").gt(upper)].copy()
            if region_col:
                grouped = df.groupby(region_col, dropna=False)[sales_col].sum().sort_values(ascending=False)
                regional_sales = {str(k): round(float(v), 2) for k, v in grouped.items()}
                if len(grouped): top_region = {"name": str(grouped.index[0]), "sales": round(float(grouped.iloc[0]), 2)}
            if product_col:
                grouped = df.groupby(product_col, dropna=False)[sales_col].sum().sort_values(ascending=False)
                product_sales = {str(k): round(float(v), 2) for k, v in grouped.items()}
                if len(grouped): top_product = {"name": str(grouped.index[0]), "sales": round(float(grouped.iloc[0]), 2)}

    # Step 5 — save reproducible outputs locally.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(safe_name).stem[:60]
    report_path = REPORT_DIR / f"{stem}_analysis_{stamp}.md"
    anomaly_path = REPORT_DIR / f"{stem}_anomalies_{stamp}.csv"
    report_lines = [
        f"# Sales Analysis Report: {safe_name}", "", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
        "## Workflow summary", "1. Loaded and validated the CSV.", "2. Profiled missing values and numeric fields.",
        "3. Compared sales across detected region/product columns where available.",
        "4. Flagged possible sales outliers using the 1.5×IQR rule.", "5. Saved this report and the anomaly rows locally.", "",
        "## Dataset overview", f"- Rows: {len(df)}", f"- Columns ({len(columns)}): {', '.join(columns)}",
        f"- Missing-value cells: {int(df.isna().sum().sum())}", f"- Columns with missing values: {json.dumps(missing, ensure_ascii=False) if missing else 'None'}", "",
        "## Sales findings", f"- Sales column: `{sales_col}`" if sales_col else "- Sales column: not detected; sales-specific findings unavailable.",
        f"- Total sales: {total_sales}" if total_sales is not None else "- Total sales: unavailable",
        f"- Average sales per row: {average_sales}" if average_sales is not None else "- Average sales per row: unavailable",
        f"- Top region: {top_region['name']} ({top_region['sales']})" if top_region else "- Top region: unavailable (no sales and region columns detected)",
        f"- Top product/category: {top_product['name']} ({top_product['sales']})" if top_product else "- Top product/category: unavailable (no sales and product/category columns detected)", "",
        "## Regional sales", "| Region | Sales |", "|---|---:|"
    ]
    report_lines += [f"| {k} | {v} |" for k, v in regional_sales.items()] or ["| Not available | — |"]
    report_lines += ["", "## Product/category sales", "| Product/category | Sales |", "|---|---:|"]
    report_lines += [f"| {k} | {v} |" for k, v in product_sales.items()] or ["| Not available | — |"]
    report_lines += ["", "## Numeric profile", "| Column | Sum | Mean | Min | Max |", "|---|---:|---:|---:|---:|"]
    for col, stats in numeric_profile.items():
        report_lines.append(f"| {col} | {stats['sum']} | {stats['mean']} | {stats['min']} | {stats['max']} |")
    if not numeric_profile: report_lines.append("| No numeric columns | — | — | — | — |")
    report_lines += ["", "## Potential anomalies", f"- Rows flagged by 1.5×IQR: {len(anomalies)}", "- This is a statistical flag for review, not proof of an error.", ""]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    if not anomalies.empty:
        anomalies.to_csv(anomaly_path, index=False)
        anomaly_file = anomaly_path.name
    else:
        anomaly_file = None

    return {
        "workflow": ["load_validate_csv", "profile_data_quality", "analyze_sales_and_groups", "detect_iqr_outliers", "save_report"],
        "filename": safe_name, "rows": int(len(df)), "columns": columns,
        "missing_values": missing, "numeric_profile": numeric_profile,
        "sales_column": sales_col, "total_sales": total_sales, "average_sales_per_row": average_sales,
        "region_column": region_col, "regional_sales": regional_sales, "top_region": top_region,
        "product_column": product_col, "product_sales": product_sales, "top_product": top_product,
        "anomaly_count": int(len(anomalies)), "anomaly_file": anomaly_file,
        "report_file": report_path.name, "report_path": "workspace/reports/" + report_path.name,
        "notes": ["Column detection is heuristic; pass explicit column names if detection is incorrect.", "Outliers are flagged with the 1.5×IQR rule and should be reviewed, not automatically removed."]
    }


@mcp.tool()
def list_analysis_reports() -> dict:
    """List Markdown analysis reports saved in workspace/reports."""
    files = sorted(p.name for p in REPORT_DIR.glob("*.md") if p.is_file())
    return {"directory": "workspace/reports", "reports": files}


if __name__ == "__main__":
    mcp.run(transport="stdio")

