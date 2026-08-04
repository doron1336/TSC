import pandas as pd
import re
import sys

def parse_score_duration(cell_value):
    """
    Parse a cell containing 'Score: X ± Y Duration: A ± B'
    Returns tuple (score_mean, score_std, duration_mean, duration_std)
    """
    if pd.isna(cell_value) or not isinstance(cell_value, str):
        return None, None, None, None

    # Extract score
    score_match = re.search(r'Score:\s*([\d.]+)\s*±\s*([\d.]+)', cell_value)
    duration_match = re.search(r'Duration:\s*([\d.]+)\s*±\s*([\d.]+)', cell_value)

    score_mean = float(score_match.group(1)) if score_match else None
    score_std = float(score_match.group(2)) if score_match else None
    duration_mean = float(duration_match.group(1)) if duration_match else None
    duration_std = float(duration_match.group(2)) if duration_match else None

    return score_mean, score_std, duration_mean, duration_std


def format_score_only(cell_value):
    """
    Extract and format just the score as 'mean ± std'
    """
    score_mean, score_std, _, _ = parse_score_duration(cell_value)
    if score_mean is not None:
        return f"{score_mean:.4f} $\\pm$ {score_std:.4f}"
    return ""


def csv_to_latex_table(csv_path, output_path, include_columns='all', score_only=True):
    """
    Convert CSV to LaTeX table format

    Parameters:
    - csv_path: Path to input CSV file
    - output_path: Path to output .tex file
    - include_columns: List of column names to include, or 'all' for all columns
    - score_only: If True, only show scores (not durations)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Determine which columns to include
    if include_columns == 'all':
        cols_to_include = df.columns.tolist()
    else:
        cols_to_include = include_columns

    # Filter dataframe
    df_filtered = df[cols_to_include]

    # Process score columns (those with numbers as column names)
    feature_cols = ['10', '30', '50', '70', '90', '110', '130', '150', '170', '190', '210']

    # Create LaTeX table string
    latex_lines = []

    # Begin table
    num_cols = len(cols_to_include)
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Results across different feature selection methods}")
    latex_lines.append("\\label{tab:results}")

    # Column specification - adjust based on needs
    col_spec = "l" * num_cols  # All left-aligned, can customize
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    # Header row
    header_row = " & ".join([str(col) for col in cols_to_include]) + " \\\\"
    latex_lines.append(header_row)
    latex_lines.append("\\midrule")

    # Data rows
    for idx, row in df_filtered.iterrows():
        row_data = []
        for col in cols_to_include:
            cell_value = row[col]

            # Check if this is a score column
            if col in feature_cols:
                if score_only:
                    formatted_value = format_score_only(cell_value)
                else:
                    formatted_value = str(cell_value)
            elif col == 'All_features':
                # Format as a regular float
                try:
                    formatted_value = f"{float(cell_value):.4f}"
                except:
                    formatted_value = str(cell_value)
            else:
                formatted_value = str(cell_value)

            row_data.append(formatted_value)

        latex_lines.append(" & ".join(row_data) + " \\\\")

    # End table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    latex_content = "\n".join(latex_lines)
    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"\nLaTeX table written to: {output_path}")
    print(f"Total rows: {len(df_filtered)}")

    return latex_content


def create_condensed_table(csv_path, output_path):
    """
    Create a more condensed table showing only key feature counts
    """
    df = pd.read_csv(csv_path)

    # Select key columns
    key_feature_cols = ['Algo_name', 'Dataset_name', 'All_features', '10', '50', '110', '170', '210']

    df_condensed = df[key_feature_cols].copy()

    # Format score columns
    for col in ['10', '50', '110', '170', '210']:
        df_condensed[col] = df_condensed[col].apply(format_score_only)

    # Format All_features
    df_condensed['All_features'] = df_condensed['All_features'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")

    # Create LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Performance comparison across algorithms and feature counts}")
    latex_lines.append("\\label{tab:results_condensed}")
    latex_lines.append("\\begin{tabular}{llcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Algorithm & Dataset & All & 10 & 50 & 110 & 170 & 210 \\\\")
    latex_lines.append("\\midrule")

    current_dataset = None
    for idx, row in df_condensed.iterrows():
        # Add separator between datasets
        if current_dataset is not None and row['Dataset_name'] != current_dataset:
            latex_lines.append("\\midrule")
        current_dataset = row['Dataset_name']

        row_str = f"{row['Algo_name']} & {row['Dataset_name']} & {row['All_features']} & {row['10']} & {row['50']} & {row['110']} & {row['170']} & {row['210']} \\\\"
        latex_lines.append(row_str)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_content = "\n".join(latex_lines)
    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"\nCondensed LaTeX table written to: {output_path}")
    return latex_content


if __name__ == "__main__":
    csv_path = "/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018/HAR_datasets/scores_dataframe_fold_final.csv"

    # Create full table with scores only
    output_path_full = "/Users/doron/Desktop/personal/thesis/TSC/results_table_full.tex"
    csv_to_latex_table(csv_path, output_path_full, score_only=True)

    # Create condensed table
    output_path_condensed = "/Users/doron/Desktop/personal/thesis/TSC/results_table_condensed.tex"
    create_condensed_table(csv_path, output_path_condensed)

    print("\n" + "="*50)
    print("Generated two LaTeX tables:")
    print(f"1. Full table: {output_path_full}")
    print(f"2. Condensed table: {output_path_condensed}")
    print("="*50)