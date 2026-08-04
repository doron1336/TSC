import pandas as pd
import re

# Read the CSV file
df = pd.read_csv('/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018/HAR_datasets/scores_dataframe_fold_final.csv')

# Remove the Type column since all datasets are HAR
df_without_type = df.drop(columns=['Type'])


def parse_score_only(cell_value):
    """Extract only the score (mean ± std) from cells with 'Score: X ± Y Duration: A ± B'"""
    if pd.isna(cell_value) or not isinstance(cell_value, str):
        return cell_value

    # Check if it's a score/duration string
    score_match = re.search(r'Score:\s*([\d.]+)\s*±\s*([\d.]+)', cell_value)
    if score_match:
        score_mean = float(score_match.group(1))
        score_std = float(score_match.group(2))
        # Use proper math mode with $...$
        return f"${score_mean:.4f} \\pm {score_std:.4f}$"
    return cell_value


def parse_duration_only(cell_value):
    """Extract only the duration (mean ± std) from cells with 'Score: X ± Y Duration: A ± B'"""
    if pd.isna(cell_value) or not isinstance(cell_value, str):
        return cell_value

    # Check if it's a score/duration string
    duration_match = re.search(r'Duration:\s*([\d.]+)\s*±\s*([\d.]+)', cell_value)
    if duration_match:
        duration_mean = float(duration_match.group(1))
        duration_std = float(duration_match.group(2))
        # Use proper math mode with $...$
        return f"${duration_mean:.4f} \\pm {duration_std:.4f}$"
    return cell_value


def format_value(x):
    """Format different types of values for LaTeX"""
    if isinstance(x, str):
        # First check if it's a score/duration string
        parsed = parse_score_only(x)
        if parsed != x:  # If it was parsed
            return parsed
        # Otherwise format as texttt
        return f"\\texttt{{{x}}}"
    elif isinstance(x, float):
        return f"{x:.4f}"
    else:
        return x


df_tex = df_without_type.applymap(format_value)

# Creating a well-formatted table with cells
output_file = '/Users/doron/Desktop/personal/thesis/TSC/utils/HAR_table.tex'
with open(output_file, 'w') as f:
    num_cols = len(df_tex.columns)

    # Create column specification with vertical lines for cells
    # Use double lines to separate logical sections: Algo | Dataset || All_features || feature counts
    col_alignment = "|l|l||c||" + "|".join(["c"] * (num_cols - 3)) + "|"

    f.write("\\begin{sidewaystable}[p]\n")
    f.write("\\centering\n")
    f.write("\\small\n")
    f.write("\\setlength{\\tabcolsep}{3pt}\n")
    f.write("\\begin{tabular}{" + col_alignment + "}\n")
    f.write("\\hline\n")

    # Write column headers with better formatting
    headers = []
    for col in df_tex.columns:
        if col in ['Algo_name', 'Dataset_name', 'All_features']:
            headers.append(f"\\textbf{{{col.replace('_', ' ')}}}")
        else:
            headers.append(f"\\textbf{{{col}}}")

    header_row = " & ".join(headers) + " \\\\\n"
    f.write(header_row)
    f.write("\\hline\n")
    f.write("\\hline\n")  # Double line after header

    # Group rows by dataset for better readability
    current_dataset = None
    for idx, row in df_tex.iterrows():
        dataset = row['Dataset_name']

        # Add separator between different datasets
        if current_dataset is not None and dataset != current_dataset:
            f.write("\\hline\n")

        current_dataset = dataset

        row_str = " & ".join([str(val) for val in row.values]) + " \\\\\n"
        f.write(row_str)
        f.write("\\hline\n")

    # End the table
    f.write("\\end{tabular}\n")
    f.write("\\caption{Performance comparison of feature selection algorithms on HAR datasets. Scores shown as mean $\\pm$ standard deviation across cross-validation folds.}\n")
    f.write("\\label{tab:har_results}\n")
    f.write("\\end{sidewaystable}\n")

print(f"LaTeX SCORES table successfully generated!")
print(f"Output file: {output_file}")
print(f"Total rows: {len(df_tex)}")
print(f"Total columns: {len(df_tex.columns)}")

# Now create the DURATION table
def format_value_duration(x):
    """Format different types of values for LaTeX - Duration version"""
    if isinstance(x, str):
        # First check if it's a score/duration string
        parsed = parse_duration_only(x)
        if parsed != x:  # If it was parsed
            return parsed
        # Otherwise format as texttt
        return f"\\texttt{{{x}}}"
    elif isinstance(x, float):
        return f"{x:.4f}"
    else:
        return x

df_tex_duration = df_without_type.applymap(format_value_duration)

output_file_duration = '/Users/doron/Desktop/personal/thesis/TSC/utils/HAR_table_duration.tex'
with open(output_file_duration, 'w') as f:
    num_cols = len(df_tex_duration.columns)

    # Create column specification with vertical lines for cells
    col_alignment = "|l|l||c||" + "|".join(["c"] * (num_cols - 3)) + "|"

    f.write("\\begin{sidewaystable}[p]\n")
    f.write("\\centering\n")
    f.write("\\small\n")
    f.write("\\setlength{\\tabcolsep}{3pt}\n")
    f.write("\\begin{tabular}{" + col_alignment + "}\n")
    f.write("\\hline\n")

    # Write column headers with better formatting
    headers = []
    for col in df_tex_duration.columns:
        if col in ['Algo_name', 'Dataset_name', 'All_features']:
            headers.append(f"\\textbf{{{col.replace('_', ' ')}}}")
        else:
            headers.append(f"\\textbf{{{col}}}")

    header_row = " & ".join(headers) + " \\\\\n"
    f.write(header_row)
    f.write("\\hline\n")
    f.write("\\hline\n")  # Double line after header

    # Group rows by dataset for better readability
    current_dataset = None
    for idx, row in df_tex_duration.iterrows():
        dataset = row['Dataset_name']

        # Add separator between different datasets
        if current_dataset is not None and dataset != current_dataset:
            f.write("\\hline\n")

        current_dataset = dataset

        row_str = " & ".join([str(val) for val in row.values]) + " \\\\\n"
        f.write(row_str)
        f.write("\\hline\n")

    # End the table
    f.write("\\end{tabular}\n")
    f.write("\\caption{Duration comparison of feature selection algorithms on HAR datasets. Durations (in seconds) shown as mean $\\pm$ standard deviation across cross-validation folds.}\n")
    f.write("\\label{tab:har_duration}\n")
    f.write("\\end{sidewaystable}\n")

print(f"\nLaTeX DURATION table successfully generated!")
print(f"Output file: {output_file_duration}")

print("\n" + "="*60)
print("INSTRUCTIONS FOR OVERLEAF:")
print("="*60)
print("1. Add to your preamble:")
print("   \\usepackage{rotating}")
print("")
print("2. Include both tables in your document:")
print("   \\input{HAR_table.tex}  % Scores table")
print("   \\input{HAR_table_duration.tex}  % Duration table")
print("")
print("Both tables are self-contained with sidewaystable environment!")
print("="*60)