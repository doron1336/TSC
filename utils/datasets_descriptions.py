import requests
from bs4 import BeautifulSoup
from enum import Enum

# List of datasets you want to fetch
datasets = ["Worms", "BeetleFly", "Car", "Coffee"]  # Replace with your datasets

# Base URL for dataset descriptions
BASE_URL = "https://www.timeseriesclassification.com/description.php?Dataset={}"

class DatasetDescriptions(Enum):
    pass

def fetch_description(dataset_name: str) -> tuple[str, str]:
    """Fetch the description of a dataset from the website."""
    url = BASE_URL.format(dataset_name)
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table', class_='table-bordered')
        table_1 = soup.find('table', {'class': 'table table-bordered'})
        rows = table_1.find_all('tr')
        cells = rows[-1].find_all('td')
        return  tables[1].text.strip(), cells[-1].get_text(strip=True)

def populate_enum():
    """Populate the DatasetDescriptions Enum with dataset names and descriptions."""
    enum_members = {}
    for dataset in datasets:
        description, type = fetch_description(dataset)
        enum_members[dataset.upper()] = description

    # Dynamically create Enum
    return Enum('DatasetDescriptions', enum_members)

if __name__ == "__main__":
    DatasetDescriptions = populate_enum()

    # Example: Access a dataset's description
    print(DatasetDescriptions.WORMS.value)
